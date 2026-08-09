"""Full-screen chat TUI (opencode/Freebuff-style) built on prompt_toolkit.

Layout (top → bottom):
    header     — fixed: session + repo + model
    transcript — scrollable, auto-pins to the bottom
    input bar  — FIXED: ``❯ `` prompt with ``/`` command and ``@file`` mentions
    status bar — fixed: model · tokens · hints

The answer pipeline (RAG + tools + LLM) runs in a background thread and
streams into the transcript via ``Application.invalidate()`` (thread-safe),
so the input bar stays responsive and pinned while the answer is generated.
"""

from __future__ import annotations

import io
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI, FormattedText, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import D
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.widgets import TextArea

from . import qa, ui
from . import config as config_mod
from .tools import TEXT_EXTS, is_ignored

# ---------------------------------------------------------------------------
# @-mention parsing
# ---------------------------------------------------------------------------

MENTION_RE = re.compile(r"@([A-Za-z0-9_./\\-]+)")
MAX_ATTACH_CHARS = 8000
MAX_ATTACH_LINES = 200

COMMANDS = [
    "/help",
    "/status",
    "/model",
    "/tools",
    "/index",
    "/reindex",
    "/clear",
    "/exit",
    "/quit",
]

# Quiet typewriter settings: the answer is revealed at a steady, readable
# pace (~120 chars/s) instead of being dumped at once.
TYPE_STEP = 3
TYPE_TICK = 0.025


def parse_mentions(text: str, project_root: Path) -> Tuple[str, List[Dict[str, str]]]:
    """Expand ``@path`` tokens into ``(clean_text, attachments)``.

    Each ``@relative/path.py`` token is resolved against ``project_root``;
    existing text files are read (truncated) into attachments and the token is
    replaced by the bare path in the returned text, so the model sees exactly
    what was referenced.  Tokens that do not resolve to a file are left as-is.
    """
    root = Path(project_root).resolve()
    attachments: List[Dict[str, str]] = []
    seen = set()

    def _replace(m: re.Match) -> str:
        raw = m.group(1)
        try:
            target = (root / raw).resolve()
            target.relative_to(root)  # reject path escapes
        except (ValueError, OSError):
            return m.group(0)
        if not target.is_file() or target.suffix.lower() not in TEXT_EXTS:
            return m.group(0)
        key = str(target)
        if key in seen:
            return raw  # already attached once
        seen.add(key)
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return m.group(0)
        lines = content.splitlines()
        truncated = False
        if len(lines) > MAX_ATTACH_LINES:
            lines = lines[:MAX_ATTACH_LINES]
            truncated = True
        content = "\n".join(lines)
        if len(content) > MAX_ATTACH_CHARS:
            content = content[:MAX_ATTACH_CHARS]
            truncated = True
        if truncated:
            content += "\n… (truncated)"
        attachments.append({"path": raw, "text": content})
        return raw  # keep the path (without @) in the question text

    clean = MENTION_RE.sub(_replace, text)
    return clean, attachments


# ---------------------------------------------------------------------------
# Completion: /commands and @files
# ---------------------------------------------------------------------------


def _fuzzy_match(query: str, text: str) -> Optional[tuple]:
    """Subsequence fuzzy match; returns ``(score, spans)`` or ``None``.

    ``spans`` is a list of ``[start, end)`` ranges into ``text`` covering the
    matched characters (used to highlight the picker row).  Bonuses: matches
    inside the basename, at word boundaries (``/``, ``.``, ``_``, ``-``),
    contiguous runs, and earlier positions.  Matching is case-insensitive,
    but spans index the ORIGINAL string so highlighting is always correct.
    """
    q = query.lower()
    t = text.lower()
    if not q:
        return (0.0, [])
    score = 0.0
    spans: List[list] = []
    last = -1
    run = 0
    basename = t.rsplit("/", 1)[-1]
    for ch in q:
        idx = t.find(ch, last + 1)
        if idx == -1:
            return None
        if last >= 0 and idx == last + 1:
            run += 1
            score += run * 2  # contiguous run bonus
            spans[-1][1] = idx + 1
        else:
            run = 0
            spans.append([idx, idx + 1])
        if idx == 0 or t[idx - 1] in "/._-":
            score += 3  # word-boundary bonus
        score += 1
        last = idx
    if basename.startswith(q):
        score += 8
    elif q in basename:
        score += 4
    # tiny preference for shorter paths (stable tie-break)
    score += 1.0 / (1.0 + len(t) * 0.01)
    return (score, spans)


def _highlight_matches(text: str, spans: List[list]) -> List[Any]:
    """Styled fragments for the picker from match spans (cyan/bold)."""
    frags: List[Any] = []
    last = 0
    for start, end in spans:
        if start > last:
            frags.append(("", text[last:start]))
        frags.append(("bold cyan", text[start:end]))
        last = end
    if last < len(text):
        frags.append(("", text[last:]))
    return frags


class ChatCompleter(Completer):
    """Tab-completion for ``/commands`` and fuzzy ``@file`` mentions.

    Typing ``@`` opens a fuzzy file picker: the typed query is matched as a
    subsequence against repo-relative paths (fzf-style scoring), matched
    characters are highlighted, and each candidate shows its line count so
    you know the size of what you are about to attach.
    """

    MAX_MENTIONS = 15

    def __init__(self, project_root: Path) -> None:
        self.root = Path(project_root)
        self._file_cache: Optional[List[str]] = None
        self._line_cache: Dict[str, int] = {}

    def _files(self) -> List[str]:
        if self._file_cache is None:
            files = []
            for p in sorted(self.root.rglob("*")):
                if p.is_dir():
                    continue
                if is_ignored(p.parts):
                    continue
                if p.suffix.lower() not in TEXT_EXTS:
                    continue
                files.append(str(p.relative_to(self.root)).replace("\\", "/"))
            self._file_cache = files
        return self._file_cache

    def _line_count(self, rel: str) -> int:
        """Count lines of a repo-relative path, cached."""
        cached = self._line_cache.get(rel)
        if cached is not None:
            return cached
        count = 0
        try:
            with (self.root / rel).open("r", encoding="utf-8", errors="replace") as fh:
                count = sum(1 for _ in fh)
        except OSError:
            count = 0
        self._line_cache[rel] = count
        return count

    def _rank_files(self, query: str) -> List[Any]:
        """All repo files sorted by fuzzy relevance to ``query``.

        Returns ``(rel, spans)`` pairs so the picker can highlight exactly
        the characters that matched.
        """
        scored = []
        for rel in self._files():
            matched = _fuzzy_match(query, rel)
            if matched is not None:
                score, spans = matched
                scored.append((rel, score, spans))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [(rel, spans) for rel, _score, spans in scored]

    def get_completions(
        self, document: Document, complete_event: Any
    ) -> Any:
        text_before = document.text_before_cursor
        # /command palette
        if text_before.startswith("/"):
            for cmd in COMMANDS:
                if cmd.startswith(text_before) and len(cmd) > len(text_before):
                    yield Completion(cmd, start_position=-len(text_before))
            return
        # @file fuzzy picker
        m = re.search(r"@([\w./\\-]*)$", text_before)
        if m:
            query = m.group(1)
            for rel, spans in self._rank_files(query)[: self.MAX_MENTIONS]:
                lines = self._line_count(rel)
                yield Completion(
                    "@" + rel,
                    start_position=-len(m.group(0)),
                    display=_highlight_matches(rel, spans),
                    display_meta=f"{lines:,} lines",
                )


# ---------------------------------------------------------------------------
# Markdown → prompt_toolkit fragments
# ---------------------------------------------------------------------------


def _md_to_ansi(text: str) -> str:
    """Render markdown to ANSI via rich (256-color, no wrapping)."""
    from rich.console import Console as RichConsole

    buf = io.StringIO()
    rcon = RichConsole(
        file=buf,
        force_terminal=True,
        legacy_windows=False,
        soft_wrap=True,
        color_system="256",
    )
    rcon.print(ui.render_markdown(text))
    return buf.getvalue()


def _ansi_fragment(text: str) -> Any:
    return ANSI(_md_to_ansi(text))


# ---------------------------------------------------------------------------
# The chat application
# ---------------------------------------------------------------------------


class ChatTUI:
    """Full-screen chat app with a fixed bottom input bar."""

    def __init__(
        self,
        project_root: Path,
        session: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        resolve_key: Optional[Callable[[str], Optional[str]]] = None,
        on_answer: Optional[Callable[[Dict[str, Any], str, qa.Answer], None]] = None,
        on_command: Optional[Callable[[str, "ChatTUI"], Optional[str]]] = None,
        build_answer: Optional[Callable[[str, Path], str]] = None,
        on_exit: Optional[Callable[[Dict[str, Any]], None]] = None,
        app_input: Any = None,
        app_output: Any = None,
    ) -> None:
        self.root = Path(project_root)
        self.session = session
        self.cfg = cfg
        self._resolve_key = resolve_key or (
            lambda p: config_mod.get_api_key(p, cfg)
        )
        self._on_answer = on_answer
        self._on_command = on_command
        self._build_answer = build_answer or (lambda q, r: "")
        self._on_exit = on_exit

        self._blocks: List[Any] = []  # transcript blocks (fragments)
        self._pending: str = ""       # full buffered streaming answer text
        self._revealed: int = 0       # chars revealed by the typewriter
        self._typewriter_stop = threading.Event()
        self._typewriter_thread: Optional[threading.Thread] = None
        self._tool_index: Optional[int] = None
        self._work_index: Optional[int] = None  # transient "working…" line
        self._busy = False
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._pinned = True
        self._vertical_scroll = 0

        primary = (cfg.get("llm") or {}).get("primary") or {}
        self._model = primary.get("model")

        self._app: Optional[Application] = None
        self.input_area = self._build_input()
        self.transcript_window = self._build_transcript()
        self._build_app(app_input=app_input, app_output=app_output)

    # -- construction ------------------------------------------------------

    def _build_input(self) -> TextArea:
        return TextArea(
            prompt="❯ ",
            multiline=False,
            height=D.exact(3),
            style="class:input",
            completer=ChatCompleter(self.root),
            complete_while_typing=True,
            accept_handler=self._on_submit,
            focus_on_click=True,
        )

    def _build_transcript(self) -> Window:
        return Window(
            content=FormattedTextControl(self._render_transcript, focusable=True),
            wrap_lines=True,
            always_hide_cursor=True,
            right_margins=[ScrollbarMargin()],
            get_vertical_scroll=self._get_scroll,
            allow_scroll_beyond_bottom=True,
        )

    def _build_app(self, app_input: Any = None, app_output: Any = None) -> None:
        header = Window(
            content=FormattedTextControl(self._render_header),
            height=D.exact(1),
            style="class:header",
        )
        status = Window(
            content=FormattedTextControl(self._render_status),
            height=D.exact(1),
            style="class:status",
        )
        body = HSplit(
            [
                header,
                self.transcript_window,  # weight 1: takes remaining space
                self.input_area,         # fixed at the bottom
                status,
            ]
        )

        kb = KeyBindings()

        @kb.add("c-c")
        def _ctrl_c(event: Any) -> None:
            if self._busy:
                self._cancel.set()
                self._typewriter_stop.set()
                self._append(("dim", "\n⏹ interrupted. Ask away or /exit.\n"))
            else:
                event.app.exit()

        @kb.add("c-d")
        def _ctrl_d(event: Any) -> None:
            event.app.exit()

        # Transcript scrolling.  The default prompt_toolkit bindings for
        # up/down are no-ops (``_ignore``) and the mouse wheel feeds an
        # up/down keypress, so these fire both from keys and the wheel — no
        # focus requirement, since the input bar is a single line and cannot
        # move a cursor vertically anyway.
        @kb.add("pageup")
        def _pgup(event: Any) -> None:
            self._bump_scroll(+10)

        @kb.add("pagedown")
        def _pgdown(event: Any) -> None:
            self._bump_scroll(-10)

        @kb.add("up")
        def _up(event: Any) -> None:
            self._bump_scroll(+1)

        @kb.add("down")
        def _down(event: Any) -> None:
            self._bump_scroll(-1)

        @kb.add("end")
        def _pin(event: Any) -> None:
            self._pinned = True
            self._invalidate()

        self._app = Application(
            layout=Layout(body, focused_element=self.input_area),
            key_bindings=kb,
            full_screen=True,
            mouse_support=True,
            input=app_input,
            output=app_output,
        )

    # -- rendering ---------------------------------------------------------

    def _get_scroll(self) -> int:
        return 10**9 if self._pinned else self._vertical_scroll

    def _bump_scroll(self, delta: int) -> None:
        info = self.transcript_window.render_info
        if info is None:
            return
        max_scroll = max(0, info.content_height - info.window_height)
        if self._pinned:
            self._pinned = False
            self._vertical_scroll = max_scroll
        self._vertical_scroll = min(
            max(0, self._vertical_scroll + delta), max_scroll
        )
        self._invalidate()

    def _render_header(self) -> Any:
        repo = self.session.get("repo_name") or Path(
            self.session.get("repo_path", ".")
        ).name
        sid = self.session.get("session_id", "")
        frags = [("bold #ff8700", "🐾 Panther"), ("", "  "), ("cyan", sid), ("", f" · {repo}")]
        if self._model:
            frags += [("", "  "), ("bold #ffaf00", self._model)]
        return frags

    def _render_status(self) -> Any:
        stats = self.session.get("stats", {})
        tokens = stats.get("total_tokens_used", 0)
        busy = " · working…" if self._busy else ""
        frags = [("dim", "model "), ("bold #ffaf00", self._model or "?")]
        frags += [("", f" · {tokens:,} tokens{busy}")]
        frags += [("", "   "), ("dim", "@file · /help · tab=complete")]
        return frags

    def _render_transcript(self) -> Any:
        with self._lock:
            blocks = list(self._blocks)
            pending = self._pending
        frags: List[Any] = []
        for b in blocks:
            # Each block is a normalized fragment list (see _append).
            frags.extend(b)
        if pending:
            cursor = "▌" if ui._supports_unicode() else "|"
            shown = pending[: self._revealed]
            # ANSI is not iterable — convert via to_formatted_text (same
            # path as _normalize) so the live streaming render never crashes.
            frags.extend(to_formatted_text(_ansi_fragment(shown + " " + cursor)))
        return FormattedText(frags)

    def _invalidate(self) -> None:
        app = self._app
        if app is not None:
            app.invalidate()

    # -- quiet typewriter (reveal buffered text at a steady pace) ----------

    def _start_typewriter(self) -> None:
        """Ensure the reveal thread is running (idempotent)."""
        if self._typewriter_thread is not None and self._typewriter_thread.is_alive():
            return
        self._typewriter_stop = threading.Event()
        self._typewriter_thread = threading.Thread(
            target=self._typewriter_loop, daemon=True
        )
        self._typewriter_thread.start()

    def _typewriter_loop(self) -> None:
        """Advance the revealed cursor at a quiet, steady pace."""
        while not self._typewriter_stop.is_set():
            advanced = False
            with self._lock:
                if self._revealed < len(self._pending):
                    self._revealed = min(
                        len(self._pending), self._revealed + TYPE_STEP
                    )
                    advanced = True
            if advanced:
                self._invalidate()
            time.sleep(TYPE_TICK)

    def _drain_typewriter(self, timeout: float = 10.0) -> None:
        """Let the reveal catch up to the full answer, then stop the thread.

        Called on the worker thread once the model stream ends, so the tail
        of the answer finishes typing out (rather than popping in) before the
        text is committed as a block.  Falls back to completing instantly
        after ``timeout`` so a very long answer can never stall the UI.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._revealed >= len(self._pending):
                    break
            time.sleep(0.01)
        self._typewriter_stop.set()
        thread = self._typewriter_thread
        self._typewriter_thread = None
        if thread is not None:
            thread.join(timeout=1)
        with self._lock:
            self._revealed = len(self._pending)
        self._invalidate()

    # -- transcript mutation (thread-safe) ---------------------------------

    @staticmethod
    def _normalize(fragment: Any) -> List[Any]:
        """Wrap a fragment (str, ANSI, HTML, tuple, list) as a fragment list."""
        if isinstance(fragment, list):
            return fragment
        if isinstance(fragment, tuple):
            return [fragment]
        return list(to_formatted_text(fragment, style=""))

    def _append(self, fragment: Any) -> None:
        with self._lock:
            self._blocks.append(self._normalize(fragment))
        self._invalidate()

    def _append_user(self, text: str) -> None:
        self._append(("bold cyan", f"❯ {text}\n"))
        self._append(("", "\n"))

    def _append_working(self) -> None:
        """Show a transient working line so the UI never looks frozen."""
        with self._lock:
            self._work_index = len(self._blocks)
            self._blocks.append([("dim", "  🤔 working…\n")])
        self._invalidate()

    def _clear_working(self) -> None:
        with self._lock:
            idx = self._work_index
            self._work_index = None
            if idx is not None and idx < len(self._blocks):
                self._blocks[idx] = [("", "")]
        self._invalidate()

    def _append_tool_start(self, name: str, args: Dict[str, Any]) -> None:
        label = _tool_label(name, args)
        with self._lock:
            self._tool_index = len(self._blocks)
            self._blocks.append([("cyan", f"  🔧 {label}…\n")])
        self._invalidate()

    def _append_tool_done(self, event: Any) -> None:
        marker = "✓" if event.ok else "✗"
        color = "green" if event.ok else "red"
        line = f"  {marker} {event.describe()}\n"
        with self._lock:
            idx = self._tool_index
            self._tool_index = None
            if idx is not None and idx < len(self._blocks):
                self._blocks[idx] = [(color, line)]
            else:
                self._blocks.append([(color, line)])
        self._invalidate()

    def _append_refs(self, chunks: Optional[List[Dict[str, Any]]]) -> None:
        refs = ui.render_chunk_refs(chunks, plain=True) if chunks else None
        if refs:
            self._append(("dim", f"\nReferenced:\n{refs}\n"))

    def _append_footer(self, answer: qa.Answer) -> None:
        model = answer.model_used or self._model
        tokens = answer.tokens or {}
        if cfg_show_tokens(self.cfg):
            self._append(
                (
                    "dim",
                    f"⚡ {model} · {tokens.get('prompt', 0)} in / "
                    f"{tokens.get('completion', 0)} out\n",
                )
            )
        self._append(("", "\n"))

    # -- input handling ----------------------------------------------------

    def _on_submit(self, event: Any) -> None:
        text = self.input_area.text.strip()
        if not text:
            return
        self.input_area.text = ""
        self._handle(text)

    def _handle(self, text: str) -> None:
        if text.startswith("/"):
            self._handle_command(text)
            return
        # Auto-scroll: jump to the bottom when a new question is asked.
        self._pinned = True
        self._vertical_scroll = 0
        self._append_user(text)
        clean, attachments = parse_mentions(text, self.root)
        if attachments:
            self._append(
                (
                    "dim",
                    f"📎 attached {len(attachments)} file(s): "
                    + ", ".join(a["path"] for a in attachments)
                    + "\n",
                )
            )
        self._busy = True
        self._cancel.clear()
        self._append_working()
        threading.Thread(
            target=self._answer_worker,
            args=(clean, attachments),
            daemon=True,
        ).start()

    def _handle_command(self, text: str) -> None:
        cmd = text.strip().lower()
        if cmd == "/clear":
            self._typewriter_stop.set()
            with self._lock:
                self._blocks = []
                self._pending = ""
                self._revealed = 0
            self._invalidate()
            return
        if self._on_command is not None:
            try:
                result = self._on_command(cmd, self)
            except Exception as exc:  # defensive: never crash the TUI
                result = f"! {exc}"
            if result:
                self._append(("", f"{result}\n"))
        else:
            self._append(("dim", f"Unknown command: {cmd}  (try /help)\n"))

    # -- answer pipeline (background thread) -------------------------------

    def _answer_worker(
        self, question: str, attachments: List[Dict[str, str]]
    ) -> None:
        try:
            tools_enabled = (self.cfg.get("tools") or {}).get("enabled", True)
            if tools_enabled:
                self._stream_agent_answer(question, attachments)
            else:
                self._stream_answer(question, attachments)
        except Exception as exc:
            self._append(("red", f"! {exc}\n"))
        finally:
            self._busy = False
            self._invalidate()

    def _stream_agent_answer(
        self, question: str, attachments: List[Dict[str, str]]
    ) -> None:
        """Streaming agent path: live text deltas + animated tool markers."""
        events = qa.stream_agent_answer(
            question,
            self.root,
            cfg=self.cfg,
            resolve_key=self._resolve_key,
            attachments=attachments,
            on_tool_start=self._append_tool_start,
            on_tool=self._append_tool_done,
        )
        answer: Optional[qa.Answer] = None
        started = False
        for kind, payload in events:
            if self._cancel.is_set():
                break
            if kind == "delta":
                if not started:
                    self._clear_working()
                    started = True
                with self._lock:
                    self._pending += payload
                self._start_typewriter()
                self._invalidate()
            else:  # "done"
                answer = payload
                break
        if not started:
            self._clear_working()
        # Let the quiet typewriter finish revealing the buffered text.
        self._drain_typewriter()
        with self._lock:
            self._pending = ""
            self._revealed = 0
        self._invalidate()
        self._finish(answer, question, streamed=started)

    def _stream_answer(
        self, question: str, attachments: List[Dict[str, str]]
    ) -> None:
        events = qa.stream_answer(
            question,
            self.root,
            cfg=self.cfg,
            resolve_key=self._resolve_key,
            attachments=attachments,
        )
        answer: Optional[qa.Answer] = None
        started = False
        for kind, payload in events:
            if self._cancel.is_set():
                break
            if kind == "delta":
                if not started:
                    self._clear_working()
                    started = True
                with self._lock:
                    self._pending += payload
                self._start_typewriter()
                self._invalidate()
            else:  # "done"
                answer = payload
                break
        if not started:
            self._clear_working()
        self._drain_typewriter()
        with self._lock:
            self._pending = ""
            self._revealed = 0
        self._invalidate()
        self._finish(answer, question, streamed=started)

    def _finish(
        self, answer: Optional[qa.Answer], question: str, streamed: bool = False
    ) -> None:
        if answer is not None and answer.text is not None:
            if self._on_answer is not None:
                self._on_answer(self.session, question, answer)
            if not streamed:
                self._reveal(answer.text)
            else:
                # Text already streamed live; commit it and add spacing.
                self._append(("", answer.text + "\n"))
                self._append(("", "\n"))
            self._append_refs(answer.retrieved_chunks)
            self._append_footer(answer)
        else:
            warning = answer.warning if answer else "The model produced no answer."
            self._append(("yellow", f"! {warning}\n"))
            text = self._build_answer(question, self.root)
            if text:
                if self._on_answer is not None:
                    self._on_answer(
                        self.session, question, qa.Answer(text=text)
                    )
                self._append(("bold yellow", f"Panther: {text}\n"))
        # Auto-scroll: after the answer is fully written, jump back to the
        # bottom so the newest content is always in view.
        self._pinned = True
        self._vertical_scroll = 0
        self._invalidate()

    def _reveal(self, text: str) -> None:
        """Typewriter-reveal the answer into the transcript (quiet pace)."""
        if not text:
            return
        if not self._app:
            self._append(("", text + "\n"))
            return
        with self._lock:
            self._pending = text
            self._revealed = 0
        self._start_typewriter()
        self._drain_typewriter()
        with self._lock:
            self._pending = ""
            self._revealed = 0
        self._append(_ansi_fragment(text))
        self._append(("", "\n"))

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> None:
        if self._app is None:
            return
        self._append(
            (
                "dim",
                "Ask about your code.  @file to attach, /help for commands, "
                "/exit to quit.\n\n",
            )
        )
        try:
            self._app.run()
        except KeyboardInterrupt:
            pass
        finally:
            if self._on_exit is not None:
                self._on_exit(self.session)


def cfg_show_tokens(cfg: Dict[str, Any]) -> bool:
    return bool((cfg.get("ui") or {}).get("show_token_counts", True))


def _tool_label(name: str, args: Dict[str, Any]) -> str:
    if name == "grep":
        return f"/{args.get('pattern', '')}/"
    if name == "read_file":
        return str(args.get("path", ""))
    if name == "list_dir":
        return str(args.get("path", "."))
    if name == "run_tests":
        return "tests"
    return name


def run_tui(
    project_root: Path,
    session: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    resolve_key: Optional[Callable[[str], Optional[str]]] = None,
    on_answer: Optional[Callable[[Dict[str, Any], str, qa.Answer], None]] = None,
    on_command: Optional[Callable[[str, ChatTUI], Optional[str]]] = None,
    build_answer: Optional[Callable[[str, Path], str]] = None,
    on_exit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """Convenience entry point used by the CLI."""
    ChatTUI(
        project_root,
        session,
        cfg,
        resolve_key=resolve_key,
        on_answer=on_answer,
        on_command=on_command,
        build_answer=build_answer,
        on_exit=on_exit,
    ).run()
