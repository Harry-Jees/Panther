"""Panther CLI — interactive codebase assistant (viva sessions).

Command surface:
    init, index, new [--no-index], resume <id>, list, delete <id>,
    reset --confirm, config [section|show], auth add|remove|list, --version
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.live import Live
from rich.table import Table

from . import __version__
from . import config as config_mod
from . import qa, rag, storage, ui

console = Console(legacy_windows=False)

# Some legacy Windows consoles (cp1252) cannot encode fancy Unicode glyphs
# (❯ ▌ 🐾 …). rich's console.encoding tells us which to use.
def _supports_unicode() -> bool:
    enc = (getattr(console, "encoding", None) or "").lower()
    return "utf" in enc


PROMPT_GLYPH = "❯" if _supports_unicode() else ">"
STREAM_CURSOR = "▌" if _supports_unicode() else "|"
THINKING_TEXT = "*thinking…*" if _supports_unicode() else "*thinking...*"


def _use_tui() -> bool:
    """True when the full-screen chat TUI should be used.

    Requires an interactive terminal AND prompt_toolkit.  Piped input
    (tests, scripts) and missing deps fall back to the classic REPL.
    """
    if os.environ.get("PANTHER_NO_TUI") == "1":
        return False
    if not sys.stdin.isatty():
        return False
    try:
        import prompt_toolkit  # noqa: F401

        return True
    except ImportError:
        return False

# Live can still raise if the model's markdown contains characters the console
# cannot encode (cp1252). Strip non-encodable chars so streaming never crashes
# on legacy Windows consoles.
def _console_safe(text: str) -> str:
    if _supports_unicode():
        return text
    try:
        return text.encode("ascii", "replace").decode("ascii")
    except Exception:
        return text

# ---------------------------------------------------------------------------
# Path / config helpers (re-exported for backward compatibility — the test
# suite patches ``panther.cli.Path.home``, which affects all modules that use
# ``Path.home()`` since it is a class attribute).
# ---------------------------------------------------------------------------


def get_panther_home() -> Path:
    return config_mod.panther_home()


def get_config_path() -> Path:
    return config_mod.config_path()


def get_db_path() -> Path:
    return config_mod.db_path()


def get_sessions_path() -> Path:
    return config_mod.sessions_path()


def ensure_panther_home() -> None:
    config_mod.ensure_home()


def load_config() -> Dict[str, Any]:
    return config_mod.load_config()


def save_config(config: Dict[str, Any]) -> None:
    config_mod.save_config(config)


def print_banner(version: Optional[str] = None) -> None:
    ui.print_banner(version or __version__)


def _resolve_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a usable config, falling back to defaults if Panther is uninitialized."""
    if cfg is not None:
        return cfg
    try:
        return load_config()
    except FileNotFoundError:
        return config_mod.defaults()


# ---------------------------------------------------------------------------
# Keyword index + fallback answers (legacy retrieval path)
# ---------------------------------------------------------------------------

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv"}
ALLOWED_EXTS = {".py", ".sql", ".md", ".txt"}
INDEX_FILE = "index.json"


def get_index_path() -> Path:
    return get_db_path() / INDEX_FILE


def write_index(project_root: Path, files: List[Dict[str, Any]]) -> None:
    from datetime import datetime, timezone

    index = {
        "project_root": str(project_root),
        "file_count": len(files),
        "indexed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
    }
    get_index_path().write_text(json.dumps(index, indent=2), encoding="utf-8")


def scan_repository(project_root: Path) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    for path in project_root.rglob("*"):
        if path.is_dir():
            if any(part in IGNORE_DIRS for part in path.parts):
                continue
            continue

        if any(part in IGNORE_DIRS for part in path.parts):
            continue

        if path.suffix.lower() not in ALLOWED_EXTS:
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        files.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "size": path.stat().st_size,
                "snippet": content[:800],
            }
        )

    return files


def build_answer(query: str, project_root: Path) -> str:
    """Lightweight keyword-retrieval answer, used when no API key is available."""
    index_path = get_db_path() / INDEX_FILE
    if not index_path.exists():
        return "Panther has not indexed this project yet. Run `panther index` first."

    index = json.loads(index_path.read_text(encoding="utf-8"))
    query_tokens = [token.lower() for token in re.findall(r"\w+", query) if len(token) > 2]
    if not query_tokens:
        return "Try asking a more specific question about your code."

    best_match: Optional[Dict[str, Any]] = None
    best_score = 0
    for item in index["files"]:
        # Search the full file content on disk when possible (the stored
        # snippet is only the first 800 characters).
        full_text = ""
        try:
            full_text = (project_root / item.get("path", "")).read_text(
                encoding="utf-8", errors="ignore"
            ).lower()
        except OSError:
            pass
        text = full_text or item.get("snippet", "").lower()
        score = sum(1 for token in query_tokens if token in text)
        if score > best_score:
            best_score = score
            best_match = item

    if best_match and best_score > 0:
        snippet = best_match["snippet"].strip().splitlines()[:10]
        snippet_text = "\n".join(snippet)
        return (
            f"I found a relevant code snippet in [bold]{best_match['path']}[/bold].\n\n"
            f"[bold]Excerpt:[/bold]\n{snippet_text}\n\n"
            "This is a simple retrieval-based answer. "
            "Configure an API key (`panther auth add nvidia_nim`) for full AI explanations."
        )

    return "I couldn't find a relevant snippet in the indexed files. Try a more specific query or run `panther index`."


# ---------------------------------------------------------------------------
# Session helpers (re-exported)
# ---------------------------------------------------------------------------


def create_session(project_root: Path) -> Dict[str, Any]:
    return storage.create_session(project_root)


def save_session(session: Dict[str, Any]) -> None:
    storage.save_session(session)


def load_session(session_id: str) -> Dict[str, Any]:
    return storage.load_session(session_id)


def list_sessions() -> List[Dict[str, str]]:
    return storage.list_sessions()


def delete_session(session_id: str) -> None:
    storage.delete_session(session_id)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def _index_progress(done: int, total: int) -> None:
    """Single-line indexing progress that erases itself when finished.

    Writes directly to stdout with a carriage return so each update overwrites
    the previous line; the final update clears the line so the terminal is
    left clean once indexing completes.
    """
    sys.stdout.write(f"\r  Indexing & embedding... {done}/{total} chunks")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\r" + " " * 64 + "\r")
        sys.stdout.flush()


def _index_project(
    project_root: Path,
    cfg: Dict[str, Any],
    verbose: bool = True,
    quiet: bool = False,
) -> int:
    """Write the keyword index and (if available) the vector index.

    Returns the number of vector chunks stored (0 when vector indexing was
    skipped).  ``quiet=True`` suppresses console output entirely — used by
    the full-screen TUI, where writing to stdout would corrupt the render.
    """
    files = scan_repository(project_root)
    write_index(project_root, files)
    if verbose and not quiet:
        console.print(f"[green]Scanned {len(files)} files in {project_root}[/green]")
    count = 0
    try:
        if quiet:
            count = rag.index_repository(project_root, cfg, progress=None)
        else:
            try:
                count = rag.index_repository(project_root, cfg, progress=_index_progress)
            finally:
                # Always clear the progress line, even if embedding fails
                # partway through and never reports completion.
                sys.stdout.write("\r" + " " * 64 + "\r")
                sys.stdout.flush()
    except rag.RAGDependencyError as exc:
        if not quiet:
            console.print(f"[yellow]! Vector index skipped: {exc}[/yellow]")
            console.print("[dim]Keyword retrieval still available for answers.[/dim]")
        return 0
    if verbose and not quiet:
        console.print(
            f"[green]Vector index ready: {count} chunks stored in {get_db_path() / 'chroma'}[/green]"
        )
    return count


def command_index(args: argparse.Namespace) -> None:
    project_root = Path.cwd()
    ensure_panther_home()
    try:
        cfg = load_config()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    _index_project(project_root, cfg)


# ---------------------------------------------------------------------------
# Init / sessions / REPL
# ---------------------------------------------------------------------------


def command_init(args: argparse.Namespace) -> None:
    config_mod.initialize()
    console.print(f"[green]Initialized Panther at {get_panther_home()}[/green]")
    console.print("Next steps:")
    console.print("  1. `panther auth add nvidia_nim`  - add a free API key")
    console.print("  2. `panther index`  - index your current repository")
    console.print("  3. `panther new`  - start a viva session")


def _maybe_auto_index(project_root: Path, cfg: Dict[str, Any]) -> None:
    if (get_db_path() / "chroma" / "chroma.sqlite3").exists():
        return
    console.print("[dim]No vector index yet - indexing this repository...[/dim]")
    _index_project(project_root, cfg, verbose=True)


def _launch_session(args: argparse.Namespace, auto_init: bool = False) -> None:
    """Start the interactive viva session for the current directory."""
    project_root = Path.cwd()
    ensure_panther_home()
    try:
        cfg = load_config()
    except FileNotFoundError:
        if not auto_init:
            console.print("[red]Panther is not initialized. Run `panther init` first.[/red]")
            return
        console.print("[dim]First run detected - initializing Panther...[/dim]")
        config_mod.initialize()
        cfg = load_config()
    print_banner()
    if not getattr(args, "no_index", False):
        _maybe_auto_index(project_root, cfg)
    session = create_session(project_root)
    save_session(session)
    repl_loop(project_root, session, cfg)


def command_new(args: argparse.Namespace) -> None:
    _launch_session(args, auto_init=False)


def command_resume(args: argparse.Namespace) -> None:
    session_id = args.session_id
    if not session_id:
        console.print("[red]Please provide a session ID to resume.[/red]")
        return
    ensure_panther_home()
    try:
        session = load_session(session_id)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    print_banner()
    cfg = None
    try:
        cfg = load_config()
    except FileNotFoundError:
        pass
    repl_loop(Path(session["repo_path"]), session, cfg)


def _record_answer(session: Dict[str, Any], question: str, result: qa.Answer) -> None:
    """Append a Q&A pair to the session transcript and persist it."""
    answer: Dict[str, Any] = {
        "role": "assistant",
        "content": result.text,
        "model_used": result.model_used,
    }
    if result.retrieved_chunks:
        answer["retrieved_chunks"] = [dict(c) for c in result.retrieved_chunks]
    if result.tokens:
        answer["tokens_used"] = result.tokens
        session["stats"]["total_tokens_used"] += sum(result.tokens.values())

    session["messages"].append({"role": "user", "content": question})
    session["messages"].append(answer)
    session["stats"]["questions"] += 1
    session["stats"]["total_messages"] = len(session["messages"])
    api_calls = session["stats"].setdefault("api_calls", {})
    key = result.model_used or "unknown"
    api_calls[key] = api_calls.get(key, 0) + 1
    save_session(session)


def _ask(question: str, project_root: Path, session: Dict[str, Any], cfg: Optional[Dict[str, Any]]) -> str:
    """Full RAG + LLM answer; falls back to keyword retrieval when unavailable."""
    cfg = _resolve_config(cfg)
    result = qa.answer_question(
        question,
        project_root,
        cfg=cfg,
        # Bind the loaded config so config-file keys are honoured:
        # get_api_key(provider, config=None) would otherwise miss them.
        resolve_key=lambda p: config_mod.get_api_key(p, cfg),
    )

    if result.text is not None:
        _record_answer(session, question, result)
        refs = ui.render_chunk_refs(result.retrieved_chunks)
        if refs:
            console.print(f"[dim]Referenced:[/dim]\n{refs}\n")
        if cfg.get("ui", {}).get("show_token_counts"):
            ui.print_answer_footer(result.model_used, result.tokens)
        return result.text

    # Fallback: no API key, missing deps, or provider failure.
    if result.warning:
        console.print(f"[yellow]! {result.warning}[/yellow]")
    text = build_answer(question, project_root)
    _record_answer(session, question, qa.Answer(text=text))
    return text


def _ask_stream(question: str, project_root: Path, session: Dict[str, Any], cfg: Optional[Dict[str, Any]]) -> None:
    """RAG + tool-calling agent answer with live markdown + loading animations."""
    cfg = _resolve_config(cfg)
    tools_enabled = (cfg.get("tools") or {}).get("enabled", True)

    if not tools_enabled:
        # Pure streaming path (no tools): SSE deltas rendered live.
        _ask_stream_sse(question, project_root, session, cfg)
        return

    spinner: Any = None

    def on_tool_start(name: str, args: Dict[str, Any]) -> None:
        nonlocal spinner
        spinner = ui.tool_spinner(f"{name} {_tool_args_label(name, args)}")
        spinner.start()

    def on_tool(event) -> None:
        nonlocal spinner
        if spinner:
            spinner.stop()
            spinner = None
        ui.print_tool_call(event)

    with ui.status_spinner("[cyan]retrieving context…[/cyan]"):
        answer = qa.agent_answer(
            question,
            project_root,
            cfg=cfg,
            resolve_key=lambda p: config_mod.get_api_key(p, cfg),
            on_tool=on_tool,
            on_tool_start=on_tool_start,
        )

    if answer.text is None:
        # Fallback: no API key, missing deps, provider failure, or the agent
        # exhausted its tool budget. Keep the agent's real token/model stats.
        if answer.warning:
            console.print(f"[yellow]! {answer.warning}[/yellow]")
        text = build_answer(question, project_root)
        fallback = qa.Answer(text=text)
        extra = answer.extra or {}
        if extra.get("tokens"):
            fallback.tokens = extra["tokens"]
        if extra.get("model_used"):
            fallback.model_used = extra["model_used"]
        _record_answer(session, question, fallback)
        console.print(f"[bold yellow]Panther:[/bold yellow] {text}\n")
        return

    _record_answer(session, question, answer)

    # Text-fill animation: the answer visibly types out.
    try:
        with Live(console=console, refresh_per_second=30, vertical_overflow="ellipsis") as live:
            buffer = ""
            last_render = 0
            step = 3 if _supports_unicode() else 6
            for i in range(0, len(answer.text), step):
                buffer = answer.text[: i + step]
                if len(buffer) - last_render >= 30 or i + step >= len(answer.text):
                    last_render = len(buffer)
                    live.update(
                        ui.render_markdown(_console_safe(f"{buffer} {STREAM_CURSOR}"))
                    )
                    time.sleep(0.004)  # visible typing speed
            live.update(ui.render_markdown(_console_safe(answer.text)))
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Answer already saved to the session.[/dim]")
    except Exception:
        console.print(ui.render_markdown(_console_safe(answer.text)))

    refs = ui.render_chunk_refs(answer.retrieved_chunks)
    if refs:
        console.print(f"\n[dim]Referenced:[/dim]\n{refs}")
    if cfg.get("ui", {}).get("show_token_counts"):
        ui.print_answer_footer(answer.model_used, answer.tokens)
    console.print("")


def _tool_args_label(name: str, args: Dict[str, Any]) -> str:
    if name == "grep":
        return f"/{args.get('pattern', '')}/"
    if name == "read_file":
        return f"{args.get('path', '')}"
    if name == "list_dir":
        return f"{args.get('path', '.')}"
    if name == "run_tests":
        return "tests"
    return ""


def _ask_stream_sse(
    question: str, project_root: Path, session: Dict[str, Any], cfg: Dict[str, Any]
) -> None:
    """SSE streaming path used when agent tools are disabled."""
    events = qa.stream_answer(
        question,
        project_root,
        cfg=cfg,
        resolve_key=lambda p: config_mod.get_api_key(p, cfg),
    )

    buffer = ""
    answer: Optional[qa.Answer] = None
    last_render = 0
    try:
        with Live(console=console, refresh_per_second=24, vertical_overflow="ellipsis") as live:
            live.update(ui.render_markdown(_console_safe(THINKING_TEXT)))
            for kind, payload in events:
                if kind == "delta":
                    buffer += payload
                    if len(buffer) - last_render >= 24:
                        last_render = len(buffer)
                        live.update(
                            ui.render_markdown(_console_safe(f"{buffer} {STREAM_CURSOR}"))
                        )
                else:  # "done"
                    answer = payload
                    if answer is not None and answer.text is not None:
                        live.update(ui.render_markdown(_console_safe(answer.text)))
                    break
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Answer not recorded.[/dim]")
        return
    except Exception as exc:
        answer = qa.Answer(None, warning=str(exc))

    if answer is None:
        answer = qa.Answer(None, warning="The model stream ended without a response.")

    if answer.text is not None:
        _record_answer(session, question, answer)
        refs = ui.render_chunk_refs(answer.retrieved_chunks)
        if refs:
            console.print(f"\n[dim]Referenced:[/dim]\n{refs}")
        if cfg.get("ui", {}).get("show_token_counts"):
            ui.print_answer_footer(answer.model_used, answer.tokens)
        console.print("")
        return

    if answer.warning:
        console.print(f"[yellow]! {answer.warning}[/yellow]")
    text = build_answer(question, project_root)
    _record_answer(session, question, qa.Answer(text=text))
    console.print(f"[bold yellow]Panther:[/bold yellow] {text}\n")


def _help_text() -> str:
    """Markup for the /help screen (shared by REPL and TUI)."""
    return (
        "Commands\n"
        "  /help       show this help\n"
        "  /status     session stats (questions, tokens, API calls)\n"
        "  /model      show the active model configuration\n"
        "  /tools      toggle agent tools (read/grep/tests) on/off\n"
        "  /reindex    re-index the current repository\n"
        "  /clear      clear the terminal\n"
        "  /exit       save and quit (/quit also works)\n\n"
        "@file mentions: type @ to fuzzy-search files (Tab to pick; line "
        "counts shown) and attach the file's content to your question.\n"
        "Panther can use tools while answering: read_file, list_dir, "
        "grep, run_tests.\n"
    )


def _print_help() -> None:
    console.print(_help_text())


def _dispatch_command(
    command: str,
    project_root: Path,
    session: Dict[str, Any],
    cfg: Dict[str, Any],
    quiet: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Run a ``/`` command.  Returns ``(should_exit, message)``.

    Shared by the classic REPL and the chat TUI so both behave identically.
    ``quiet=True`` (the TUI) suppresses console output — /reindex then runs
    without writing to stdout, which would corrupt the full-screen render.
    """
    cmd = command.strip().lower()
    if cmd in {"/exit", "/quit"}:
        return True, None  # caller prints its own goodbye
    if cmd == "/help":
        return False, _help_text()
    if cmd == "/status":
        return False, storage.format_stats(session)
    if cmd == "/model":
        llm = cfg.get("llm") or {}
        p = llm.get("primary") or {}
        return False, f"Provider: {p.get('provider')} / {p.get('model')}"
    if cmd == "/tools":
        tools_cfg = cfg.setdefault("tools", {})
        tools_cfg["enabled"] = not tools_cfg.get("enabled", True)
        state = "ON" if tools_cfg["enabled"] else "OFF"
        save_config(cfg)
        return False, f"Agent tools: {state}"
    if cmd == "/reindex":
        count = _index_project(project_root, cfg, quiet=quiet)
        return False, (
            f"Re-indexed the repository ({count} vector chunks)."
            if count
            else "Re-indexed keyword index (vector indexing unavailable)."
        )
    return False, f"Unknown command: {command}  (try /help)"


def repl_loop(project_root: Path, session: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> None:
    cfg = _resolve_config(cfg)

    # Full-screen chat TUI on interactive terminals with prompt_toolkit.
    if _use_tui():
        from . import tui

        def on_command(cmd: str, tui_app: Any) -> Optional[str]:
            exit_now, message = _dispatch_command(
                cmd, project_root, session, cfg, quiet=True
            )
            if exit_now:
                save_session(session)
                tui_app._app.exit()
                return None
            return message

        def on_answer(sess: Dict[str, Any], question: str, result: qa.Answer) -> None:
            _record_answer(sess, question, result)

        def on_exit(sess: Dict[str, Any]) -> None:
            save_session(sess)

        tui.run_tui(
            project_root,
            session,
            cfg,
            resolve_key=lambda p: config_mod.get_api_key(p, cfg),
            on_answer=on_answer,
            on_command=on_command,
            build_answer=build_answer,
            on_exit=on_exit,
        )
        return

    primary = (cfg.get("llm") or {}).get("primary") or {}
    ui.print_session_header(session, model=primary.get("model"))
    console.print("Type /help for commands, /exit to quit.\n")

    while True:
        try:
            user_input = console.input(f"[bold cyan]{PROMPT_GLYPH} [/bold cyan]")
        except EOFError:
            # Piped input (scripts, CI): stdin exhausted, so end the session.
            console.print("\n[dim]Input closed. Session saved.[/dim]")
            save_session(session)
            break
        except KeyboardInterrupt:
            console.print("\n[dim]Session interrupted. Saving...[/dim]")
            save_session(session)
            break

        if not user_input.strip():
            continue

        if user_input.startswith("/"):
            command = user_input.strip().lower()
            if command == "/clear":
                console.clear()
                continue
            exit_now, message = _dispatch_command(command, project_root, session, cfg)
            if message:
                console.print(message)
            if exit_now:
                save_session(session)
                console.print("Goodbye! Session saved.")
                break
            continue

        _ask_stream(user_input, project_root, session, cfg)


# ---------------------------------------------------------------------------
# Info commands
# ---------------------------------------------------------------------------


def command_list(args: argparse.Namespace) -> None:
    ensure_panther_home()
    sessions = list_sessions()
    if not sessions:
        console.print("[yellow]No saved sessions found.[/yellow]")
        return
    table = Table(title="Saved Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Created At", style="green")
    table.add_column("Repo Path", style="white")
    table.add_column("Messages", style="magenta")
    for session in sessions:
        table.add_row(
            session["session_id"],
            session["created_at"],
            session["repo_path"],
            session.get("messages", ""),
        )
    console.print(table)


def command_config(args: argparse.Namespace) -> None:
    try:
        cfg = load_config()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    section = getattr(args, "section", None)
    if section and section != "show":
        if section in cfg:
            console.print_json(data=cfg[section])
            return
        console.print(
            f"[yellow]Unknown config section: {section}. "
            f"Available: {', '.join(cfg)}[/yellow]"
        )
        return
    console.print_json(data=cfg)


def command_auth(args: argparse.Namespace) -> None:
    """Manage API keys: add, remove, list."""
    ensure_panther_home()
    try:
        cfg = load_config()
    except FileNotFoundError:
        cfg = config_mod.initialize()

    api_keys = cfg.setdefault("api_keys", {})
    action = getattr(args, "action", None) or "list"

    if action == "list":
        console.print("\nConfigured providers:")
        if not api_keys:
            console.print("  (none)")
        for provider, value in api_keys.items():
            env_set = bool(os.environ.get(config_mod.ENV_KEY_MAP.get(provider, "")))
            console.print(f"  - {provider}: {'(set)' if value or env_set else '(not set)'}")
        console.print(
            "\nHint: keys can also be provided via the NVIDIA_NIM_API_KEY "
            "environment variable."
        )
        return

    if action == "add":
        provider = getattr(args, "provider", None)
        if not provider:
            console.print("[red]Usage: panther auth add <provider> [key][/red]")
            return
        if provider not in config_mod.ENV_KEY_MAP:
            console.print(
                f"[yellow]Unknown provider '{provider}'. "
                f"Known providers: {', '.join(config_mod.ENV_KEY_MAP)}[/yellow]"
            )
            return
        key = getattr(args, "key", None)
        if not key:
            key = console.input(f"Enter API key for {provider}: ")
        if not key:
            console.print("[red]No key provided.[/red]")
            return
        api_keys[provider] = key
        save_config(cfg)
        console.print(f"[green]Saved API key for {provider} to config.[/green]")
        return

    if action == "remove":
        provider = getattr(args, "provider", None)
        if not provider:
            console.print("[red]Usage: panther auth remove <provider>[/red]")
            return
        if api_keys.get(provider):
            api_keys[provider] = None
            save_config(cfg)
            console.print(f"[green]Removed API key for {provider}.[/green]")
        else:
            console.print(f"[yellow]No API key configured for {provider}.[/yellow]")
        return

    console.print(f"[red]Unknown auth action: {action}[/red]")


def command_delete(args: argparse.Namespace) -> None:
    session_id = args.session_id
    if not session_id:
        console.print("[red]Please provide a session ID to delete.[/red]")
        return
    ensure_panther_home()
    try:
        delete_session(session_id)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    console.print(f"[green]Deleted session {session_id}.[/green]")


def command_reset(args: argparse.Namespace) -> None:
    if not getattr(args, "confirm", False):
        console.print("[red]Refusing to reset without confirmation. Use: panther reset --confirm[/red]")
        return
    home = get_panther_home()
    if home.exists():
        shutil.rmtree(home)
    console.print(f"[green]Removed {home}. Run `panther init` to start fresh.[/green]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(prog="panther", description="Panther CLI - interactive codebase assistant")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Initialize Panther configuration")
    subparsers.add_parser("index", help="Index the current repository for retrieval")

    new_parser = subparsers.add_parser("new", help="Start a new interactive session (auto-indexes if needed)")
    new_parser.add_argument("--no-index", action="store_true", help="Skip auto-indexing on start")

    resume_parser = subparsers.add_parser("resume", help="Resume a saved session")
    resume_parser.add_argument("session_id", nargs="?", help="Session ID to resume")

    subparsers.add_parser("list", help="List saved sessions")

    delete_parser = subparsers.add_parser("delete", help="Delete a saved session")
    delete_parser.add_argument("session_id", nargs="?", help="Session ID to delete")

    reset_parser = subparsers.add_parser("reset", help="Remove all Panther data (caution)")
    reset_parser.add_argument("--confirm", action="store_true", help="Actually perform the reset")

    config_parser = subparsers.add_parser("config", help="Show current configuration")
    config_parser.add_argument("section", nargs="?", help="Section to show (or 'show')")

    auth_parser = subparsers.add_parser("auth", help="Manage API keys")
    auth_sub = auth_parser.add_subparsers(dest="action")
    auth_sub.add_parser("list", help="List configured providers")
    auth_add = auth_sub.add_parser("add", help="Add an API key")
    auth_add.add_argument("provider", nargs="?", help="Provider (nvidia_nim)")
    auth_add.add_argument("key", nargs="?", help="API key (prompted if omitted)")
    auth_remove = auth_sub.add_parser("remove", help="Remove an API key")
    auth_remove.add_argument("provider", nargs="?", help="Provider (nvidia_nim)")

    args = parser.parse_args()
    command = args.command
    if command is None:
        # Bare `panther` launches the app (guide: "$ panther" -> viva mode),
        # auto-initializing on first run.
        _launch_session(argparse.Namespace(no_index=False), auto_init=True)
        return 0
    if command == "init":
        command_init(args)
    elif command == "index":
        command_index(args)
    elif command == "new":
        command_new(args)
    elif command == "resume":
        command_resume(args)
    elif command == "list":
        command_list(args)
    elif command == "delete":
        command_delete(args)
    elif command == "reset":
        command_reset(args)
    elif command == "config":
        command_config(args)
    elif command == "auth":
        command_auth(args)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
