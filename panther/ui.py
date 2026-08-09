"""Terminal UI components: gradient banner, markdown, progress, code highlighting."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

# legacy_windows=False avoids rich's cp1252-based legacy Windows renderer,
# which cannot encode the block characters used in the banner (it crashes
# with UnicodeEncodeError on non-ASCII output).
console = Console(legacy_windows=False)

# Legacy Windows consoles (cp1252) cannot encode fancy Unicode glyphs like ⚡ 🐾 ❯.
# rich reports the effective output encoding; "utf" means we can use them safely.
def _supports_unicode() -> bool:
    enc = (getattr(console, "encoding", None) or "").lower()
    return "utf" in enc


ASCII_ROWS = [
    "██████╗  █████╗ ███╗   ██╗████████╗██╗  ██╗███████╗██████╗    ▄▀  ▄▀  ▄▀ ",
    "██╔══██╗██╔══██╗████╗  ██║╚══██╔══╝██║  ██║██╔════╝██╔══██╗  ▄▀  ▄▀  ▄▀  ",
    "██████╔╝███████║██╔██╗ ██║   ██║   ███████║█████╗  ██████╔╝ ▐█▌ ▐█▌ ▐█▌  ",
    "██╔═══╝ ██╔══██║██║╚██╗██║   ██║   ██╔══██║██╔══╝  ██╔══██╗ █▌  █▌  █▌   ",
    "██║     ██║  ██║██║ ╚████║   ██║   ██║  ██║███████╗██║  ██║ ▀   ▀   ▀    ",
    "╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝              ",
]

GRADIENT_START = (230, 115, 0)
GRADIENT_END = (220, 20, 20)


def _print_ascii_banner(version: str) -> None:
    """Plain-ASCII fallback for pipes / legacy consoles that cannot encode Unicode."""
    print("")
    print("==========================================")
    print("   PANTHER CLI  |  Coding Agent Terminal")
    print(f"   v{version}  |  NVIDIA  |  codebase Q&A")
    print("==========================================")
    print("")


def print_banner(version: str = "1.0.0b0") -> None:
    """Render the Panther ASCII banner with a red→orange gradient."""
    # rich buffers non-TTY output and flushes on context exit, so a
    # UnicodeEncodeError from the block-glyph rows would surface later at an
    # unrelated point (e.g. a Status spinner stopping).  Gate on encoding
    # up front so cp1252 pipes/consoles get the plain-ASCII banner directly.
    if not _supports_unicode():
        _print_ascii_banner(version)
        return
    try:
        console.print("")
        num_rows = len(ASCII_ROWS)
        for index, row in enumerate(ASCII_ROWS):
            factor = index / max(1, num_rows - 1)
            r = round(GRADIENT_START[0] + factor * (GRADIENT_END[0] - GRADIENT_START[0]))
            g = round(GRADIENT_START[1] + factor * (GRADIENT_END[1] - GRADIENT_START[1]))
            b = round(GRADIENT_START[2] + factor * (GRADIENT_END[2] - GRADIENT_START[2]))
            console.print(Text(row, style=f"bold rgb({r},{g},{b})"))

        console.print(f"  -- v{version} --  Coding Agent Terminal | NVIDIA --\n")
    except (UnicodeEncodeError, OSError):
        _print_ascii_banner(version)


def highlight_code(code: str, language: str = "python") -> Text:
    """Syntax-highlight a code block, falling back to plain text when Pygments is missing."""
    from rich.syntax import Syntax

    try:
        return Syntax(code, language, theme="monokai", line_numbers=False)
    except Exception:
        return Text(code)


def print_code_block(code: str, language: str = "python") -> None:
    console.print(highlight_code(code, language))


def render_markdown(text: str) -> Markdown:
    """Render assistant output as Markdown with syntax-highlighted code blocks."""
    return Markdown(text, code_theme="monokai", inline_code_lexer="python")


def print_session_header(session: Dict[str, Any], model: Optional[str] = None) -> None:
    """Print a compact session header panel (Gemini-CLI-style)."""
    repo = session.get("repo_name") or Path(session.get("repo_path", ".")).name
    sid = session.get("session_id", "")
    if _supports_unicode():
        try:
            frags = [
                ("🐾 ", "bold bright_yellow"),
                (repo, "bold cyan"),
                (f"   {sid}", "dim"),
                ("\n", ""),
                ("❯ ", "cyan"),
                ("ask about your code  ·  /help  ·  /exit", "dim"),
            ]
            if model:
                frags += [("\n", ""), ("model", "dim"), (f"  {model}", "bright_yellow")]
            console.print(
                Panel.fit(
                    Text.assemble(*frags),
                    border_style="cyan",
                    padding=(0, 2),
                )
            )
            return
        except (UnicodeEncodeError, OSError):
            pass  # fall through to the ASCII version below
    # ASCII fallback (plain print avoids rich's dirty-buffer issue on cp1252).
    model_str = f"  ({model})" if model else ""
    print(f"PANTHER  {repo}  {sid}{model_str}")
    print("ask about your code  |  /help  |  /exit")


def print_answer_footer(model: Optional[str], tokens: Optional[Dict[str, int]]) -> None:
    """Print a subtle one-line footer with model + token usage."""
    if not model and not tokens:
        return
    bits = []
    if model:
        bits.append(f"{model}")
    if tokens:
        bits.append(f"{tokens.get('prompt', 0)} in / {tokens.get('completion', 0)} out")
    if not bits:
        return
    if _supports_unicode():
        try:
            console.print(f"[dim]⚡ {' · '.join(bits)}[/dim]")
            return
        except (UnicodeEncodeError, OSError):
            pass  # fall through to the ASCII version below
    print(f"{' | '.join(bits)}")


def print_progress_bar(description: str, total: int, completed: int) -> None:
    """Print a lightweight progress line (ASCII-safe for any console)."""
    width = 40
    if total <= 0:
        return
    fraction = min(1.0, completed / total)
    filled = int(fraction * width)
    bar = "#" * filled + "-" * (width - filled)
    pct = int(fraction * 100)
    console.print(f"  {description}  {bar} {pct}%  ({completed}/{total})")


def animate_text_fill(text: str, delay: float = 0.002) -> None:
    """Typewriter-style text-fill effect for non-streamed output.

    Prints in small chunks so the answer visibly 'fills in'.  The chunk size
    and tick rate adapt to the length of ``text`` so long answers still
    finish in roughly a second.  Falls back to a single print when not a TTY
    (pipes, CI) or when encoding blocks it.
    """
    if not console.is_terminal or not text:
        console.print(text)
        return
    try:
        step = max(2, min(8, len(text) // 150))  # 2–8 chars per tick
        ticks = max(1, len(text) // step)
        tick_delay = min(delay, 1.0 / ticks)
        for i in range(0, len(text), step):
            console.print(text[i : i + step], end="")
            time.sleep(tick_delay)
        console.print("")
    except (UnicodeEncodeError, OSError):
        console.print(text)


def print_tool_call(event: Any) -> None:
    """Render one tool execution result as a compact status line.

    Called after the tool has executed (the loading spinner is handled by
    :func:`tool_spinner`, which wraps the execution itself).
    """
    if _supports_unicode():
        marker = "[green]✓[/green]" if event.ok else "[red]✗[/red]"
        icon = "🔧"
    else:
        marker = "[green]OK[/green]" if event.ok else "[red]X[/red]"
        icon = "tool:"
    label = f"[bold cyan]{icon} {event.describe()}[/bold cyan]"
    try:
        console.print(f"  {marker} {label}")
    except (UnicodeEncodeError, OSError):
        print(f"  {'OK' if event.ok else 'X'} {event.describe()}")


def _ascii_safe(text: str) -> str:
    """Replace non-encodable glyphs for legacy (cp1252) consoles/pipes.

    rich's Status renders its label even on ``stop()``, so a ``…`` or ``🔧``
    in the message would crash with UnicodeEncodeError exactly when the
    spinner finishes on such a console.  The label is already wrapped in
    rich markup; ASCII-replacing the whole string keeps the tags intact.
    """
    if _supports_unicode():
        return text
    try:
        return text.encode("ascii", "replace").decode("ascii")
    except Exception:
        return text


_STATUS_TRANSIENT_KWARGS: Optional[Dict[str, bool]] = None


def _spinner_kwargs() -> Dict[str, bool]:
    """Spinner options for the installed rich version (cached).

    rich>=13.6 accepts ``transient=True`` so the spinner erases itself when
    it stops (the progress indicator is cleared once the phase completes).
    rich 14 is always transient (its ``Status`` hardcodes it), so the kwarg
    must NOT be passed there — it would raise TypeError.
    """
    global _STATUS_TRANSIENT_KWARGS
    if _STATUS_TRANSIENT_KWARGS is None:
        try:
            import inspect
            from rich.status import Status

            if "transient" in inspect.signature(Status.__init__).parameters:
                _STATUS_TRANSIENT_KWARGS = {"transient": True}
            else:
                _STATUS_TRANSIENT_KWARGS = {}
        except (ImportError, ValueError, TypeError):
            _STATUS_TRANSIENT_KWARGS = {}
    return _STATUS_TRANSIENT_KWARGS


def tool_spinner(description: str) -> Any:
    """Status spinner that wraps an in-progress tool execution."""
    from rich.status import Status

    return Status(
        _ascii_safe(f"[bold cyan]🔧 {description}[/bold cyan]"),
        console=console,
        spinner="dots",
        **_spinner_kwargs(),
    )


def status_spinner(message: str) -> Any:
    """Context-manager spinner for loading phases (indexing, retrieval)."""
    from rich.status import Status

    return Status(
        _ascii_safe(message), console=console, spinner="dots", **_spinner_kwargs()
    )


def render_chunk_refs(chunks: List[Dict[str, Any]], plain: bool = False) -> Optional[str]:
    """Format retrieved chunk references as a compact string.

    ``plain=True`` omits rich markup so the string can be embedded directly
    into prompt_toolkit fragments (which do not parse ``[cyan]`` tags).
    """
    if not chunks:
        return None
    lines = []
    for chunk in chunks:
        loc = chunk.get("file_path", "?")
        if chunk.get("start_line") is not None:
            loc += f":{chunk['start_line']}"
        sim = chunk.get("similarity")
        if sim is not None:
            loc += f" ({sim:.2f})"
        lines.append(f">> {loc}" if plain else f"  [cyan]>> {loc}[/cyan]")
    return "\n".join(lines)
