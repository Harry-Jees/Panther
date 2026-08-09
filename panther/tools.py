"""Agent tools: safe, read-only operations the LLM can call while answering.

Exposed as OpenAI-compatible function schemas so the model can request them,
then executed locally with paths resolved against the project root (no writes,
no escapes outside the repo).
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Maximum result size returned to the model (keeps prompts small).
MAX_OUTPUT_CHARS = 4000
MAX_LINES = 200

# File extensions the tools are willing to read/search.
TEXT_EXTS = {
    ".py", ".sql", ".txt", ".md", ".toml", ".yaml", ".yml", ".json",
    ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".sh", ".cfg", ".ini",
}
IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", ".panther", "dist",
    "build", "__pycache__",
}


def is_ignored(parts: List[str]) -> bool:
    """True if any path part is an ignored dir (or a ``*.egg-info`` package)."""
    return any(
        part in IGNORED_DIRS or part.endswith(".egg-info") for part in parts
    )


class ToolError(Exception):
    """Raised when a tool cannot execute (bad path, missing file, etc.)."""


def _resolve(project_root: Path, raw_path: str) -> Path:
    """Resolve ``raw_path`` inside ``project_root``, rejecting escapes."""
    if not raw_path:
        raise ToolError("A path is required.")
    base = project_root.resolve()
    candidate = (base / raw_path).resolve()
    if not candidate.is_relative_to(base):
        raise ToolError(f"Path escapes the project root: {raw_path}")
    return candidate


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _truncate(text: str) -> str:
    lines = text.splitlines()
    truncated = False
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES]
        truncated = True
    out = "\n".join(lines)
    if len(out) > MAX_OUTPUT_CHARS:
        out = out[:MAX_OUTPUT_CHARS]
        truncated = True
    return out + ("\n… (truncated)" if truncated else "")


def read_file(project_root: Path, args: Dict[str, Any]) -> str:
    path = _resolve(project_root, str(args.get("path", "")))
    if not path.is_file():
        raise ToolError(f"Not a file: {args.get('path')}")
    if path.suffix.lower() not in TEXT_EXTS:
        # Still allow reading, but note it may be binary.
        content = _read_text(path)
        return f"Note: '{path.name}' may be binary.\n\n{_truncate(content)}"
    content = _read_text(path)
    return _truncate(content)


def list_dir(project_root: Path, args: Dict[str, Any]) -> str:
    path = _resolve(project_root, str(args.get("path", ".")))
    if not path.is_dir():
        raise ToolError(f"Not a directory: {args.get('path', '.')}")
    entries = []
    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if is_ignored([child.name]):
            continue
        marker = "/" if child.is_dir() else ""
        entries.append(f"{child.name}{marker}")
    if not entries:
        return "(empty directory)"
    return "\n".join(entries)


def grep(project_root: Path, args: Dict[str, Any]) -> str:
    pattern = str(args.get("pattern", ""))
    # Models often wrap patterns in vim-style slashes (/foo/); strip them so
    # the regex matches the bare pattern the user intended.
    if len(pattern) >= 2 and pattern.startswith("/") and pattern.endswith("/"):
        pattern = pattern[1:-1]
    if not pattern:
        raise ToolError("A pattern is required.")
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ToolError(f"Invalid regex: {exc}") from exc

    root = project_root.resolve()  # long form; avoids Windows 8.3 short-name
    # mismatches when comparing relative_to() against resolved paths.
    search_paths: List[Path] = []
    single_file = False
    if args.get("path"):
        target = _resolve(root, str(args["path"]))
        if target.is_dir():
            search_paths = sorted(target.rglob("*"))
        elif target.is_file():
            search_paths = [target]  # allow pointing at a single file
            single_file = True
        else:
            raise ToolError(f"Not a file or directory: {args['path']}")
    else:
        search_paths = sorted(root.rglob("*"))
    matches: List[str] = []
    for path in search_paths:
        if path.is_dir():
            continue
        if is_ignored(path.parts):
            continue
        # An explicitly targeted file is searched regardless of extension;
        # directory scans stay limited to known text extensions.
        if not single_file and path.suffix.lower() not in TEXT_EXTS:
            continue
        try:
            for lineno, line in enumerate(_read_text(path).splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{path.relative_to(root).as_posix()}:{lineno}: {line.strip()[:200]}")
                    if len(matches) >= 60:
                        return "\n".join(matches) + "\n… (more matches truncated)"
        except OSError:
            continue
    return "\n".join(matches) if matches else f"No matches for /{pattern}/"


def run_tests(project_root: Path, args: Dict[str, Any]) -> str:
    """Run the project's test suite (unittest discovery), capturing output."""
    cmd = args.get("command")
    if not cmd:
        # Default: discover tests under the conventional tests/ folder.
        test_dir = project_root / "tests"
        if not test_dir.is_dir():
            raise ToolError("No 'tests' directory found; pass a command explicitly.")
        cmd = f'"{_python()}" -m unittest discover -s tests'
    else:
        # Reject shell metacharacters outside quotes. Even with shell=False
        # they are inert, but failing fast is clearer and guards against
        # future refactors. Quote-aware so `pytest -k "a|b"` still works.
        if _has_unquoted_shell_meta(cmd):
            raise ToolError("Shell metacharacters are not allowed in test commands.")
    try:
        argv = shlex.split(cmd)
    except ValueError as exc:
        raise ToolError(f"Invalid command: {exc}") from exc
    if not argv:
        raise ToolError("Empty command.")
    # Only test commands are allowed. We never use a shell, so shell
    # metacharacters (;, &&, |, $(), backticks…) cannot inject anything.
    if not _is_test_invocation(argv):
        raise ToolError("Only test commands are allowed (unittest/pytest).")
    try:
        proc = subprocess.run(
            argv,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Tests timed out after 120s."
    output = (proc.stdout or "") + (proc.stderr or "")
    status = "PASS" if proc.returncode == 0 else "FAIL"
    return f"[tests exit code {proc.returncode} — {status}]\n{_truncate(output)}"


def _has_unquoted_shell_meta(cmd: str) -> bool:
    """Return True if a shell metacharacter appears outside quotes.

    Quote-aware so legitimate arguments like ``pytest -k "a|b"`` are not
    rejected, while unquoted ``;``, ``&&``, ``|``, backticks, ``$(`` and
    ``${`` (command substitution / chaining) still fail fast.
    """
    quote: Optional[str] = None
    i = 0
    n = len(cmd)
    while i < n:
        ch = cmd[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in ";|`&":
            return True
        elif ch == "$" and cmd[i : i + 2] in ("$(", "${"):
            return True
        i += 1
    return False


def _is_test_invocation(argv: List[str]) -> bool:
    """Return True if ``argv`` is a python/py/pytest test invocation.

    Allows ``python -m unittest``, ``python -m pytest``, ``py -3.11 -m
    unittest`` (version flag before ``-m``), and bare ``pytest``.  Runs with
    ``shell=False`` so nothing here can chain commands.
    """
    prog = Path(argv[0]).name.lower()
    if prog.startswith("pytest"):
        return True
    if not (prog.startswith("python") or prog in {"py", "py.exe"}):
        return False
    if "-m" not in argv:
        return False
    m_idx = argv.index("-m")
    # Everything between the interpreter and `-m` must be flags (version
    # selectors for the py launcher), not a script path — otherwise
    # `python foo.py -m unittest` would run an arbitrary script.
    if not all(a.startswith("-") for a in argv[1:m_idx]):
        return False
    return m_idx + 1 < len(argv) and argv[m_idx + 1] in {"unittest", "pytest"}


def _python() -> str:
    import sys

    return sys.executable


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _tool_schema(name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    _tool_schema(
        "read_file",
        "Read a text file from the project (paths are relative to the project root).",
        {"path": {"type": "string", "description": "Relative path to the file, e.g. 'panther/api.py'."}},
        ["path"],
    ),
    _tool_schema(
        "list_dir",
        "List the contents of a directory in the project.",
        {"path": {"type": "string", "description": "Relative directory path; default '.' (project root)."}},
        [],
    ),
    _tool_schema(
        "grep",
        "Search project text files for a regex pattern; returns file:line matches.",
        {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "path": {"type": "string", "description": "Optional subdirectory to restrict the search."},
        },
        ["pattern"],
    ),
    _tool_schema(
        "run_tests",
        "Run the project's test suite (unittest or pytest) and report results.",
        {"command": {"type": "string", "description": "Optional test command, e.g. 'python -m unittest discover -s tests'."}},
        [],
    ),
]

TOOL_NAMES = {schema["function"]["name"] for schema in TOOL_SCHEMAS}

_EXECUTORS: Dict[str, Callable[[Path, Dict[str, Any]], str]] = {
    "read_file": read_file,
    "list_dir": list_dir,
    "grep": grep,
    "run_tests": run_tests,
}


def execute_tool(name: str, args: Dict[str, Any], project_root: Path) -> str:
    """Execute ``name`` with ``args``; returns a string result for the model."""
    if name not in _EXECUTORS:
        raise ToolError(f"Unknown tool: {name}")
    return _EXECUTORS[name](project_root, args or {})


def parse_tool_arguments(raw: str) -> Dict[str, Any]:
    """Best-effort JSON parse of a tool call's arguments string."""
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {}
