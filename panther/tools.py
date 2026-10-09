from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

IGNORED_PARTS = {
    ".git",
    ".panther",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    risk_level: str
    schema: dict[str, Any]
    execute: Callable[[dict[str, Any], Path], dict[str, Any]]


def _result(success: bool, *, data: str = "", error: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {"success": success}
    if data:
        result["data"] = data
    if error:
        result["error"] = error
    return result


def _project_path(root: Path, value: str) -> Path:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Path must remain within the project root") from exc
    return candidate


def read_file(args: dict[str, Any], root: Path) -> dict[str, Any]:
    file_path = args.get("filePath")
    if not isinstance(file_path, str) or not file_path:
        return _result(False, error="Missing required argument: filePath")
    try:
        return _result(True, data=_project_path(root, file_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        return _result(False, error=f"Failed to read file: {exc}")


def write_file(args: dict[str, Any], root: Path) -> dict[str, Any]:
    file_path, content = args.get("filePath"), args.get("content")
    if not isinstance(file_path, str) or not file_path:
        return _result(False, error="Missing required argument: filePath")
    if not isinstance(content, str):
        return _result(False, error="Missing required argument: content")
    try:
        target = _project_path(root, file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return _result(True, data=f"File written: {file_path}")
    except (OSError, ValueError) as exc:
        return _result(False, error=f"Failed to write file: {exc}")


def list_dir(args: dict[str, Any], root: Path) -> dict[str, Any]:
    directory = args.get("path", ".")
    if not isinstance(directory, str):
        return _result(False, error="path must be a string")
    try:
        target = _project_path(root, directory)
        entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        return _result(True, data="\n".join(
            f"{'[dir] ' if entry.is_dir() else ''}{entry.name}"
            for entry in entries if entry.name not in IGNORED_PARTS
        ))
    except (OSError, ValueError) as exc:
        return _result(False, error=f"Failed to list directory: {exc}")


def grep(args: dict[str, Any], root: Path) -> dict[str, Any]:
    pattern = args.get("pattern")
    search_path = args.get("path", ".")
    glob = args.get("glob", "*")
    if not isinstance(pattern, str) or not pattern:
        return _result(False, error="Missing required argument: pattern")
    if not isinstance(search_path, str) or not isinstance(glob, str):
        return _result(False, error="path and glob must be strings")
    try:
        base = _project_path(root, search_path)
        matcher = re.compile(pattern)
        matches: list[str] = []
        for path in ([base] if base.is_file() else base.rglob("*")):
            if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
                continue
            if not fnmatch.fnmatch(path.name, glob):
                continue
            try:
                for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if matcher.search(line):
                        matches.append(f"{path.relative_to(root)}:{line_no}:{line}")
                        if len(matches) >= 100:
                            return _result(True, data="\n".join(matches))
            except (OSError, UnicodeError):
                continue
        return _result(True, data="\n".join(matches) if matches else "No matches found.")
    except (OSError, ValueError, re.error) as exc:
        return _result(False, error=f"Search failed: {exc}")


def run_command(args: dict[str, Any], root: Path) -> dict[str, Any]:
    command, timeout = args.get("command"), args.get("timeout", 30)
    if not isinstance(command, str) or not command.strip():
        return _result(False, error="Missing required argument: command")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        return _result(False, error="timeout must be a positive number of seconds")
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=min(timeout, 300),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _result(False, error=f"Command failed: {exc}")
    output = (completed.stdout + completed.stderr).strip()
    return {
        "success": completed.returncode == 0,
        "data": output or "(command completed with no output)",
        "meta": {"exitCode": completed.returncode},
    }


def run_tests(args: dict[str, Any], root: Path) -> dict[str, Any]:
    command = args.get("command", f'"{sys.executable}" -m unittest discover -s tests -v')
    if not isinstance(command, str) or not command.strip():
        return _result(False, error="command must be a non-empty string")
    return run_command({"command": command, "timeout": args.get("timeout", 300)}, root)


def apply_diff(args: dict[str, Any], root: Path) -> dict[str, Any]:
    file_path, diff = args.get("filePath"), args.get("diff")
    if not isinstance(file_path, str) or not file_path:
        return _result(False, error="Missing required argument: filePath")
    if not isinstance(diff, str) or not diff:
        return _result(False, error="Missing required argument: diff")
    try:
        target = _project_path(root, file_path)
        original = target.read_text(encoding="utf-8") if target.exists() else ""
        source = original.splitlines(keepends=True)
        output: list[str] = []
        lines = diff.splitlines(keepends=True)
        cursor = 0
        hunks = 0
        position = 0
        while position < len(lines):
            line = lines[position]
            position += 1
            if not line.startswith("@@"):
                continue
            match = re.match(
                r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@",
                line,
            )
            if not match:
                return _result(False, error=f"Invalid diff hunk: {line.strip()}")
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) is not None else 1
            new_count = int(match.group(4)) if match.group(4) is not None else 1
            start = old_start if old_count == 0 else old_start - 1
            if start < cursor or start > len(source):
                return _result(False, error="Diff hunk is outside the target file")
            output.extend(source[cursor:start])
            cursor = start
            consumed = 0
            produced = 0
            while position < len(lines) and not lines[position].startswith("@@"):
                hunk_line = lines[position]
                if hunk_line.startswith("\\"):
                    position += 1
                    continue
                if hunk_line.startswith((" ", "-")):
                    expected = hunk_line[1:]
                    if cursor >= len(source) or source[cursor].rstrip("\r\n") != expected.rstrip("\r\n"):
                        return _result(False, error=f"Diff context mismatch near line {cursor + 1}")
                    if hunk_line.startswith(" "):
                        output.append(source[cursor])
                        produced += 1
                    cursor += 1
                    consumed += 1
                elif hunk_line.startswith("+"):
                    output.append(hunk_line[1:])
                    produced += 1
                else:
                    break
                position += 1
            if consumed != old_count or produced != new_count:
                return _result(False, error="Diff hunk line counts do not match its header")
            hunks += 1
        if hunks == 0:
            return _result(False, error="Diff contains no valid hunks")
        output.extend(source[cursor:])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(output), encoding="utf-8")
        return _result(True, data=f"Diff applied to: {file_path}")
    except (OSError, ValueError) as exc:
        return _result(False, error=f"Failed to apply diff: {exc}")


TOOLS = [
    Tool(
        "read_file",
        "Read a project file",
        "safe",
        {"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]},
        read_file,
    ),
    Tool(
        "list_dir",
        "List project directory contents",
        "safe",
        {"type": "object", "properties": {"path": {"type": "string"}}},
        list_dir,
    ),
    Tool(
        "grep",
        "Search project files using a regular expression",
        "safe",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
            },
            "required": ["pattern"],
        },
        grep,
    ),
    Tool(
        "run_command",
        "Run a shell command in the project root",
        "risky",
        {
            "type": "object",
            "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}},
            "required": ["command"],
        },
        run_command,
    ),
    Tool(
        "run_tests",
        "Run the project's Python unit tests",
        "risky",
        {
            "type": "object",
            "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}},
        },
        run_tests,
    ),
    Tool(
        "write_file",
        "Write a file inside the project root",
        "risky",
        {
            "type": "object",
            "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}},
            "required": ["filePath", "content"],
        },
        write_file,
    ),
    Tool(
        "apply_diff",
        "Apply a unified diff to a project file",
        "risky",
        {
            "type": "object",
            "properties": {"filePath": {"type": "string"}, "diff": {"type": "string"}},
            "required": ["filePath", "diff"],
        },
        apply_diff,
    ),
]
