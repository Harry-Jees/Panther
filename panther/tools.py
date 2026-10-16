from __future__ import annotations

import fnmatch
import html
from html.parser import HTMLParser
import json
import re
import shlex
import subprocess
import sys
import secrets
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .mysql import mysql_execute, mysql_read

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
    capability: str = "general"


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


class _WebTextParser(HTMLParser):
    """Extract readable text while ignoring scripts and page styling."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0
        self._link: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            self._link = dict(attrs).get("href")
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a":
            self._link = None
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            text = " ".join(html.unescape(data).split())
            if text:
                self.parts.append(f"{text} ({self._link})" if self._link else text)


def _download_web_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        content_type = response.headers.get_content_type()
        payload = response.read(1_500_000)
    if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
        return f"Fetched {url} ({content_type}, {len(payload)} bytes)."
    if content_type == "text/plain":
        return payload.decode("utf-8", errors="replace")[:30_000]
    parser = _WebTextParser()
    parser.feed(payload.decode("utf-8", errors="replace"))
    return "\n".join(parser.parts)[:30_000]


def _plain_web_text(value: str) -> str:
    parser = _WebTextParser()
    parser.feed(value)
    return " ".join(parser.parts)


def _bing_rss_search(query: str) -> str:
    url = "https://www.bing.com/search?" + urllib.parse.urlencode(
        {"q": query, "format": "rss"}
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Panther/0.1 (desktop coding assistant)",
            "Accept": "application/rss+xml, application/xml, text/xml",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read(1_500_000)
    feed = ET.fromstring(payload)
    items = feed.findall("./channel/item")
    results: list[str] = []
    for item in items:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = _plain_web_text(item.findtext("description") or "")
        parsed_link = urllib.parse.urlparse(link)
        if (
            not title
            or parsed_link.scheme not in {"http", "https"}
            or not parsed_link.netloc
        ):
            continue
        results.append(
            f"- {title}\n  {link}"
            + (f"\n  {description}" if description else "")
        )
    if not results:
        raise ValueError("Bing RSS returned no parseable search results.")
    return "Search provider: Bing RSS\n" + "\n".join(results[:10])


def _wikipedia_search(query: str) -> str:
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 8,
            "format": "json",
        }
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Panther/0.1 (desktop coding assistant)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.loads(response.read(1_500_000).decode("utf-8"))
    pages = result.get("query", {}).get("search", [])
    if not isinstance(pages, list):
        raise ValueError("Wikipedia search returned an invalid response.")
    results: list[str] = []
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("title"), str):
            continue
        title = page["title"]
        snippet = _plain_web_text(str(page.get("snippet", "")))
        page_url = (
            "https://en.wikipedia.org/wiki/"
            + urllib.parse.quote(title.replace(" ", "_"))
        )
        results.append(
            f"- {title}\n  {page_url}" + (f"\n  {snippet}" if snippet else "")
        )
    if not results:
        raise ValueError("Wikipedia search returned no matching pages.")
    return "Search provider: Wikipedia\n" + "\n".join(results[:8])


def _external_search_sources(data: str) -> list[str]:
    sources: list[str] = []
    candidates = [
        urllib.parse.unquote(value)
        for value in re.findall(r"uddg=([^&\s)]+)", data)
    ]
    candidates.extend(re.findall(r"https?://[^\s)<>\"']+", data))
    search_hosts = {
        "bing.com",
        "duckduckgo.com",
        "google.com",
        "wikipedia.org",
    }
    for candidate in candidates:
        source = candidate.rstrip(".,;]")
        parsed = urllib.parse.urlparse(source)
        hostname = (parsed.hostname or "").lower()
        if (
            parsed.scheme not in {"http", "https"}
            or not hostname
            or any(hostname == host or hostname.endswith("." + host) for host in search_hosts)
            or source in sources
        ):
            continue
        sources.append(source)
    return sources


def _search_web(query: str) -> str:
    """Search public providers and reject blocked pages masquerading as results."""
    failures: list[str] = []
    try:
        return _bing_rss_search(query)
    except (OSError, ValueError, UnicodeError, ET.ParseError) as exc:
        failures.append(f"Bing RSS: {exc}")

    providers = (
        ("DuckDuckGo", "https://html.duckduckgo.com/html/?"),
        ("Google", "https://www.google.com/search?"),
    )
    for name, endpoint in providers:
        try:
            data = _download_web_text(
                endpoint + urllib.parse.urlencode({"q": query, "num": 8})
            )
            normalized = data.lower()
            blocked_google = (
                name == "Google"
                and "trouble accessing google search" in normalized
            )
            blocked_duckduckgo = (
                name == "DuckDuckGo"
                and any(
                    phrase in normalized
                    for phrase in ("error-lite", "all regions")
                )
            )
            sources = _external_search_sources(data)
            if data.strip() and sources and not blocked_google and not blocked_duckduckgo:
                source_block = "\nSources:\n" + "\n".join(
                    f"- {source}" for source in sources[:12]
                )
                return f"Search provider: {name}\n{data}{source_block}"
            reason = (
                "blocked response"
                if blocked_google or blocked_duckduckgo
                else "no external result links"
            )
            failures.append(f"{name}: {reason}")
        except (OSError, ValueError, UnicodeError, ET.ParseError) as exc:
            failures.append(f"{name}: {exc}")
    try:
        return _wikipedia_search(query)
    except (OSError, ValueError, UnicodeError, ET.ParseError) as exc:
        failures.append(f"Wikipedia: {exc}")
    raise RuntimeError("; ".join(failures) or "no search providers configured")


def browse_web(args: dict[str, Any], root: Path) -> dict[str, Any]:
    """Search the public web or fetch a supplied HTTP(S) page."""
    query = args.get("query")
    url = args.get("url")
    if query is not None and (not isinstance(query, str) or not query.strip()):
        return _result(False, error="query must be a non-empty string")
    if url is not None and (not isinstance(url, str) or not url.strip()):
        return _result(False, error="url must be a non-empty string")
    if query and url:
        return _result(False, error="Provide either query or url, not both")
    try:
        if query:
            data = _search_web(query)
            return _result(True, data=f"Search results for: {query}\n{data}")
        if not url:
            return _result(False, error="Provide a query or url")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return _result(False, error="url must be an absolute http(s) URL")
        return _result(True, data=f"Source: {url}\n{_download_web_text(url)}")
    except (OSError, ValueError, UnicodeError, RuntimeError) as exc:
        return _result(False, error=f"Web request failed: {exc}")


def read_file(args: dict[str, Any], root: Path) -> dict[str, Any]:
    file_path = args.get("filePath")
    if not isinstance(file_path, str) or not file_path:
        return _result(False, error="Missing required argument: filePath")
    try:
        return _result(True, data=_project_path(root, file_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        return _result(False, error=f"Failed to read file: {exc}")


def read_file_range(args: dict[str, Any], root: Path) -> dict[str, Any]:
    file_path = args.get("filePath")
    start_line = args.get("startLine", 1)
    end_line = args.get("endLine", start_line + 200)
    if not isinstance(file_path, str) or not file_path:
        return _result(False, error="Missing required argument: filePath")
    if not isinstance(start_line, int) or not isinstance(end_line, int) or start_line < 1 or end_line < start_line:
        return _result(False, error="startLine and endLine must be valid positive line numbers")
    try:
        lines = _project_path(root, file_path).read_text(encoding="utf-8").splitlines()
        selected = lines[start_line - 1:end_line]
        data = "\n".join(f"{number}: {line}" for number, line in enumerate(selected, start_line))
        return _result(True, data=data or f"{file_path} is empty in the requested range")
    except (OSError, ValueError, UnicodeError) as exc:
        return _result(False, error=f"Failed to read file range: {exc}")


def find_files(args: dict[str, Any], root: Path) -> dict[str, Any]:
    pattern = args.get("pattern", "*")
    directory = args.get("path", ".")
    if not isinstance(pattern, str) or not isinstance(directory, str):
        return _result(False, error="pattern and path must be strings")
    try:
        base = _project_path(root, directory)
        root_resolved = root.resolve()
        matches = [
            str(path.relative_to(root_resolved)).replace("\\", "/")
            for path in base.rglob(pattern)
            if path.is_file() and not any(part in IGNORED_PARTS for part in path.parts)
        ][:500]
        return _result(True, data="\n".join(matches) if matches else "No files found.")
    except (OSError, ValueError) as exc:
        return _result(False, error=f"File search failed: {exc}")


def query_index(args: dict[str, Any], root: Path) -> dict[str, Any]:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return _result(False, error="query must be a non-empty string")
    index_path = root / ".panther" / "index.json"
    try:
        value = json.loads(index_path.read_text(encoding="utf-8"))
        files = value.get("files", []) if isinstance(value, dict) else []
        query_lower = query.lower()
        matches = []
        for item in files:
            if not isinstance(item, dict):
                continue
            haystack = json.dumps(item, ensure_ascii=False).lower()
            if query_lower in haystack:
                matches.append(
                    f"{item.get('path', '?')}"
                    + (f" symbols={item['symbols']}" if item.get("symbols") else "")
                )
            if len(matches) >= 100:
                break
        return _result(True, data="\n".join(matches) if matches else "No indexed matches found.")
    except FileNotFoundError:
        return _result(False, error="Repository index not found. Run the index command first.")
    except (OSError, json.JSONDecodeError) as exc:
        return _result(False, error=f"Could not query repository index: {exc}")


def _git(args: list[str], root: Path, *, timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _result(False, error=f"Git command failed: {exc}")
    output = (completed.stdout + completed.stderr).strip()
    return {
        "success": completed.returncode == 0,
        "data": output or "(git completed with no output)",
        "meta": {"exitCode": completed.returncode},
    }


def git_status(args: dict[str, Any], root: Path) -> dict[str, Any]:
    _ = args
    return _git(["status", "--short", "--branch"], root)


def git_diff(args: dict[str, Any], root: Path) -> dict[str, Any]:
    path = args.get("path")
    if path is not None and not isinstance(path, str):
        return _result(False, error="path must be a string")
    command = ["diff", "--"] + ([path] if path else [])
    return _git(command, root)


def git_push(args: dict[str, Any], root: Path) -> dict[str, Any]:
    if args:
        return _result(False, error="git_push does not accept arguments")
    return _git(["push"], root, timeout=300)


def git_pull(args: dict[str, Any], root: Path) -> dict[str, Any]:
    if args:
        return _result(False, error="git_pull does not accept arguments")
    return _git(["pull"], root, timeout=300)


def create_checkpoint(args: dict[str, Any], root: Path) -> dict[str, Any]:
    """Save the current tracked Git diff before an agent edit."""
    _ = args
    result = _git(["diff", "--binary", "HEAD"], root)
    if not result["success"]:
        return result
    checkpoint_id = secrets.token_hex(6)
    directory = root / ".panther" / "checkpoints"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{checkpoint_id}.patch"
        target.write_text(result.get("data", ""), encoding="utf-8")
        return _result(True, data=f"Checkpoint created: {checkpoint_id}")
    except OSError as exc:
        return _result(False, error=f"Could not create checkpoint: {exc}")


def restore_checkpoint(args: dict[str, Any], root: Path) -> dict[str, Any]:
    checkpoint_id = args.get("checkpointId")
    if not isinstance(checkpoint_id, str) or not re.fullmatch(r"[a-f0-9]{12}", checkpoint_id):
        return _result(False, error="checkpointId must be a valid checkpoint identifier")
    target = root / ".panther" / "checkpoints" / f"{checkpoint_id}.patch"
    if not target.is_file():
        return _result(False, error=f"Checkpoint not found: {checkpoint_id}")
    return _git(["apply", "--binary", "--3way", str(target)], root)


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
        root_resolved = root.resolve()
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
                        matches.append(f"{path.relative_to(root_resolved)}:{line_no}:{line}")
                        if len(matches) >= 100:
                            return _result(True, data="\n".join(matches))
            except (OSError, UnicodeError):
                continue
        return _result(True, data="\n".join(matches) if matches else "No matches found.")
    except (OSError, ValueError, re.error) as exc:
        return _result(False, error=f"Search failed: {exc}")


def run_command(args: dict[str, Any], root: Path) -> dict[str, Any]:
    command, timeout = args.get("command"), args.get("timeout", 30)
    use_shell = args.get("shell", False)
    if not isinstance(command, str) or not command.strip():
        return _result(False, error="Missing required argument: command")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        return _result(False, error="timeout must be a positive number of seconds")
    if not isinstance(use_shell, bool):
        return _result(False, error="shell must be a boolean")
    try:
        executable: str | list[str] = command if use_shell else shlex.split(command, posix=sys.platform != "win32")
        if not executable:
            return _result(False, error="command cannot be empty")
        completed = subprocess.run(
            executable,
            cwd=root,
            shell=use_shell,
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
    return run_command({"command": command, "timeout": args.get("timeout", 300), "shell": args.get("shell", False)}, root)


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
        "browse_web",
        "Search the public web with query or fetch readable text from an HTTP(S) URL",
        "safe",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "url": {"type": "string"},
            },
        },
        browse_web,
    ),
    Tool(
        "mysql_read",
        "Run one bounded, read-only SELECT, SHOW, DESCRIBE, or EXPLAIN query against the configured project MySQL database; bind values with %s and parameters",
        "safe",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "parameters": {"type": "array"},
            },
            "required": ["query"],
        },
        mysql_read,
        capability="mysql_read",
    ),
    Tool(
        "mysql_execute",
        "Execute one data or schema change against the configured project MySQL database; requires fresh approval; bind values with %s and parameters",
        "risky",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "parameters": {"type": "array"},
            },
            "required": ["query"],
        },
        mysql_execute,
        capability="mysql_write",
    ),
    Tool(
        "read_file",
        "Read a project file",
        "safe",
        {"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]},
        read_file,
    ),
    Tool(
        "read_file_range",
        "Read a bounded line range from a project file with line numbers",
        "safe",
        {
            "type": "object",
            "properties": {
                "filePath": {"type": "string"},
                "startLine": {"type": "integer"},
                "endLine": {"type": "integer"},
            },
            "required": ["filePath"],
        },
        read_file_range,
    ),
    Tool(
        "find_files",
        "Find project files by glob pattern",
        "safe",
        {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
        },
        find_files,
    ),
    Tool(
        "query_index",
        "Search the parallel repository index for files and symbols",
        "safe",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        query_index,
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
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "number"},
                "shell": {"type": "boolean"},
            },
            "required": ["command"],
        },
        run_command,
    ),
    Tool(
        "git_status",
        "Show the current Git branch and working-tree changes",
        "safe",
        {"type": "object", "properties": {}},
        git_status,
    ),
    Tool(
        "git_diff",
        "Show uncommitted Git changes, optionally limited to a path",
        "safe",
        {"type": "object", "properties": {"path": {"type": "string"}}},
        git_diff,
    ),
    Tool(
        "git_push",
        "Push the current branch to its configured upstream; requires fresh interactive approval",
        "risky",
        {"type": "object", "properties": {}},
        git_push,
        capability="git_remote",
    ),
    Tool(
        "git_pull",
        "Pull from the current branch's configured upstream; requires fresh interactive approval",
        "risky",
        {"type": "object", "properties": {}},
        git_pull,
        capability="git_remote",
    ),
    Tool(
        "create_checkpoint",
        "Save the current tracked Git diff before making edits",
        "safe",
        {"type": "object", "properties": {}},
        create_checkpoint,
    ),
    Tool(
        "restore_checkpoint",
        "Restore a previously saved Git checkpoint",
        "risky",
        {
            "type": "object",
            "properties": {"checkpointId": {"type": "string"}},
            "required": ["checkpointId"],
        },
        restore_checkpoint,
    ),
    Tool(
        "run_tests",
        "Run the project's Python unit tests",
        "risky",
        {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "number"},
                "shell": {"type": "boolean"},
            },
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
