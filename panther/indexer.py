from __future__ import annotations

import json
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .tools import IGNORED_PARTS


@dataclass(frozen=True)
class IndexEvent:
    worker: str
    status: str
    completed: int
    total: int
    message: str = ""


def _files(root: Path) -> list[Path]:
    return sorted(
        (
            path for path in root.rglob("*")
            if path.is_file() and not any(part in IGNORED_PARTS for part in path.parts)
        ),
        key=lambda path: path.as_posix().lower(),
    )


def _symbols(text: str) -> list[str]:
    patterns = (
        r"^\s*(?:class|def|async\s+def)\s+([A-Za-z_]\w*)",
        r"^\s*(?:export\s+)?(?:class|function)\s+([A-Za-z_]\w*)",
    )
    found: list[str] = []
    for line in text.splitlines():
        for pattern in patterns:
            match = re.match(pattern, line)
            if match and match.group(1) not in found:
                found.append(match.group(1))
    return found[:200]


def _index_file(
    path: Path,
    root: Path,
    previous: dict[str, object] | None = None,
) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    stat = path.stat()
    suffix = path.suffix.lower()
    if previous and previous.get("mtimeNs") == stat.st_mtime_ns and previous.get("bytes") == stat.st_size:
        cached = dict(previous)
        cached["cached"] = True
        return cached
    item: dict[str, object] = {
        "path": relative,
        "suffix": suffix,
        "bytes": stat.st_size,
        "mtimeNs": stat.st_mtime_ns,
        "language": {
            ".py": "python", ".js": "javascript", ".jsx": "javascript",
            ".ts": "typescript", ".tsx": "typescript", ".md": "markdown",
            ".java": "java", ".go": "go", ".rs": "rust",
        }.get(suffix, "text"),
    }
    if suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".md"}:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:120_000]
            item["lines"] = text.count("\n") + (1 if text else 0)
            item["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if suffix != ".md":
                item["symbols"] = _symbols(text)
                item["imports"] = sorted(set(re.findall(r"^(?:from|import)\s+([\w.]+)", text, re.MULTILINE)))[:100]
        except OSError as exc:
            item["error"] = str(exc)
    return item


def build_index(
    root: Path,
    *,
    on_event: Callable[[IndexEvent], None] | None = None,
    worker_count: int | None = None,
) -> dict[str, object]:
    """Build a compact repository index using parallel indexing workers."""
    root = root.resolve()
    paths = _files(root)
    previous_items: dict[str, dict[str, object]] = {}
    previous_path = root / ".panther" / "index.json"
    try:
        previous_value = json.loads(previous_path.read_text(encoding="utf-8"))
        previous_items = {
            str(item["path"]): item
            for item in previous_value.get("files", [])
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    total = len(paths)
    count = worker_count or min(8, max(2, (os.cpu_count() or 2)))
    chunks = [paths[index::count] for index in range(count)]
    entries: list[dict[str, object]] = []
    completed = 0

    def work(worker_number: int, chunk: list[Path]) -> list[dict[str, object]]:
        name = f"index-{worker_number}"
        if on_event:
            on_event(IndexEvent(name, "running", 0, len(chunk), "scanning files"))
        result: list[dict[str, object]] = []
        for completed, path in enumerate(chunk, 1):
            result.append(_index_file(path, root, previous_items.get(path.relative_to(root).as_posix())))
            if on_event and (completed == len(chunk) or completed % 10 == 0):
                on_event(IndexEvent(name, "running", completed, len(chunk), f"indexed {completed}/{len(chunk)}"))
        if on_event:
            on_event(IndexEvent(name, "complete", len(chunk), len(chunk), "ready"))
        return result

    with ThreadPoolExecutor(max_workers=count, thread_name_prefix="panther-index") as pool:
        futures = [pool.submit(work, number + 1, chunk) for number, chunk in enumerate(chunks) if chunk]
        for future in as_completed(futures):
            batch = future.result()
            entries.extend(batch)
            completed += len(batch)
            if on_event:
                on_event(IndexEvent("coordinator", "running", completed, total, "merging results"))

    entries.sort(key=lambda item: str(item["path"]).lower())
    result: dict[str, object] = {
        "version": 2,
        "root": str(root),
        "fileCount": len(entries),
        "files": entries,
    }
    destination = root / ".panther" / "index.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if on_event:
        on_event(IndexEvent("coordinator", "complete", total, total, str(destination)))
    return result
