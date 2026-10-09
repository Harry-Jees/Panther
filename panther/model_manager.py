from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .config import panther_dir

MODEL_MANIFEST = {
    "small": {
        "name": "Qwen3-Coder-4B-Instruct-Q4_K_M",
        "repo": "Qwen/Qwen3-Coder-4B-Instruct-GGUF",
        "filename": "Qwen3-Coder-4B-Instruct-Q4_K_M.gguf",
        "contextWindow": 32768,
        "expectedHash": "",
    },
    "large": {
        "name": "Qwen3-Coder-80B-A3B-Q4_K_M",
        "repo": "Qwen/Qwen3-Coder-80B-A3B-GGUF",
        "filename": "Qwen3-Coder-80B-A3B-Q4_K_M.gguf",
        "contextWindow": 32768,
        "expectedHash": "",
    },
}


def context_window(tier: str) -> int:
    entry = MODEL_MANIFEST.get(tier)
    if entry is None:
        raise ValueError(f"Unknown model tier: {tier}")
    value = entry["contextWindow"]
    if not isinstance(value, int):
        raise ValueError(f"Invalid context window for model tier: {tier}")
    return value


def models_dir() -> Path:
    return panther_dir() / "models"


def model_path(tier: str) -> Path:
    if tier not in MODEL_MANIFEST:
        raise ValueError(f"Unknown model tier: {tier}")
    return models_dir() / MODEL_MANIFEST[tier]["filename"]


def list_models() -> list[dict[str, object]]:
    return [
        {
            "tier": tier,
            "name": details["name"],
            "path": str(model_path(tier)),
            "downloaded": model_path(tier).is_file(),
        }
        for tier, details in MODEL_MANIFEST.items()
    ]


def ensure_model(
    tier: str,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    entry = MODEL_MANIFEST.get(tier)
    if entry is None:
        raise ValueError(f"Unknown model tier: {tier}")
    destination = model_path(tier)
    if destination.is_file():
        return destination

    models_dir().mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".download")
    offset = partial.stat().st_size if partial.exists() else 0
    url = f"https://huggingface.co/{entry['repo']}/resolve/main/{entry['filename']}"
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    try:
        response = urllib.request.urlopen(request, timeout=60)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not download model: {exc}") from exc

    with response:
        status = getattr(response, "status", 200)
        if offset and status != 206:
            offset = 0
        content_range = response.headers.get("Content-Range", "")
        try:
            total = int(content_range.rsplit("/", 1)[1]) if content_range else int(response.headers.get("Content-Length", "0")) + offset
        except ValueError:
            total = 0
        digest = hashlib.sha256()
        if offset:
            with partial.open("rb") as existing:
                for block in iter(lambda: existing.read(1024 * 1024), b""):
                    digest.update(block)
        mode = "ab" if offset else "wb"
        downloaded = offset
        with partial.open(mode) as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
                downloaded += len(block)
                if on_progress:
                    on_progress(downloaded, total)

    expected_hash = str(entry["expectedHash"])
    actual_hash = digest.hexdigest()
    if expected_hash and actual_hash != expected_hash:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {entry['filename']}")
    partial.replace(destination)
    metadata = {
        "name": entry["name"],
        "tier": tier,
        "contextWindow": entry["contextWindow"],
        "path": str(destination),
        "sha256": actual_hash,
    }
    (models_dir() / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def switch_model(
    tier: str,
    *,
    confirm_delete: bool,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    new_model = ensure_model(tier, on_progress)
    other_tier = "large" if tier == "small" else "small"
    old_model = model_path(other_tier)
    if confirm_delete and old_model.exists():
        old_model.unlink()
    return new_model
