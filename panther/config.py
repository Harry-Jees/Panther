from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "permissionMode": "ask_when_risky",
    "modelTier": "small",
    "modelFamily": "qwen",
    "activeProvider": "local",
    # NVIDIA advertises this fast agent-focused model as a free endpoint.
    # Users can override it with `panther config --model`.
    "remoteModel": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "telemetryOptIn": False,
}
PERMISSION_MODES = {"always_ask", "ask_when_risky", "never_ask"}
MODEL_TIERS = {"small", "large"}
PROVIDERS = {"local", "nvidia"}


def panther_dir() -> Path:
    return Path(os.environ.get("PANTHER_HOME", Path.home() / ".panther")).expanduser()


def config_path(project_root: Path | None = None) -> Path:
    override = os.environ.get("PANTHER_CONFIG_PATH")
    if override:
        return Path(override).expanduser()
    if project_root is not None:
        project_config = project_root / ".panther" / "config.json"
        if project_config.exists():
            return project_config
    return panther_dir() / "config.json"


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Config must contain a JSON object: {path}")
    return value


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = {**DEFAULT_CONFIG, **config}
    if merged["permissionMode"] not in PERMISSION_MODES:
        raise ValueError(f"Unknown permission mode: {merged['permissionMode']}")
    if merged["modelTier"] not in MODEL_TIERS:
        raise ValueError(f"Unknown model tier: {merged['modelTier']}")
    if merged["activeProvider"] not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {merged['activeProvider']}"
        )
    if not isinstance(merged["remoteModel"], str) or not merged["remoteModel"].strip():
        raise ValueError("remoteModel must be a non-empty string")
    if not isinstance(merged["telemetryOptIn"], bool):
        raise ValueError("telemetryOptIn must be a boolean")
    return merged


def load_config(project_root: Path | None = None) -> dict[str, Any]:
    global_config = _read_config(config_path())
    project_config: dict[str, Any] = {}
    if project_root is not None:
        project_path = project_root / ".panther" / "config.json"
        project_config = _read_config(project_path)
    return validate_config({**global_config, **project_config})


def save_config(config: dict[str, Any], project_root: Path | None = None) -> Path:
    validated = validate_config(config)
    destination = (
        project_root / ".panther" / "config.json"
        if project_root is not None
        else config_path()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    overrides = {
        key: value
        for key, value in validated.items()
        if value != DEFAULT_CONFIG[key]
    }
    destination.write_text(
        json.dumps(overrides, indent=2) + "\n", encoding="utf-8"
    )
    return destination
