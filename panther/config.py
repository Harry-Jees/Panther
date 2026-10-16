from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "permissionMode": "ask_when_risky",
    "autopilot": False,
    "rememberedPermissions": [],
    "modelTier": "small",
    "modelFamily": "qwen",
    "activeProvider": "local",
    "activeBot": "",
    "bots": {},
    "apiBaseUrl": "",
    "apiModel": "",
    # NVIDIA advertises this fast agent-focused model as a free endpoint.
    # Users can override it with `panther config --model`.
    "remoteModel": "nvidia/nemotron-3.5-lightning-30b-a3b",
    # Dedicated coding route. This endpoint was verified against the current
    # configured NVIDIA account; users can override it with --code-model.
    "codeModel": "openai/gpt-oss-20b",
    # Multimodal model used for image questions.
    "visionModel": "google/gemma-4-31b-it",
    "telemetryOptIn": False,
    "onboardingComplete": False,
    "shareProfileWithNvidia": False,
}
PERMISSION_MODES = {"always_ask", "ask_when_risky", "never_ask"}
MODEL_TIERS = {"small", "large"}
PROVIDERS = {"local", "nvidia", "gemini", "openai-compatible"}
BOT_PROVIDERS = {"nvidia", "gemini", "openai-compatible"}


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
    if not isinstance(merged["autopilot"], bool):
        raise ValueError("autopilot must be a boolean")
    remembered_permissions = merged["rememberedPermissions"]
    if (
        not isinstance(remembered_permissions, list)
        or any(
            not isinstance(tool_name, str) or not tool_name.strip()
            for tool_name in remembered_permissions
        )
    ):
        raise ValueError("rememberedPermissions must be a list of tool names")
    if merged["modelTier"] not in MODEL_TIERS:
        raise ValueError(f"Unknown model tier: {merged['modelTier']}")
    active_provider = merged["activeProvider"]
    if not isinstance(active_provider, str) or active_provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {active_provider}"
        )
    if not isinstance(merged["remoteModel"], str) or not merged["remoteModel"].strip():
        raise ValueError("remoteModel must be a non-empty string")
    if not isinstance(merged["codeModel"], str) or not merged["codeModel"].strip():
        raise ValueError("codeModel must be a non-empty string")
    if not isinstance(merged["visionModel"], str) or not merged["visionModel"].strip():
        raise ValueError("visionModel must be a non-empty string")
    if not isinstance(merged["apiBaseUrl"], str):
        raise ValueError("apiBaseUrl must be a string")
    if not isinstance(merged["apiModel"], str):
        raise ValueError("apiModel must be a string")
    if active_provider == "openai-compatible":
        if not merged["apiBaseUrl"].strip():
            raise ValueError("Set apiBaseUrl before activating the OpenAI-compatible provider")
        if not merged["apiModel"].strip():
            raise ValueError("Set apiModel before activating the OpenAI-compatible provider")
    bots = merged["bots"]
    if not isinstance(bots, dict):
        raise ValueError("bots must be an object")
    for name, profile in bots.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Bot names must be non-empty strings")
        if not isinstance(profile, dict):
            raise ValueError(f"Bot profile '{name}' must be an object")
        bot_provider = profile.get("provider")
        if not isinstance(bot_provider, str) or bot_provider not in BOT_PROVIDERS:
            raise ValueError(f"Unknown provider for bot '{name}': {bot_provider}")
        if not isinstance(profile.get("model"), str) or not profile["model"].strip():
            raise ValueError(f"Bot '{name}' must have a non-empty model")
        if bot_provider == "openai-compatible":
            base_url = profile.get("baseUrl")
            if not isinstance(base_url, str) or not base_url.strip():
                raise ValueError(f"Bot '{name}' must have a baseUrl")
    active_bot = merged["activeBot"]
    if not isinstance(active_bot, str):
        raise ValueError("activeBot must be a string")
    if active_bot and active_bot not in bots:
        raise ValueError(f"Active bot does not exist: {active_bot}")
    if not isinstance(merged["telemetryOptIn"], bool):
        raise ValueError("telemetryOptIn must be a boolean")
    if not isinstance(merged["onboardingComplete"], bool):
        raise ValueError("onboardingComplete must be a boolean")
    if not isinstance(merged["shareProfileWithNvidia"], bool):
        raise ValueError("shareProfileWithNvidia must be a boolean")
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
    existing_config = _read_config(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    overrides = {
        key: value
        for key, value in validated.items()
        if value != DEFAULT_CONFIG[key]
    }
    if "onboardingComplete" in existing_config:
        onboarding_complete = (
            existing_config["onboardingComplete"] is True
            or validated["onboardingComplete"] is True
        )
    else:
        onboarding_complete = (
            destination.exists() or validated["onboardingComplete"] is True
        )
    overrides["onboardingComplete"] = onboarding_complete
    destination.write_text(
        json.dumps(overrides, indent=2) + "\n", encoding="utf-8"
    )
    return destination
