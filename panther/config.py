"""Configuration management for Panther CLI.

Configuration lives in ``~/.panther/config.json``.  It covers LLM provider
selection, retrieval settings, token budgeting, indexing rules, UI
preferences and API keys.  API keys can also be supplied through environment
variables, which always take precedence over the config file.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "llm": {
        "primary": {
            "provider": "nvidia_nim",
            "model": "nvidia/nemotron-3-super-120b-a12b",
            "temperature": 0.7,
            "max_tokens": 2048,
        },
    },
    "retrieval": {
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k_chunks": 5,
        # all-MiniLM-L6-v2 cosine similarities for code vs natural-language
        # queries typically land in the 0.25-0.55 range; 0.5 filters out
        # everything (verified empirically), so keep the default lower.
        "similarity_threshold": 0.25,
        # Optional: re-score retrieved chunks with a local cross-encoder for
        # better precision. Best-effort — falls back gracefully if the model
        # cannot be loaded.
        "rerank": False,
    },
    "tools": {
        "enabled": True,
        "max_turns": 6,
    },
    "token_budget": {
        "soft_limit_tpm": 30000,
        "max_prompt_tokens": 28000,
        "token_counter": "tiktoken",
    },
    "indexing": {
        "ignore_patterns": [".git", "node_modules", "__pycache__", ".venv", ".panther", "build", "dist", ".idea"],
        "file_extensions": [".py", ".sql", ".txt", ".md", ".toml", ".yaml", ".yml", ".json"],
    },
    "ui": {
        "gradient_enabled": True,
        "verbose_logging": False,
        "show_token_counts": True,
        "theme": "dark",
    },
    "storage": {
        "panther_home": "~/.panther",
        "db_type": "chroma",
        "session_format": "json",
    },
    "api_keys": {"nvidia_nim": None},
}

# Maps provider names to environment variable names.
ENV_KEY_MAP = {"nvidia_nim": "NVIDIA_NIM_API_KEY"}


def panther_home() -> Path:
    """Path to the global Panther data directory (``~/.panther``)."""
    return Path.home() / ".panther"


def config_path() -> Path:
    return panther_home() / "config.json"


def db_path() -> Path:
    return panther_home() / "db"


def sessions_path() -> Path:
    return panther_home() / "sessions"


def ensure_home() -> None:
    """Create the Panther directory structure if it does not exist yet."""
    panther_home().mkdir(parents=True, exist_ok=True)
    db_path().mkdir(parents=True, exist_ok=True)
    sessions_path().mkdir(parents=True, exist_ok=True)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge ``override`` into ``base`` (override wins per key)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config() -> Dict[str, Any]:
    """Load the merged configuration.

    Missing files raise ``FileNotFoundError`` (callers are expected to hint at
    ``panther init``).  Corrupted files fall back to defaults instead of
    crashing, mirroring the guide's error-handling strategy.
    """
    path = config_path()
    if not path.exists():
        raise FileNotFoundError("Panther is not initialized. Run `panther init` first.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return copy.deepcopy(DEFAULT_CONFIG)
    return _deep_merge(DEFAULT_CONFIG, raw)


def save_config(config: Dict[str, Any]) -> None:
    config_path().write_text(json.dumps(config, indent=2), encoding="utf-8")


def defaults() -> Dict[str, Any]:
    """A fresh copy of the default configuration (safe to mutate)."""
    return copy.deepcopy(DEFAULT_CONFIG)


def initialize() -> Dict[str, Any]:
    """Create the Panther home directory and write a default config file."""
    ensure_home()
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["storage"]["panther_home"] = str(panther_home())
    save_config(config)
    return config


def get_api_key(provider: str, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Resolve an API key: environment variable first, then config file."""
    env_var = ENV_KEY_MAP.get(provider)
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    if config:
        return (config.get("api_keys") or {}).get(provider)
    return None
