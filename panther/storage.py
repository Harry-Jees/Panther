"""Session persistence for Panther CLI.

Sessions are stored as JSON transcripts under ``~/.panther/sessions/`` and can
be resumed later with ``panther resume <session-id>``.
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def _short_session_id() -> str:
    """Short, time-sortable session id (~7 chars), e.g. ``2h9k4cq``.

    Six base36 digits of the epoch timestamp keep ids chronological (same
    length ⇒ lexicographic order ≈ creation order) and unique per second;
    a random trailing digit guards against two sessions created in the same
    second.
    """

    def _b36(value: int) -> str:
        digits = []
        while value:
            value, rem = divmod(value, 36)
            digits.append(_BASE36[rem])
        return "".join(reversed(digits)) or "0"

    return _b36(int(time.time())) + _b36(random.randrange(36))


def create_session(project_root: Path) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    session_id = _short_session_id()
    return {
        "session_id": session_id,
        "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_path": str(project_root),
        "repo_name": Path(project_root).name or str(project_root),
        "messages": [],
        "stats": {
            "questions": 0,
            "total_messages": 0,
            "total_tokens_used": 0,
            "api_calls": {},
        },
    }


def _sessions_dir() -> Path:
    from . import config

    return config.sessions_path()


def save_session(session: Dict[str, Any]) -> None:
    """Persist a session to ``~/.panther/sessions/<id>.json``."""
    path = _sessions_dir() / f"{session['session_id']}.json"
    path.write_text(json.dumps(session, indent=2), encoding="utf-8")


def load_session(session_id: str) -> Dict[str, Any]:
    path = _sessions_dir() / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def delete_session(session_id: str) -> None:
    path = _sessions_dir() / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    path.unlink()


def list_sessions() -> List[Dict[str, str]]:
    sessions: List[Dict[str, str]] = []
    # Old sessions used a ``session_YYYYMMDD_HHMMSS`` prefix; new ones use a
    # short random id. Match every JSON file and read the real id from its
    # contents (skipping non-session files defensively).
    for path in sorted(_sessions_dir().glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sessions.append(
                {
                    "session_id": data.get("session_id", path.stem),
                    "created_at": data.get("created_at", ""),
                    "repo_path": data.get("repo_path", ""),
                    "messages": str(len(data.get("messages", []))),
                }
            )
        except (json.JSONDecodeError, OSError):
            continue
    return sessions


def format_stats(session: Dict[str, Any]) -> str:
    """Human-readable summary of a session's stats."""
    stats = session.get("stats", {})
    calls = stats.get("api_calls") or {}
    call_str = ", ".join(f"{k}: {v}" for k, v in calls.items()) or "none"
    return (
        f"Questions asked: {stats.get('questions', 0)} | "
        f"Tokens used: {stats.get('total_tokens_used', 0):,} | "
        f"API calls ({call_str})"
    )
