from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionManager:
    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir

    def _path(self, session_id: str) -> Path:
        if not session_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in session_id):
            raise ValueError("Invalid session ID")
        return self.sessions_dir / f"{session_id}.json"

    def create(self) -> dict[str, Any]:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        now = _now()
        session = {
            "id": secrets.token_urlsafe(9),
            "createdAt": now,
            "updatedAt": now,
            "history": [],
        }
        self.save(session)
        return session

    def save(self, session: dict[str, Any]) -> None:
        if not isinstance(session.get("history"), list):
            raise ValueError("Session history must be a list")
        session["updatedAt"] = _now()
        path = self._path(str(session.get("id", "")))
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")

    def load(self, session_id: str) -> dict[str, Any] | None:
        path = self._path(session_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid session file {path}: {exc}") from exc
        if not isinstance(data, dict) or data.get("id") != session_id or not isinstance(data.get("history"), list):
            raise ValueError(f"Invalid session data: {path}")
        return data

    def list(self) -> list[dict[str, Any]]:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        sessions = []
        for path in self.sessions_dir.glob("*.json"):
            session = self.load(path.stem)
            if session is not None:
                sessions.append(session)
        return sorted(sessions, key=lambda item: item.get("updatedAt", ""), reverse=True)

    def delete(self, session_id: str) -> bool:
        try:
            self._path(session_id).unlink()
        except FileNotFoundError:
            return False
        return True
