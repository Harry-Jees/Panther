from __future__ import annotations

from typing import Literal

PermissionDecision = Literal["approved", "denied", "needs_prompt"]
_MODES = {
    "always_ask": {"safe": "needs_prompt", "risky": "needs_prompt"},
    "ask_when_risky": {"safe": "approved", "risky": "needs_prompt"},
    "never_ask": {"safe": "approved", "risky": "approved"},
}


def check_permission(risk_level: str, mode: str, *, one_shot: bool = False) -> PermissionDecision:
    if risk_level not in {"safe", "risky"}:
        raise ValueError(f"Unknown tool risk level: {risk_level}")
    decision = _MODES.get(mode, _MODES["always_ask"])[risk_level]
    if one_shot and decision == "needs_prompt":
        return "denied"
    return decision  # type: ignore[return-value]
