from __future__ import annotations

from typing import Literal

PermissionDecision = Literal["approved", "denied", "needs_prompt"]
_MODES = {
    "always_ask": {"safe": "needs_prompt", "risky": "needs_prompt"},
    "ask_when_risky": {"safe": "approved", "risky": "needs_prompt"},
    "never_ask": {"safe": "approved", "risky": "approved"},
}


def check_permission(
    risk_level: str,
    mode: str,
    *,
    capability: str = "general",
    one_shot: bool = False,
    remembered: bool = False,
    autopilot: bool = False,
) -> PermissionDecision:
    """Evaluate a tool capability under the configured policy.

    Capability is intentionally explicit even when the legacy two-level policy
    is used; it gives the CLI and future policy stores a stable audit label.
    """
    if risk_level not in {"safe", "risky"}:
        raise ValueError(f"Unknown tool risk level: {risk_level}")
    if not capability or not isinstance(capability, str):
        raise ValueError("Tool capability must be a non-empty string")
    if capability == "git_remote":
        return "denied" if one_shot else "needs_prompt"
    if autopilot or remembered:
        return "approved"
    decision = _MODES.get(mode, _MODES["always_ask"])[risk_level]
    if one_shot and decision == "needs_prompt":
        return "denied"
    return decision  # type: ignore[return-value]
