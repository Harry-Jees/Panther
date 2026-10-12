from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .permissions import check_permission
from .tools import TOOLS


SYSTEM_PROMPT = """You are Panther, a helpful local-first coding assistant.

Panther was created by Harry Jees.

Accuracy rules:
- Do not hallucinate. Never invent facts, capabilities, tool results, file contents, actions, citations, or project details.
- Ground claims in the conversation, repository files, tool results, or other evidence actually available to you.
- If information is missing or uncertain, say so clearly and ask the user for clarification instead of guessing.
- Do not claim that an action succeeded unless a tool result confirms it.
- When the user asks for a preference or product detail that has not been provided, ask them rather than choosing silently.
"""


def _parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    candidates = [text.strip()]
    match = re.search(r"<function=(\w+)>([\s\S]*?)</function>", text)
    if match:
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            args = {}
        return match.group(1), args if isinstance(args, dict) else {}
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("name"), str):
            args = parsed.get("arguments", {})
            return parsed["name"], args if isinstance(args, dict) else {}
    return None


def run_agent(
    task: str,
    provider: Any,
    history: list[dict[str, str]],
    project_root: Path,
    permission_mode: str,
    *,
    one_shot: bool = True,
    approve: Callable[[str, dict[str, Any]], bool] | None = None,
    on_response: Callable[[str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    max_iterations: int = 50,
) -> str:
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    if not history:
        history.append(system_message)
    elif history[0].get("role") == "system":
        history[0] = system_message
    else:
        history.insert(0, system_message)
    history.append({"role": "user", "content": task})
    tool_schemas = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.schema,
        }
        for tool in TOOLS
    ]

    for iteration in range(max_iterations):
        if on_status:
            on_status("thinking" if iteration == 0 else "continuing")
        response = provider.generate(history, tool_schemas)
        if on_response:
            on_response(response)
        call = _parse_tool_call(response)
        if call is None:
            history.append({"role": "assistant", "content": response})
            return response

        tool_name, arguments = call
        tool = next((candidate for candidate in TOOLS if candidate.name == tool_name), None)
        if tool is None:
            history.append({"role": "assistant", "content": response})
            history.append({"role": "user", "content": f"Unknown tool '{tool_name}'. Choose one of: {', '.join(t.name for t in TOOLS)}"})
            continue

        decision = check_permission(tool.risk_level, permission_mode, one_shot=one_shot)
        if decision == "needs_prompt":
            if on_status:
                on_status("waiting for permission")
            decision = "approved" if approve and approve(tool_name, arguments) else "denied"
        if decision == "denied":
            result = {"success": False, "error": "Tool denied by permission mode or user."}
        else:
            result = tool.execute(arguments, project_root)
        history.append({"role": "assistant", "content": response})
        history.append({"role": "tool", "content": json.dumps(result), "toolName": tool_name})

    raise RuntimeError(f"Agent loop reached the {max_iterations}-iteration safety limit.")
