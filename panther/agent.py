"""Tool-calling agent loop: let the LLM inspect the codebase before answering.

The loop calls the provider with tool schemas, executes any requested tools
locally, feeds results back, and repeats until the model answers without
further tool calls (or the turn budget is exhausted).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import tools
from .api import LLMError, call_llm, stream_agent_chat

DEFAULT_MAX_TURNS = 6


class ToolEvent:
    """A single tool execution, surfaced to the UI for animation."""

    def __init__(self, name: str, args: Dict[str, Any], ok: bool, output: str) -> None:
        self.name = name
        self.args = args
        self.ok = ok
        self.output = output

    def describe(self) -> str:
        if self.name == "grep":
            return f"grep /{self.args.get('pattern', '')}/"
        if self.name == "read_file":
            return f"read {self.args.get('path', '')}"
        if self.name == "list_dir":
            return f"list {self.args.get('path', '.')}"
        if self.name == "run_tests":
            return "run tests"
        return f"{self.name}({self.args})"


class AgentResult:
    """Outcome of an agent turn."""

    def __init__(
        self,
        content: Optional[str],
        tool_events: List[ToolEvent],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        warning: Optional[str] = None,
        turns_used: int = 0,
    ) -> None:
        self.content = content
        self.tool_events = tool_events
        self.model = model
        self.provider = provider
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.warning = warning
        self.turns_used = turns_used


def run_agent(
    messages: List[Dict[str, str]],
    project_root: Path,
    cfg: Dict[str, Any],
    resolve_key: Optional[Callable[[str], Optional[str]]] = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    on_tool: Optional[Callable[[ToolEvent], None]] = None,
    on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> AgentResult:
    """Run the tool-calling loop and return the final answer.

    ``on_tool`` (if given) is called with a :class:`ToolEvent` after each tool
    executes.  ``on_tool_start(name, args)`` is called just before execution so
    the UI can wrap the real work in a loading spinner.
    """
    tools_enabled = (cfg.get("tools") or {}).get("enabled", True)
    schemas = tools.TOOL_SCHEMAS if tools_enabled else None

    history = list(messages)
    events: List[ToolEvent] = []
    total_prompt = 0
    total_completion = 0
    model: Optional[str] = None
    provider: Optional[str] = None

    for turn in range(max_turns):
        try:
            result = call_llm(history, cfg, resolve_key=resolve_key, tools=schemas)
        except LLMError as exc:
            return AgentResult(None, events, warning=str(exc), turns_used=turn + 1)

        total_prompt += result.get("prompt_tokens", 0)
        total_completion += result.get("completion_tokens", 0)
        model = result.get("model") or model
        provider = result.get("provider") or provider

        tool_calls = result.get("tool_calls") or []
        if not tool_calls:
            return AgentResult(
                result.get("content"),
                events,
                model=model,
                provider=provider,
                prompt_tokens=total_prompt,
                completion_tokens=total_completion,
                turns_used=turn + 1,
            )

        # Record the assistant's tool-call message, then execute each call.
        history.append(
            {
                "role": "assistant",
                "content": result.get("content") or None,
                "tool_calls": tool_calls,
            }
        )
        for call in tool_calls:
            name = (call.get("function") or {}).get("name", "")
            call_id = call.get("id") or ""
            raw_args = (call.get("function") or {}).get("arguments", "{}")
            args = tools.parse_tool_arguments(raw_args)
            if on_tool_start:
                on_tool_start(name, args)
            try:
                output = tools.execute_tool(name, args, project_root)
                event = ToolEvent(name, args, ok=True, output=output)
            except Exception as exc:
                output = f"Error: {exc}"
                event = ToolEvent(name, args, ok=False, output=output)
            events.append(event)
            if on_tool:
                on_tool(event)
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output,
                }
            )

    return AgentResult(
        None,
        events,
        model=model,
        provider=provider,
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        warning=f"Reached the {max_turns}-turn tool budget without a final answer.",
        turns_used=max_turns,
    )


def run_agent_stream(
    messages: List[Dict[str, str]],
    project_root: Path,
    cfg: Dict[str, Any],
    resolve_key: Optional[Callable[[str], Optional[str]]] = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    on_tool: Optional[Callable[[ToolEvent], None]] = None,
    on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Any:
    """Streaming variant of :func:`run_agent`.

    Yields ``("delta", text)`` tuples for the final answer text as the LLM
    streams it, plus ``("tool", ToolEvent)`` after each tool executes, then a
    final ``("done", AgentResult)`` event.  Tool-start notifications still go
    through ``on_tool_start`` so the UI can show a spinner while a tool runs.
    """
    tools_enabled = (cfg.get("tools") or {}).get("enabled", True)
    schemas = tools.TOOL_SCHEMAS if tools_enabled else None

    history = list(messages)
    events: List[ToolEvent] = []
    total_prompt = 0
    total_completion = 0
    model: Optional[str] = None
    provider: Optional[str] = None

    for turn in range(max_turns):
        try:
            stream = stream_agent_chat(history, cfg, resolve_key=resolve_key, tools=schemas)
            content = ""
            tool_calls: List[Dict[str, Any]] = []
            done: Dict[str, Any] = {}
            for kind, payload in stream:
                if kind == "delta":
                    content += payload
                    yield ("delta", payload)
                elif kind == "tool_calls":
                    tool_calls = payload
                elif kind == "done":
                    done = payload
        except LLMError as exc:
            yield ("done", AgentResult(None, events, warning=str(exc), turns_used=turn + 1))
            return

        total_prompt += done.get("prompt_tokens", 0)
        total_completion += done.get("completion_tokens", 0)
        model = done.get("model") or model
        provider = done.get("provider") or provider

        # Providers may attach tool calls either as a dedicated stream event
        # ("tool_calls") or inside the final "done" payload; honor both.
        if not tool_calls:
            tool_calls = done.get("tool_calls") or []

        if not tool_calls:
            yield (
                "done",
                AgentResult(
                    content or done.get("content"),
                    events,
                    model=model,
                    provider=provider,
                    prompt_tokens=total_prompt,
                    completion_tokens=total_completion,
                    turns_used=turn + 1,
                ),
            )
            return

        # Record the assistant's tool-call message, then execute each call.
        history.append(
            {
                "role": "assistant",
                "content": content or None,
                "tool_calls": tool_calls,
            }
        )
        for call in tool_calls:
            name = (call.get("function") or {}).get("name", "")
            call_id = call.get("id") or ""
            raw_args = (call.get("function") or {}).get("arguments", "{}")
            args = tools.parse_tool_arguments(raw_args)
            if on_tool_start:
                on_tool_start(name, args)
            try:
                output = tools.execute_tool(name, args, project_root)
                event = ToolEvent(name, args, ok=True, output=output)
            except Exception as exc:
                output = f"Error: {exc}"
                event = ToolEvent(name, args, ok=False, output=output)
            events.append(event)
            if on_tool:
                on_tool(event)
            yield ("tool", event)
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output,
                }
            )

    yield (
        "done",
        AgentResult(
            None,
            events,
            model=model,
            provider=provider,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            warning=f"Reached the {max_turns}-turn tool budget without a final answer.",
            turns_used=max_turns,
        ),
    )
