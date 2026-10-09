from __future__ import annotations

from importlib import import_module
from typing import Any


class LocalProvider:
    """In-process llama.cpp provider, loaded only when a local model is used."""

    def __init__(self, model_path: str, context_window: int = 32768) -> None:
        try:
            llama_cpp = import_module("llama_cpp")
        except ImportError as exc:
            raise RuntimeError(
                "Local inference requires llama-cpp-python. Install Panther with "
                "`pip install 'panther-cli[local]'`."
            ) from exc
        llama_class = getattr(llama_cpp, "Llama", None)
        if not callable(llama_class):
            raise RuntimeError(
                "The installed llama-cpp-python package does not expose Llama."
            )
        self.model_path = model_path
        self.context_window = context_window
        try:
            self._model: Any = llama_class(
                model_path=model_path,
                n_ctx=context_window,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load local model {model_path}: {exc}") from exc

    def generate(self, history: list[dict[str, str]], tools: list[dict[str, Any]]) -> str:
        prompt = (
            "You are Panther, a local coding agent. If a tool is needed, respond with "
            "only JSON in the form {\"name\":\"tool_name\",\"arguments\":{...}}. "
            "Available tools:\n"
            + "\n".join(f"- {tool['name']}: {tool['description']} {tool['parameters']}" for tool in tools)
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": prompt}]
        for item in history:
            role = item.get("role", "user")
            if role == "system":
                continue
            if role == "tool":
                tool_name = item.get("toolName", "tool")
                messages.append(
                    {
                        "role": "user",
                        "content": f"Result from {tool_name}:\n{item.get('content', '')}",
                    }
                )
            elif role in {"user", "assistant"}:
                messages.append(
                    {"role": role, "content": item.get("content", "")}
                )
        try:
            response = self._model.create_chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=2048,
            )
            return str(response["choices"][0]["message"]["content"] or "")
        except Exception as exc:
            raise RuntimeError(f"Local model generation failed: {exc}") from exc

    def close(self) -> None:
        close = getattr(self._model, "close", None)
        if callable(close):
            close()
