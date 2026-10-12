from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from importlib import import_module
from typing import Any


NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODELS_URL = "https://integrate.api.nvidia.com/v1/models"
NVIDIA_KEYRING_SERVICE = "panther-cli"
NVIDIA_KEYRING_USERNAME = "nvidia"


def _conversation(
    history: list[dict[str, str]],
    tools: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system_context = next(
        (
            item.get("content", "")
            for item in history
            if item.get("role") == "system" and item.get("content")
        ),
        "You are Panther, a local coding agent.",
    )
    prompt = (
        system_context
        + "\n\nIf a tool is needed, respond with "
        'only JSON in the form {"name":"tool_name","arguments":{...}}. '
        "Available tools:\n"
        + "\n".join(
            f"- {tool['name']}: {tool['description']} {tool['parameters']}"
            for tool in tools
        )
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
            messages.append({"role": role, "content": item.get("content", "")})
    return messages


class NvidiaProvider:
    """NVIDIA NIM chat-completions provider."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self._api_key = (os.environ.get("NVIDIA_API_KEY") or api_key or "").strip()
        if not self._api_key:
            try:
                keyring = import_module("keyring")
            except ImportError as exc:
                raise RuntimeError(
                    "No NVIDIA API key found. Add NVIDIA_API_KEY to the project "
                    ".env file or install keyring and run `panther auth add nvidia`."
                ) from exc
            try:
                self._api_key = keyring.get_password(
                    NVIDIA_KEYRING_SERVICE,
                    NVIDIA_KEYRING_USERNAME,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not read NVIDIA credentials from the OS keyring: {exc}"
                ) from exc
        if not self._api_key:
            raise RuntimeError(
                "No NVIDIA API key is configured. Add NVIDIA_API_KEY to the "
                "project .env file or run `panther auth add nvidia`."
            )

    def generate(self, history: list[dict[str, str]], tools: list[dict[str, Any]]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": _conversation(history, tools),
                "temperature": 0.2,
                "max_tokens": 2048,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            NVIDIA_API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            return str(result["choices"][0]["message"]["content"] or "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404 and "not found for account" in detail.lower():
                raise RuntimeError(
                    "NVIDIA lists this model, but its serving function is not "
                    "available to this API account. Check NVIDIA API account "
                    "access and key permissions."
                    f" NVIDIA response: {detail}"
                ) from exc
            raise RuntimeError(
                f"NVIDIA API request failed ({exc.code}): {detail or exc.reason}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Could not connect to the NVIDIA API: {exc}") from exc
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"NVIDIA API returned an invalid response: {exc}") from exc

    def list_models(self) -> list[str]:
        """Return model IDs advertised by NVIDIA for this API key."""
        request = urllib.request.Request(
            NVIDIA_MODELS_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            models = result.get("data", [])
            if not isinstance(models, list):
                raise TypeError("data is not a list")
            return sorted(
                str(item["id"])
                for item in models
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"NVIDIA model listing failed ({exc.code}): {detail or exc.reason}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Could not connect to the NVIDIA API: {exc}") from exc
        except (AttributeError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"NVIDIA API returned an invalid model list: {exc}") from exc


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
        try:
            response = self._model.create_chat_completion(
                messages=_conversation(history, tools),
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
