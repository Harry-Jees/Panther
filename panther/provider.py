from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from urllib.parse import quote, urlsplit
from importlib import import_module
from typing import Any


NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODELS_URL = "https://integrate.api.nvidia.com/v1/models"
NVIDIA_KEYRING_SERVICE = "panther-cli"
NVIDIA_KEYRING_USERNAME = "nvidia"
OPENAI_COMPATIBLE_KEYRING_USERNAME = "openai-compatible"
GEMINI_KEYRING_USERNAME = "gemini"


def _nvidia_connection_error(exc: BaseException) -> RuntimeError:
    """Turn low-level network failures into an actionable provider error."""
    reason = getattr(exc, "reason", None)
    detail = str(reason or exc)
    if isinstance(reason, socket.gaierror) or "getaddrinfo failed" in detail.lower():
        return RuntimeError(
            "Could not resolve integrate.api.nvidia.com. Check the computer's "
            "DNS/internet connection or configure the HTTPS proxy used by Python."
        )
    return RuntimeError(f"Could not connect to the NVIDIA API: {exc}")


def _conversation(
    history: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
    messages: list[dict[str, Any]] = [{"role": "system", "content": prompt}]
    for item in history:
        role = item.get("role", "user")
        if role == "system":
            continue
        if role == "tool":
            tool_name = item.get("toolName", "tool")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("toolCallId", tool_name),
                    "content": (
                        f"<untrusted_tool_result name=\"{tool_name}\">"
                        f"{item.get('content', '')}</untrusted_tool_result>"
                    ),
                }
            )
        elif role in {"user", "assistant"}:
            content = item.get("content", "")
            if role == "assistant" and isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict) and isinstance(parsed.get("name"), str):
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": item.get("toolCallId", "call_1"),
                            "type": "function",
                            "function": {
                                "name": parsed["name"],
                                "arguments": json.dumps(parsed.get("arguments", {})),
                            },
                        }],
                    })
                    continue
            messages.append({"role": role, "content": content})
    return messages


def _api_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Panther tool metadata to OpenAI-compatible tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]


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

    def generate(self, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        return self._generate(history, tools, timeout=120)

    def probe(self) -> str:
        """Verify the model can serve a small real chat completion."""
        response = self._generate(
            [{"role": "user", "content": "Reply with exactly: Panther is ready."}],
            [],
            timeout=20,
        ).strip()
        if not response:
            raise RuntimeError("NVIDIA returned an empty response during model check.")
        return response

    def _generate(
        self,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        timeout: int,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _conversation(history, tools),
            "temperature": 0.2,
            "max_tokens": 64 if not tools else 2048,
        }
        if tools:
            payload["tools"] = _api_tools(tools)
            payload["tool_choice"] = "auto"
        payload = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            NVIDIA_API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        request.add_header("Authorization", "Bearer " + self._api_key)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            message = result["choices"][0]["message"]
            native_calls = message.get("tool_calls", [])
            if native_calls:
                call = native_calls[0]
                function = call.get("function", {})
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                return json.dumps({
                    "name": function.get("name", ""),
                    "arguments": arguments,
                })
            return str(message.get("content") or "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404 and "not found for account" in detail.lower():
                raise RuntimeError(
                    "NVIDIA lists this model, but its serving function is not "
                    "available to this API account. Check NVIDIA API account "
                    "access and key permissions. Run `panther nvidia check` to "
                    "find and select a model that passes a live inference test."
                    f" NVIDIA response: {detail}"
                ) from exc
            raise RuntimeError(
                f"NVIDIA API request failed ({exc.code}): {detail or exc.reason}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise _nvidia_connection_error(exc) from exc
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"NVIDIA API returned an invalid response: {exc}") from exc

    def list_models(self) -> list[str]:
        """Return model IDs advertised by NVIDIA for this API key."""
        request = urllib.request.Request(
            NVIDIA_MODELS_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            method="GET",
        )
        request.add_header("Authorization", "Bearer " + self._api_key)
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
            raise _nvidia_connection_error(exc) from exc
        except (AttributeError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"NVIDIA API returned an invalid model list: {exc}") from exc


class OpenAICompatibleProvider:
    """Provider for user-configured OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = self._validate_base_url(base_url)
        self._api_key = (
            os.environ.get("OPENAI_COMPATIBLE_API_KEY") or api_key or ""
        ).strip()
        if not self._api_key:
            try:
                keyring = import_module("keyring")
            except ImportError as exc:
                raise RuntimeError(
                    "No API key found. Set OPENAI_COMPATIBLE_API_KEY or install "
                    "keyring and run `panther auth add openai-compatible`."
                ) from exc
            try:
                self._api_key = keyring.get_password(
                    NVIDIA_KEYRING_SERVICE,
                    OPENAI_COMPATIBLE_KEYRING_USERNAME,
                ) or ""
            except Exception as exc:
                raise RuntimeError(
                    f"Could not read API credentials from the OS keyring: {exc}"
                ) from exc
        if not self._api_key:
            raise RuntimeError(
                "No API key is configured. Set OPENAI_COMPATIBLE_API_KEY or "
                "run `panther auth add openai-compatible`."
            )

    @staticmethod
    def _validate_base_url(base_url: str) -> str:
        value = base_url.strip().rstrip("/")
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "API base URL must be an http(s) URL without credentials, "
                "query parameters, or a fragment."
            )
        return value

    def generate(self, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": _conversation(history, tools),
                "tools": _api_tools(tools),
                "tool_choice": "auto",
                "temperature": 0.2,
                "max_tokens": 2048,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
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
            message = result["choices"][0]["message"]
            native_calls = message.get("tool_calls", [])
            if native_calls:
                call = native_calls[0]
                function = call.get("function", {})
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("tool-call arguments are not an object")
                return json.dumps({
                    "name": function.get("name", ""),
                    "arguments": arguments,
                })
            return str(message.get("content") or "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible API request failed ({exc.code}): "
                f"{detail or exc.reason}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Could not connect to the OpenAI-compatible API: {exc}"
            ) from exc
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"OpenAI-compatible API returned an invalid response: {exc}"
            ) from exc


class GeminiProvider:
    """Google Gemini generateContent provider with native function calling."""

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model.strip()
        self._api_key = api_key.strip()
        if not self.model:
            raise ValueError("Gemini model cannot be empty.")
        if not self._api_key:
            raise ValueError("Gemini API key cannot be empty.")

    @staticmethod
    def _schema(value: Any) -> Any:
        if isinstance(value, list):
            return [GeminiProvider._schema(item) for item in value]
        if not isinstance(value, dict):
            return value
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"additionalProperties", "$schema", "title"}:
                continue
            if key == "type" and isinstance(item, str):
                result[key] = item.upper()
            else:
                result[key] = GeminiProvider._schema(item)
        return result

    @staticmethod
    def _content_part(item: dict[str, Any]) -> list[dict[str, Any]]:
        content = item.get("content", "")
        if isinstance(content, str):
            return [{"text": content}]
        if not isinstance(content, list):
            raise TypeError("message content must be text or a list")
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                parts.append({"text": str(part.get("text", ""))})
            elif part.get("type") == "image_url":
                image = part.get("image_url", {})
                url = image.get("url", "") if isinstance(image, dict) else ""
                if not isinstance(url, str) or not url.startswith("data:") or "," not in url:
                    raise ValueError("Gemini image content must use a data URL")
                header, encoded = url[5:].split(",", 1)
                mime_type = header.split(";", 1)[0]
                parts.append({
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": encoded,
                    }
                })
        return parts

    def _contents(self, history: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        system = next(
            (
                str(item.get("content", ""))
                for item in history
                if item.get("role") == "system" and item.get("content")
            ),
            "You are Panther, a local coding agent.",
        )
        contents: list[dict[str, Any]] = []
        for item in history:
            role = item.get("role")
            if role == "system":
                continue
            if role == "tool":
                tool_name = str(item.get("toolName", "tool"))
                contents.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": tool_name,
                            "response": {
                                "result": (
                                    f"<untrusted_tool_result name=\"{tool_name}\">"
                                    f"{item.get('content', '')}</untrusted_tool_result>"
                                )
                            },
                        }
                    }],
                })
                continue
            if role not in {"user", "assistant"}:
                continue
            content = item.get("content", "")
            if role == "assistant" and isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict) and isinstance(parsed.get("name"), str):
                    contents.append({
                        "role": "model",
                        "parts": [{
                            "functionCall": {
                                "name": parsed["name"],
                                "args": parsed.get("arguments", {}),
                            }
                        }],
                    })
                    continue
            contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": self._content_part({"content": content}),
            })
        return {"parts": [{"text": system}]}, contents

    def generate(self, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        system_instruction, contents = self._contents(history)
        payload: dict[str, Any] = {
            "systemInstruction": system_instruction,
            "contents": contents,
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
        }
        if tools:
            payload["tools"] = [{
                "functionDeclarations": [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": self._schema(tool["parameters"]),
                    }
                    for tool in tools
                ]
            }]
        model_id = quote(self.model, safe="-_.")
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-goog-api-key": self._api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            parts = result["candidates"][0]["content"]["parts"]
            for part in parts:
                function_call = part.get("functionCall")
                if isinstance(function_call, dict):
                    return json.dumps({
                        "name": function_call.get("name", ""),
                        "arguments": function_call.get("args", {}),
                    })
            return "".join(
                str(part.get("text", ""))
                for part in parts
                if isinstance(part, dict)
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Gemini API request failed ({exc.code}): {detail or exc.reason}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Could not connect to the Gemini API: {exc}") from exc
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Gemini API returned an invalid response: {exc}") from exc


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

    def generate(self, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
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
