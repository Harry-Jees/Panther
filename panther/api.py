"""LLM API client for the NVIDIA NIM free tier (~40 RPM).

The endpoint is OpenAI-compatible, so a single client class serves it.  On
rate limits or server errors Panther retries with exponential backoff; there
is no secondary provider.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

ENDPOINTS = {
    "nvidia_nim": "https://integrate.api.nvidia.com/v1/chat/completions",
}


class LLMError(Exception):
    """Raised when an LLM provider call fails."""


class RateLimitError(LLMError):
    def __init__(self, message: str, retry_after: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def estimate_tokens(text: str) -> int:
    """Estimate token count for ``text``.

    Uses ``tiktoken`` (cl100k_base) when available; falls back to a heuristic
    of ~4 characters per token otherwise.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


class LLMClient:
    """Minimal OpenAI-compatible chat completions client."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 60.0,
    ) -> None:
        if provider not in ENDPOINTS:
            raise LLMError(f"Unknown provider: {provider}")
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(
        self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        if requests is None:  # pragma: no cover
            raise LLMError("The `requests` package is required. Run: pip install requests")
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                self.endpoint(), json=payload, headers=headers, timeout=self.timeout
            )
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"Network error calling {self.provider}: {exc}") from exc

        if resp.status_code == 429:
            try:
                retry_after = float(resp.headers.get("Retry-After", 1))
            except (TypeError, ValueError):
                retry_after = 1.0
            raise RateLimitError(
                f"{self.provider} rate limited (429): {resp.text[:200]}", retry_after
            )
        if resp.status_code >= 400:
            raise LLMError(f"{self.provider} error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response from {self.provider}: {data}") from exc
        content = message.get("content")
        tool_calls = message.get("tool_calls") or []
        if content is None and not tool_calls:
            raise LLMError(f"Unexpected response from {self.provider}: {data}")
        usage = data.get("usage") or {}
        return {
            "content": content,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "tool_calls": tool_calls,
        }

    def complete_stream(
        self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Iterator[tuple]:
        """Stream a chat completion (SSE), yielding content deltas.

        With ``tools`` given, ``delta.tool_calls`` fragments are accumulated
        per index and emitted as a final ``("tool_calls", [...])`` event before
        any ``("usage", ...)`` event.  Raises :class:`RateLimitError` /
        :class:`LLMError` exactly like :meth:`complete`.
        """
        if requests is None:  # pragma: no cover
            raise LLMError("The `requests` package is required. Run: pip install requests")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                self.endpoint(),
                json=payload,
                headers=headers,
                timeout=self.timeout,
                stream=True,
            )
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"Network error calling {self.provider}: {exc}") from exc

        if resp.status_code == 429:
            try:
                retry_after = float(resp.headers.get("Retry-After", 1))
            except (TypeError, ValueError):
                retry_after = 1.0
            raise RateLimitError(
                f"{self.provider} rate limited (429): {resp.text[:200]}", retry_after
            )
        if resp.status_code >= 400:
            raise LLMError(f"{self.provider} error {resp.status_code}: {resp.text[:300]}")

        usage: Dict[str, int] = {}
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}
        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield ("delta", content)
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    entry = tool_calls_acc.setdefault(
                        idx, {"id": "", "function": {"name": "", "arguments": ""}}
                    )
                    if tc.get("id"):
                        entry["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        entry["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        entry["function"]["arguments"] += fn["arguments"]
        finally:
            # Ensure the connection is released even if the caller abandons
            # the generator mid-stream (e.g. on interrupt).
            resp.close()
        if tool_calls_acc:
            yield ("tool_calls", [tool_calls_acc[i] for i in sorted(tool_calls_acc)])
        if usage:
            yield ("usage", usage)

    def endpoint(self) -> str:
        return ENDPOINTS[self.provider]


def _complete_with_retries(
    client: LLMClient,
    messages: List[Dict[str, str]],
    max_retries: int = 5,
    max_backoff: float = 60.0,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Call ``client`` with exponential backoff on 429/5xx.

    A rate-limit whose ``Retry-After`` exceeds ``max_backoff`` propagates
    immediately so the caller can switch providers.
    """
    base_backoff = 1.0
    attempt = 0
    while True:
        try:
            return client.complete(messages, tools=tools)
        except RateLimitError as exc:
            if exc.retry_after >= max_backoff:
                raise
            wait = min(max_backoff, exc.retry_after * (2 ** attempt), 10.0)
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(wait)
        except LLMError as exc:
            # Server/network errors: a couple of retries, then give up.
            attempt += 1
            if attempt > min(max_retries, 2):
                raise
            time.sleep(min(max_backoff, base_backoff * (2 ** attempt)))


def call_llm(
    messages: List[Dict[str, str]],
    cfg: Dict[str, Any],
    resolve_key: Optional[Callable[[str], Optional[str]]] = None,
    max_retries: int = 5,
    max_backoff: float = 60.0,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Call the configured LLM provider (NVIDIA NIM).

    Returns ``{"content", "provider", "model", "prompt_tokens",
    "completion_tokens", "tool_calls"}``. Raises :class:`LLMError` when the
    provider is unconfigured or unavailable.
    """
    _REQUEST_LIMITER.wait_if_needed()
    resolve_key = resolve_key or (lambda p: (cfg.get("api_keys") or {}).get(p))
    spec = (cfg.get("llm") or {}).get("primary") or {}
    provider = spec.get("provider")
    if not provider:
        raise LLMError("No LLM provider configured. Run `panther init`.")
    api_key = resolve_key(provider)
    if not api_key:
        raise LLMError(
            f"No API key configured for {provider} (set one with `panther auth add` "
            f"or {_env_hint(provider)})."
        )
    client = LLMClient(
        provider,
        api_key,
        spec.get("model", ""),
        spec.get("temperature", 0.7),
        spec.get("max_tokens", 2048),
    )
    try:
        result = _complete_with_retries(
            client, messages, max_retries, max_backoff, tools=tools
        )
    except LLMError as exc:
        raise LLMError(f"{provider} failed: {exc}") from exc
    result["provider"] = provider
    result["model"] = client.model
    result.setdefault("tool_calls", [])
    return result


def stream_chat(
    messages: List[Dict[str, str]],
    cfg: Dict[str, Any],
    resolve_key: Optional[Callable[[str], Optional[str]]] = None,
) -> Iterator[tuple]:
    """Stream a chat completion from the configured LLM provider.

    Yields ``("delta", text)`` tuples as tokens arrive, then a final
    ``("done", {...})`` event carrying ``provider``/``model``/``content`` and
    token counts.  Raises :class:`LLMError` when the provider is unconfigured
    or unavailable.
    """
    _REQUEST_LIMITER.wait_if_needed()
    resolve_key = resolve_key or (lambda p: (cfg.get("api_keys") or {}).get(p))
    spec = (cfg.get("llm") or {}).get("primary") or {}
    provider = spec.get("provider")
    if not provider:
        raise LLMError("No LLM provider configured. Run `panther init`.")
    api_key = resolve_key(provider)
    if not api_key:
        raise LLMError(
            f"No API key configured for {provider} "
            f"(set one with `panther auth add` or {_env_hint(provider)})."
        )
    client = LLMClient(
        provider,
        api_key,
        spec.get("model", ""),
        spec.get("temperature", 0.7),
        spec.get("max_tokens", 2048),
    )
    try:
        parts: List[str] = []
        usage: Dict[str, int] = {}
        for kind, payload in client.complete_stream(messages):
            if kind == "delta":
                parts.append(payload)
                yield ("delta", payload)
            elif kind == "usage":
                usage = payload
        yield (
            "done",
            {
                "provider": provider,
                "model": client.model,
                "content": "".join(parts),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        )
    except LLMError as exc:
        raise LLMError(f"{provider} failed: {exc}") from exc


def stream_agent_chat(
    messages: List[Dict[str, str]],
    cfg: Dict[str, Any],
    resolve_key: Optional[Callable[[str], Optional[str]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Iterator[tuple]:
    """Stream a chat completion with tool schemas attached.

    Like :func:`stream_chat` but accepts ``tools`` and forwards the
    accumulated ``("tool_calls", [...])`` event so the streaming agent loop
    can execute tools without a separate blocking call.  Final event is
    ``("done", {...})`` carrying ``tool_calls``.
    """
    _REQUEST_LIMITER.wait_if_needed()
    resolve_key = resolve_key or (lambda p: (cfg.get("api_keys") or {}).get(p))
    spec = (cfg.get("llm") or {}).get("primary") or {}
    provider = spec.get("provider")
    if not provider:
        raise LLMError("No LLM provider configured. Run `panther init`.")
    api_key = resolve_key(provider)
    if not api_key:
        raise LLMError(
            f"No API key configured for {provider} "
            f"(set one with `panther auth add` or {_env_hint(provider)})."
        )
    client = LLMClient(
        provider,
        api_key,
        spec.get("model", ""),
        spec.get("temperature", 0.7),
        spec.get("max_tokens", 2048),
    )
    try:
        parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        usage: Dict[str, int] = {}
        for kind, payload in client.complete_stream(messages, tools=tools):
            if kind == "delta":
                parts.append(payload)
                yield ("delta", payload)
            elif kind == "tool_calls":
                tool_calls = payload
                yield ("tool_calls", payload)
            elif kind == "usage":
                usage = payload
        yield (
            "done",
            {
                "provider": provider,
                "model": client.model,
                "content": "".join(parts),
                "tool_calls": tool_calls,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        )
    except LLMError as exc:
        raise LLMError(f"{provider} failed: {exc}") from exc


def _env_hint(provider: str) -> str:
    from . import config

    env_var = config.ENV_KEY_MAP.get(provider)
    return f"the {env_var} environment variable" if env_var else "an environment variable"


class RequestRateLimiter:
    """Rolling-window requests-per-minute tracker (soft cap enforcement).

    The free NVIDIA tier is roughly 40 RPM; pacing just under that avoids
    429s and the long backoff sleeps that make answers feel like they never
    arrive.
    """

    def __init__(self, rpm: float = 30.0) -> None:
        self.rpm = rpm
        self._times: List[float] = []

    def wait_if_needed(self) -> None:
        """Sleep until the rolling window drops below the RPM cap, if needed."""
        now = time.monotonic()
        self._times = [t for t in self._times if now - t < 60.0]
        if self.rpm <= 0:
            return
        if len(self._times) >= self.rpm:
            wait = 60.0 - (now - self._times[0]) + 0.05
            if wait > 0:
                time.sleep(wait)
        self._times.append(now)


# Process-wide request rate limiter (RPM soft cap, below the free tier limit).
_REQUEST_LIMITER = RequestRateLimiter()


class TokenRateLimiter:
    """Rolling-window tokens-per-minute tracker (soft cap enforcement)."""

    def __init__(self, soft_limit_tpm: int = 30000) -> None:
        self.soft_limit = soft_limit_tpm
        self._events: List[tuple] = []

    def add(self, tokens: int) -> None:
        self._events.append((time.monotonic(), max(0, tokens)))

    def used_tpm(self) -> int:
        now = time.monotonic()
        self._events = [(ts, t) for ts, t in self._events if now - ts < 60.0]
        return sum(t for _, t in self._events)

    def wait_if_needed(self) -> None:
        """Sleep until the rolling window drops below the soft cap, if needed."""
        if self.soft_limit <= 0:
            return
        while self.used_tpm() >= self.soft_limit:
            time.sleep(1.0)
