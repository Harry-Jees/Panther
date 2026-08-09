import json
import unittest
from unittest.mock import MagicMock, patch

from panther import api


class FakeRequestException(Exception):
    pass


def _make_response(status_code=200, payload=None, headers=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = text
    if payload is not None:
        resp.json.return_value = payload
    return resp


def _install_fake_requests():
    fake = MagicMock()
    fake.exceptions.RequestException = FakeRequestException
    return patch("panther.api.requests", fake), fake


def _tool_chunk(tool_calls):
    """Build an SSE 'data:' line carrying a tool_calls delta."""
    return "data: " + json.dumps({"choices": [{"delta": {"tool_calls": tool_calls}}]})


class TestEstimateTokens(unittest.TestCase):
    def test_heuristic_floor(self):
        self.assertEqual(api.estimate_tokens("a"), 1)
        self.assertGreaterEqual(api.estimate_tokens("hello world " * 100), 1)

    def test_uses_tiktoken_when_available(self):
        fake_tiktoken = MagicMock()
        fake_tiktoken.get_encoding.return_value.encode.return_value = [0] * 42
        with patch.dict("sys.modules", {"tiktoken": fake_tiktoken}):
            self.assertEqual(api.estimate_tokens("some text"), 42)

    def test_falls_back_when_tiktoken_fails(self):
        fake_tiktoken = MagicMock()
        fake_tiktoken.get_encoding.side_effect = Exception("no data file")
        with patch.dict("sys.modules", {"tiktoken": fake_tiktoken}):
            self.assertEqual(api.estimate_tokens("abcd" * 10), 10)


class TestLLMClient(unittest.TestCase):
    def test_complete_success(self):
        patcher, fake = _install_fake_requests()
        with patcher:
            fake.post.return_value = _make_response(
                200,
                {
                    "choices": [{"message": {"content": "hello from nim"}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 4},
                },
            )
            client = api.LLMClient("nvidia_nim", "nvapi-1", "nemotron")
            result = client.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(result["content"], "hello from nim")
        self.assertEqual(result["prompt_tokens"], 11)
        self.assertEqual(result["completion_tokens"], 4)

    def test_complete_429_raises_rate_limit(self):
        patcher, fake = _install_fake_requests()
        with patcher:
            fake.post.return_value = _make_response(
                429, text="rate limited", headers={"Retry-After": "3"}
            )
            client = api.LLMClient("nvidia_nim", "nvapi-x", "some-model")
            with self.assertRaises(api.RateLimitError) as ctx:
                client.complete([])
        self.assertEqual(ctx.exception.retry_after, 3.0)

    def test_complete_5xx_raises_llm_error(self):
        patcher, fake = _install_fake_requests()
        with patcher:
            fake.post.return_value = _make_response(500, text="server error")
            client = api.LLMClient("nvidia_nim", "nvapi-1", "m")
            with self.assertRaises(api.LLMError):
                client.complete([])

    def test_complete_network_error_raises_llm_error(self):
        patcher, fake = _install_fake_requests()
        with patcher:
            fake.post.side_effect = FakeRequestException("connection refused")
            client = api.LLMClient("nvidia_nim", "nvapi-1", "m")
            with self.assertRaises(api.LLMError):
                client.complete([])

    def test_unknown_provider_rejected(self):
        with self.assertRaises(api.LLMError):
            api.LLMClient("nope", "k", "m")


class TestCallLLM(unittest.TestCase):
    def _cfg(self):
        return {
            "llm": {
                "primary": {"provider": "nvidia_nim", "model": "m1"},
            },
            "api_keys": {"nvidia_nim": "nvapi-1"},
        }

    def test_uses_primary_when_healthy(self):
        with patch("panther.api.LLMClient") as mock_client:
            mock_client.return_value.complete.return_value = {
                "content": "ok",
                "prompt_tokens": 1,
                "completion_tokens": 1,
            }
            result = api.call_llm([], self._cfg(), max_retries=0)
        self.assertEqual(result["provider"], "nvidia_nim")
        self.assertEqual(result["content"], "ok")
        mock_client.assert_called_once()

    def test_no_keys_raises_with_hint(self):
        cfg = self._cfg()
        cfg["api_keys"] = {}
        with self.assertRaises(api.LLMError) as ctx:
            api.call_llm([], cfg)
        self.assertIn("panther auth add", str(ctx.exception))

    def test_primary_failure_raises(self):
        with patch("panther.api.LLMClient") as mock_client:
            mock_client.return_value.complete.side_effect = api.LLMError("boom")
            with self.assertRaises(api.LLMError):
                api.call_llm([], self._cfg(), max_retries=0)


class TestCompleteStream(unittest.TestCase):
    def _sse_lines(self, *chunks):
        lines = []
        for text in chunks:
            if text is None:
                lines.append('data: {"choices":[{"delta":{"content":null}}]}')
            else:
                lines.append(f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}')
        lines.append("data: [DONE]")
        return lines

    def test_stream_yields_content_deltas(self):
        patcher, fake = _install_fake_requests()
        with patcher:
            resp = _make_response(200, payload=None)
            resp.iter_lines.return_value = iter(self._sse_lines("Hel", None, "lo ", "world"))
            fake.post.return_value = resp
            client = api.LLMClient("nvidia_nim", "nvapi-x", "m")
            self.assertEqual(
                list(client.complete_stream([])),
                [("delta", "Hel"), ("delta", "lo "), ("delta", "world")],
            )

    def test_stream_429_raises_rate_limit(self):
        patcher, fake = _install_fake_requests()
        with patcher:
            fake.post.return_value = _make_response(
                429, text="slow down", headers={"Retry-After": "2"}
            )
            client = api.LLMClient("nvidia_nim", "nvapi-x", "m")
            with self.assertRaises(api.RateLimitError):
                list(client.complete_stream([]))

    def test_stream_uses_stream_flag(self):
        patcher, fake = _install_fake_requests()
        with patcher:
            resp = _make_response(200, payload=None)
            resp.iter_lines.return_value = iter(self._sse_lines("ok"))
            fake.post.return_value = resp
            client = api.LLMClient("nvidia_nim", "nvapi-1", "m")
            list(client.complete_stream([]))
        _, kwargs = fake.post.call_args
        self.assertTrue(kwargs["json"]["stream"])

    def test_stream_yields_usage_from_final_chunk(self):
        patcher, fake = _install_fake_requests()
        with patcher:
            resp = _make_response(200, payload=None)
            resp.iter_lines.return_value = iter(
                [
                    'data: {"choices":[{"delta":{"content":"hi"}}]}',
                    'data: {"choices":[],"usage":{"prompt_tokens": 11, "completion_tokens": 4}}',
                    "data: [DONE]",
                ]
            )
            fake.post.return_value = resp
            client = api.LLMClient("nvidia_nim", "nvapi-1", "m")
            events = list(client.complete_stream([]))
        self.assertIn(("usage", {"prompt_tokens": 11, "completion_tokens": 4}), events)

    def test_stream_closes_response_on_completion(self):
        patcher, fake = _install_fake_requests()
        with patcher:
            resp = _make_response(200, payload=None)
            resp.iter_lines.return_value = iter(self._sse_lines("ok"))
            fake.post.return_value = resp
            client = api.LLMClient("nvidia_nim", "nvapi-1", "m")
            list(client.complete_stream([]))
        resp.close.assert_called_once()


class TestStreamChat(unittest.TestCase):
    def _cfg(self):
        return {
            "llm": {
                "primary": {"provider": "nvidia_nim", "model": "m1"},
            },
            "api_keys": {"nvidia_nim": "nvapi-1"},
        }

    def test_stream_chat_yields_deltas_then_done(self):
        with patch("panther.api.LLMClient") as mock_client:
            mock_client.return_value.complete_stream.return_value = iter(
                [("delta", "a"), ("delta", "b"), ("delta", "c")]
            )
            events = list(api.stream_chat([], self._cfg()))
        self.assertEqual(events[0], ("delta", "a"))
        self.assertEqual(events[2], ("delta", "c"))
        kind, payload = events[3]
        self.assertEqual(kind, "done")
        self.assertEqual(payload["provider"], "nvidia_nim")
        self.assertEqual(payload["content"], "abc")

    def test_stream_chat_forwards_usage(self):
        with patch("panther.api.LLMClient") as mock_client:
            mock_client.return_value.complete_stream.return_value = iter(
                [("delta", "ok"), ("usage", {"prompt_tokens": 7, "completion_tokens": 3})]
            )
            events = list(api.stream_chat([], self._cfg()))
        payload = events[1][1]
        self.assertEqual(payload["prompt_tokens"], 7)
        self.assertEqual(payload["completion_tokens"], 3)

    def test_stream_chat_no_keys_raises(self):
        cfg = self._cfg()
        cfg["api_keys"] = {}
        with self.assertRaises(api.LLMError):
            list(api.stream_chat([], cfg))


class TestStreamAgentChat(unittest.TestCase):
    def _cfg(self):
        return {
            "llm": {
                "primary": {"provider": "nvidia_nim", "model": "m1"},
            },
            "api_keys": {"nvidia_nim": "nvapi-1"},
        }

    def test_forwards_tool_calls_then_done(self):
        with patch("panther.api.LLMClient") as mock_client:
            mock_client.return_value.complete_stream.return_value = iter(
                [
                    ("delta", "think"),
                    ("tool_calls", [{"id": "c1", "function": {"name": "grep", "arguments": '{"pattern": "x"}'}}]),
                    ("usage", {"prompt_tokens": 7, "completion_tokens": 3}),
                ]
            )
            events = list(api.stream_agent_chat([], self._cfg(), tools=[{"type": "function"}]))
        self.assertEqual(events[0], ("delta", "think"))
        kind, calls = events[1]
        self.assertEqual(kind, "tool_calls")
        self.assertEqual(calls[0]["function"]["name"], "grep")
        done = events[2][1]
        self.assertEqual(done["tool_calls"][0]["function"]["name"], "grep")
        self.assertEqual(done["prompt_tokens"], 7)

    def test_forwards_tools_to_client(self):
        with patch("panther.api.LLMClient") as mock_client:
            mock_client.return_value.complete_stream.return_value = iter(
                [("delta", "ok")]
            )
            list(api.stream_agent_chat([], self._cfg(), tools=[{"type": "function"}]))
        args, kwargs = mock_client.return_value.complete_stream.call_args
        self.assertEqual(kwargs["tools"], [{"type": "function"}])

    def test_complete_stream_accumulates_tool_call_fragments(self):
        patcher, fake = _install_fake_requests()
        with patcher:
            resp = _make_response(200, payload=None)
            resp.iter_lines.return_value = iter(
                [
                    _tool_chunk([{"index": 0, "id": "c1", "function": {"name": "gr", "arguments": '{"p'}}]),
                    _tool_chunk([{"index": 0, "function": {"name": "ep", "arguments": 'attern": "x"}'}}]),
                    'data: {"choices":[],"usage":{"prompt_tokens": 7, "completion_tokens": 3}}',
                    "data: [DONE]",
                ]
            )
            fake.post.return_value = resp
            client = api.LLMClient("nvidia_nim", "nvapi-x", "m")
            events = list(client.complete_stream([], tools=[{"type": "function"}]))
        kinds = [k for k, _ in events]
        self.assertIn("tool_calls", kinds)
        calls = next(p for k, p in events if k == "tool_calls")
        self.assertEqual(calls[0]["id"], "c1")
        self.assertEqual(calls[0]["function"]["name"], "grep")
        self.assertEqual(calls[0]["function"]["arguments"], '{"pattern": "x"}')
        self.assertIn(("usage", {"prompt_tokens": 7, "completion_tokens": 3}), events)


class TestRetries(unittest.TestCase):
    def test_retries_on_429_then_succeeds(self):
        client = MagicMock()
        client.complete.side_effect = [
            api.RateLimitError("429", retry_after=0.0),
            {"content": "done"},
        ]
        with patch("time.sleep") as mock_sleep:
            result = api._complete_with_retries(client, [], max_retries=3)
        self.assertEqual(result["content"], "done")
        self.assertEqual(client.complete.call_count, 2)
        mock_sleep.assert_called_once()

    def test_high_retry_after_propagates(self):
        client = MagicMock()
        client.complete.side_effect = api.RateLimitError("429", retry_after=300)
        with patch("time.sleep") as mock_sleep:
            with self.assertRaises(api.RateLimitError):
                api._complete_with_retries(client, [], max_retries=5)
        mock_sleep.assert_not_called()


class TestTokenRateLimiter(unittest.TestCase):
    def test_used_tpm_tracks_window(self):
        limiter = api.TokenRateLimiter(soft_limit_tpm=100)
        limiter.add(30)
        limiter.add(20)
        self.assertEqual(limiter.used_tpm(), 50)

    def test_wait_if_needed_sleeps_when_over_cap(self):
        limiter = api.TokenRateLimiter(soft_limit_tpm=10)
        limiter.add(100)
        # Over-cap twice, then the window clears and the loop exits.
        with patch.object(limiter, "used_tpm", side_effect=[100, 100, 0]), patch(
            "panther.api.time.sleep"
        ) as mock_sleep:
            limiter.wait_if_needed()
        self.assertEqual(mock_sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
