import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from panther import agent


def _cfg(tools_enabled=True):
    return {
        "llm": {
            "primary": {"provider": "nvidia_nim", "model": "m1"},
        },
        "api_keys": {"nvidia_nim": "nvapi-1"},
        "tools": {"enabled": tools_enabled, "max_turns": 6},
    }


def _tool_call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class TestRunAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "app.py").write_text("def helper():\n    return 1\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_direct_answer_without_tools(self):
        with patch("panther.agent.call_llm", return_value={
            "content": "42",
            "model": "m1",
            "provider": "nvidia_nim",
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "tool_calls": [],
        }) as mock_call:
            result = agent.run_agent([{"role": "user", "content": "hi"}], self.root, _cfg())
        self.assertEqual(result.content, "42")
        self.assertEqual(result.tool_events, [])
        self.assertEqual(result.model, "m1")
        mock_call.assert_called_once()

    def test_executes_tool_and_continues(self):
        responses = [
            {
                "content": None,
                "model": "m1",
                "provider": "nvidia_nim",
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "tool_calls": [_tool_call("grep", '{"pattern": "helper"}')],
            },
            {
                "content": "Found it in app.py",
                "model": "m1",
                "provider": "nvidia_nim",
                "prompt_tokens": 6,
                "completion_tokens": 3,
                "tool_calls": [],
            },
        ]
        with patch("panther.agent.call_llm", side_effect=responses) as mock_call:
            events = []
            result = agent.run_agent(
                [{"role": "user", "content": "find helper"}], self.root, _cfg(), on_tool=events.append
            )
        self.assertEqual(result.content, "Found it in app.py")
        self.assertEqual(len(result.tool_events), 1)
        self.assertEqual(result.tool_events[0].name, "grep")
        self.assertTrue(result.tool_events[0].ok)
        self.assertEqual(mock_call.call_count, 2)
        # Tool result was fed back to the model.
        last_messages = mock_call.call_args_list[1][0][0]
        self.assertEqual(last_messages[-1]["role"], "tool")
        self.assertIn("app.py", last_messages[-1]["content"])

    def test_tool_error_is_reported_to_model(self):
        responses = [
            {
                "content": None,
                "model": "m1",
                "provider": "nvidia_nim",
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "tool_calls": [_tool_call("read_file", '{"path": "missing.py"}')],
            },
            {
                "content": "ok",
                "model": "m1",
                "provider": "nvidia_nim",
                "prompt_tokens": 6,
                "completion_tokens": 1,
                "tool_calls": [],
            },
        ]
        with patch("panther.agent.call_llm", side_effect=responses):
            result = agent.run_agent([{"role": "user", "content": "read it"}], self.root, _cfg())
        self.assertFalse(result.tool_events[0].ok)
        self.assertIn("Error", result.tool_events[0].output)

    def test_respects_turn_budget(self):
        tool_call = {
            "content": None,
            "model": "m1",
            "provider": "nvidia_nim",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "tool_calls": [_tool_call("list_dir", "{}")],
        }
        with patch("panther.agent.call_llm", return_value=tool_call):
            result = agent.run_agent([{"role": "user", "content": "loop"}], self.root, _cfg(), max_turns=3)
        self.assertIsNone(result.content)
        self.assertIn("budget", result.warning)
        self.assertEqual(result.turns_used, 3)

    def test_tools_disabled_never_calls_llm_with_schemas(self):
        with patch("panther.agent.call_llm", return_value={
            "content": "hi",
            "model": "m1",
            "provider": "nvidia_nim",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "tool_calls": [],
        }) as mock_call:
            agent.run_agent([{"role": "user", "content": "hi"}], self.root, _cfg(tools_enabled=False))
        _, kwargs = mock_call.call_args
        self.assertIsNone(kwargs.get("tools"))

    def test_llm_failure_returns_warning(self):
        from panther.api import LLMError

        with patch("panther.agent.call_llm", side_effect=LLMError("boom")):
            result = agent.run_agent([{"role": "user", "content": "hi"}], self.root, _cfg())
        self.assertIsNone(result.content)
        self.assertIn("boom", result.warning)


class TestRunAgentStream(unittest.TestCase):
    """Streaming agent loop: deltas + tool events + done."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "app.py").write_text("def helper():\n    return 1\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_streams_deltas_then_done(self):
        with patch("panther.agent.stream_agent_chat", return_value=iter(
            [("delta", "Hel"), ("delta", "lo"), ("done", {
                "model": "m1", "provider": "nvidia_nim",
                "prompt_tokens": 5, "completion_tokens": 2, "tool_calls": [],
            })]
        )):
            events = list(
                agent.run_agent_stream([{"role": "user", "content": "hi"}], self.root, _cfg())
            )
        self.assertEqual(events[0], ("delta", "Hel"))
        self.assertEqual(events[1], ("delta", "lo"))
        kind, result = events[2]
        self.assertEqual(kind, "done")
        self.assertEqual(result.content, "Hello")
        self.assertEqual(result.turns_used, 1)

    def test_tool_events_yielded_between_turns(self):
        def fake_stream(messages, cfg, **kwargs):
            if len(messages) == 1:
                return iter([("done", {
                    "model": "m1", "provider": "nvidia_nim",
                    "prompt_tokens": 5, "completion_tokens": 1,
                    "tool_calls": [_tool_call("grep", '{"pattern": "helper"}')],
                })])
            return iter([("delta", "found"), ("done", {
                "model": "m1", "provider": "nvidia_nim",
                "prompt_tokens": 6, "completion_tokens": 1, "tool_calls": [],
            })])

        with patch("panther.agent.stream_agent_chat", side_effect=fake_stream):
            events = list(
                agent.run_agent_stream([{"role": "user", "content": "find"}], self.root, _cfg())
            )
        kinds = [k for k, _ in events]
        self.assertIn("tool", kinds)
        tool_event = next(p for k, p in events if k == "tool")
        self.assertEqual(tool_event.name, "grep")
        self.assertTrue(tool_event.ok)
        final = events[-1][1]
        self.assertEqual(final.content, "found")
        self.assertEqual(len(final.tool_events), 1)

    def test_stream_failure_returns_warning(self):
        from panther.api import LLMError

        with patch("panther.agent.stream_agent_chat", side_effect=LLMError("boom")):
            events = list(
                agent.run_agent_stream([{"role": "user", "content": "hi"}], self.root, _cfg())
            )
        kind, result = events[-1]
        self.assertEqual(kind, "done")
        self.assertIsNone(result.content)
        self.assertIn("boom", result.warning)


if __name__ == "__main__":
    unittest.main()
