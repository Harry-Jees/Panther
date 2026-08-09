"""Tests for the chat TUI: @mention parsing, completion, and CLI fallback."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from panther.tui import ChatCompleter, ChatTUI, parse_mentions
from prompt_toolkit.document import Document


class MentionTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "app.py").write_text(
            "def hello():\n    return 42\n\nprint(hello())\n" * 80,
            encoding="utf-8",
        )
        (self.root / "notes.md").write_text("# Notes\n\nSome markdown here.\n", encoding="utf-8")


class TestParseMentions(MentionTestBase):
    def test_resolves_file_and_cleans_token(self):
        clean, atts = parse_mentions("what does @app.py do?", self.root)
        self.assertEqual(clean, "what does app.py do?")
        self.assertEqual(len(atts), 1)
        self.assertEqual(atts[0]["path"], "app.py")
        self.assertIn("def hello()", atts[0]["text"])

    def test_multiple_mentions(self):
        clean, atts = parse_mentions("see @app.py and @notes.md", self.root)
        self.assertEqual(clean, "see app.py and notes.md")
        self.assertEqual([a["path"] for a in atts], ["app.py", "notes.md"])

    def test_duplicate_mention_attached_once(self):
        clean, atts = parse_mentions("@app.py and @app.py again", self.root)
        self.assertEqual(len(atts), 1)

    def test_missing_file_left_alone(self):
        clean, atts = parse_mentions("check @nope.py please", self.root)
        self.assertEqual(clean, "check @nope.py please")
        self.assertEqual(atts, [])

    def test_path_escape_rejected(self):
        clean, atts = parse_mentions("look at @../../secret.txt", self.root)
        self.assertEqual(atts, [])

    def test_truncates_long_files(self):
        # 80 * 4 lines = 320 lines, well over the 200-line cap.
        clean, atts = parse_mentions("big @app.py", self.root)
        self.assertLessEqual(len(atts[0]["text"].splitlines()), 201)
        self.assertIn("truncated", atts[0]["text"])


class TestChatCompleter(MentionTestBase):
    def setUp(self):
        super().setUp()
        # nested files so fuzzy matching has something interesting to find
        (self.root / "src").mkdir(exist_ok=True)
        (self.root / "src" / "client.py").write_text(
            "class Client:\n    pass\n" * 30, encoding="utf-8"
        )
        self.comp = ChatCompleter(self.root)

    def test_command_completion(self):
        docs = list(self.comp.get_completions(Document(text="/he"), None))
        self.assertTrue(any(c.text == "/help" for c in docs))

    def test_prefix_file_completion(self):
        docs = list(self.comp.get_completions(Document(text="what is @app"), None))
        self.assertTrue(any(c.text == "@app.py" for c in docs))

    def test_fuzzy_subsequence_completion(self):
        # "cl" is a subsequence of src/client.py but not a prefix of any file
        docs = list(self.comp.get_completions(Document(text="@cl"), None))
        self.assertTrue(any(c.text == "@src/client.py" for c in docs))

    def test_basename_match_ranks_first(self):
        docs = list(self.comp.get_completions(Document(text="@client"), None))
        self.assertEqual(docs[0].text, "@src/client.py")

    def test_no_fuzzy_match(self):
        docs = list(self.comp.get_completions(Document(text="@zzzzz"), None))
        self.assertEqual(docs, [])

    def test_completions_show_line_counts(self):
        # src/client.py holds "class Client:\n    pass\n" * 30 → exactly 60 lines
        docs = list(self.comp.get_completions(Document(text="@client"), None))
        meta = str(docs[0].display_meta)
        self.assertIn("60 lines", meta)

    def test_plain_text_no_completions(self):
        docs = list(self.comp.get_completions(Document(text="hello world"), None))
        self.assertEqual(docs, [])


class TestTuiRender(unittest.TestCase):
    """Header/status must never leak rich ``[cyan]``-style markup."""

    def setUp(self):
        self.session = {
            "session_id": "s1",
            "repo_name": "x",
            "repo_path": ".",
            "stats": {"questions": 1, "total_tokens_used": 123, "api_calls": {}},
            "messages": [],
        }
        self.cfg = {
            "llm": {"primary": {"provider": "nvidia_nim", "model": "m1"},
                    "fallback": {"provider": None, "model": None}},
            "tools": {"enabled": True},
            "ui": {"show_token_counts": True},
        }

    def test_header_has_no_markup_leak(self):
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.formatted_text import to_formatted_text

        tui = ChatTUI(Path("."), self.session, self.cfg, app_output=DummyOutput())
        text = "".join(frag for _, frag in to_formatted_text(tui._render_header()))
        self.assertNotIn("[", text)
        self.assertNotIn("]", text)
        self.assertIn("🐾 Panther", text)
        self.assertIn("m1", text)

    def test_status_has_no_markup_leak(self):
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.formatted_text import to_formatted_text

        tui = ChatTUI(Path("."), self.session, self.cfg, app_output=DummyOutput())
        text = "".join(frag for _, frag in to_formatted_text(tui._render_status()))
        self.assertNotIn("[", text)
        self.assertNotIn("]", text)
        self.assertIn("123 tokens", text)


class TestTuiCliFallback(unittest.TestCase):
    """The TUI must only engage on a real interactive terminal."""

    def test_use_tui_false_when_piped(self):
        from panther import cli

        with patch("panther.cli.sys.stdin") as stdin, patch.dict(
            "os.environ", {}, clear=False
        ):
            stdin.isatty.return_value = False
            self.assertFalse(cli._use_tui())

    def test_use_tui_false_when_disabled_env(self):
        from panther import cli

        with patch("panther.cli.sys.stdin") as stdin, patch.dict(
            "os.environ", {"PANTHER_NO_TUI": "1"}, clear=False
        ):
            stdin.isatty.return_value = True
            self.assertFalse(cli._use_tui())


class TestTuiEventLoop(unittest.TestCase):
    """Headless run of the real prompt_toolkit event loop (pipe input).

    Guards the fragment-normalization and threading paths that crashed in
    development (bare-tuple to_formatted_text, Win32 output detection).
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "app.py").write_text("def hello():\n    return 42\n", encoding="utf-8")
        self.session = {
            "session_id": "tui_test",
            "repo_name": "x",
            "repo_path": str(self.root),
            "stats": {"questions": 0, "total_tokens_used": 0, "api_calls": {}},
            "messages": [],
        }
        self.cfg = {
            "llm": {"primary": {"provider": "nvidia_nim", "model": "m1"},
                    "fallback": {"provider": None, "model": None}},
            "tools": {"enabled": True},
            "ui": {"show_token_counts": True},
            "token_budget": {"max_prompt_tokens": 28000},
        }

    def _run_chat(self, text: str) -> ChatTUI:
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        import threading
        import panther.qa as qa

        original_agent_stream = qa.stream_agent_answer

        def fake_agent_stream(*a, **k):
            yield ("delta", "stream")
            yield ("done", qa.Answer(text="streamed", model_used="fake", tokens={"prompt": 2, "completion": 3}))

        qa.stream_agent_answer = fake_agent_stream
        recorded = []
        holder = {}

        def run():
            with create_pipe_input() as pipe:
                tui = ChatTUI(
                    self.root,
                    self.session,
                    self.cfg,
                    on_answer=lambda s, q, a: recorded.append((q, a.text)),
                    on_command=lambda cmd, t: "mock: " + cmd,
                    build_answer=lambda q, r: "fallback",
                    app_output=DummyOutput(),
                )
                holder["tui"] = tui
                tui._app.input = pipe

                def feed():
                    import time

                    time.sleep(0.15)
                    pipe.send_text(text + "\r")
                    time.sleep(0.4)
                    pipe.send_text("\x03")
                    time.sleep(0.2)
                    pipe.close()

                threading.Thread(target=feed, daemon=True).start()
                tui.run()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=10)
        qa.stream_agent_answer = original_agent_stream
        self.assertEqual(recorded, [("what is app.py?", "streamed")])
        return holder["tui"]

    def test_ask_with_mention_records_answer(self):
        self._run_chat("what is @app.py?")

    def test_render_transcript_with_pending_stream(self):
        """Regression: live streaming render crashed with
        ``'ANSI' object is not iterable`` (list(ANSI(...)) in
        ``_render_transcript``).  Rendering while ``_pending`` is set must
        never raise, and must include the streamed text."""
        from prompt_toolkit.output import DummyOutput

        tui = ChatTUI(self.root, self.session, self.cfg, app_output=DummyOutput())
        tui._pending = "streaming answer text"
        frags = tui._render_transcript()  # must not raise
        joined = "".join(frag for _, frag in frags)
        self.assertIn("streaming answer text", joined)

    def test_command_path_renders_message(self):
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        import panther.qa as qa
        import threading

        holder = {}
        messages = []

        def run():
            with create_pipe_input() as pipe:
                tui = ChatTUI(
                    self.root, self.session, self.cfg,
                    on_command=lambda cmd, t: messages.append(cmd) or "mock: " + cmd,
                    app_output=DummyOutput(),
                )
                holder["tui"] = tui
                tui._app.input = pipe

                def feed():
                    import time

                    time.sleep(0.15)
                    pipe.send_text("/status\r")
                    time.sleep(0.3)
                    pipe.send_text("\x03")
                    time.sleep(0.2)
                    pipe.close()

                threading.Thread(target=feed, daemon=True).start()
                tui.run()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=10)
        self.assertEqual(messages, ["/status"])
        blocks = holder["tui"]._blocks
        self.assertTrue(any("mock: /status" in str(b) for b in blocks))


class TestDispatchCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.session = {
            "session_id": "s1",
            "repo_name": "x",
            "repo_path": str(self.root),
            "stats": {"questions": 1, "total_tokens_used": 10, "api_calls": {}},
            "messages": [],
        }

    def _cfg(self):
        return {
            "llm": {"primary": {"provider": "nvidia_nim", "model": "m1"},
                    "fallback": {"provider": None, "model": None}},
            "tools": {"enabled": True},
        }

    def test_exit_command(self):
        from panther import cli

        exit_now, msg = cli._dispatch_command("/exit", self.root, self.session, self._cfg())
        self.assertTrue(exit_now)
        self.assertIsNone(msg)  # caller prints its own goodbye

    def test_status_command(self):
        from panther import cli

        exit_now, msg = cli._dispatch_command("/status", self.root, self.session, self._cfg())
        self.assertFalse(exit_now)
        self.assertIn("Questions asked: 1", msg)

    def test_tools_toggle(self):
        from panther import cli

        cfg = self._cfg()
        with patch("panther.cli.save_config"):  # never touch the real config
            exit_now, msg = cli._dispatch_command("/tools", self.root, self.session, cfg)
        self.assertFalse(exit_now)
        self.assertIn("OFF", msg)
        self.assertFalse(cfg["tools"]["enabled"])

    def test_unknown_command(self):
        from panther import cli

        exit_now, msg = cli._dispatch_command("/bogus", self.root, self.session, self._cfg())
        self.assertFalse(exit_now)
        self.assertIn("Unknown command", msg)


if __name__ == "__main__":
    unittest.main()
