import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from panther import cli


class TestPantherCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home_dir = Path(self.tmp.name)
        self.project_root = self.home_dir / "project"
        self.project_root.mkdir()
        (self.project_root / "main.py").write_text("def foo():\n    return 42\n", encoding="utf-8")
        (self.project_root / "README.md").write_text("# hello\n", encoding="utf-8")
        (self.project_root / "notes.txt").write_text("hello world\n", encoding="utf-8")
        git_dir = self.project_root / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]", encoding="utf-8")
        node_dir = self.project_root / "node_modules"
        node_dir.mkdir()
        (node_dir / "x.js").write_text("console.log('ignore');\n", encoding="utf-8")

        patcher = patch("panther.cli.Path.home", return_value=self.home_dir)
        self.addCleanup(patcher.stop)
        self.mock_home = patcher.start()

    def tearDown(self):
        self.tmp.cleanup()

    def test_init_creates_home_and_config(self):
        cli.command_init(argparse.Namespace())
        panther_home = self.home_dir / ".panther"
        self.assertTrue(panther_home.exists())
        self.assertTrue((panther_home / "config.json").exists())
        config = json.loads((panther_home / "config.json").read_text(encoding="utf-8"))
        self.assertIn("llm", config)
        self.assertIn("storage", config)

    def test_scan_repository_ignores_git_and_node_modules(self):
        files = cli.scan_repository(self.project_root)
        file_paths = {item["path"] for item in files}
        self.assertIn("main.py", file_paths)
        self.assertIn("README.md", file_paths)
        self.assertIn("notes.txt", file_paths)
        self.assertNotIn(".git/config", file_paths)
        self.assertNotIn("node_modules/x.js", file_paths)

    def test_write_index_and_build_answer(self):
        cli.ensure_panther_home()
        files = cli.scan_repository(self.project_root)
        cli.write_index(self.project_root, files)
        answer = cli.build_answer("foo", self.project_root)
        self.assertIn("relevant code snippet", answer)
        self.assertIn("main.py", answer)

    def test_build_answer_without_index(self):
        cli.ensure_panther_home()
        index_path = cli.get_index_path()
        if index_path.exists():
            index_path.unlink()
        answer = cli.build_answer("foo", self.project_root)
        self.assertIn("Panther has not indexed this project yet", answer)

    def test_create_session_formats_session(self):
        session = cli.create_session(self.project_root)
        self.assertIn("session_", session["session_id"])
        self.assertEqual(session["repo_path"], str(self.project_root))
        self.assertEqual(session["stats"]["questions"], 0)

    def test_auth_add_and_remove(self):
        cli.command_init(argparse.Namespace())
        cli.command_auth(argparse.Namespace(action="add", provider="nvidia_nim", key="nvapi_test"))
        config = json.loads((self.home_dir / ".panther" / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["api_keys"]["nvidia_nim"], "nvapi_test")

        cli.command_auth(argparse.Namespace(action="remove", provider="nvidia_nim"))
        config = json.loads((self.home_dir / ".panther" / "config.json").read_text(encoding="utf-8"))
        self.assertIsNone(config["api_keys"]["nvidia_nim"])

    def test_auth_rejects_unknown_provider(self):
        cli.command_init(argparse.Namespace())
        cli.command_auth(argparse.Namespace(action="add", provider="bogus", key="k"))
        config = json.loads((self.home_dir / ".panther" / "config.json").read_text(encoding="utf-8"))
        self.assertNotIn("bogus", config["api_keys"])

    def test_delete_session(self):
        cli.ensure_panther_home()
        session = cli.create_session(self.project_root)
        cli.save_session(session)
        session_file = self.home_dir / ".panther" / "sessions" / f"{session['session_id']}.json"
        self.assertTrue(session_file.exists())

        cli.command_delete(argparse.Namespace(session_id=session["session_id"]))
        self.assertFalse(session_file.exists())

    def test_config_section(self):
        cli.command_init(argparse.Namespace())
        with patch("panther.cli.console") as mock_console:
            cli.command_config(argparse.Namespace(section="llm"))
        mock_console.print_json.assert_called_once()

    def test_unknown_config_section(self):
        cli.command_init(argparse.Namespace())
        with patch("panther.cli.console") as mock_console:
            cli.command_config(argparse.Namespace(section="nope"))
        self.assertTrue(mock_console.print.called)

    def test_entry_point_main_runs_without_recursion(self):
        from panther import __main__ as entry

        with patch("sys.argv", ["panther", "list"]):
            self.assertEqual(entry.main(), 0)

    def test_bare_main_launches_session_after_auto_init(self):
        with patch("sys.argv", ["panther"]), patch(
            "panther.cli.repl_loop"
        ) as mock_repl, patch("panther.cli._maybe_auto_index") as mock_index:
            result = cli.main()
        self.assertEqual(result, 0)
        mock_repl.assert_called_once()
        # First run: no config existed, so auto-init must have written one.
        self.assertTrue((self.home_dir / ".panther" / "config.json").exists())

    def test_main_version_flag(self):
        with patch("sys.argv", ["panther", "--version"]), self.assertRaises(SystemExit):
            cli.main()

    def test_config_merge_keeps_defaults(self):
        cli.command_init(argparse.Namespace())
        config = cli.load_config()
        self.assertIn("retrieval", config)
        self.assertIn("token_budget", config)
        self.assertIn("chunk_size", config["retrieval"])

    def test_ask_resolves_config_file_key(self):
        """The CLI binds the loaded config into resolve_key so config-file keys work."""
        cli.command_init(argparse.Namespace())
        cli.command_auth(argparse.Namespace(action="add", provider="nvidia_nim", key="nvapi_config_key"))
        session = cli.create_session(self.project_root)
        cfg = cli.load_config()
        with patch(
            "panther.qa.answer_question",
            return_value=cli.qa.Answer(text="ok", model_used="m", tokens={"prompt": 1, "completion": 1}),
        ) as mock_answer:
            cli._ask("hi", self.project_root, session, cfg)
        _, kwargs = mock_answer.call_args
        resolver = kwargs["resolve_key"]
        self.assertEqual(resolver("nvidia_nim"), "nvapi_config_key")
        self.assertIsNone(resolver("bogus"))


if __name__ == "__main__":
    unittest.main()
