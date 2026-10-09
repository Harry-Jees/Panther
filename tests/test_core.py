from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from panther.agent import run_agent
from panther.config import load_config, save_config
from panther.permissions import check_permission
from panther.session import SessionManager
from panther.tools import TOOLS, apply_diff, read_file, write_file

REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_project_config_overrides_global_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            home = root / "home"
            project = root / "project"
            with patch.dict(os.environ, {"PANTHER_HOME": str(home)}):
                save_config({"modelTier": "large"}, None)
                project_config = project / ".panther" / "config.json"
                project_config.parent.mkdir(parents=True)
                project_config.write_text('{"permissionMode":"never_ask"}', encoding="utf-8")
                actual = load_config(project)
            self.assertEqual(actual["modelTier"], "large")
            self.assertEqual(actual["permissionMode"], "never_ask")

    def test_invalid_config_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.json"
            config.write_text("{", encoding="utf-8")
            with patch.dict(os.environ, {"PANTHER_CONFIG_PATH": str(config)}):
                with self.assertRaisesRegex(ValueError, "Could not load config"):
                    load_config()


class PermissionTests(unittest.TestCase):
    def test_permission_decision_matrix_and_one_shot_denial(self) -> None:
        self.assertEqual(check_permission("safe", "ask_when_risky"), "approved")
        self.assertEqual(check_permission("risky", "ask_when_risky"), "needs_prompt")
        self.assertEqual(
            check_permission("risky", "ask_when_risky", one_shot=True),
            "denied",
        )
        self.assertEqual(check_permission("risky", "never_ask"), "approved")


class ToolTests(unittest.TestCase):
    def test_file_tools_are_confined_to_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            root.mkdir()
            result = write_file({"filePath": "src/new.txt", "content": "hello"}, root)
            self.assertTrue(result["success"])
            self.assertEqual(read_file({"filePath": "src/new.txt"}, root)["data"], "hello")
            outside = write_file({"filePath": "../escape.txt", "content": "x"}, root)
            self.assertFalse(outside["success"])

    def test_tool_names_are_unique(self) -> None:
        names = [tool.name for tool in TOOLS]
        self.assertEqual(len(names), len(set(names)))

    def test_apply_diff_replaces_existing_text_and_creates_new_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "existing.txt").write_text("before\nafter\n", encoding="utf-8")
            changed = apply_diff(
                {
                    "filePath": "existing.txt",
                    "diff": (
                        "--- a/existing.txt\n+++ b/existing.txt\n"
                        "@@ -1,2 +1,2 @@\n-before\n+updated\n after\n"
                    ),
                },
                root,
            )
            self.assertTrue(changed["success"], changed.get("error"))
            self.assertEqual(
                (root / "existing.txt").read_text(encoding="utf-8"),
                "updated\nafter\n",
            )

            created = apply_diff(
                {
                    "filePath": "created.txt",
                    "diff": "--- /dev/null\n+++ b/created.txt\n@@ -0,0 +1 @@\n+new\n",
                },
                root,
            )
            self.assertTrue(created["success"], created.get("error"))
            self.assertEqual((root / "created.txt").read_text(encoding="utf-8"), "new\n")


class SessionTests(unittest.TestCase):
    def test_session_round_trip_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager = SessionManager(Path(temp))
            session = manager.create()
            session["history"].append({"role": "user", "content": "hello"})
            manager.save(session)
            loaded = manager.load(session["id"])
            if loaded is None:
                self.fail("Saved session could not be loaded")
            self.assertEqual(loaded["history"][0]["content"], "hello")
            self.assertTrue(manager.delete(session["id"]))
            self.assertIsNone(manager.load(session["id"]))

    def test_session_id_cannot_escape_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager = SessionManager(Path(temp))
            with self.assertRaises(ValueError):
                manager.load("../outside")


class AgentTests(unittest.TestCase):
    def test_agent_executes_safe_tool_and_returns_final_answer(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.responses = [
                    '{"name":"read_file","arguments":{"filePath":"note.txt"}}',
                    "The file says hello.",
                ]

            def generate(self, history, tools):
                _ = history, tools
                return self.responses.pop(0)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "note.txt").write_text("hello", encoding="utf-8")
            history: list[dict[str, str]] = []
            answer = run_agent(
                "Read note.txt",
                FakeProvider(),
                history,
                root,
                "ask_when_risky",
            )
            self.assertEqual(answer, "The file says hello.")
            self.assertTrue(any(item["role"] == "tool" for item in history))

    def test_agent_denies_risky_tool_in_one_shot(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.responses = [
                    '{"name":"write_file","arguments":{"filePath":"new.txt","content":"x"}}',
                    "I cannot write files without interactive approval.",
                ]

            def generate(self, history, tools):
                _ = history, tools
                return self.responses.pop(0)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            answer = run_agent(
                "Write a file",
                FakeProvider(),
                [],
                root,
                "ask_when_risky",
            )
            self.assertFalse((root / "new.txt").exists())
            self.assertIn("cannot write", answer)

    def test_interactive_agent_can_run_approved_risky_tool(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.responses = [
                    '{"name":"write_file","arguments":{"filePath":"new.txt","content":"approved"}}',
                    "The file was written.",
                ]

            def generate(self, history, tools):
                _ = history, tools
                return self.responses.pop(0)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            answer = run_agent(
                "Write a file",
                FakeProvider(),
                [],
                root,
                "ask_when_risky",
                one_shot=False,
                approve=lambda name, arguments: isinstance(name, str) and isinstance(arguments, dict),
            )
            self.assertEqual((root / "new.txt").read_text(encoding="utf-8"), "approved")
            self.assertEqual(answer, "The file was written.")


class CLITests(unittest.TestCase):
    def run_cli(self, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        command_env = os.environ.copy()
        if env:
            command_env.update(env)
        return subprocess.run(
            [sys.executable, "-m", "panther.cli", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=command_env,
            check=False,
        )

    def test_help_and_version(self) -> None:
        help_result = self.run_cli("--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("run", help_result.stdout)
        version_result = self.run_cli("--version")
        self.assertEqual(version_result.returncode, 0)
        self.assertIn("0.1.0", version_result.stdout)

    def test_config_command_creates_default_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_cli("config", env={"PANTHER_HOME": temp})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["activeProvider"], "local")
            self.assertTrue((Path(temp) / "config.json").exists())

    def test_analyze_command(self) -> None:
        result = self.run_cli("analyze", str(REPO_ROOT))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Repository analysis", result.stdout)
        self.assertIn("README.md", result.stdout)

    def test_run_reports_missing_inference_backend_as_json_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp) / "models"
            model_dir.mkdir()
            (model_dir / "Qwen3-Coder-4B-Instruct-Q4_K_M.gguf").write_bytes(b"")
            result = self.run_cli(
                "run",
                "--json",
                "Explain this project",
                env={"PANTHER_HOME": temp},
            )
        self.assertEqual(result.returncode, 1)
        records = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(records[0]["type"], "start")
        self.assertEqual(records[-1]["type"], "error")
        self.assertIn("llama-cpp-python", records[-1]["message"])
        self.assertNotIn("Checking for local model", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
