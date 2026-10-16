from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import time
import urllib.error
from email.message import Message
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from prompt_toolkit.document import Document

from panther.agent import (
    CODING_WORKERS,
    AgentEvent,
    AgentPhase,
    MultiAgentOrchestrator,
    TeamChatMessage,
    _system_prompt,
    _plan_coding_tasks,
    brainstorm_with_workers,
    review_with_workers,
    run_agent,
    select_coding_worker_models,
)
from panther.cli import (
    _AgentDashboard,
    _auth_command,
    _bot_command,
    _create_prompt_session,
    _SlashCommandCompleter,
    _is_coding_task,
    _is_agent_brainstorm_task,
    _is_agent_review_task,
    _transcript_cursor_position,
    _image_content_for_task,
    _nvidia_api_key,
    _nvidia_chat_model_candidates,
    _nvidia_check,
    _nvidia_provider,
    _mysql_command,
    _repl,
    _run_command,
    main,
)
from panther.config import load_config, save_config
from panther.indexer import build_index
from panther.permissions import check_permission
from panther.mysql import (
    connect_mysql,
    mysql_config_path,
    mysql_keyring_username,
    mysql_read,
    mysql_execute,
)
from panther.provider import (
    GeminiProvider,
    NvidiaProvider,
    OpenAICompatibleProvider,
)
from panther.session import SessionManager
from panther.skills import discover_skills, install_skill, load_skill_instructions
from panther.tools import (
    TOOLS,
    _bing_rss_search,
    apply_diff,
    browse_web,
    create_checkpoint,
    git_diff,
    git_status,
    grep,
    list_dir,
    _search_web,
    find_files,
    query_index,
    read_file,
    read_file_range,
    restore_checkpoint,
    run_command,
    run_tests,
    write_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_nvidia_provider_and_model_can_be_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_config(
                {
                    "activeProvider": "nvidia",
                    "remoteModel": "nvidia/test-model",
                },
                root,
            )
            config = load_config(root)
        self.assertEqual(config["activeProvider"], "nvidia")
        self.assertEqual(config["remoteModel"], "nvidia/test-model")

    def test_named_bot_profiles_round_trip_in_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_config(
                {
                    "activeBot": "Gemini Work",
                    "bots": {
                        "Gemini Work": {
                            "name": "Gemini Work",
                            "provider": "gemini",
                            "model": "gemini-test",
                        },
                        "Custom API": {
                            "name": "Custom API",
                            "provider": "openai-compatible",
                            "model": "custom-test",
                            "baseUrl": "https://api.example.test/v1",
                        },
                    },
                },
                root,
            )
            config = load_config(root)
        self.assertEqual(config["activeBot"], "Gemini Work")
        self.assertEqual(config["bots"]["Gemini Work"]["provider"], "gemini")
        self.assertEqual(config["bots"]["Custom API"]["baseUrl"], "https://api.example.test/v1")

    def test_named_bots_can_be_switched_without_exposing_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            root.mkdir()
            home = Path(temp) / "home"
            profile = {
                "name": "Gemini Work",
                "provider": "gemini",
                "model": "gemini-test",
            }
            with patch.dict(os.environ, {"PANTHER_HOME": str(home)}):
                save_config(
                    {
                        "bots": {"Gemini Work": profile},
                        "activeBot": "",
                    },
                    root,
                )
                self.assertEqual(
                    _bot_command(
                        Namespace(action="use", name="Gemini Work"),
                        root,
                    ),
                    0,
                )
                config = load_config(root)
            self.assertEqual(config["activeBot"], "Gemini Work")
            self.assertNotIn("apiKey", config["bots"]["Gemini Work"])

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

    def test_project_autopilot_and_remembered_permissions_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {
                "autopilot": True,
                "rememberedPermissions": ["write_file", "run_tests"],
            }
            save_config(config, root)
            actual = load_config(root)
        self.assertTrue(actual["autopilot"])
        self.assertEqual(
            actual["rememberedPermissions"],
            ["write_file", "run_tests"],
        )

    def test_invalid_config_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.json"
            config.write_text("{", encoding="utf-8")
            with patch.dict(os.environ, {"PANTHER_CONFIG_PATH": str(config)}):
                with self.assertRaisesRegex(ValueError, "Could not load config"):
                    load_config()


class AuthTests(unittest.TestCase):
    def test_add_stores_key_in_keyring_and_activates_nvidia(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            keyring = SimpleNamespace(set_password=Mock())
            with patch("panther.cli.import_module", return_value=keyring), patch(
                "panther.cli.getpass.getpass", return_value="test-secret"
            ) as getpass:
                self.assertEqual(
                    _auth_command(Namespace(action="add"), root),
                    0,
                )
            getpass.assert_called_once()
            keyring.set_password.assert_called_once_with(
                "panther-cli",
                "nvidia",
                "test-secret",
            )
            config_path = root / ".panther" / "config.json"
            self.assertNotIn("test-secret", config_path.read_text(encoding="utf-8"))
            self.assertEqual(load_config(root)["activeProvider"], "nvidia")

    def test_bot_setup_saves_named_profile_and_tests_sample_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            root.mkdir()
            keyring = SimpleNamespace(set_password=Mock())
            answers = iter(["2", "gemini-test"])
            with patch.dict(os.environ, {"PANTHER_HOME": str(Path(temp) / "home")}), patch(
                "panther.cli.sys.stdin",
                SimpleNamespace(isatty=lambda: True),
            ), patch("builtins.input", side_effect=lambda _prompt="": next(answers)), patch(
                "panther.cli.getpass.getpass",
                return_value="gemini-secret",
            ), patch("panther.cli.import_module", return_value=keyring), patch(
                "panther.cli._test_bot",
                return_value="Gemini is ready.",
            ) as test_bot:
                result = _bot_command(
                    Namespace(action="add", name="Gemini Work"),
                    root,
                )
            self.assertEqual(result, 0)
            with patch.dict(
                os.environ,
                {"PANTHER_HOME": str(Path(temp) / "home")},
            ):
                config = load_config(root)
            self.assertEqual(config["activeBot"], "Gemini Work")
            self.assertEqual(config["bots"]["Gemini Work"]["provider"], "gemini")
            keyring.set_password.assert_called_once()
            test_bot.assert_called_once()


class PermissionTests(unittest.TestCase):
    def test_permission_decision_matrix_and_one_shot_denial(self) -> None:
        self.assertEqual(check_permission("safe", "ask_when_risky"), "approved")
        self.assertEqual(check_permission("risky", "ask_when_risky"), "needs_prompt")
        self.assertEqual(
            check_permission("risky", "ask_when_risky", one_shot=True),
            "denied",
        )
        self.assertEqual(check_permission("risky", "never_ask"), "approved")
        self.assertEqual(
            check_permission(
                "risky",
                "never_ask",
                capability="git_remote",
                autopilot=True,
                remembered=True,
            ),
            "needs_prompt",
        )
        self.assertEqual(
            check_permission(
                "risky",
                "ask_when_risky",
                capability="git_remote",
                one_shot=True,
            ),
            "denied",
        )

    def test_remembered_and_autopilot_approvals_cover_one_shot_tools(self) -> None:
        self.assertEqual(
            check_permission(
                "risky",
                "ask_when_risky",
                one_shot=True,
                remembered=True,
            ),
            "approved",
        )
        self.assertEqual(
            check_permission(
                "risky",
                "ask_when_risky",
                one_shot=True,
                autopilot=True,
            ),
            "approved",
        )


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
        self.assertEqual(
            set(names),
            {
                "browse_web", "mysql_read", "mysql_execute", "read_file", "read_file_range", "find_files",
                "query_index", "list_dir", "grep", "run_command", "git_status",
                "git_diff", "git_push", "git_pull", "create_checkpoint", "restore_checkpoint", "run_tests",
                "write_file", "apply_diff",
            },
        )

    def test_bounded_file_reads_include_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "note.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            result = read_file_range({"filePath": "note.txt", "startLine": 2, "endLine": 3}, root)
            self.assertTrue(result["success"])
            self.assertEqual(result["data"], "2: two\n3: three")

    def test_find_files_ignores_generated_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            (root / "src/main.py").write_text("", encoding="utf-8")
            (root / "__pycache__").mkdir()
            (root / "__pycache__/main.pyc").write_bytes(b"")
            result = find_files({"pattern": "*.py"}, root)
            self.assertEqual(result["data"], "src/main.py")

    def test_parallel_index_contains_file_metadata_and_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "main.py").write_text("class Panther:\n    pass\n", encoding="utf-8")
            events = []
            index = build_index(root, worker_count=2, on_event=events.append)
            self.assertEqual(index["fileCount"], 1)
            self.assertEqual(index["files"][0]["symbols"], ["Panther"])
            self.assertTrue((root / ".panther" / "index.json").is_file())
            self.assertIn("coordinator", {event.worker for event in events})

    def test_web_search_falls_back_to_structured_wikipedia_results(self) -> None:
        with (
            patch("panther.tools._bing_rss_search", side_effect=OSError("offline")),
            patch(
                "panther.tools._download_web_text",
                side_effect=[
                    "DuckDuckGo unavailable",
                    "Google Search If you're having trouble accessing Google Search",
                ],
            ),
            patch(
                "panther.tools._wikipedia_search",
                return_value="Search provider: Wikipedia\n- Example\n  https://en.wikipedia.org/wiki/Example",
            ),
        ):
            result = _search_web("example")
        self.assertIn("Search provider: Wikipedia", result)
        self.assertIn("https://en.wikipedia.org/wiki/Example", result)

    def test_web_search_rejects_engine_landing_pages_without_result_links(self) -> None:
        with (
            patch("panther.tools._bing_rss_search", side_effect=OSError("offline")),
            patch(
                "panther.tools._download_web_text",
                side_effect=[
                    "DuckDuckGo All Regions (https://duckduckgo.com/?q=example)",
                    "Google Search (https://www.google.com/search?q=example)",
                ],
            ),
            patch("panther.tools._wikipedia_search", side_effect=OSError("offline")),
        ):
            with self.assertRaisesRegex(RuntimeError, "no external result links"):
                _search_web("example")

    def test_bing_rss_search_extracts_actual_result_items(self) -> None:
        feed = b"""<?xml version="1.0"?>
        <rss version="2.0"><channel><item>
          <title>Python documentation</title>
          <link>https://docs.python.org/3/</link>
          <description>The &lt;b&gt;official&lt;/b&gt; documentation.</description>
        </item></channel></rss>"""
        response = io.BytesIO(feed)
        with patch("panther.tools.urllib.request.urlopen", return_value=response):
            result = _bing_rss_search("python documentation")
        self.assertIn("Search provider: Bing RSS", result)
        self.assertIn("https://docs.python.org/3/", result)
        self.assertIn("The official documentation.", result)

    @patch("panther.tools._download_web_text", return_value="Readable page text")
    def test_browse_web_fetches_http_content(self, download: Mock) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = browse_web({"url": "https://example.com/page"}, Path(temp))
        self.assertTrue(result["success"])
        self.assertIn("Readable page text", result["data"])
        download.assert_called_once_with("https://example.com/page")

    @patch("panther.tools._search_web", return_value="Search provider: test")
    def test_browse_web_dispatches_search_queries(self, search: Mock) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = browse_web({"query": "Panther project"}, Path(temp))
        self.assertTrue(result["success"])
        self.assertIn("Panther project", result["data"])
        search.assert_called_once_with("Panther project")

    def test_index_directory_and_grep_tools_return_project_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("def launch():\n    return True\n", encoding="utf-8")
            (root / ".panther").mkdir()
            (root / ".panther" / "index.json").write_text(
                json.dumps({"files": [{"path": "src/app.py", "symbols": ["launch"]}]}),
                encoding="utf-8",
            )

            directory = list_dir({"path": "src"}, root)
            indexed = query_index({"query": "launch"}, root)
            matches = grep({"pattern": "def launch", "glob": "*.py"}, root)

        self.assertTrue(directory["success"])
        self.assertIn("app.py", directory["data"])
        self.assertIn("src/app.py", indexed["data"])
        self.assertIn("src\\app.py:1:def launch()", matches["data"])

    def test_command_and_test_runner_execute_requested_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            command_result = run_command({"command": "git --version"}, root)
            test_result = run_tests(
                {"command": "git --version"},
                root,
            )
        self.assertTrue(command_result["success"], command_result.get("error"))
        self.assertIn("git version", command_result["data"])
        self.assertTrue(test_result["success"], test_result.get("error"))

    @patch("panther.tools._git")
    def test_git_and_checkpoint_tools_use_git_and_persist_checkpoint(self, git: Mock) -> None:
        git.return_value = {"success": True, "data": "## main"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            status = git_status({}, root)
            diff = git_diff({"path": "src/app.py"}, root)

            git.return_value = {"success": True, "data": "binary-patch"}
            checkpoint = create_checkpoint({}, root)
            checkpoint_id = checkpoint["data"].split(": ", 1)[1]
            saved_patch = root / ".panther" / "checkpoints" / f"{checkpoint_id}.patch"
            saved_content = saved_patch.read_text(encoding="utf-8")

            git.return_value = {"success": True, "data": "restored"}
            restored = restore_checkpoint({"checkpointId": checkpoint_id}, root)

        self.assertTrue(status["success"])
        self.assertTrue(diff["success"])
        self.assertEqual(saved_content, "binary-patch")
        self.assertTrue(restored["success"])
        self.assertEqual(git.call_args.args[0][:2], ["apply", "--binary"])

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


class SkillTests(unittest.TestCase):
    def test_local_skill_file_installs_and_is_loaded_into_agent_with_web_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"PANTHER_HOME": str(Path(temp) / "home")},
        ):
            root = Path(temp) / "project"
            root.mkdir()
            source = Path(temp) / "web-research.md"
            source.write_text(
                "---\nname: web-research\n---\n"
                "For current documentation, browse the official public web and cite sources.\n",
                encoding="utf-8",
            )
            installed = install_skill(str(source), root)
            discovered = discover_skills(root)
            prompt_text = load_skill_instructions(root)

            class FakeProvider:
                def __init__(self) -> None:
                    self.tools: set[str] = set()
                    self.system_prompt = ""

                def generate(self, history, tools):
                    self.tools = {tool["name"] for tool in tools}
                    self.system_prompt = history[0]["content"]
                    return "Done."

            provider = FakeProvider()
            answer = run_agent(
                "Find current docs",
                provider,
                [],
                root,
                "never_ask",
                enable_subagents=False,
            )

        self.assertEqual(installed.name, "web-research")
        self.assertEqual(len(discovered), 1)
        self.assertIn("browse the official public web", prompt_text)
        self.assertIn("browse the official public web", provider.system_prompt)
        self.assertIn("browse_web", provider.tools)
        self.assertEqual(answer, "Done.")

    def test_global_skill_is_discovered_for_any_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"PANTHER_HOME": str(Path(temp) / "home")},
        ):
            project = Path(temp) / "project"
            source = Path(temp) / "global.md"
            source.write_text("# Global helper\nShared guidance.", encoding="utf-8")
            installed = install_skill(str(source), project, global_install=True)
            discovered = discover_skills(project)

        self.assertEqual(installed.parent, Path(temp) / "home" / "skills")
        self.assertEqual([item.parent.name for item in discovered], ["global"])

    def test_local_skill_directory_copies_supporting_files_without_git_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            source = Path(temp) / "skills" / "sample-skill"
            (source / "references").mkdir(parents=True)
            (source / ".git").mkdir()
            (source / "SKILL.md").write_text("# Sample\nUse the reference.", encoding="utf-8")
            (source / "references" / "guide.md").write_text("Reference text", encoding="utf-8")
            (source / ".git" / "config").write_text("metadata", encoding="utf-8")

            installed = install_skill(str(source), root, name="sample")
            installed_skill = (installed / "SKILL.md").read_text(encoding="utf-8")
            installed_reference = (installed / "references" / "guide.md").read_text(encoding="utf-8")
            copied_git_metadata = (installed / ".git").exists()

        self.assertEqual(installed_skill, "# Sample\nUse the reference.")
        self.assertEqual(installed_reference, "Reference text")
        self.assertFalse(copied_git_metadata)

    @patch("panther.skills.subprocess.run")
    def test_git_skill_install_selects_subdirectory_and_clones_without_submodules(
        self,
        git_run: Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"PANTHER_HOME": str(Path(temp) / "home")},
        ):
            project = Path(temp) / "project"

            def clone(arguments, **kwargs):
                clone_target = Path(arguments[-1])
                skill = clone_target / "catalog" / "web"
                skill.mkdir(parents=True)
                (skill / "SKILL.md").write_text(
                    "---\nname: web-browsing\n---\nSearch the web when asked.",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(arguments, 0, "", "")

            git_run.side_effect = clone
            installed = install_skill(
                "https://example.test/org/skills.git",
                project,
                subdir="catalog/web",
            )
            clone_arguments = git_run.call_args.args[0]
            copied_skill = (installed / "SKILL.md").is_file()

        self.assertEqual(installed.name, "web-browsing")
        self.assertIn("--no-recurse-submodules", clone_arguments)
        self.assertTrue(copied_skill)

    def test_local_multi_skill_repository_requires_subdirectory_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repository = Path(temp) / "repo"
            for name in ("first", "second"):
                skill = repository / "skills" / name
                skill.mkdir(parents=True)
                (skill / "SKILL.md").write_text(f"# {name}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "--subdir"):
                install_skill(str(repository), Path(temp) / "project")

            selected = install_skill(
                str(repository),
                Path(temp) / "project",
                subdir="skills/second",
            )

        self.assertEqual(selected.name, "second")

    def test_skills_cli_installs_local_skill(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"PANTHER_HOME": str(Path(temp) / "home")},
        ):
            root = Path(temp) / "project"
            root.mkdir()
            source = Path(temp) / "quick-skill.md"
            source.write_text("# Quick skill\nDo the thing.", encoding="utf-8")
            with patch("panther.cli.Path.cwd", return_value=root), patch(
                "panther.cli.sys.stdout", io.StringIO()
            ):
                result = main(["skills", "add", str(source), "--name", "quick"])
            installed = root / ".panther" / "skills" / "quick" / "SKILL.md"
            installed_exists = installed.is_file()

        self.assertEqual(result, 0)
        self.assertTrue(installed_exists)

    def test_repl_skill_command_handles_windows_paths_and_lists_install(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            root.mkdir()
            source = Path(temp) / "skill files" / "helper.md"
            source.parent.mkdir()
            source.write_text("# Helper\nInstalled from the interactive prompt.", encoding="utf-8")
            output = io.StringIO()
            with patch(
                "builtins.input",
                side_effect=[
                    f'/skill add "{source}" --name helper',
                    "/skills",
                    "exit",
                ],
            ), patch("panther.cli.load_config", return_value={"activeBot": ""}), patch(
                "panther.cli._ensure_repository_index"
            ), patch("panther.cli._print_banner"), patch(
                "panther.cli._print_slash_commands"
            ), patch("panther.cli._create_prompt_session", return_value=None), patch(
                "panther.cli.sys.stdout", output
            ):
                result = _repl(root, show_banner=False)
            installed_exists = (root / ".panther" / "skills" / "helper" / "SKILL.md").is_file()

        self.assertEqual(result, 0)
        self.assertTrue(installed_exists)
        self.assertIn("helper (project)", output.getvalue())


class MysqlTests(unittest.TestCase):
    def test_mysql_write_permission_display_hides_parameter_values(self) -> None:
        from panther.cli import _permission_summary

        rendered = _permission_summary(
            {
                "query": "INSERT INTO users (password) VALUES (%s)",
                "parameters": ["sensitive-value"],
            },
            "mysql_execute",
        )
        self.assertIn("INSERT INTO users", rendered)
        self.assertIn("redacted SQL parameter values", rendered)
        self.assertNotIn("sensitive-value", rendered)

    def test_project_mysql_setup_tests_connection_and_keeps_password_in_keyring(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            root.mkdir()
            keyring = SimpleNamespace(
                get_password=Mock(return_value=None),
                set_password=Mock(),
            )
            with patch("panther.cli.import_module", return_value=keyring), patch(
                "panther.cli.getpass.getpass", return_value="database-secret"
            ), patch("panther.cli.test_mysql_connection") as test_connection:
                result = _mysql_command(
                    Namespace(
                        action="setup",
                        host="127.0.0.1",
                        port=3307,
                        database="panther",
                        username="panther_user",
                    ),
                    root,
                )
            settings = json.loads(mysql_config_path(root).read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        test_connection.assert_called_once_with(
            root,
            "database-secret",
            {
                "host": "127.0.0.1",
                "port": 3307,
                "database": "panther",
                "username": "panther_user",
            },
        )
        keyring.set_password.assert_called_once_with(
            "panther-cli",
            mysql_keyring_username(root),
            "database-secret",
        )
        self.assertEqual(settings["database"], "panther")
        self.assertNotIn("password", settings)

    def test_mysql_connection_uses_project_keyring_and_never_persists_password(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            settings = {
                "host": "localhost",
                "port": 3306,
                "database": "app",
                "username": "app_user",
            }
            config_path = mysql_config_path(root)
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(settings), encoding="utf-8")
            keyring = SimpleNamespace(get_password=Mock(return_value="secret-value"))
            connector = SimpleNamespace(connect=Mock(return_value=object()))
            with patch("panther.mysql._mysql_connector", return_value=connector), patch(
                "panther.mysql.importlib.import_module",
                return_value=keyring,
            ) as imported:
                connection = connect_mysql(root)
            persisted_config = config_path.read_text(encoding="utf-8")

        self.assertIsNotNone(connection)
        imported.assert_called_once_with("keyring")
        connector.connect.assert_called_once_with(
            host="localhost",
            port=3306,
            database="app",
            user="app_user",
            password="secret-value",
            connection_timeout=5,
            read_timeout=30,
            write_timeout=30,
        )
        self.assertNotIn("secret-value", persisted_config)

    def test_mysql_read_is_readonly_bounded_and_parameterized(self) -> None:
        rows = [(f"row-{number}",) for number in range(101)]
        cursor = SimpleNamespace(
            description=[("name",)],
            execute=Mock(),
            fetchmany=Mock(return_value=rows),
            close=Mock(),
        )
        connection = SimpleNamespace(
            start_transaction=Mock(),
            cursor=Mock(return_value=cursor),
            rollback=Mock(),
            close=Mock(),
        )
        with patch("panther.mysql.connect_mysql", return_value=connection) as connect:
            result = mysql_read(
                {
                    "query": "SELECT name FROM users WHERE id > %s",
                    "parameters": [10],
                },
                Path("project"),
            )

        payload = json.loads(result["data"])
        connect.assert_called_once_with(Path("project"))
        connection.start_transaction.assert_called_once_with(readonly=True)
        cursor.execute.assert_called_once_with(
            "SELECT * FROM (SELECT name FROM users WHERE id > %s) AS `_panther_bounded_result` LIMIT 101",
            (10,),
        )
        self.assertEqual(len(payload["rows"]), 100)
        self.assertTrue(payload["truncated"])
        connection.rollback.assert_called_once()
        connection.close.assert_called_once()

    def test_mysql_read_rejects_writes_and_file_or_lock_clauses(self) -> None:
        for query in (
            "DELETE FROM users",
            "SELECT * FROM users INTO OUTFILE '/tmp/dump'",
            "SELECT * FROM users FOR UPDATE",
            "SELECT LOAD_FILE('/etc/passwd')",
            "SELECT GET_LOCK('panther', 10)",
            "SELECT 1; DELETE FROM users",
        ):
            with self.subTest(query=query), patch("panther.mysql.connect_mysql") as connect:
                result = mysql_read({"query": query}, Path("project"))
                self.assertFalse(result["success"])
                connect.assert_not_called()

    def test_mysql_execute_commits_single_approved_statement_and_blocks_admin(self) -> None:
        cursor = SimpleNamespace(
            execute=Mock(),
            rowcount=2,
            lastrowid=17,
            close=Mock(),
        )
        connection = SimpleNamespace(
            cursor=Mock(return_value=cursor),
            commit=Mock(),
            rollback=Mock(),
            close=Mock(),
        )
        with patch("panther.mysql.connect_mysql", return_value=connection):
            result = mysql_execute(
                {"query": "UPDATE users SET active = %s WHERE id = %s", "parameters": [True, 7]},
                Path("project"),
            )
            denied = mysql_execute(
                {"query": "CREATE USER 'other'@'localhost'"},
                Path("project"),
            )
            denied_replace = mysql_execute(
                {"query": "CREATE OR REPLACE USER 'other'@'localhost'"},
                Path("project"),
            )

        self.assertTrue(result["success"])
        self.assertEqual(json.loads(result["data"]), {"affectedRows": 2, "lastInsertId": 17})
        connection.commit.assert_called_once()
        connection.rollback.assert_not_called()
        self.assertFalse(denied["success"])
        self.assertIn("administrative statement", denied["error"])
        self.assertFalse(denied_replace["success"])
        self.assertEqual(connection.cursor.call_count, 1)


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
    def test_system_prompt_supports_general_chat_without_finance_assumptions(self) -> None:
        prompt = _system_prompt()
        self.assertIn("general-purpose assistant", prompt)
        self.assertIn("do not assume an ambiguous topic is about stocks", prompt)
        self.assertIn("Cite only source URLs returned by the web tool", prompt)

    def test_image_is_attached_and_prompted_for_direct_visual_analysis(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.user_content = None

            def generate(self, history, tools):
                _ = tools
                self.user_content = next(
                    item["content"]
                    for item in reversed(history)
                    if item.get("role") == "user"
                )
                return "The image shows a chart."

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
            provider = FakeProvider()
            answer = run_agent(
                "What does this image show?",
                provider,
                [],
                root,
                "ask_when_risky",
                image_content=image,
            )

        self.assertEqual(answer, "The image shows a chart.")
        self.assertIn(
            "Analyze the attached image directly",
            provider.user_content[0]["text"],
        )
        self.assertEqual(provider.user_content[1], image)

    def test_subagent_model_events_are_forwarded_with_role(self) -> None:
        captured_events: list[AgentEvent] = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env").write_text("NVIDIA_API_KEY=private", encoding="utf-8")

            def fake_run_agent(*args, **kwargs):
                workspace = args[3]
                self.assertFalse((workspace / ".env").exists())
                self.assertEqual(len(kwargs["required_tool_groups"]), 2)
                self.assertIn("browse_web", kwargs["allowed_tool_names"])
                self.assertTrue(
                    kwargs["required_tool_groups"][0] & kwargs["allowed_tool_names"]
                )
                self.assertTrue(
                    kwargs["required_tool_groups"][1] & kwargs["allowed_tool_names"]
                )
                (workspace / "calculator.py").write_text(
                    "def add(a, b):\n    return a + b\n",
                    encoding="utf-8",
                )
                kwargs["on_event"](
                    AgentEvent("model_started", AgentPhase.UNDERSTANDING, 0, {})
                )
                return "Implemented calculator."

            orchestrator = MultiAgentOrchestrator(
                SimpleNamespace(model="nvidia/test"),
                root,
                worker_models={"mochi": "nvidia/mochi-model"},
                provider_factory=lambda model: SimpleNamespace(model=model),
            )
            with patch("panther.agent.run_agent", side_effect=fake_run_agent):
                report = orchestrator._run_role(
                    "mochi",
                    {
                        "title": "Build calculator",
                        "task": "Create calculator.py",
                        "main_request": "Create a calculator",
                    },
                    captured_events.append,
                )

        self.assertTrue(report.success)
        self.assertEqual(report.changed_files, ("calculator.py",))
        self.assertIn("def add(a, b)", report.diff)
        self.assertFalse((root / "calculator.py").exists())
        self.assertTrue(
            any(
                event.name == "model_started"
                and event.data["subagent"] == "mochi"
                and event.data["model"] == "nvidia/mochi-model"
                for event in captured_events
            )
        )

    def test_coding_worker_model_selection_prefers_distinct_kimi_family(self) -> None:
        models = select_coding_worker_models(
            [
                "moonshotai/kimi-k2.5",
                "deepseek-ai/deepseek-v3",
                "nvidia/nemotron-3.5",
            ],
            "nvidia/default",
        )
        self.assertEqual(models["mochi"], "moonshotai/kimi-k2.5")
        self.assertEqual(models["boba"], "deepseek-ai/deepseek-v3")
        self.assertEqual(models["pip"], "nvidia/nemotron-3.5")
        self.assertEqual(len(set(models.values())), 3)

    def test_coding_planner_retries_malformed_json(self) -> None:
        plan = {
            "tasks": [
                {"agent": "mochi", "title": "Core", "task": "Implement core behavior."},
                {"agent": "boba", "title": "UI", "task": "Integrate user-facing changes."},
                {"agent": "pip", "title": "Tests", "task": "Cover edge cases."},
            ]
        }

        class PlannerProvider:
            def __init__(self) -> None:
                self.responses = [
                    '{"tasks":[{"agent":"mochi","task":"unescaped "quote""}',
                    json.dumps(plan),
                ]
                self.calls = 0

            def generate(self, history, tools):
                _ = history, tools
                self.calls += 1
                return self.responses.pop(0)

        provider = PlannerProvider()
        with tempfile.TemporaryDirectory() as temp:
            parsed, warning = _plan_coding_tasks(
                provider,
                "Make the tic-tac-toe AI work",
                Path(temp),
            )

        self.assertEqual(parsed["mochi"]["title"], "Core")
        self.assertIsNone(warning)
        self.assertEqual(provider.calls, 2)

    def test_coding_planner_falls_back_after_repeated_invalid_json(self) -> None:
        class PlannerProvider:
            model = "test-model"

            def __init__(self) -> None:
                self.calls = 0

            def generate(self, history, tools):
                _ = history, tools
                self.calls += 1
                return '{"tasks": [invalid'

        provider = PlannerProvider()
        with tempfile.TemporaryDirectory() as temp:
            plan, warning = _plan_coding_tasks(
                provider,
                "Make the tic-tac-toe AI work",
                Path(temp),
            )

        self.assertEqual(set(plan), {"mochi", "boba", "pip"})
        self.assertIn("built-in task split", warning or "")
        self.assertTrue(all("tic-tac-toe AI work" in item["task"] for item in plan.values()))
        self.assertEqual(provider.calls, 2)

    def test_coding_team_runs_parallel_workers_with_distinct_model_providers(self) -> None:
        plan = {
            "tasks": [
                {"agent": "mochi", "title": "Build", "task": "Implement the main feature."},
                {"agent": "boba", "title": "Integrate", "task": "Implement a separate integration."},
                {"agent": "pip", "title": "Test", "task": "Add focused tests."},
            ]
        }

        class PlannerProvider:
            model = "nvidia/lead"

            def generate(self, history, tools):
                _ = history, tools
                return json.dumps(plan)

        models = {
            "mochi": "moonshotai/kimi-k2.5",
            "boba": "deepseek-ai/deepseek-v3",
            "pip": "nvidia/nemotron-3.5",
        }
        created_models: list[str] = []
        events: list[AgentEvent] = []
        chat_calls: list[tuple[str, list[dict[str, str]], list[dict[str, str]]]] = []

        def provider_factory(model: str):
            created_models.append(model)

            class WorkerProvider:
                def __init__(self) -> None:
                    self.model = model

                def generate(self, history, tools):
                    chat_calls.append((model, history, tools))
                    return f"A useful note from {model}."

            return WorkerProvider()

        def fake_run_agent(prompt, provider, history, workspace, permission_mode, **kwargs):
            _ = history, permission_mode
            self.assertIn("isolated coding worker", prompt)
            self.assertNotIn("run_command", kwargs["allowed_tool_names"])
            (workspace / f"{provider.model.split('/')[-1]}.py").write_text(
                "VALUE = True\n",
                encoding="utf-8",
            )
            kwargs["on_event"](
                AgentEvent("model_started", AgentPhase.UNDERSTANDING, 0, {})
            )
            return "Implementation complete."

        with tempfile.TemporaryDirectory() as temp:
            orchestrator = MultiAgentOrchestrator(
                PlannerProvider(),
                Path(temp),
                worker_models=models,
                provider_factory=provider_factory,
            )
            with patch("panther.agent.run_agent", side_effect=fake_run_agent):
                reports = orchestrator.run("Build a calculator", on_event=events.append)

        self.assertEqual(set(created_models), set(models.values()))
        self.assertEqual(len(reports), 3)
        self.assertTrue(all(report.success and report.diff for report in reports))
        messages = [event for event in events if event.name == "team_message"]
        self.assertEqual(
            [event.data["role"] for event in messages],
            ["mochi", "boba", "pip"],
        )
        self.assertEqual(len(chat_calls), 3)
        self.assertEqual(chat_calls[0][2], [])
        self.assertIn("A useful note from moonshotai/kimi-k2.5.", chat_calls[1][1][-1]["content"])
        self.assertIn("A useful note from deepseek-ai/deepseek-v3.", chat_calls[2][1][-1]["content"])
        self.assertTrue(
            all(
                any(
                    event.name == "subagent_completed"
                    and event.data["role"] == report.role
                    and event.data["success"]
                    for event in events
                )
                for report in reports
            )
        )

    def test_brainstorm_workers_share_prior_ideas_without_tools(self) -> None:
        models = {"mochi": "model/mochi", "boba": "model/boba", "pip": "model/pip"}
        calls: list[tuple[str, list[dict[str, str]], list[dict[str, str]]]] = []
        events: list[AgentEvent] = []

        def provider_factory(model: str):
            class WorkerProvider:
                def generate(self, history, tools):
                    calls.append((model, history, tools))
                    return f"Build idea from {model}."

            return WorkerProvider()

        messages, failures = brainstorm_with_workers(
            "What do the three agents suggest I build?",
            SimpleNamespace(model="default"),
            worker_models=models,
            provider_factory=provider_factory,
            on_event=events.append,
        )

        self.assertEqual([message.role for message in messages], ["mochi", "boba", "pip"])
        self.assertEqual(failures, [])
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(not tools for _, _, tools in calls))
        self.assertIn("Build idea from model/mochi.", calls[1][1][-1]["content"])
        self.assertIn("Build idea from model/boba.", calls[2][1][-1]["content"])
        self.assertEqual(
            [event.name for event in events],
            [
                "brainstorm_started",
                "brainstorm_agent_started",
                "brainstorm_message",
                "brainstorm_agent_started",
                "brainstorm_message",
                "brainstorm_agent_started",
                "brainstorm_message",
            ],
        )

    def test_review_workers_only_receive_read_tools_and_report_named_findings(self) -> None:
        events: list[AgentEvent] = []
        received_tools: list[set[str]] = []

        def fake_run_agent(prompt, provider, history, root, permission_mode, **kwargs):
            self.assertIn("read-only Panther project reviewer", prompt)
            self.assertEqual(permission_mode, "never_ask")
            allowed = kwargs["allowed_tool_names"]
            required_groups = kwargs["required_tool_groups"]
            received_tools.append(allowed)
            self.assertNotIn("write_file", allowed)
            self.assertNotIn("apply_diff", allowed)
            self.assertIn("browse_web", allowed)
            self.assertEqual(len(required_groups), 1)
            self.assertTrue(required_groups[0] & allowed)
            kwargs["on_event"](
                AgentEvent(
                    "tool_requested",
                    AgentPhase.INSPECTING,
                    1,
                    {"tool": "read_file", "arguments": {"filePath": "game.py"}},
                )
            )
            return f"Verified finding by {provider.model}."

        with tempfile.TemporaryDirectory() as temp:
            with patch("panther.agent.run_agent", side_effect=fake_run_agent):
                findings, failures = review_with_workers(
                    "Analyze the tic tac toe project",
                    SimpleNamespace(model="test-model"),
                    Path(temp),
                    on_event=events.append,
                )

        self.assertEqual(len(findings), 3)
        self.assertEqual(failures, [])
        self.assertEqual(len(received_tools), 3)
        self.assertEqual([item.role for item in findings], ["mochi", "boba", "pip"])
        self.assertTrue(all(event.data.get("subagent") for event in events if event.name == "tool_requested"))
        self.assertEqual(
            [event.name for event in events if event.name == "review_message"],
            ["review_message"] * 3,
        )

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

    def test_agent_requires_successful_required_tool_before_final_answer(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.responses = [
                    "I can answer without inspecting the file.",
                    '{"name":"read_file","arguments":{"filePath":"note.txt"}}',
                    "The file says hello.",
                ]
                self.saw_requirement = False

            def generate(self, history, tools):
                _ = tools
                self.assert_requirement_message(history)
                return self.responses.pop(0)

            def assert_requirement_message(self, history):
                self.saw_requirement = self.saw_requirement or any(
                    item.get("role") == "user"
                    and "must use the required project tools" in item.get("content", "")
                    for item in history
                )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "note.txt").write_text("hello", encoding="utf-8")
            provider = FakeProvider()
            events: list[AgentEvent] = []
            answer = run_agent(
                "Read note.txt",
                provider,
                [],
                root,
                "never_ask",
                enable_subagents=False,
                required_tool_groups=(frozenset({"read_file"}),),
                on_event=events.append,
            )

        self.assertTrue(provider.saw_requirement)
        self.assertEqual(answer, "The file says hello.")
        self.assertTrue(any(event.name == "required_tool_missing" for event in events))

    def test_agent_rejects_tools_outside_worker_allowlist(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.responses = [
                    '{"name":"run_command","arguments":{"command":"echo unsafe"}}',
                    "I cannot run commands in this worker.",
                ]
                self.offered_tools: list[str] = []

            def generate(self, history, tools):
                _ = history
                self.offered_tools = [tool["name"] for tool in tools]
                return self.responses.pop(0)

        with tempfile.TemporaryDirectory() as temp:
            provider = FakeProvider()
            answer = run_agent(
                "Do isolated work",
                provider,
                [],
                Path(temp),
                "never_ask",
                enable_subagents=False,
                allowed_tool_names={"read_file"},
            )
        self.assertNotIn("run_command", provider.offered_tools)
        self.assertIn("cannot run commands", answer)

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

    def test_agent_requires_approval_for_mysql_changes(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.responses = [
                    '{"name":"mysql_execute","arguments":{"query":"DELETE FROM users WHERE id = %s","parameters":[4]}}',
                    "The database change was denied.",
                ]
                self.offered_tools: set[str] = set()

            def generate(self, history, tools):
                _ = history
                self.offered_tools = {tool["name"] for tool in tools}
                return self.responses.pop(0)

        with tempfile.TemporaryDirectory() as temp:
            provider = FakeProvider()
            events: list[AgentEvent] = []
            answer = run_agent(
                "Delete a user",
                provider,
                [],
                Path(temp),
                "ask_when_risky",
                one_shot=False,
                approve=lambda tool, arguments: False,
                autopilot=True,
                remembered_permissions={"mysql_execute"},
                enable_subagents=False,
                on_event=events.append,
            )

        self.assertIn("mysql_execute", provider.offered_tools)
        request = next(event for event in events if event.name == "permission_requested")
        self.assertEqual(request.data["tool"], "mysql_execute")
        self.assertEqual(request.data["arguments"]["query"], "DELETE FROM users WHERE id = %s")
        self.assertIn("denied", answer)

    def test_one_shot_mysql_change_is_denied_without_prompt(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.responses = [
                    '{"name":"mysql_execute","arguments":{"query":"DELETE FROM users"}}',
                    "The database change was denied.",
                ]

            def generate(self, history, tools):
                _ = history, tools
                return self.responses.pop(0)

        with tempfile.TemporaryDirectory() as temp:
            events: list[AgentEvent] = []
            answer = run_agent(
                "Delete users",
                FakeProvider(),
                [],
                Path(temp),
                "never_ask",
                one_shot=True,
                approve=lambda _tool, _arguments: self.fail(
                    "one-shot database writes must not prompt for approval"
                ),
                autopilot=True,
                enable_subagents=False,
                on_event=events.append,
            )

        self.assertIn("denied", answer)
        self.assertFalse(any(event.name == "permission_requested" for event in events))
        completed = next(event for event in events if event.name == "tool_completed")
        self.assertFalse(completed.data["result"]["success"])

    def test_agent_autopilot_executes_risky_tool_without_prompt(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.responses = [
                    '{"name":"write_file","arguments":{"filePath":"new.txt","content":"autopilot"}}',
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
                autopilot=True,
            )
            self.assertEqual((root / "new.txt").read_text(encoding="utf-8"), "autopilot")
            self.assertEqual(answer, "The file was written.")

    def test_agent_executes_project_remembered_tool_without_prompt(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.responses = [
                    '{"name":"write_file","arguments":{"filePath":"new.txt","content":"remembered"}}',
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
                remembered_permissions={"write_file"},
            )
            self.assertEqual((root / "new.txt").read_text(encoding="utf-8"), "remembered")
            self.assertEqual(answer, "The file was written.")

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


class NvidiaProviderTests(unittest.TestCase):
    def test_probe_sends_minimal_chat_request_with_short_timeout(self) -> None:
        response = io.BytesIO(
            b'{"choices":[{"message":{"content":"Panther is ready."}}]}'
        )
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-secret"}), patch(
            "panther.provider.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            provider = NvidiaProvider("nvidia/test-model")
            self.assertEqual(provider.probe(), "Panther is ready.")

        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["model"], "nvidia/test-model")
        self.assertNotIn("tools", payload)
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 20)

    def test_nvidia_candidate_order_prefers_chat_models_and_excludes_services(self) -> None:
        models = [
            "nvidia/nemotron-instruct",
            "nvidia/llama-instruct",
            "nvidia/qwen-embed",
            "nvidia/rerank",
            "nvidia/other-chat",
        ]

        self.assertEqual(
            _nvidia_chat_model_candidates(
                models,
                ["nvidia/llama-instruct", "nvidia/qwen-embed"],
            ),
            [
                "nvidia/llama-instruct",
                "nvidia/nemotron-instruct",
                "nvidia/other-chat",
            ],
        )

    def test_nvidia_check_switches_only_to_a_model_that_passed_inference(self) -> None:
        config = {
            "activeProvider": "nvidia",
            "activeBot": "",
            "bots": {},
            "remoteModel": "nvidia/unavailable",
            "codeModel": "nvidia/working",
            "visionModel": "nvidia/vision",
        }
        instances: list[SimpleNamespace] = []

        def provider_factory(
            model: str,
            api_key: str | None = None,
        ) -> SimpleNamespace:
            def probe() -> str:
                if model == "nvidia/unavailable":
                    raise RuntimeError(
                        "NVIDIA lists this model, but its serving function is not "
                        "available to this API account."
                    )
                return "ready"

            instance = SimpleNamespace(
                model=model,
                list_models=lambda: [
                    "nvidia/unavailable",
                    "nvidia/working",
                    "nvidia/vision",
                ],
                probe=probe,
            )
            instances.append(instance)
            return instance

        with patch("panther.cli._nvidia_api_key", return_value="test-secret"), patch(
            "panther.cli.NvidiaProvider",
            side_effect=provider_factory,
        ), patch(
            "panther.cli.save_config",
            return_value=Path("config.json"),
        ) as save_config, patch(
            "panther.cli.sys.stdin",
            SimpleNamespace(isatty=lambda: True),
        ), patch("builtins.input", return_value="1"):
            self.assertEqual(_nvidia_check(config, Path(".")), 0)

        self.assertEqual(config["remoteModel"], "nvidia/working")
        self.assertEqual(config["activeProvider"], "nvidia")
        self.assertEqual(save_config.call_args.args[0]["remoteModel"], "nvidia/working")
        self.assertEqual(
            [instance.model for instance in instances],
            [
                "nvidia/unavailable",
                "nvidia/unavailable",
                "nvidia/working",
                "nvidia/vision",
            ],
        )

    def test_project_env_reads_nvidia_key_without_exporting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env").write_text(
                "NVIDIA_API_KEY=project-secret\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(_nvidia_api_key(root), "project-secret")
                self.assertNotIn("NVIDIA_API_KEY", os.environ)
            with patch.dict(os.environ, {"NVIDIA_API_KEY": "shell-secret"}):
                self.assertEqual(_nvidia_api_key(root), "shell-secret")

    def test_project_env_key_is_passed_to_nvidia_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env").write_text(
                "NVIDIA_API_KEY=project-secret\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True), patch(
                "panther.cli.NvidiaProvider"
            ) as provider_class:
                _nvidia_provider("nvidia/test-model", root)
        provider_class.assert_called_once_with(
            "nvidia/test-model",
            api_key="project-secret",
        )

    def test_generate_uses_environment_key_without_keyring(self) -> None:
        response = io.BytesIO(
            b'{"choices":[{"message":{"content":"Hello"}}]}'
        )
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "env-secret"}), patch(
            "panther.provider.import_module",
            side_effect=AssertionError("keyring should not be accessed"),
        ), patch(
            "panther.provider.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            provider = NvidiaProvider("nvidia/test-model")
            answer = provider.generate([{"role": "user", "content": "Hi"}], [])
        self.assertEqual(answer, "Hello")
        self.assertEqual(
            urlopen.call_args.args[0].get_header("Authorization"),
            "Bearer env-secret",
        )

    def test_generate_uses_keyring_credential_and_nvidia_api(self) -> None:
        keyring = SimpleNamespace(get_password=Mock(return_value="test-secret"))
        response = io.BytesIO(
            b'{"choices":[{"message":{"content":"Hello"}}]}'
        )
        with patch.dict(sys.modules, {"keyring": keyring}), patch(
            "panther.provider.urllib.request.urlopen", return_value=response
        ) as urlopen:
            provider = NvidiaProvider("nvidia/test-model")
            answer = provider.generate([{"role": "user", "content": "Hi"}], [])
        self.assertEqual(answer, "Hello")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "nvidia/test-model")
        self.assertEqual(payload["messages"][-1]["content"], "Hi")

    def test_account_specific_404_explains_model_access_issue(self) -> None:
        error = urllib.error.HTTPError(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            404,
            "Not Found",
            Message(),
            io.BytesIO(b'{"detail":"Function not found for account"}'),
        )
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-secret"}), patch(
            "panther.provider.urllib.request.urlopen",
            side_effect=error,
        ):
            provider = NvidiaProvider("nvidia/test-model")
            with self.assertRaisesRegex(
                RuntimeError,
                "not available to this API account",
            ) as context:
                provider.generate([{"role": "user", "content": "Hi"}], [])
        self.assertIn("panther nvidia check", str(context.exception))


class AdditionalProviderTests(unittest.TestCase):
    def test_openai_compatible_provider_uses_custom_base_url_and_tool_calls(self) -> None:
        response = io.BytesIO(
            b'{"choices":[{"message":{"tool_calls":[{"function":{"name":"read_file",'
            b'"arguments":"{\\"filePath\\":\\"README.md\\"}"}}]}}]}'
        )
        with patch(
            "panther.provider.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            provider = OpenAICompatibleProvider(
                "example-model",
                "https://api.example.test/v1/",
                api_key="custom-secret",
            )
            answer = provider.generate([{"role": "user", "content": "Read README"}], [])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.example.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer custom-secret")
        self.assertEqual(
            json.loads(answer),
            {"name": "read_file", "arguments": {"filePath": "README.md"}},
        )

    def test_openai_compatible_provider_rejects_unsafe_base_urls(self) -> None:
        for base_url in (
            "file:///etc/passwd",
            "https://user:password@example.test/v1",
            "https://example.test/v1?key=secret",
        ):
            with self.subTest(base_url=base_url), self.assertRaises(ValueError):
                OpenAICompatibleProvider("model", base_url, api_key="secret")

    def test_gemini_provider_sends_native_function_declarations(self) -> None:
        response = io.BytesIO(
            b'{"candidates":[{"content":{"parts":[{"functionCall":{"name":"read_file",'
            b'"args":{"filePath":"README.md"}}}]}}]}'
        )
        tools = [{
            "name": "read_file",
            "description": "Read a project file",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string"}},
                "required": ["filePath"],
            },
        }]
        with patch(
            "panther.provider.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            provider = GeminiProvider("gemini-test", "gemini-secret")
            answer = provider.generate(
                [
                    {"role": "system", "content": "You are Panther."},
                    {"role": "user", "content": "Read README"},
                ],
                tools,
            )
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("X-goog-api-key"), "gemini-secret")
        self.assertIn("/models/gemini-test:generateContent", request.full_url)
        payload = json.loads(request.data)
        self.assertEqual(payload["tools"][0]["functionDeclarations"][0]["name"], "read_file")
        self.assertEqual(
            json.loads(answer),
            {"name": "read_file", "arguments": {"filePath": "README.md"}},
        )


class CLITests(unittest.TestCase):
    def test_coding_loading_indicator_animates_during_model_wait(self) -> None:
        dashboard = _AgentDashboard()
        output = io.StringIO()
        with patch("panther.cli.sys.stdout", output):
            dashboard.enabled = True
            dashboard.transient = True
            dashboard.update(
                AgentEvent(
                    "coding_plan_generating",
                    AgentPhase.INSPECTING,
                    0,
                    {},
                )
            )
            time.sleep(0.65)
            dashboard.update(
                AgentEvent(
                    "coding_plan_started",
                    AgentPhase.INSPECTING,
                    0,
                    {
                        "workers": [
                            {
                                "role": "mochi",
                                "name": "Mochi",
                                "specialty": "implementation",
                                "model": "test-model",
                            }
                        ]
                    },
                )
            )
            dashboard.finish()

        rendered = output.getvalue()
        self.assertIn("| Dividing the task", rendered)
        self.assertIn("/ Dividing the task", rendered)
        self.assertIn("\\ Dividing the task", rendered)
        self.assertIsNone(dashboard._spinner_thread)

    def test_transient_coding_huddle_clears_named_worker_activity_on_finish(self) -> None:
        dashboard = _AgentDashboard()
        output = io.StringIO()
        with patch("panther.cli.sys.stdout", output):
            dashboard.enabled = True
            dashboard.transient = True
            dashboard.update(
                AgentEvent(
                    "coding_plan_started",
                    AgentPhase.INSPECTING,
                    0,
                    {
                        "workers": [
                            {
                                "role": role,
                                "name": name,
                                "specialty": "implementation",
                                "model": "test-model",
                            }
                            for role, (name, _) in CODING_WORKERS.items()
                        ]
                    },
                )
            )
            dashboard.update(
                AgentEvent(
                    "subagent_started",
                    AgentPhase.INSPECTING,
                    0,
                    {
                        "role": "mochi",
                        "name": "Mochi",
                        "title": "Inspect game",
                        "model": "test-model",
                    },
                )
            )
            dashboard.update(
                AgentEvent(
                    "model_started",
                    AgentPhase.INSPECTING,
                    1,
                    {"subagent": "mochi"},
                )
            )
            dashboard.update(
                AgentEvent(
                    "tool_requested",
                    AgentPhase.INSPECTING,
                    1,
                    {
                        "subagent": "mochi",
                        "tool": "read_file",
                        "arguments": {"filePath": "john doe/tic_tac_toe.py"},
                    },
                )
            )
            dashboard.finish()

        rendered = output.getvalue()
        for name in ("Mochi", "Boba", "Pip"):
            self.assertIn(name, rendered)
        self.assertIn("Reading john doe/tic_tac_toe.py", rendered)
        self.assertIn("\033[2K", rendered)
        self.assertGreaterEqual(rendered.count("\033["), 2)
        self.assertFalse(dashboard._lines)
        self.assertEqual(dashboard._drawn_lines, 0)

    def test_dashboard_shows_active_subagent_model_work(self) -> None:
        dashboard = _AgentDashboard()
        output = io.StringIO()
        with patch("panther.cli.sys.stdout", output):
            dashboard.enabled = True
            dashboard.update(
                AgentEvent(
                    "coding_plan_started",
                    AgentPhase.INSPECTING,
                    0,
                    {
                        "workers": [
                            {
                                "role": "mochi",
                                "name": "Mochi",
                                "specialty": "implementation",
                                "model": "moonshotai/kimi-k2.5",
                            }
                        ]
                    },
                )
            )
            dashboard.update(
                AgentEvent(
                    "subagent_started",
                    AgentPhase.INSPECTING,
                    0,
                    {
                        "role": "mochi",
                        "name": "Mochi",
                        "title": "Build",
                        "model": "moonshotai/kimi-k2.5",
                    },
                )
            )
            dashboard.update(
                AgentEvent(
                    "model_started",
                    AgentPhase.INSPECTING,
                    1,
                    {"subagent": "mochi"},
                )
            )
            dashboard.update(
                AgentEvent(
                    "tool_requested",
                    AgentPhase.INSPECTING,
                    1,
                    {
                        "subagent": "mochi",
                        "tool": "read_file",
                        "arguments": {"filePath": "README.md"},
                    },
                )
            )
            dashboard.update(
                AgentEvent(
                    "tool_completed",
                    AgentPhase.INSPECTING,
                    1,
                    {
                        "subagent": "mochi",
                        "tool": "read_file",
                        "result": {"success": True, "data": "README contents"},
                    },
                )
            )
            dashboard.update(
                AgentEvent(
                    "subagent_completed",
                    AgentPhase.INSPECTING,
                    0,
                    {
                        "role": "mochi",
                        "success": True,
                        "changed_files": ["calculator.py"],
                        "summary": "Added the calculator module.",
                    },
                )
            )
            dashboard.update(
                AgentEvent(
                    "team_chat_started",
                    AgentPhase.INSPECTING,
                    0,
                    {"workers": ["mochi", "boba", "pip"]},
                )
            )
            dashboard.update(
                AgentEvent(
                    "team_message",
                    AgentPhase.INSPECTING,
                    0,
                    {"role": "boba", "name": "Boba", "message": "Nice work, Mochi!"},
                )
            )
            dashboard.update(
                AgentEvent(
                    "brainstorm_started",
                    AgentPhase.UNDERSTANDING,
                    0,
                    {"workers": ["mochi", "boba", "pip"]},
                )
            )
            dashboard.update(
                AgentEvent(
                    "brainstorm_agent_started",
                    AgentPhase.UNDERSTANDING,
                    0,
                    {"role": "pip", "name": "Pip", "model": "test-model"},
                )
            )
            dashboard.update(
                AgentEvent(
                    "brainstorm_message",
                    AgentPhase.UNDERSTANDING,
                    0,
                    {
                        "role": "pip",
                        "name": "Pip",
                        "model": "test-model",
                        "message": "Build a tiny local task garden.",
                    },
                )
            )
        self.assertEqual(dashboard._workers["mochi"]["mark"], "✓")
        self.assertEqual(dashboard._workers["mochi"]["status"], "complete")
        rendered = output.getvalue()
        self.assertIn("Mochi", rendered)
        self.assertIn("Build", rendered)
        self.assertIn("Reading README.md", rendered)
        self.assertIn("Thinking through the next step", rendered)
        self.assertIn("read file finished", rendered)
        self.assertIn("calculator.py", rendered)
        self.assertIn("✓", rendered)
        self.assertIn("Post-build huddle", rendered)
        self.assertIn("Boba:", rendered)
        self.assertIn("Nice work, Mochi!", rendered)
        self.assertIn("Brainstorm huddle", rendered)
        self.assertIn("no files will be changed", rendered)
        self.assertIn("Pip: Build a tiny local task garden.", rendered)
        self.assertIn("Thinking up a build idea", rendered)
        self.assertNotIn("MODEL:", rendered)
        self.assertNotIn("TOOLS:", rendered)
        self.assertNotIn("TESTS:", rendered)
        self.assertNotIn("PHASE:", rendered)
        self.assertNotIn("Panther |", rendered)
        self.assertNotIn("\033[1A", rendered)
        self.assertIn("\n", rendered)

    def test_dashboard_shows_named_read_only_review_results(self) -> None:
        dashboard = _AgentDashboard()
        output = io.StringIO()
        with patch("panther.cli.sys.stdout", output):
            dashboard.enabled = True
            dashboard.update(
                AgentEvent(
                    "review_started",
                    AgentPhase.INSPECTING,
                    0,
                    {
                        "workers": [
                            {
                                "role": role,
                                "name": name,
                                "model": "test-model",
                            }
                            for role, (name, _) in CODING_WORKERS.items()
                        ]
                    },
                )
            )
            dashboard.update(
                AgentEvent(
                    "review_message",
                    AgentPhase.COMPLETE,
                    0,
                    {
                        "role": "boba",
                        "name": "Boba",
                        "model": "test-model",
                        "message": "Verified: the reset button clears the board.",
                    },
                )
            )

        rendered = output.getvalue()
        self.assertIn("Read-only review", rendered)
        self.assertIn("Mochi", rendered)
        self.assertIn("Boba", rendered)
        self.assertIn("Pip", rendered)
        self.assertIn("Verified: the reset button clears the board.", rendered)

    def test_activity_updates_do_not_print_spinner_noise_for_chat(self) -> None:
        from panther.cli import _run_with_provider

        class FakeProvider:
            def generate(self, history, tools):
                _ = history, tools
                return "Hi! How can I help?"

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = {"id": "test", "history": []}
            config = {
                "rememberedPermissions": [],
                "permissionMode": "ask_when_risky",
                "autopilot": False,
            }
            output = io.StringIO()
            with patch("panther.cli.sys.stdout", output), patch(
                "panther.cli.sys.stderr", io.StringIO()
            ), patch("panther.cli._manager") as manager:
                _run_with_provider(
                    "hi",
                    root,
                    session,
                    config,
                    FakeProvider(),
                    json_output=False,
                    one_shot=False,
                )

        rendered = output.getvalue()
        self.assertIn("[YOU]", rendered)
        self.assertIn("[PANTHER]", rendered)
        self.assertNotIn("MODEL:", rendered)
        self.assertNotIn("thinking...", rendered)

    def test_coding_worker_actions_are_printed_without_tty(self) -> None:
        from panther.cli import _run_with_provider

        class FakeProvider:
            model = "test-model"

        def fake_run_agent(*args, **kwargs):
            on_event = kwargs["on_event"]
            self.assertIsNotNone(on_event)
            on_event(
                AgentEvent(
                    "coding_plan_started",
                    AgentPhase.INSPECTING,
                    0,
                    {
                        "workers": [
                            {
                                "role": "mochi",
                                "name": "Mochi",
                                "specialty": "implementation",
                                "model": "test-model",
                            }
                        ]
                    },
                )
            )
            on_event(
                AgentEvent(
                    "subagent_started",
                    AgentPhase.INSPECTING,
                    0,
                    {
                        "role": "mochi",
                        "name": "Mochi",
                        "title": "Build game",
                        "model": "test-model",
                    },
                )
            )
            on_event(
                AgentEvent(
                    "tool_requested",
                    AgentPhase.INSPECTING,
                    1,
                    {
                        "subagent": "mochi",
                        "tool": "read_file",
                        "arguments": {"filePath": "john doe/tic_tac_toe.py"},
                    },
                )
            )
            on_event(
                AgentEvent(
                    "tool_completed",
                    AgentPhase.INSPECTING,
                    1,
                    {
                        "subagent": "mochi",
                        "tool": "read_file",
                        "result": {"success": True},
                    },
                )
            )
            return "Done."

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = {"id": "test", "history": []}
            config = {
                "rememberedPermissions": [],
                "permissionMode": "ask_when_risky",
                "autopilot": False,
            }
            output = io.StringIO()
            with patch("panther.cli.sys.stdout", output), patch(
                "panther.cli.run_agent", side_effect=fake_run_agent
            ), patch("panther.cli._manager"):
                _run_with_provider(
                    "Improve the tic-tac-toe game",
                    root,
                    session,
                    config,
                    FakeProvider(),
                    json_output=False,
                    one_shot=False,
                    code_mode=True,
                )

        self.assertIn("Mochi:", output.getvalue())
        self.assertIn("Reading john doe/tic_tac_toe.py", output.getvalue())
        self.assertIn("read file finished", output.getvalue())

    def test_nvidia_coding_mode_can_discover_worker_models(self) -> None:
        from panther.cli import _run_with_provider

        provider = NvidiaProvider("nvidia/test-model", api_key="test-key")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = {"id": "test", "history": []}
            config = {
                "rememberedPermissions": [],
                "permissionMode": "ask_when_risky",
                "autopilot": False,
            }
            with patch.object(
                provider,
                "list_models",
                return_value=[
                    "moonshotai/kimi-k2.5",
                    "deepseek-ai/deepseek-v3",
                    "nvidia/nemotron-3.5",
                ],
            ), patch("panther.cli.run_agent", return_value="Implementation complete.") as run, patch(
                "panther.cli._manager"
            ):
                _run_with_provider(
                    "Create a calculator module",
                    root,
                    session,
                    config,
                    provider,
                    json_output=True,
                    one_shot=True,
                    code_mode=True,
                )

        self.assertEqual(
            run.call_args.kwargs["worker_models"],
            {
                "mochi": "moonshotai/kimi-k2.5",
                "boba": "deepseek-ai/deepseek-v3",
                "pip": "nvidia/nemotron-3.5",
            },
        )
        self.assertEqual(len(run.call_args.kwargs["required_tool_groups"]), 3)

    def test_build_idea_request_passes_agent_suggestions_to_chat_without_code_mode(self) -> None:
        from panther.cli import _run_with_provider

        suggestions = [
            TeamChatMessage(
                "mochi",
                "Mochi",
                "test-model",
                "Build a local project idea board.",
            ),
            TeamChatMessage(
                "boba",
                "Boba",
                "test-model",
                "Build a cheerful command-line habit garden.",
            ),
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = {"id": "test", "history": []}
            config = {
                "rememberedPermissions": [],
                "permissionMode": "ask_when_risky",
                "autopilot": False,
            }
            with patch(
                "panther.cli.brainstorm_with_workers",
                return_value=(suggestions, []),
            ) as brainstorm, patch(
                "panther.cli.run_agent",
                return_value="The team has two ideas.",
            ) as run, patch("panther.cli._manager"):
                _run_with_provider(
                    "i wnat to ask what the 3 agents suggest to build",
                    root,
                    session,
                    config,
                    SimpleNamespace(model="test-model"),
                    json_output=True,
                    one_shot=True,
                )

        brainstorm.assert_called_once()
        self.assertIsNone(brainstorm.call_args.kwargs["on_event"])
        self.assertFalse(run.call_args.kwargs["code_mode"])
        self.assertEqual(run.call_args.kwargs["allowed_tool_names"], set())
        final_task = run.call_args.args[0]
        self.assertIn("Mochi (test-model): Build a local project idea board.", final_task)
        self.assertIn("Boba (test-model): Build a cheerful command-line habit garden.", final_task)

    def test_analysis_request_runs_read_only_agent_reviews(self) -> None:
        from panther.cli import _run_with_provider

        self.assertTrue(
            _is_agent_review_task("hi anlyse tic tac toe in john doe")
        )
        findings = [
            TeamChatMessage(role, name, "test-model", f"{name} reviewed the game.")
            for role, name in (
                ("mochi", "Mochi"),
                ("boba", "Boba"),
                ("pip", "Pip"),
            )
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = {"id": "test", "history": []}
            config = {
                "rememberedPermissions": [],
                "permissionMode": "ask_when_risky",
                "autopilot": False,
            }
            with patch(
                "panther.cli.review_with_workers",
                return_value=(findings, []),
            ) as review, patch(
                "panther.cli.run_agent",
                return_value="The team completed a read-only review.",
            ) as run, patch("panther.cli._manager"):
                _run_with_provider(
                    "hi anlyse tic tac toe in john doe",
                    root,
                    session,
                    config,
                    SimpleNamespace(model="test-model"),
                    json_output=True,
                    one_shot=True,
                )

        review.assert_called_once()
        self.assertFalse(run.call_args.kwargs["code_mode"])
        self.assertEqual(run.call_args.kwargs["allowed_tool_names"], set())
        self.assertIn("Mochi (test-model): Mochi reviewed the game.", run.call_args.args[0])

    def test_coding_mode_detection_does_not_match_test_substrings(self) -> None:
        self.assertFalse(_is_coding_task("search google for testa current value"))
        self.assertFalse(_is_coding_task("search google for Tesla current value"))
        prompt = "i wnat to ask what the 3 agents suggest to build"
        self.assertTrue(_is_agent_brainstorm_task(prompt))
        self.assertFalse(_is_coding_task(prompt))
        self.assertTrue(_is_coding_task("run tests for the parser"))
        self.assertTrue(_is_coding_task("fix this Python bug"))

    def test_inline_prompt_leaves_mouse_wheel_available_for_terminal_scrollback(self) -> None:
        prompt_session = Mock()
        with patch(
            "panther.cli.sys.stdin",
            SimpleNamespace(isatty=lambda: True),
        ), patch(
            "panther.cli.PromptSession",
            return_value=prompt_session,
        ) as prompt_factory:
            _create_prompt_session({}, Path("."))

        self.assertFalse(prompt_factory.call_args.kwargs["mouse_support"])

    def test_transcript_cursor_position_uses_formatted_text_fragments(self) -> None:
        position = _transcript_cursor_position([
            {"role": "user", "content": "first line\nsecond line"},
            {"role": "assistant", "content": "response"},
        ])
        self.assertEqual(position.x, 0)
        self.assertEqual(position.y, 6)

    def test_slash_commands_complete_after_existing_query_text(self) -> None:
        completions = list(
            _SlashCommandCompleter().get_completions(
                Document("explain this first /sta"),
                None,
            )
        )
        self.assertEqual([item.text for item in completions], ["/status"])
        self.assertEqual(completions[0].start_position, -4)

    def run_cli(
        self,
        *args: str,
        env: dict[str, str] | None = None,
        cwd: Path = REPO_ROOT,
        module: str = "panther.cli",
    ) -> subprocess.CompletedProcess[str]:
        command_env = os.environ.copy()
        project_path = str(REPO_ROOT)
        existing_pythonpath = command_env.get("PYTHONPATH")
        command_env["PYTHONPATH"] = (
            os.pathsep.join((project_path, existing_pythonpath))
            if existing_pythonpath
            else project_path
        )
        if env:
            command_env.update(env)
        return subprocess.run(
            [sys.executable, "-m", module, *args],
            cwd=cwd,
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
        module_help = self.run_cli("--help", module="panther")
        self.assertEqual(module_help.returncode, 0, module_help.stderr)
        self.assertIn("run", module_help.stdout)

    def test_config_command_creates_default_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_cli(
                "config",
                env={"PANTHER_HOME": temp},
                cwd=Path(temp),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["activeProvider"], "local")
            self.assertTrue((Path(temp) / "config.json").exists())

    def test_analyze_command(self) -> None:
        result = self.run_cli("analyze", str(REPO_ROOT))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Repository analysis", result.stdout)
        self.assertIn("README.md", result.stdout)

    def test_plain_image_question_attaches_matching_project_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_dir = root / "panther"
            image_dir.mkdir()
            (image_dir / "1.PNG").write_bytes(b"image data")

            image_content = _image_content_for_task(
                root,
                "analyse image 1.png and say what it shows",
            )

        self.assertIsNotNone(image_content)
        self.assertEqual(
            image_content["image_url"]["url"],
            "data:image/png;base64,aW1hZ2UgZGF0YQ==",
        )

    def test_plain_image_question_runs_with_attached_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_dir = root / "panther"
            image_dir.mkdir()
            (image_dir / "1.PNG").write_bytes(b"image data")
            config = {
                "activeProvider": "nvidia",
                "permissionMode": "ask_when_risky",
            }
            args = Namespace(
                task="analyse image 1.png and say what it shows",
                session=None,
                json=False,
                code=False,
                image=None,
            )
            with patch("panther.cli.load_config", return_value=config), patch(
                "panther.cli._get_session", return_value={"id": "test-session"}
            ), patch("panther.cli._execute_task", return_value="A chart") as execute:
                self.assertEqual(_run_command(args, root), 0)

        image_content = execute.call_args.kwargs["image_content"]
        self.assertIsNotNone(image_content)
        self.assertEqual(
            image_content["image_url"]["url"],
            "data:image/png;base64,aW1hZ2UgZGF0YQ==",
        )

    def test_repl_continues_after_referenced_image_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            error_output = io.StringIO()
            with patch(
                "builtins.input",
                side_effect=["analyse i.png", "/help", "exit"],
            ) as prompt, patch(
                "sys.stderr",
                error_output,
            ), patch(
                "panther.cli.load_config",
                return_value={
                    "activeProvider": "nvidia",
                    "remoteModel": "nvidia/test",
                    "codeModel": "nvidia/test-code",
                    "visionModel": "nvidia/test-vision",
                    "permissionMode": "ask_when_risky",
                },
            ), patch("panther.cli._ensure_repository_index"), patch(
                "panther.cli._print_banner"
            ), patch("panther.cli._print_slash_commands") as show_commands, patch(
                "panther.cli._manager"
            ), patch(
                "panther.cli._create_prompt_session",
                return_value=None,
            ):
                result = _repl(root, show_banner=False)

        self.assertEqual(result, 0)
        self.assertEqual(prompt.call_count, 3)
        self.assertEqual(show_commands.call_count, 2)
        self.assertIn("Image file not found in the project: i.png", error_output.getvalue())

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
                cwd=Path(temp),
            )
        self.assertEqual(result.returncode, 1)
        records = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(records[0]["type"], "start")
        self.assertEqual(records[-1]["type"], "error")
        self.assertIn("llama-cpp-python", records[-1]["message"])
        self.assertNotIn("Checking for local model", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
