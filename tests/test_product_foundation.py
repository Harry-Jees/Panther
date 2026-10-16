from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from panther.cli import (
    USER_PROFILE_KEYRING_USERNAME,
    _NvidiaProfileProvider,
    _first_run_onboarding,
    _needs_first_run_onboarding,
    _print_banner,
)
from panther.config import save_config
from panther.tools import git_pull, git_push


class ResponsiveBannerTests(unittest.TestCase):
    @patch("panther.cli._supports_truecolor", return_value=False)
    @patch("panther.cli.shutil.get_terminal_size", return_value=SimpleNamespace(columns=12))
    def test_banner_stays_visible_and_fits_a_narrow_terminal(
        self,
        _terminal_size: Mock,
        _truecolor: Mock,
    ) -> None:
        output = io.StringIO()
        with patch("panther.cli.sys.stdout", output):
            _print_banner()

        lines = output.getvalue().splitlines()
        self.assertIn("PANTHER", lines)
        self.assertTrue(all(len(line) <= 12 for line in lines))


class GitRemoteToolTests(unittest.TestCase):
    @patch("panther.tools._git")
    def test_push_and_pull_use_configured_upstream_and_long_timeout(
        self,
        git: Mock,
    ) -> None:
        git.return_value = {"success": True, "data": "done"}

        self.assertTrue(git_push({}, Path("."))["success"])
        self.assertEqual(git.call_args.args[0], ["push"])
        self.assertEqual(git.call_args.kwargs["timeout"], 300)

        self.assertTrue(git_pull({}, Path("."))["success"])
        self.assertEqual(git.call_args.args[0], ["pull"])
        self.assertEqual(git.call_args.kwargs["timeout"], 300)

    @patch("panther.tools._git")
    def test_push_and_pull_reject_arguments_instead_of_forwarding_git_flags(
        self,
        git: Mock,
    ) -> None:
        self.assertFalse(git_push({"force": True}, Path("."))["success"])
        self.assertFalse(git_pull({"remote": "--upload-pack=cmd"}, Path("."))["success"])
        git.assert_not_called()


class DesktopOnboardingTests(unittest.TestCase):
    @patch("panther.cli.NvidiaProvider")
    @patch("panther.cli.getpass.getpass", return_value="nvidia-secret")
    @patch("panther.cli._nvidia_api_key", return_value=None)
    @patch("panther.cli.import_module")
    def test_onboarding_saves_secret_and_profile_only_to_keyring(
        self,
        import_module: Mock,
        _environment_key: Mock,
        _secret_prompt: Mock,
        nvidia_provider: Mock,
    ) -> None:
        keyring = SimpleNamespace(
            get_password=Mock(return_value=None),
            set_password=Mock(),
        )
        import_module.return_value = keyring
        nvidia_provider.return_value.list_models.return_value = [
            "nvidia/model-a",
            "nvidia/model-b",
        ]
        nvidia_provider.return_value.probe.return_value = "ready"
        answers = iter(
            ["2", "Ada", "Build software", "Prefers concise answers", "n"]
        )
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "config.json"
            with patch.dict(
                "os.environ",
                {"PANTHER_CONFIG_PATH": str(config_path)},
            ), patch("builtins.input", side_effect=lambda _prompt="": next(answers)):
                config = {
                    "remoteModel": "nvidia/model-a",
                    "activeProvider": "local",
                    "activeBot": "",
                    "onboardingComplete": False,
                    "shareProfileWithNvidia": False,
                }
                _first_run_onboarding(config, Path(temp))
            stored_config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertNotIn("nvidia-secret", json.dumps(stored_config))
        self.assertNotIn("Ada", json.dumps(stored_config))
        self.assertEqual(config["remoteModel"], "nvidia/model-b")
        nvidia_provider.assert_called_with(
            "nvidia/model-b",
            api_key="nvidia-secret",
        )
        nvidia_provider.return_value.probe.assert_called_once()
        self.assertEqual(config["activeProvider"], "nvidia")
        self.assertTrue(config["onboardingComplete"])
        self.assertTrue(stored_config["onboardingComplete"])
        self.assertFalse(config["shareProfileWithNvidia"])
        keyring.set_password.assert_any_call(
            "panther-cli",
            "nvidia",
            "nvidia-secret",
        )
        profile_call = next(
            call for call in keyring.set_password.call_args_list
            if call.args[1] == USER_PROFILE_KEYRING_USERNAME
        )
        self.assertEqual(
            json.loads(profile_call.args[2]),
            {
                "userName": "Ada",
                "mainPurpose": "Build software",
                "personalFact": "Prefers concise answers",
            },
        )

    def test_legacy_install_is_not_mistaken_for_a_first_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text('{"activeProvider":"local"}', encoding="utf-8")
            with patch.dict("os.environ", {"PANTHER_CONFIG_PATH": str(path)}):
                self.assertFalse(_needs_first_run_onboarding({"onboardingComplete": False}))
                path.unlink()
                self.assertTrue(_needs_first_run_onboarding({"onboardingComplete": False}))

    def test_newly_created_config_keeps_first_run_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            with patch.dict("os.environ", {"PANTHER_CONFIG_PATH": str(path)}):
                save_config({"onboardingComplete": False})
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertFalse(saved["onboardingComplete"])
                self.assertTrue(_needs_first_run_onboarding(saved))

    def test_opted_in_profile_is_sent_without_mutating_session_history(self) -> None:
        provider = SimpleNamespace(
            model="nvidia/test",
            generate=Mock(return_value="ok"),
        )
        history = [{"role": "system", "content": "System"}, {"role": "user", "content": "Hi"}]
        wrapper = _NvidiaProfileProvider(
            provider,
            {"userName": "Ada", "mainPurpose": "Build software", "personalFact": ""},
        )

        self.assertEqual(wrapper.generate(history, []), "ok")

        sent_history = provider.generate.call_args.args[0]
        self.assertIn("Ada", sent_history[0]["content"])
        self.assertEqual(history[0]["content"], "System")
        self.assertEqual(history[1]["content"], "Hi")


if __name__ == "__main__":
    unittest.main()
