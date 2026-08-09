import json
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from panther import tools


class ToolTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "app.py").write_text(
            "def hello():\n    return 42\n\n# TODO: greet\nprint(hello())\n", encoding="utf-8"
        )
        (self.root / "README.md").write_text("# Panther\n\nUseful notes.\n", encoding="utf-8")
        (self.root / "src").mkdir()
        (self.root / "src" / "mod.py").write_text("class Thing:\n    pass\n", encoding="utf-8")
        # Ignored + binary-ish
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "x.js").write_text("ignored", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()


class TestReadFile(ToolTestBase):
    def test_reads_text_file(self):
        out = tools.read_file(self.root, {"path": "app.py"})
        self.assertIn("def hello", out)
        self.assertIn("return 42", out)

    def test_rejects_escape(self):
        with self.assertRaises(tools.ToolError):
            tools.read_file(self.root, {"path": "../../etc/passwd"})

    def test_missing_file(self):
        with self.assertRaises(tools.ToolError):
            tools.read_file(self.root, {"path": "nope.py"})


class TestListDir(ToolTestBase):
    def test_lists_root(self):
        out = tools.list_dir(self.root, {})
        self.assertIn("app.py", out)
        self.assertIn("README.md", out)
        self.assertIn("src/", out)
        self.assertNotIn("node_modules", out)  # ignored dir

    def test_lists_subdir(self):
        out = tools.list_dir(self.root, {"path": "src"})
        self.assertIn("mod.py", out)


class TestGrep(ToolTestBase):
    def test_finds_matches(self):
        out = tools.grep(self.root, {"pattern": "hello"})
        self.assertIn("app.py:1", out)

    def test_no_matches(self):
        out = tools.grep(self.root, {"pattern": "zzzz_nope"})
        self.assertIn("No matches", out)

    def test_case_insensitive(self):
        out = tools.grep(self.root, {"pattern": "TODO"})
        self.assertIn("app.py", out)

    def test_invalid_regex(self):
        with self.assertRaises(tools.ToolError):
            tools.grep(self.root, {"pattern": "("})


class TestRunTests(ToolTestBase):
    def test_runs_unittest_suite(self):
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_x.py").write_text(
            "import unittest\nclass T(unittest.TestCase):\n"
            "    def test_ok(self):\n        self.assertEqual(1, 1)\n",
            encoding="utf-8",
        )
        out = tools.run_tests(self.root, {})
        self.assertIn("PASS", out)

    def test_rejects_non_test_commands(self):
        with self.assertRaises(tools.ToolError):
            tools.run_tests(self.root, {"command": "rm -rf /"})

    def test_rejects_shell_metacharacter_injection(self):
        # `startswith` guard alone would let this through; shell=False must not.
        with self.assertRaises(tools.ToolError):
            tools.run_tests(self.root, {"command": "python -m unittest; rm -rf /"})
        with self.assertRaises(tools.ToolError):
            tools.run_tests(self.root, {"command": "python -m unittest && echo pwned"})
        with self.assertRaises(tools.ToolError):
            tools.run_tests(self.root, {"command": "pytest $(rm -rf /)"})

    def test_quoted_pipe_in_filter_is_allowed(self):
        # `|` inside quotes is a legit pytest -k filter, not an injection.
        self.assertFalse(
            tools._has_unquoted_shell_meta('pytest -k "test_a|test_b"')
        )

    def test_invocation_whitelist(self):
        ok = [
            "python -m unittest",
            "python -m pytest -q",
            "py -3.11 -m unittest -v",
            "pytest tests/",
        ]
        bad = [
            "notpython -m unittest",
            "python script.py",
            "python -m pip install x",
            "python foo.py -m unittest",  # script disguised as a test run
        ]
        for cmd in ok:
            self.assertTrue(
                tools._is_test_invocation(shlex.split(cmd)), f"should allow: {cmd}"
            )
        for cmd in bad:
            self.assertFalse(
                tools._is_test_invocation(shlex.split(cmd)), f"should reject: {cmd}"
            )


class TestIgnoreDirs(unittest.TestCase):
    def test_ignored_dirs_and_egg_info(self):
        self.assertTrue(tools.is_ignored(["panther", ".git", "x"]))
        self.assertTrue(tools.is_ignored(["panther_cli.egg-info"]))
        self.assertTrue(tools.is_ignored(["node_modules"]))
        self.assertFalse(tools.is_ignored(["panther", "api.py"]))


class TestSchemaAndExecute(ToolTestBase):
    def test_schemas_are_well_formed(self):
        names = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
        self.assertEqual(names, {"read_file", "list_dir", "grep", "run_tests"})
        for schema in tools.TOOL_SCHEMAS:
            self.assertEqual(schema["type"], "function")
            self.assertIn("parameters", schema["function"])

    def test_execute_tool_dispatch(self):
        out = tools.execute_tool("grep", {"pattern": "Thing"}, self.root)
        self.assertIn("src/mod.py", out)

    def test_grep_strips_vim_style_slashes(self):
        # Models often wrap patterns in /…/ like vim; must still match.
        out = tools.grep(self.root, {"pattern": "/Thing/"})
        self.assertIn("src/mod.py", out)

    def test_grep_can_target_a_single_file(self):
        out = tools.grep(self.root, {"pattern": "Thing", "path": "src/mod.py"})
        self.assertIn("src/mod.py:1", out)

    def test_grep_no_matches_returns_message(self):
        out = tools.grep(self.root, {"pattern": "NoSuchThingEver"})
        self.assertIn("No matches", out)

    def test_execute_unknown_tool(self):
        with self.assertRaises(tools.ToolError):
            tools.execute_tool("bogus", {}, self.root)

    def test_parse_tool_arguments(self):
        self.assertEqual(tools.parse_tool_arguments('{"path": "a.py"}'), {"path": "a.py"})
        self.assertEqual(tools.parse_tool_arguments("not json"), {})


if __name__ == "__main__":
    unittest.main()
