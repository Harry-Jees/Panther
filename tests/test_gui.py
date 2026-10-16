from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from panther.gui import _snapshot_workspace, _workspace_changes


class WorkspaceChangeTests(unittest.TestCase):
    def test_workspace_changes_report_added_modified_and_deleted_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            modified = root / "src" / "app.py"
            modified.parent.mkdir()
            modified.write_text("keep\nremove this\n", encoding="utf-8")
            deleted = root / "old.txt"
            deleted.write_text("old line\n", encoding="utf-8")
            before, before_errors = _snapshot_workspace(root)

            modified.write_text("keep\nadd this\n", encoding="utf-8")
            deleted.unlink()
            added = root / "new.txt"
            added.write_text("new line\n", encoding="utf-8")
            after, after_errors = _snapshot_workspace(root)

        changes = {change.path: change for change in _workspace_changes(before, after)}
        self.assertEqual(before_errors, [])
        self.assertEqual(after_errors, [])
        self.assertEqual(set(changes), {"new.txt", "old.txt", "src/app.py"})
        self.assertEqual(changes["new.txt"].status, "A")
        self.assertEqual(changes["new.txt"].added, 1)
        self.assertEqual(changes["old.txt"].status, "D")
        self.assertEqual(changes["old.txt"].removed, 1)
        self.assertEqual(changes["src/app.py"].status, "M")
        self.assertEqual(changes["src/app.py"].added, 1)
        self.assertEqual(changes["src/app.py"].removed, 1)
        self.assertIn("-remove this", changes["src/app.py"].diff)
        self.assertIn("+add this", changes["src/app.py"].diff)

    def test_sensitive_file_change_is_reported_without_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            secret_file = root / ".env"
            secret_file.write_text("API_KEY=do-not-display\n", encoding="utf-8")
            before, _ = _snapshot_workspace(root)
            secret_file.write_text("API_KEY=changed-secret\n", encoding="utf-8")
            after, _ = _snapshot_workspace(root)

        changes = _workspace_changes(before, after)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].path, ".env")
        self.assertIn("hidden for safety", changes[0].note)
        self.assertNotIn("changed-secret", changes[0].diff)

    def test_ignored_directories_are_not_in_the_change_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ignored = root / ".git"
            ignored.mkdir()
            (ignored / "internal").write_text("ignored\n", encoding="utf-8")
            files, errors = _snapshot_workspace(root)

        self.assertEqual(files, {})
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
