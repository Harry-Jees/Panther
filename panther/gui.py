from __future__ import annotations

import difflib
import codecs
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from .config import load_config, save_config
from .tools import IGNORED_PARTS

_MAX_VIEW_BYTES = 2 * 1024 * 1024
_MAX_DIFF_BYTES = 2 * 1024 * 1024
_MAX_SNAPSHOT_CONTENT_BYTES = 64 * 1024 * 1024
_MAX_TASK_DIFF_OUTPUT_CHARS = 120_000
_SNAPSHOT_IGNORED_PARTS = IGNORED_PARTS | {
    "node_modules",
    ".next",
    ".pytest_cache",
    ".tox",
    "target",
}


@dataclass(frozen=True)
class _WorkspaceFile:
    content: bytes | None
    digest: bytes
    size: int
    sensitive: bool
    omission_reason: str = ""


@dataclass(frozen=True)
class _FileChange:
    path: str
    status: str
    diff: str
    added: int
    removed: int
    note: str = ""


def _is_sensitive_file(path: Path) -> bool:
    name = path.name.casefold()
    return (
        name.startswith(".env")
        or name.startswith("id_rsa")
        or path.suffix.casefold() in {".key", ".pem", ".p12", ".pfx"}
    )


def _snapshot_workspace(
    root: Path,
) -> tuple[dict[str, _WorkspaceFile], list[str]]:
    files: dict[str, _WorkspaceFile] = {}
    errors: list[str] = []
    retained_content_bytes = 0

    def on_walk_error(error: OSError) -> None:
        errors.append(str(error))

    for current, directories, names in os.walk(root, onerror=on_walk_error):
        current_path = Path(current)
        directories[:] = sorted(
            name
            for name in directories
            if name not in _SNAPSHOT_IGNORED_PARTS
            and not (current_path / name).is_symlink()
        )
        for name in sorted(names):
            path = current_path / name
            if path.is_symlink():
                continue
            try:
                if not path.is_file():
                    continue
                sensitive = _is_sensitive_file(path)
                stat = path.stat()
                capture_content = (
                    not sensitive
                    and stat.st_size <= _MAX_DIFF_BYTES
                    and retained_content_bytes + stat.st_size
                    <= _MAX_SNAPSHOT_CONTENT_BYTES
                )
                buffer = bytearray() if capture_content else None
                digest = hashlib.sha256()
                size = 0
                if sensitive or stat.st_size <= _MAX_DIFF_BYTES:
                    with path.open("rb") as source:
                        while chunk := source.read(64 * 1024):
                            digest.update(chunk)
                            size += len(chunk)
                            if buffer is not None:
                                buffer.extend(chunk)
                                if len(buffer) > _MAX_DIFF_BYTES:
                                    buffer = None
                else:
                    size = stat.st_size
                    digest.update(f"{size}:{stat.st_mtime_ns}".encode("ascii"))
                if buffer is not None:
                    retained_content_bytes += size
                omission_reason = ""
                if sensitive:
                    omission_reason = "Changed; contents hidden for safety."
                elif size > _MAX_DIFF_BYTES:
                    omission_reason = (
                        f"Diff omitted (file exceeds {_MAX_DIFF_BYTES // (1024 * 1024)} MiB)."
                    )
                elif buffer is None:
                    omission_reason = "Diff omitted (workspace snapshot size limit reached)."
                files[path.relative_to(root).as_posix()] = _WorkspaceFile(
                    content=bytes(buffer) if buffer is not None else None,
                    digest=digest.digest(),
                    size=size,
                    sensitive=sensitive,
                    omission_reason=omission_reason,
                )
            except OSError as exc:
                errors.append(f"{path.relative_to(root).as_posix()}: {exc}")
    return files, errors


def _workspace_changes(
    before: dict[str, _WorkspaceFile],
    after: dict[str, _WorkspaceFile],
) -> list[_FileChange]:
    changes: list[_FileChange] = []
    for path in sorted(before.keys() | after.keys()):
        old = before.get(path)
        new = after.get(path)
        if old is not None and new is not None and old.digest == new.digest:
            continue
        status = "A" if old is None else "D" if new is None else "M"
        old_file = old or _WorkspaceFile(b"", b"", 0, False)
        new_file = new or _WorkspaceFile(b"", b"", 0, False)
        if old_file.sensitive or new_file.sensitive:
            changes.append(
                _FileChange(
                    path,
                    status,
                    "",
                    0,
                    0,
                    old_file.omission_reason or new_file.omission_reason,
                )
            )
            continue
        if old_file.content is None or new_file.content is None:
            changes.append(
                _FileChange(
                    path,
                    status,
                    "",
                    0,
                    0,
                    old_file.omission_reason
                    or new_file.omission_reason
                    or "Diff omitted (file content was not captured).",
                )
            )
            continue
        try:
            old_text = old_file.content.decode("utf-8")
            new_text = new_file.content.decode("utf-8")
        except UnicodeDecodeError:
            changes.append(
                _FileChange(path, status, "", 0, 0, "Diff omitted (not UTF-8 text).")
            )
            continue
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        diff = "".join(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                n=3,
            )
        )
        added = sum(
            line.startswith("+") and not line.startswith("+++")
            for line in diff.splitlines()
        )
        removed = sum(
            line.startswith("-") and not line.startswith("---")
            for line in diff.splitlines()
        )
        changes.append(_FileChange(path, status, diff, added, removed))
    return changes


class PantherDesktop:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Panther")
        self.root.geometry("1500x900")
        self.root.minsize(1050, 650)
        self.project_root: Path | None = None
        self.sessions: dict[Path, dict[str, Any]] = {}
        self._tree_paths: dict[str, Path] = {}
        self._selected_file: Path | None = None
        self._editor_dirty = False
        self._busy = False
        self._shutdown = threading.Event()
        self._ui_queue: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self._setup_dialog: tk.Toplevel | None = None
        self._setup_key: tk.Entry | None = None
        self._setup_model: ttk.Combobox | None = None
        self._setup_name: tk.Entry | None = None
        self._setup_purpose: tk.Entry | None = None
        self._setup_fact: tk.Entry | None = None
        self._setup_share: tk.BooleanVar | None = None
        self._verified_key: str | None = None
        self._verified_models: list[str] = []
        self._active_changes: list[_FileChange] = []
        self._last_scan_errors: list[str] = []
        self._cli_process: subprocess.Popen[bytes] | None = None
        self._cli_root: Path | None = None
        self._cli_lock = threading.Lock()
        self._cli_snapshot_lock = threading.Lock()
        self._cli_request_snapshot: tuple[
            Path,
            dict[str, _WorkspaceFile],
            list[str],
        ] | None = None
        self._cli_ready = False
        self._cli_at_root_prompt = False
        self._cli_prompt_secret = False
        self._terminal_process: subprocess.Popen[str] | None = None
        self._terminal_busy = False
        self._terminal_cancel_requested = threading.Event()
        self._terminal_lock = threading.Lock()
        self._build()
        self.workspace_label.configure(text="Choose a project folder to begin")
        self.root.protocol("WM_DELETE_WINDOW", self._close_window)
        self.root.after(50, self._drain_ui_queue)
        self.root.after(150, self._check_first_run)

    def _build(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")

        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(fill=tk.X)
        self.open_folder_button = ttk.Button(
            toolbar, text="Open Folder", command=self._choose_workspace
        )
        self.open_folder_button.pack(side=tk.LEFT)
        self.refresh_button = ttk.Button(
            toolbar, text="Refresh Files", command=self._refresh_files
        )
        self.refresh_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="NVIDIA Setup", command=self._show_setup).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.workspace_label = ttk.Label(toolbar, text="No folder selected")
        self.workspace_label.pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

        workbench = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
        workbench.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        upper = ttk.Panedwindow(workbench, orient=tk.HORIZONTAL)
        workbench.add(upper, weight=4)

        explorer = ttk.Frame(upper, padding=4)
        upper.add(explorer, weight=1)
        tree_frame = ttk.Frame(explorer)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(tree_frame, text="PROJECT FILES").pack(anchor=tk.W, pady=(0, 5))
        self.file_tree = ttk.Treeview(tree_frame, show="tree", selectmode="browse")
        tree_scroll = ttk.Scrollbar(
            tree_frame, orient=tk.VERTICAL, command=self.file_tree.yview
        )
        self.file_tree.configure(yscrollcommand=tree_scroll.set)
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_tree.bind("<<TreeviewOpen>>", self._tree_opened)
        self.file_tree.bind("<<TreeviewSelect>>", self._tree_selected)

        viewer = ttk.Frame(upper, padding=4)
        upper.add(viewer, weight=4)
        viewer_header = ttk.Frame(viewer)
        viewer_header.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(viewer_header, text="EDITOR").pack(side=tk.LEFT)
        self.save_button = ttk.Button(
            viewer_header, text="Save", command=self._save_file, state=tk.DISABLED
        )
        self.save_button.pack(side=tk.RIGHT)
        self.file_label = ttk.Label(viewer, text="Select a project file to view it.")
        self.file_label.pack(fill=tk.X, pady=(0, 4))
        editor_frame = ttk.Frame(viewer)
        editor_frame.pack(fill=tk.BOTH, expand=True)
        self.editor = tk.Text(
            editor_frame,
            wrap=tk.NONE,
            undo=True,
            font=("Consolas", 10),
            state=tk.DISABLED,
        )
        editor_y = ttk.Scrollbar(
            editor_frame, orient=tk.VERTICAL, command=self.editor.yview
        )
        editor_x = ttk.Scrollbar(
            viewer, orient=tk.HORIZONTAL, command=self.editor.xview
        )
        self.editor.configure(yscrollcommand=editor_y.set, xscrollcommand=editor_x.set)
        self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        editor_y.pack(side=tk.RIGHT, fill=tk.Y)
        editor_x.pack(fill=tk.X)
        self.editor.bind("<<Modified>>", self._editor_changed)

        right = ttk.Frame(upper, padding=4)
        upper.add(right, weight=3)
        console_header = ttk.Frame(right, padding=(6, 4))
        console_header.pack(fill=tk.X)
        ttk.Label(console_header, text="PANTHER CLI").pack(side=tk.LEFT)
        self.restart_cli_button = ttk.Button(
            console_header,
            text="Restart CLI",
            command=self._restart_cli_session,
            state=tk.DISABLED,
        )
        self.restart_cli_button.pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(console_header, text="Clear", command=self._clear_console).pack(
            side=tk.RIGHT
        )
        self.status = tk.StringVar(
            value="Save NVIDIA setup and choose a folder to start Panther CLI."
        )
        ttk.Label(right, textvariable=self.status, padding=(8, 2)).pack(fill=tk.X)

        console_frame = ttk.Frame(right)
        console_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.console = tk.Text(
            console_frame,
            state=tk.DISABLED,
            wrap=tk.WORD,
            background="#111318",
            foreground="#e6e8ed",
            insertbackground="#e6e8ed",
            font=("Consolas", 10),
            padx=10,
            pady=8,
        )
        console_scroll = ttk.Scrollbar(
            console_frame, orient=tk.VERTICAL, command=self.console.yview
        )
        self.console.configure(yscrollcommand=console_scroll.set)
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        console_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.console.tag_configure("prompt", foreground="#7dd3fc")
        self.console.tag_configure("assistant", foreground="#f5a65b")
        self.console.tag_configure("system", foreground="#9ca3af")
        self.console.tag_configure("error", foreground="#f87171")
        self.console.tag_configure("diff_added", foreground="#86efac")
        self.console.tag_configure("diff_removed", foreground="#fca5a5")
        self.console.tag_configure("diff_header", foreground="#7dd3fc")

        task_row = ttk.Frame(right, padding=(6, 4))
        task_row.pack(fill=tk.X)
        self.task_entry = ttk.Entry(task_row)
        self.task_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.task_entry.bind("<Return>", self._send_cli_input)
        self.send_button = ttk.Button(
            task_row,
            text="Send",
            command=self._send_cli_input,
            state=tk.DISABLED,
        )
        self.send_button.pack(side=tk.RIGHT)

        terminal = ttk.Frame(workbench, padding=4)
        workbench.add(terminal, weight=1)
        terminal_header = ttk.Frame(terminal)
        terminal_header.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(terminal_header, text="TERMINAL").pack(side=tk.LEFT)
        self.terminal_cwd_label = ttk.Label(terminal_header, text="No folder selected")
        self.terminal_cwd_label.pack(side=tk.LEFT, padx=10)
        self.terminal_cancel_button = ttk.Button(
            terminal_header,
            text="Stop",
            command=self._stop_terminal_command,
            state=tk.DISABLED,
        )
        self.terminal_cancel_button.pack(side=tk.RIGHT)
        ttk.Button(
            terminal_header, text="Clear", command=self._clear_terminal
        ).pack(side=tk.RIGHT, padx=(0, 6))
        terminal_output_frame = ttk.Frame(terminal)
        terminal_output_frame.pack(fill=tk.BOTH, expand=True)
        self.terminal_output = tk.Text(
            terminal_output_frame,
            state=tk.DISABLED,
            wrap=tk.NONE,
            background="#111318",
            foreground="#e6e8ed",
            insertbackground="#e6e8ed",
            font=("Consolas", 10),
            padx=8,
            pady=6,
        )
        terminal_y = ttk.Scrollbar(
            terminal_output_frame,
            orient=tk.VERTICAL,
            command=self.terminal_output.yview,
        )
        terminal_x = ttk.Scrollbar(
            terminal, orient=tk.HORIZONTAL, command=self.terminal_output.xview
        )
        self.terminal_output.configure(
            yscrollcommand=terminal_y.set,
            xscrollcommand=terminal_x.set,
        )
        self.terminal_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        terminal_y.pack(side=tk.RIGHT, fill=tk.Y)
        terminal_x.pack(fill=tk.X)
        self.terminal_output.tag_configure("command", foreground="#7dd3fc")
        self.terminal_output.tag_configure("error", foreground="#f87171")
        terminal_input = ttk.Frame(terminal, padding=(0, 5, 0, 0))
        terminal_input.pack(fill=tk.X)
        self.terminal_prompt = ttk.Label(
            terminal_input, text="> ", font=("Consolas", 10)
        )
        self.terminal_prompt.pack(side=tk.LEFT)
        self.terminal_entry = ttk.Entry(terminal_input, font=("Consolas", 10))
        self.terminal_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.terminal_entry.bind("<Return>", self._run_terminal_command)
        self.terminal_entry.configure(state=tk.DISABLED)
        self.terminal_run_button = ttk.Button(
            terminal_input,
            text="Run",
            command=self._run_terminal_command,
            state=tk.DISABLED,
        )
        self.terminal_run_button.pack(side=tk.LEFT, padx=(6, 0))

        self._append_console(
            "PANTHER",
            "Panther's Python CLI, banner, and slash commands will start here "
            "after NVIDIA setup and workspace selection.",
            "system",
        )
        self._append_terminal("Select a project folder to enable commands.\n")

    def _check_first_run(self) -> None:
        try:
            from .cli import _needs_first_run_onboarding

            if _needs_first_run_onboarding(load_config()):
                self._show_setup()
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror("Panther setup error", str(exc), parent=self.root)

    def _select_workspace(self, path: Path) -> None:
        if self._editor_dirty and not self._confirm_discard_or_save():
            return
        selected = path.expanduser().resolve()
        if not selected.is_dir():
            messagebox.showerror("Invalid folder", f"Not a folder:\n{selected}", parent=self.root)
            return
        self.project_root = selected
        self.workspace_label.configure(text=str(selected))
        self.terminal_cwd_label.configure(text=str(selected))
        self.terminal_prompt.configure(text=f"{selected.name or selected.drive}> ")
        self._set_busy(self._busy)
        self._selected_file = None
        self._editor_dirty = False
        self._set_editor_text("", selected_path=None)
        self._populate_root()
        self._append_console("PANTHER", f"Workspace: {selected}", "system")
        self._start_cli_session()

    def _choose_workspace(self) -> None:
        initial = str(self.project_root or Path.cwd())
        folder = filedialog.askdirectory(
            parent=self.root,
            title="Choose Panther project folder",
            initialdir=initial,
            mustexist=True,
        )
        if folder:
            self._select_workspace(Path(folder))

    def _populate_root(self) -> None:
        self.file_tree.delete(*self.file_tree.get_children())
        self._tree_paths.clear()
        if self.project_root is None:
            return
        root_id = self.file_tree.insert(
            "",
            tk.END,
            text=self.project_root.name or str(self.project_root),
            open=True,
        )
        self._tree_paths[root_id] = self.project_root
        self._populate_directory(root_id, self.project_root)

    def _populate_directory(self, item_id: str, directory: Path) -> None:
        self.file_tree.delete(*self.file_tree.get_children(item_id))
        try:
            entries = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
        except OSError as exc:
            self._append_console("ERROR", f"Cannot list {directory}: {exc}", "error")
            return
        for entry in entries:
            if entry.name in IGNORED_PARTS or entry.is_symlink():
                continue
            try:
                is_directory = entry.is_dir()
            except OSError:
                continue
            child_id = self.file_tree.insert(
                item_id,
                tk.END,
                text=entry.name,
                open=False,
            )
            self._tree_paths[child_id] = entry
            if is_directory:
                self.file_tree.insert(child_id, tk.END, text="…")

    def _tree_opened(self, _event: tk.Event[Any]) -> None:
        item_id = self.file_tree.focus()
        path = self._tree_paths.get(item_id)
        children = self.file_tree.get_children(item_id)
        if path is not None and path.is_dir() and children:
            if len(children) == 1 and self.file_tree.item(children[0], "text") == "…":
                self._populate_directory(item_id, path)

    def _tree_selected(self, _event: tk.Event[Any]) -> None:
        selection = self.file_tree.selection()
        if not selection:
            return
        path = self._tree_paths.get(selection[0])
        if path is not None and path.is_file():
            self._load_file(path)

    def _load_file(self, path: Path) -> None:
        if self._editor_dirty and not self._confirm_discard_or_save():
            return
        if self.project_root is None:
            return
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.project_root)
            if resolved.stat().st_size > _MAX_VIEW_BYTES:
                raise ValueError("File is larger than the 2 MiB viewer limit.")
            content = resolved.read_text(encoding="utf-8")
        except UnicodeError:
            messagebox.showwarning(
                "Unsupported file",
                "This viewer opens UTF-8 text files only.",
                parent=self.root,
            )
            return
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not open file", str(exc), parent=self.root)
            return
        self._set_editor_text(content, resolved)

    def _set_editor_text(self, content: str, selected_path: Path | None) -> None:
        self.editor.configure(state=tk.NORMAL)
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", content)
        self.editor.edit_modified(False)
        self.editor.configure(state=tk.NORMAL if selected_path else tk.DISABLED)
        self._selected_file = selected_path
        self._editor_dirty = False
        self.file_label.configure(text=str(selected_path) if selected_path else "Select a project file to view it.")
        self.save_button.configure(state=tk.NORMAL if selected_path else tk.DISABLED)

    def _editor_changed(self, _event: tk.Event[Any]) -> None:
        if self.editor.edit_modified():
            self._editor_dirty = True
            self.editor.edit_modified(False)

    def _save_file(self) -> None:
        if self._selected_file is None or self.project_root is None:
            return
        try:
            target = self._selected_file.resolve(strict=True)
            target.relative_to(self.project_root)
            target.write_text(self.editor.get("1.0", "end-1c"), encoding="utf-8")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not save file", str(exc), parent=self.root)
            return
        self._editor_dirty = False
        self._append_console("PANTHER", f"Saved {target.relative_to(self.project_root)}", "system")
        self.status.set(f"Saved {target.name}")

    def _confirm_discard_or_save(self) -> bool:
        choice = messagebox.askyesnocancel(
            "Unsaved changes",
            "Save changes to the selected file before continuing?",
            parent=self.root,
        )
        if choice is None:
            return False
        if choice:
            self._save_file()
            return not self._editor_dirty
        self._editor_dirty = False
        return True

    def _refresh_files(self) -> None:
        self._populate_root()

    def _append_console(self, label: str, text: str, tag: str = "") -> None:
        self.console.configure(state=tk.NORMAL)
        self.console.insert(tk.END, f"\n[{label}]\n", tag)
        self.console.insert(tk.END, f"{text.rstrip()}\n\n")
        self.console.see(tk.END)
        self.console.configure(state=tk.DISABLED)

    def _clear_console(self) -> None:
        self.console.configure(state=tk.NORMAL)
        self.console.delete("1.0", tk.END)
        self.console.configure(state=tk.DISABLED)

    def _show_task_changes(
        self,
        project_root: Path,
        changes: list[_FileChange],
        scan_errors: list[str],
    ) -> None:
        self._active_changes = changes
        self._last_scan_errors = scan_errors

        if changes:
            added = sum(change.added for change in changes)
            removed = sum(change.removed for change in changes)
            summary = (
                f"{len(changes)} changed file(s); {added} line(s) added, "
                f"{removed} line(s) removed."
            )
            file_summary = "\n".join(
                f"{change.status} {change.path} (+{change.added}/-{change.removed})"
                for change in changes
            )
            self._append_console(
                "TASK CHANGES",
                f"{summary}\n{file_summary}",
                "system",
            )
        else:
            summary = "No workspace file changes detected for this task."
            self._append_console("TASK CHANGES", summary, "system")
        if changes:
            self.console.configure(state=tk.NORMAL)
            self.console.insert(tk.END, "[MODIFIED LINES]\n", "system")
            remaining = _MAX_TASK_DIFF_OUTPUT_CHARS
            truncated = False
            for change in changes:
                self.console.insert(
                    tk.END,
                    f"\n[{change.status} +{change.added}/-{change.removed}] "
                    f"{change.path}\n",
                    "system",
                )
                if change.note:
                    self.console.insert(tk.END, f"{change.note}\n", "error")
                    continue
                if not change.diff:
                    self.console.insert(tk.END, "No line-level diff available.\n")
                    continue
                for line in change.diff.splitlines(keepends=True):
                    if len(line) > remaining:
                        self.console.insert(
                            tk.END,
                            "\n[Diff display truncated. Use the file editor or Git diff "
                            "for the complete changes.]\n",
                            "system",
                        )
                        truncated = True
                        break
                    tag = (
                        "diff_added"
                        if line.startswith("+") and not line.startswith("+++ ")
                        else "diff_removed"
                        if line.startswith("-") and not line.startswith("--- ")
                        else "diff_header"
                        if line.startswith(("+++", "---", "@@"))
                        else ""
                    )
                    self.console.insert(tk.END, line, tag)
                    remaining -= len(line)
                if truncated:
                    break
            self.console.insert(tk.END, "\n")
            self.console.see(tk.END)
            self.console.configure(state=tk.DISABLED)
        if scan_errors:
            self._append_console(
                "CHANGE SCAN WARNINGS",
                "Some workspace files could not be inspected:\n"
                + "\n".join(scan_errors),
                "error",
            )

        if self.project_root == project_root:
            self._populate_root()
            if self._selected_file is not None:
                relative_path = self._selected_file.relative_to(project_root).as_posix()
                changed = next(
                    (change for change in changes if change.path == relative_path),
                    None,
                )
                if changed is not None and changed.status == "D":
                    self._set_editor_text("", selected_path=None)
                elif changed is not None and not self._editor_dirty:
                    self._load_file(self._selected_file)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        input_locked = (
            busy
            or self._terminal_busy
            or not self._cli_ready
            or self._cli_process is None
        )
        self.send_button.configure(state=tk.DISABLED if input_locked else tk.NORMAL)
        self.task_entry.configure(state=tk.DISABLED if input_locked else tk.NORMAL)
        self.task_entry.configure(show="*" if self._cli_prompt_secret else "")
        workspace_locked = busy or self._terminal_busy
        self.open_folder_button.configure(
            state=tk.DISABLED if workspace_locked else tk.NORMAL
        )
        self.refresh_button.configure(
            state=tk.DISABLED if workspace_locked else tk.NORMAL
        )
        self.terminal_run_button.configure(
            state=tk.DISABLED if input_locked or self.project_root is None else tk.NORMAL
        )
        self.terminal_entry.configure(
            state=(
                tk.DISABLED
                if input_locked or self.project_root is None
                else tk.NORMAL
            )
        )

    def _config_uses_nvidia(self, config: dict[str, Any]) -> bool:
        active_bot = config.get("activeBot")
        bots = config.get("bots", {})
        profile = bots.get(active_bot, {}) if isinstance(bots, dict) else {}
        return (
            config.get("activeProvider") == "nvidia"
            or (
                isinstance(profile, dict)
                and profile.get("provider") == "nvidia"
            )
        )

    def _start_cli_session(self) -> None:
        project_root = self.project_root
        if project_root is None or self._shutdown.is_set():
            return
        self._stop_cli_session()
        try:
            config = load_config(project_root)
            if not self._config_uses_nvidia(config):
                self.status.set("Save NVIDIA setup to start the Panther CLI.")
                return
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "panther"],
                cwd=project_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=env,
            )
        except (OSError, ValueError) as exc:
            self._append_console("CLI ERROR", str(exc), "error")
            self.status.set("Could not start Panther CLI")
            return

        with self._cli_lock:
            self._cli_process = process
            self._cli_root = project_root
        with self._cli_snapshot_lock:
            self._cli_request_snapshot = None
        self._cli_ready = False
        self._cli_at_root_prompt = False
        self._cli_prompt_secret = False
        self.status.set(f"Starting Panther CLI in {project_root}")
        self._set_busy(False)
        threading.Thread(
            target=self._read_cli_output,
            args=(process,),
            daemon=True,
            name="panther-cli-output",
        ).start()

    def _restart_cli_session(self) -> None:
        if self.project_root is None:
            self.status.set("Choose a folder before starting Panther CLI.")
            return
        self._start_cli_session()

    def _stop_cli_session(self) -> None:
        with self._cli_lock:
            process = self._cli_process
            self._cli_process = None
            self._cli_root = None
        if process is not None and process.poll() is None:
            process.terminate()
        self._cli_ready = False
        self._cli_at_root_prompt = False
        self._cli_prompt_secret = False
        self._set_busy(False)

    def _read_cli_output(self, process: subprocess.Popen[bytes]) -> None:
        if process.stdout is None:
            return
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        prompt_tail = ""
        stream_error: str | None = None
        try:
            while chunk := process.stdout.read1(4096):
                text = decoder.decode(chunk)
                if not text:
                    continue
                self._post(lambda text=text: self._append_cli_output(text))
                prompt_tail = (prompt_tail + text)[-512:]
                current_line = prompt_tail.rsplit("\n", 1)[-1].rsplit("\r", 1)[-1]
                if current_line.endswith("panther> "):
                    self._cli_repl_prompt(process)
                elif current_line.endswith((": ", "? ")):
                    self._cli_interactive_prompt(process, current_line)
        except OSError as exc:
            stream_error = str(exc)
        finally:
            final_text = decoder.decode(b"", final=True)
            if final_text:
                self._post(lambda text=final_text: self._append_cli_output(text))
            return_code = process.wait()
            self._post(
                lambda: self._cli_session_ended(
                    process,
                    return_code,
                    stream_error,
                )
            )

    def _append_cli_output(self, text: str) -> None:
        self.console.configure(state=tk.NORMAL)
        self.console.insert(tk.END, text)
        self.console.see(tk.END)
        self.console.configure(state=tk.DISABLED)

    def _cli_repl_prompt(self, process: subprocess.Popen[bytes]) -> None:
        with self._cli_lock:
            if process is not self._cli_process:
                return
            root = self._cli_root
        with self._cli_snapshot_lock:
            pending = self._cli_request_snapshot
            self._cli_request_snapshot = None
        if root is not None and pending is not None:
            before_root, before, before_errors = pending
            threading.Thread(
                target=self._finish_cli_request,
                args=(before_root, before, before_errors),
                daemon=True,
                name="panther-cli-change-scan",
            ).start()
        self._post(lambda: self._set_cli_prompt_ready(process, True, False))

    def _cli_interactive_prompt(
        self,
        process: subprocess.Popen[bytes],
        prompt: str,
    ) -> None:
        with self._cli_lock:
            if process is not self._cli_process:
                return
        secret = any(term in prompt.casefold() for term in ("password", "api key"))
        self._post(lambda: self._set_cli_prompt_ready(process, False, secret))

    def _set_cli_prompt_ready(
        self,
        process: subprocess.Popen[bytes],
        root_prompt: bool,
        secret: bool,
    ) -> None:
        with self._cli_lock:
            if process is not self._cli_process:
                return
        self._cli_at_root_prompt = root_prompt
        self._cli_prompt_secret = secret
        self._cli_ready = True
        self._set_busy(False)
        self.status.set(
            "Panther CLI ready" if root_prompt else "Panther CLI input required"
        )
        self.task_entry.focus_set()

    def _finish_cli_request(
        self,
        project_root: Path,
        before: dict[str, _WorkspaceFile],
        before_errors: list[str],
    ) -> None:
        after, after_errors = _snapshot_workspace(project_root)
        changes = _workspace_changes(before, after)
        self._post(
            lambda: self._show_task_changes(
                project_root,
                changes,
                before_errors + after_errors,
            )
        )

    def _cli_session_ended(
        self,
        process: subprocess.Popen[bytes],
        return_code: int,
        stream_error: str | None,
    ) -> None:
        with self._cli_lock:
            if process is not self._cli_process:
                return
            self._cli_process = None
            self._cli_root = None
        self._cli_ready = False
        self._cli_at_root_prompt = False
        self._cli_prompt_secret = False
        if stream_error:
            self._append_console("CLI OUTPUT ERROR", stream_error, "error")
        if return_code:
            self._append_console(
                "CLI EXIT",
                f"Panther CLI exited with code {return_code}.",
                "error",
            )
        self.status.set("Panther CLI stopped")
        self.restart_cli_button.configure(
            state=tk.NORMAL if self.project_root is not None else tk.DISABLED
        )
        self._set_busy(False)

    def _send_cli_input(self, _event: tk.Event[Any] | None = None) -> str:
        text = self.task_entry.get()
        if not text.strip() or not self._cli_ready:
            return "break"
        with self._cli_lock:
            process = self._cli_process
            project_root = self._cli_root
        if (
            process is None
            or process.stdin is None
            or process.poll() is not None
            or project_root is None
        ):
            self.status.set("Panther CLI is not running. Restart the CLI session.")
            return "break"

        self.task_entry.delete(0, tk.END)
        secret = self._cli_prompt_secret
        self._append_console(
            "YOU",
            "[hidden input]" if secret else text,
            "prompt",
        )
        at_root_prompt = self._cli_at_root_prompt
        self._cli_ready = False
        self._cli_at_root_prompt = False
        self._cli_prompt_secret = False
        self._set_busy(True)

        def submit() -> None:
            try:
                if at_root_prompt:
                    before, errors = _snapshot_workspace(project_root)
                    with self._cli_snapshot_lock:
                        self._cli_request_snapshot = (project_root, before, errors)
                with self._cli_lock:
                    if process is not self._cli_process or process.stdin is None:
                        return
                    process.stdin.write((text + "\n").encode("utf-8"))
                    process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                error = str(exc)
                self._post(
                    lambda: self._append_console("CLI INPUT ERROR", error, "error")
                )
                self._post(
                    lambda: self.status.set("Panther CLI could not receive input.")
                )

        threading.Thread(
            target=submit,
            daemon=True,
            name="panther-cli-input",
        ).start()
        return "break"

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        input_locked = (
            busy
            or self._terminal_busy
            or not self._cli_ready
            or self._cli_process is None
        )
        self.send_button.configure(state=tk.DISABLED if input_locked else tk.NORMAL)
        self.task_entry.configure(state=tk.DISABLED if input_locked else tk.NORMAL)
        self.task_entry.configure(show="*" if self._cli_prompt_secret else "")
        workspace_locked = busy or self._terminal_busy
        self.open_folder_button.configure(
            state=tk.DISABLED if workspace_locked else tk.NORMAL
        )
        self.refresh_button.configure(
            state=tk.DISABLED if workspace_locked else tk.NORMAL
        )
        self.terminal_run_button.configure(
            state=tk.DISABLED if input_locked or self.project_root is None else tk.NORMAL
        )
        self.terminal_entry.configure(
            state=(
                tk.DISABLED
                if input_locked or self.project_root is None
                else tk.NORMAL
            )
        )

    def _append_terminal(self, text: str, tag: str = "") -> None:
        self.terminal_output.configure(state=tk.NORMAL)
        self.terminal_output.insert(tk.END, text, tag)
        self.terminal_output.see(tk.END)
        self.terminal_output.configure(state=tk.DISABLED)

    def _clear_terminal(self) -> None:
        self.terminal_output.configure(state=tk.NORMAL)
        self.terminal_output.delete("1.0", tk.END)
        self.terminal_output.configure(state=tk.DISABLED)

    def _run_terminal_command(self, _event: tk.Event[Any] | None = None) -> str:
        command = self.terminal_entry.get().strip()
        if not command or self._terminal_busy:
            return "break"
        if self.project_root is None:
            self._append_terminal("Open a project folder before running commands.\n", "error")
            return "break"
        project_root = self.project_root
        self._terminal_busy = True
        self._terminal_cancel_requested.clear()
        self.terminal_entry.delete(0, tk.END)
        self._append_terminal(f"{project_root.name or project_root}> {command}\n", "command")
        self.terminal_entry.configure(state=tk.DISABLED)
        self.terminal_cancel_button.configure(state=tk.NORMAL)
        self._set_busy(self._busy)
        worker = threading.Thread(
            target=self._execute_terminal_command,
            args=(command, project_root),
            daemon=True,
            name="panther-terminal-command",
        )
        worker.start()
        return "break"

    def _execute_terminal_command(self, command: str, project_root: Path) -> None:
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=project_root,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with self._terminal_lock:
                self._terminal_process = process
            if self._terminal_cancel_requested.is_set():
                process.terminate()
            if process.stdout is not None:
                for line in process.stdout:
                    self._post(lambda line=line: self._append_terminal(line))
            return_code = process.wait()
            if return_code:
                self._post(
                    lambda: self._append_terminal(
                        f"\n[Process exited with code {return_code}]\n",
                        "error",
                    )
                )
        except OSError as exc:
            error = str(exc)
            self._post(lambda: self._append_terminal(f"\n{error}\n", "error"))
        finally:
            with self._terminal_lock:
                self._terminal_process = None
            self._terminal_busy = False
            self._post(self._terminal_command_finished)

    def _terminal_command_finished(self) -> None:
        if self._shutdown.is_set():
            return
        self.terminal_entry.configure(state=tk.NORMAL)
        self.terminal_run_button.configure(state=tk.NORMAL)
        self.terminal_cancel_button.configure(state=tk.DISABLED)
        self._set_busy(self._busy)
        self.terminal_entry.focus_set()

    def _stop_terminal_command(self) -> None:
        self._terminal_cancel_requested.set()
        with self._terminal_lock:
            process = self._terminal_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError as exc:
                messagebox.showerror("Could not stop command", str(exc), parent=self.root)

    def _send_task(self, _event: tk.Event[Any] | None = None) -> str:
        task = self.task_entry.get().strip()
        if not task or self._busy:
            return "break"
        if self.project_root is None:
            messagebox.showinfo("Choose a folder", "Open a project folder first.", parent=self.root)
            return "break"
        project_root = self.project_root
        self.task_entry.delete(0, tk.END)
        self._append_console("YOU", task, "prompt")
        self.status.set("Starting Panther…")
        self._set_busy(True)
        worker = threading.Thread(
            target=self._run_task,
            args=(task, project_root, self.code_mode.get()),
            daemon=True,
            name="panther-agent",
        )
        worker.start()
        return "break"

    def _post(self, callback: Callable[[], None]) -> None:
        if self._shutdown.is_set():
            return
        self._ui_queue.put(callback)

    def _drain_ui_queue(self) -> None:
        if self._shutdown.is_set():
            return
        try:
            while True:
                callback = self._ui_queue.get_nowait()
                callback()
        except queue.Empty:
            pass
        finally:
            if not self._shutdown.is_set():
                try:
                    self.root.after(50, self._drain_ui_queue)
                except tk.TclError:
                    self._shutdown.set()

    def _ask_permission(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        from .cli import _permission_summary

        answer: list[bool] = []
        completed = threading.Event()

        def show_dialog() -> None:
            try:
                if self._shutdown.is_set():
                    return
                accepted = messagebox.askyesno(
                    "Panther permission request",
                    f"Allow this tool once?\n\nTool: {tool_name}\n"
                    f"Arguments: {_permission_summary(arguments, tool_name)}",
                    parent=self.root,
                )
                answer.append(accepted)
            finally:
                completed.set()

        self._post(show_dialog)
        while not completed.wait(0.1):
            if self._shutdown.is_set():
                return False
        return bool(answer and answer[0])

    def _run_task(self, task: str, project_root: Path, code_mode: bool) -> None:
        from .cli import (
            _execute_task,
            _get_session,
            _is_coding_task,
            _manager,
        )

        before, before_errors = _snapshot_workspace(project_root)
        succeeded = False
        try:
            config = load_config(project_root)
            session = self.sessions.get(project_root)
            if session is None:
                session = _get_session(_manager(project_root), None)
                self.sessions[project_root] = session
            effective_code_mode = code_mode or _is_coding_task(task)
            _execute_task(
                task,
                project_root,
                session,
                config,
                json_output=True,
                one_shot=False,
                code_mode=effective_code_mode,
                ui_on_response=lambda text: self._post(
                    lambda: self._append_console("PANTHER", text, "assistant")
                ),
                ui_on_status=lambda status: self._post(
                    lambda: self.status.set(status.capitalize())
                ),
                ui_on_event=lambda event: self._post(lambda: self._handle_event(event)),
                ui_approve=self._ask_permission,
            )
            succeeded = True
        except Exception as exc:
            error = str(exc)
            self._post(lambda: self._append_console("ERROR", error, "error"))
        finally:
            after, after_errors = _snapshot_workspace(project_root)
            changes = _workspace_changes(before, after)
            scan_errors = before_errors + after_errors
            self._post(
                lambda: self._show_task_changes(
                    project_root,
                    changes,
                    scan_errors,
                )
            )
            self._post(
                lambda: self.status.set("Ready" if succeeded else "Task failed")
            )
            self._post(lambda: self._set_busy(False))

    def _handle_event(self, event: Any) -> None:
        event_name = event.name
        if event_name in {
            "agent_started",
            "model_started",
            "permission_requested",
            "tool_started",
            "tool_completed",
            "agent_completed",
            "agent_error",
        }:
            self.status.set(event_name.replace("_", " ").capitalize())

    def _close_window(self) -> None:
        self._shutdown.set()
        with self._terminal_lock:
            terminal_process = self._terminal_process
        if terminal_process is not None and terminal_process.poll() is None:
            terminal_process.terminate()
        if self._setup_dialog is not None:
            self._close_setup()
        self.root.destroy()

    def _show_setup(self) -> None:
        if self._setup_dialog is not None and self._setup_dialog.winfo_exists():
            self._setup_dialog.lift()
            return
        dialog = tk.Toplevel(self.root)
        self._setup_dialog = dialog
        dialog.title("Panther NVIDIA setup")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            frame,
            text="Connect Panther to NVIDIA NIM",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        ttk.Label(frame, text="API key").grid(row=1, column=0, sticky=tk.W, pady=3)
        self._setup_key = ttk.Entry(frame, show="*", width=46)
        self._setup_key.grid(row=1, column=1, sticky=tk.EW, pady=3)
        ttk.Label(frame, text="Your name").grid(row=2, column=0, sticky=tk.W, pady=3)
        self._setup_name = ttk.Entry(frame, width=46)
        self._setup_name.grid(row=2, column=1, sticky=tk.EW, pady=3)
        ttk.Label(frame, text="Main purpose").grid(row=3, column=0, sticky=tk.W, pady=3)
        self._setup_purpose = ttk.Entry(frame, width=46)
        self._setup_purpose.grid(row=3, column=1, sticky=tk.EW, pady=3)
        ttk.Label(frame, text="Personal fact (optional)").grid(row=4, column=0, sticky=tk.W, pady=3)
        self._setup_fact = ttk.Entry(frame, width=46)
        self._setup_fact.grid(row=4, column=1, sticky=tk.EW, pady=3)
        ttk.Label(frame, text="Model").grid(row=5, column=0, sticky=tk.W, pady=3)
        self._setup_model = ttk.Combobox(frame, state="disabled", width=43)
        self._setup_model.grid(row=5, column=1, sticky=tk.EW, pady=3)
        self._setup_share = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Allow sending my profile to NVIDIA for personalization",
            variable=self._setup_share,
        ).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=6)
        self._setup_status = tk.StringVar(value="Verify your key to load available models.")
        ttk.Label(frame, textvariable=self._setup_status, wraplength=400).grid(
            row=7, column=0, columnspan=2, sticky=tk.W, pady=5
        )
        actions = ttk.Frame(frame)
        actions.grid(row=8, column=0, columnspan=2, sticky=tk.EW, pady=(8, 0))
        self._setup_verify_button = ttk.Button(
            actions, text="Verify key and load models", command=self._verify_setup_key
        )
        self._setup_verify_button.pack(side=tk.LEFT)
        self._setup_finish_button = ttk.Button(
            actions, text="Save setup", command=self._finish_setup, state=tk.DISABLED
        )
        self._setup_finish_button.pack(side=tk.RIGHT)
        frame.columnconfigure(1, weight=1)
        dialog.protocol("WM_DELETE_WINDOW", self._close_setup)

    def _verify_setup_key(self) -> None:
        from .provider import NvidiaProvider

        dialog = self._setup_dialog
        if dialog is None or not dialog.winfo_exists():
            return
        assert self._setup_key is not None
        api_key = self._setup_key.get().strip()
        if not api_key:
            messagebox.showerror("Missing key", "Enter your NVIDIA API key.", parent=dialog)
            return
        self._setup_verify_button.configure(state=tk.DISABLED)
        self._setup_status.set("Checking the key and loading models…")

        def query_models() -> None:
            try:
                models = NvidiaProvider("nvidia/placeholder", api_key=api_key).list_models()
                if not models:
                    raise RuntimeError("NVIDIA returned no models for this key.")
                self._post(lambda: self._models_loaded(api_key, models))
            except Exception as exc:
                error = str(exc)
                self._post(lambda: self._model_load_failed(error))

        threading.Thread(target=query_models, daemon=True, name="panther-model-list").start()

    def _models_loaded(self, api_key: str, models: list[str]) -> None:
        if self._setup_dialog is None or not self._setup_dialog.winfo_exists():
            return
        self._verified_key = api_key
        self._verified_models = models
        assert self._setup_model is not None
        config = load_config()
        current_model = config["remoteModel"]
        self._setup_model.configure(values=models, state="readonly")
        self._setup_model.set(current_model if current_model in models else models[0])
        self._setup_status.set(f"Loaded {len(models)} model(s). Select one and save setup.")
        self._setup_verify_button.configure(state=tk.NORMAL)
        self._setup_finish_button.configure(state=tk.NORMAL)

    def _model_load_failed(self, error: str) -> None:
        if self._setup_dialog is None or not self._setup_dialog.winfo_exists():
            return
        self._setup_status.set(f"Could not load models: {error}")
        self._setup_verify_button.configure(state=tk.NORMAL)

    def _finish_setup(self) -> None:
        from .cli import USER_PROFILE_KEYRING_USERNAME
        from .provider import NVIDIA_KEYRING_SERVICE, NVIDIA_KEYRING_USERNAME

        dialog = self._setup_dialog
        if dialog is None or not dialog.winfo_exists():
            return
        if not self._verified_models:
            messagebox.showerror("Models not loaded", "Verify your key first.", parent=dialog)
            return
        if not self._verified_key:
            messagebox.showerror("Key not verified", "Verify your NVIDIA key first.", parent=dialog)
            return
        assert self._setup_name is not None
        assert self._setup_purpose is not None
        assert self._setup_fact is not None
        assert self._setup_model is not None
        assert self._setup_share is not None
        name = self._setup_name.get().strip()
        purpose = self._setup_purpose.get().strip()
        model = self._setup_model.get().strip()
        if not name or not purpose:
            messagebox.showerror(
                "Profile incomplete",
                "Enter your name and main purpose.",
                parent=dialog,
            )
            return
        if model not in self._verified_models:
            messagebox.showerror(
                "Invalid model",
                "Choose a model returned for this NVIDIA key.",
                parent=dialog,
            )
            return
        try:
            from importlib import import_module

            keyring = import_module("keyring")
            keyring.set_password(
                NVIDIA_KEYRING_SERVICE,
                NVIDIA_KEYRING_USERNAME,
                self._verified_key,
            )
            keyring.set_password(
                NVIDIA_KEYRING_SERVICE,
                USER_PROFILE_KEYRING_USERNAME,
                json.dumps(
                    {
                        "userName": name,
                        "mainPurpose": purpose,
                        "personalFact": self._setup_fact.get().strip(),
                    },
                    ensure_ascii=False,
                ),
            )
            config = load_config()
            config["activeProvider"] = "nvidia"
            config["activeBot"] = ""
            config["remoteModel"] = model
            config["onboardingComplete"] = True
            config["shareProfileWithNvidia"] = self._setup_share.get()
            save_config(config)
        except Exception as exc:
            messagebox.showerror("Setup failed", str(exc), parent=dialog)
            return
        self._close_setup()
        self._append_console("PANTHER", f"NVIDIA configured with {model}.", "system")
        self.status.set("Ready")

    def _close_setup(self) -> None:
        if self._setup_dialog is not None:
            try:
                self._setup_dialog.grab_release()
                if self._setup_key is not None:
                    self._setup_key.delete(0, tk.END)
                self._setup_dialog.destroy()
            except tk.TclError:
                pass
        self._setup_dialog = None
        self._setup_key = None
        self._verified_key = None
        self._verified_models.clear()


def main() -> int:
    root = tk.Tk()
    PantherDesktop(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
