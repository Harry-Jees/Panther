from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Sequence

from dotenv import dotenv_values

from .agent import run_agent
from .config import config_path, load_config, panther_dir, save_config
from .model_manager import context_window, ensure_model, list_models, switch_model
from .provider import (
    NVIDIA_KEYRING_SERVICE,
    NVIDIA_KEYRING_USERNAME,
    LocalProvider,
    NvidiaProvider,
)
from .session import SessionManager
from .tools import IGNORED_PARTS


ANSI_RESET = "\033[0m"
COLOR_START = (255, 45, 37)
COLOR_END = (255, 165, 45)
PANTHER_CLI_LOGO_LINES = [
    r"██████╗  █████╗ ███╗   ██╗████████╗██╗  ██╗███████╗██████╗      ██████╗██╗     ██╗",
    r"██╔══██╗██╔══██╗████╗  ██║╚══██╔══╝██║  ██║██╔════╝██╔══██╗    ██╔════╝██║     ██║",
    r"██████╔╝███████║██╔██╗ ██║   ██║   ███████║█████╗  ██████╔╝    ██║     ██║     ██║",
    r"██╔═══╝ ██╔══██║██║╚██╗██║   ██║   ██╔══██║██╔══╝  ██╔══██╗    ██║     ██║     ██║",
    r"██║     ██║  ██║██║ ╚████║   ██║   ██║  ██║███████╗██║  ██║    ╚██████╗███████╗██║",
    r"╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝     ╚═════╝╚══════╝╚═╝",
]
_ASCII_LOGO_LINES = ["PANTHER CLI"]


def _supports_truecolor() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _apply_gradient(text: str) -> str:
    if not _supports_truecolor():
        return text
    count = max(len(text) - 1, 1)
    rendered: list[str] = []
    for index, character in enumerate(text):
        ratio = index / count
        red = int(COLOR_START[0] + (COLOR_END[0] - COLOR_START[0]) * ratio)
        green = int(COLOR_START[1] + (COLOR_END[1] - COLOR_START[1]) * ratio)
        blue = int(COLOR_START[2] + (COLOR_END[2] - COLOR_START[2]) * ratio)
        rendered.append(f"\033[38;2;{red};{green};{blue}m{character}")
    return "".join(rendered) + ANSI_RESET


def _print_banner(show_banner: bool = True) -> None:
    """Print the interactive startup banner without polluting scripted output."""
    if not show_banner:
        return
    encoding = sys.stdout.encoding or "utf-8"
    try:
        unicode_logo = all(
            line.encode(encoding, errors="strict") for line in PANTHER_CLI_LOGO_LINES
        )
    except UnicodeEncodeError:
        unicode_logo = False
    logo_lines = PANTHER_CLI_LOGO_LINES if unicode_logo else _ASCII_LOGO_LINES
    for line in logo_lines:
        print(_apply_gradient(line))
    print(_apply_gradient("  Intelligent Execution. Relentless Precision."))
    width = shutil.get_terminal_size((80, 24)).columns
    divider = "─" if unicode_logo else "-"
    divider_length = min(max(width - 4, 30), 60)
    print(f"  {divider * divider_length}{ANSI_RESET if _supports_truecolor() else ''}")
    print("  Panther Engine - Ready")
    print("  Type 'help' to explore.\n")


def _repository_files(root: Path) -> list[Path]:
    return sorted(
        (
            path for path in root.rglob("*")
            if path.is_file() and not any(part in IGNORED_PARTS for part in path.parts)
        ),
        key=lambda path: path.as_posix().lower(),
    )


def _summarize_repository(root: Path) -> dict[str, Any]:
    files = _repository_files(root)
    top_files = files[:20]
    readme = next((path for path in files if path.name.lower() == "readme.md"), None)
    if readme is not None and readme not in top_files:
        top_files[-1] = readme
    return {
        "root": str(root),
        "file_count": len(files),
        "top_files": [str(path.relative_to(root)) for path in top_files],
        "summary": "Repository summary generated from the available files.",
    }


def _analyze_repository(root: Path) -> str:
    summary = _summarize_repository(root)
    lines = [
        "Repository analysis",
        "===================",
        f"Root: {summary['root']}",
        f"Files discovered: {summary['file_count']}",
        "Top files:",
        *(f"- {name}" for name in summary["top_files"]),
        "",
        "Repository summary",
        "-------------------",
        summary["summary"],
    ]
    return "\n".join(lines)


def _write_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False))


def _nvidia_api_key(project_root: Path) -> str | None:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if api_key and api_key.strip():
        return api_key
    values = dotenv_values(project_root / ".env")
    api_key = values.get("NVIDIA_API_KEY")
    return api_key if isinstance(api_key, str) else None


def _nvidia_provider(model: str, project_root: Path) -> NvidiaProvider:
    return NvidiaProvider(model, api_key=_nvidia_api_key(project_root))


def _progress(downloaded: int, total: int) -> None:
    percent = min(round(downloaded * 100 / total), 100) if total else 0
    size = f"{downloaded / 1_048_576:.1f} MB"
    if total:
        size += f" / {total / 1_048_576:.1f} MB"
    print(f"\rDownloading model: {percent}% ({size})", end="", file=sys.stderr, flush=True)


def _manager(project_root: Path) -> SessionManager:
    return SessionManager(project_root / ".panther" / "sessions")


def _get_session(manager: SessionManager, session_id: str | None) -> dict[str, Any]:
    session = manager.load(session_id) if session_id else None
    if session_id and session is None:
        raise ValueError(f"Session not found: {session_id}")
    return session if session is not None else manager.create()


def _run_with_provider(
    task: str,
    project_root: Path,
    session: dict[str, Any],
    config: dict[str, Any],
    provider: LocalProvider | NvidiaProvider,
    *,
    json_output: bool,
    one_shot: bool,
) -> str:
    def approve(tool_name: str, arguments: dict[str, Any]) -> bool:
        if json_output:
            return False
        answer = input(f"\nAllow {tool_name}({json.dumps(arguments)})? [y/N] ")
        return answer.strip().lower() == "y"

    result = run_agent(
        task,
        provider,
        session["history"],
        project_root,
        config["permissionMode"],
        one_shot=one_shot,
        approve=approve,
        on_response=None if json_output else lambda text: print(text, end="", flush=True),
    )
    _manager(project_root).save(session)
    return result


def _execute_task(
    task: str,
    project_root: Path,
    session: dict[str, Any],
    config: dict[str, Any],
    *,
    json_output: bool,
    one_shot: bool = True,
) -> str:
    if config["activeProvider"] == "local":
        if not json_output:
            print("Checking for local model...", file=sys.stderr)
        model_file = ensure_model(config["modelTier"], None if json_output else _progress)
        if not json_output:
            print("\nLoading local model...", file=sys.stderr)
        provider: LocalProvider | NvidiaProvider = LocalProvider(
            str(model_file), context_window(config["modelTier"])
        )
    else:
        if not json_output:
            print("Connecting to NVIDIA...", file=sys.stderr)
        provider = _nvidia_provider(config["remoteModel"], project_root)
    try:
        return _run_with_provider(
            task,
            project_root,
            session,
            config,
            provider,
            json_output=json_output,
            one_shot=one_shot,
        )
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()


def _add_commands(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Execute a task with the local agent")
    run_parser.add_argument("task", nargs="?", help="Task description")
    run_parser.add_argument("--json", action="store_true", help="Emit newline-delimited JSON")
    run_parser.add_argument("--session", help="Resume a project session")

    subparsers.add_parser("analyze", help="Summarize repository files").add_argument(
        "path", nargs="?", default=".", help="Repository path"
    )
    subparsers.add_parser("config", help="Show or update Panther configuration").add_argument(
        "--permissions", action="store_true", help="Set the permission mode"
    )
    config_parser = subparsers.choices["config"]
    config_parser.add_argument(
        "--provider", choices=["local", "nvidia"], help="Set the active model provider"
    )
    config_parser.add_argument(
        "--model", help="Set the NVIDIA model ID used for remote requests"
    )
    subparsers.add_parser("model", help="Manage local model downloads").add_argument(
        "action", choices=["list", "switch"]
    )
    model_parser = subparsers.choices["model"]
    model_parser.add_argument("tier", nargs="?", choices=["small", "large"])
    model_parser.add_argument("--yes", action="store_true", help="Confirm deletion of the other model")

    session_parser = subparsers.add_parser("session", help="Manage project sessions")
    session_parser.add_argument("action", nargs="?", default="list", choices=["list", "show", "delete"])
    session_parser.add_argument("id", nargs="?")

    subparsers.add_parser("index", help="Build a local file inventory").add_argument(
        "--path", default=".", help="Project root"
    )
    subparsers.add_parser("skills", help="List discovered SKILL.md files")
    auth_parser = subparsers.add_parser("auth", help="Manage provider credentials")
    auth_parser.add_argument("action", choices=["add", "remove", "status"])
    auth_parser.add_argument("provider", choices=["nvidia"])
    nvidia_parser = subparsers.add_parser(
        "nvidia", help="Inspect NVIDIA API access"
    )
    nvidia_parser.add_argument("action", choices=["models"])


def _run_command(args: argparse.Namespace, root: Path) -> int:
    config = load_config(root)
    manager = _manager(root)
    task = args.task
    if not task:
        if not sys.stdin.isatty():
            raise ValueError("A task argument is required when stdin is not interactive.")
        task = input("Panther task: ").strip()
    if not task:
        raise ValueError("No task provided.")
    session = _get_session(manager, args.session)
    if args.json:
        _write_json({"type": "start", "task": task, "session": session["id"]})
    else:
        print(f"Session: {session['id']}\n")
    try:
        result = _execute_task(task, root, session, config, json_output=args.json)
    except Exception as exc:
        if args.json:
            _write_json({"type": "error", "message": str(exc)})
        raise
    if args.json:
        _write_json({"type": "complete", "result": result})
    else:
        print("\n\nTask complete.")
    return 0


def _auth_command(args: argparse.Namespace, root: Path) -> int:
    try:
        keyring = import_module("keyring")
    except ImportError as exc:
        raise RuntimeError(
            "Credential storage requires Panther's keyring dependency. "
            "Reinstall Panther with `python -m pip install --upgrade panther-cli`."
        ) from exc
    if args.action == "add":
        api_key = getpass.getpass("NVIDIA API key (input hidden): ").strip()
        if not api_key:
            raise ValueError("NVIDIA API key cannot be empty.")
        try:
            keyring.set_password(
                NVIDIA_KEYRING_SERVICE,
                NVIDIA_KEYRING_USERNAME,
                api_key,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not save NVIDIA credentials to the OS keyring: {exc}"
            ) from exc
        config = load_config(root)
        config["activeProvider"] = "nvidia"
        save_config(config, root)
        print("NVIDIA credentials saved to the OS keyring; NVIDIA is now active.")
        return 0
    if args.action == "remove":
        try:
            keyring.delete_password(
                NVIDIA_KEYRING_SERVICE,
                NVIDIA_KEYRING_USERNAME,
            )
        except keyring.errors.PasswordDeleteError:
            print("No NVIDIA credentials were stored.")
        except Exception as exc:
            raise RuntimeError(
                f"Could not remove NVIDIA credentials from the OS keyring: {exc}"
            ) from exc
        config = load_config(root)
        if config["activeProvider"] == "nvidia" and not _nvidia_api_key(root):
            config["activeProvider"] = "local"
            save_config(config, root)
        print("NVIDIA keyring credentials removed.")
        return 0
    stored = _nvidia_api_key(root) or keyring.get_password(
        NVIDIA_KEYRING_SERVICE, NVIDIA_KEYRING_USERNAME
    )
    print("NVIDIA API key: configured" if stored else "NVIDIA API key: not configured")
    return 0


def _repl(
    root: Path,
    session_id: str | None = None,
    *,
    show_banner: bool = True,
) -> int:
    config = load_config(root)
    manager = _manager(root)
    session = _get_session(manager, session_id)
    _print_banner(show_banner=show_banner)
    print(f"Panther session {session['id']} (type 'exit' to quit)")
    provider: LocalProvider | NvidiaProvider | None = None
    try:
        while True:
            try:
                task = input("panther> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if task.lower() in {"exit", "quit"}:
                return 0
            if not task:
                continue
            try:
                if provider is None:
                    if config["activeProvider"] == "local":
                        model_file = ensure_model(config["modelTier"], _progress)
                        provider = LocalProvider(
                            str(model_file),
                            context_window(config["modelTier"]),
                        )
                    else:
                        provider = _nvidia_provider(config["remoteModel"], root)
                _run_with_provider(
                    task,
                    root,
                    session,
                    config,
                    provider,
                    json_output=False,
                    one_shot=False,
                )
                print("\n")
            except Exception as exc:
                print(f"\nPanther error: {exc}", file=sys.stderr)
    finally:
        if provider is not None:
            close = getattr(provider, "close", None)
            if callable(close):
                close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="panther",
        description="A local-first coding agent powered by Python.",
    )
    parser.add_argument("--version", action="version", version="panther-cli 0.1.0")
    parser.add_argument(
        "--no-banner", action="store_true", help="Hide the interactive startup banner"
    )
    _add_commands(parser)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    try:
        if args.command is None:
            return _repl(root, show_banner=not args.no_banner)
        if args.command == "run":
            return _run_command(args, root)
        if args.command == "analyze":
            target = Path(args.path).resolve()
            print(_analyze_repository(target))
            return 0
        if args.command == "config":
            config = load_config(root)
            if args.permissions:
                print("Permission modes: always_ask, ask_when_risky, never_ask")
                selection = input(f"Permission mode [{config['permissionMode']}]: ").strip()
                if selection:
                    config["permissionMode"] = selection
                    print(f"Saved: {save_config(config, root)}")
            elif args.provider:
                config["activeProvider"] = args.provider
                print(f"Saved: {save_config(config, root)}")
            elif args.model:
                config["remoteModel"] = args.model
                print(f"Saved: {save_config(config, root)}")
            else:
                if not config_path().exists():
                    save_config(config)
                print(json.dumps(config, indent=2))
            return 0
        if args.command == "auth":
            return _auth_command(args, root)
        if args.command == "nvidia":
            provider = _nvidia_provider(load_config(root)["remoteModel"], root)
            for model in provider.list_models():
                print(model)
            return 0
        if args.command == "model":
            if args.action == "list":
                for model in list_models():
                    status = "installed" if model["downloaded"] else "not downloaded"
                    print(f"{model['tier']}: {model['name']} ({status})")
                return 0
            if not args.tier:
                raise ValueError("Specify a model tier: small or large.")
            old_model = "large" if args.tier == "small" else "small"
            installed = next(item["downloaded"] for item in list_models() if item["tier"] == old_model)
            confirm = args.yes
            if installed and not confirm:
                confirm = input(f"Delete the installed {old_model} model after switching? [y/N] ").strip().lower() == "y"
            path = switch_model(args.tier, confirm_delete=confirm, on_progress=_progress)
            config = load_config(root)
            config["modelTier"] = args.tier
            save_config(config, root)
            print(f"\nActive model tier set to {args.tier}: {path}")
            return 0
        if args.command == "session":
            manager = _manager(root)
            if args.action == "list":
                for session in manager.list()[:20]:
                    last = session["history"][-1]["content"][:80] if session["history"] else "(empty)"
                    print(f"{session['id']}  {session['updatedAt'][:10]}  {last}")
                return 0
            if not args.id:
                raise ValueError(f"session {args.action} requires an ID")
            if args.action == "delete":
                print("Deleted." if manager.delete(args.id) else "Session not found.")
                return 0
            session = manager.load(args.id)
            if session is None:
                raise ValueError("Session not found.")
            print(json.dumps(session, indent=2))
            return 0
        if args.command == "index":
            project_root = Path(args.path).resolve()
            files = [str(path.relative_to(project_root)) for path in _repository_files(project_root)]
            index_path = project_root / ".panther" / "index.json"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8")
            print(f"Indexed {len(files)} files into {index_path}")
            return 0
        if args.command == "skills":
            skill_roots = [root / ".panther" / "skills", panther_dir() / "skills"]
            skills = sorted(path for base in skill_roots if base.exists() for path in base.rglob("SKILL.md"))
            if not skills:
                print("No skills found.")
            for skill in skills:
                print(f"{skill.parent.name}: {skill}")
            return 0
        parser.print_help()
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Panther error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
