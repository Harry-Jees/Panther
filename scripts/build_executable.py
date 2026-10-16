#!/usr/bin/env python3
"""Cross-platform build script for packaging Panther as a standalone executable.

Usage:
    python scripts/build_executable.py [--clean] [--no-archive]

Supported platforms:
    - Windows -> dist/panther.exe (bundled into dist/panther-windows-x64.zip)
    - macOS   -> dist/panther (bundled into dist/panther-macos-<arch>.tar.gz)
    - Linux   -> dist/panther (bundled into dist/panther-linux-<arch>.tar.gz)
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC_FILE = ROOT / "panther.spec"
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"


def check_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("[!] PyInstaller is not installed. Installing it now...", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller>=6.5.0"])


def clean_previous_builds() -> None:
    print("[*] Cleaning build artifacts...")
    for directory in (BUILD_DIR, DIST_DIR):
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)


def build_binary() -> Path:
    check_pyinstaller()
    clean_previous_builds()

    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64"}:
        arch_label = "x64"
    elif machine in {"arm64", "aarch64"}:
        arch_label = "arm64"
    else:
        arch_label = machine

    print(f"[*] Building Panther executable for {platform.system()} ({arch_label})...")
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        str(SPEC_FILE),
    ]

    subprocess.check_call(cmd, cwd=ROOT)

    exe_name = "panther.exe" if system == "windows" else "panther"
    output_binary = DIST_DIR / exe_name

    if not output_binary.exists():
        raise RuntimeError(f"Build failed: expected executable not found at {output_binary}")

    print(f"[+] Successfully built: {output_binary}")

    # Ensure executable permissions on POSIX
    if system != "windows":
        output_binary.chmod(0o755)

    # Smoke test the built binary
    print("[*] Running binary smoke test...")
    test_run = subprocess.run(
        [str(output_binary), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if test_run.returncode != 0:
        raise RuntimeError(
            f"Smoke test failed with exit code {test_run.returncode}:\n{test_run.stderr}"
        )
    print(f"[+] Smoke test passed: {test_run.stdout.strip()}")

    return output_binary


def create_release_archive(binary: Path) -> Path:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch_label = "x64" if machine in {"amd64", "x86_64"} else ("arm64" if machine in {"arm64", "aarch64"} else machine)
    
    if system == "windows":
        archive_name = f"panther-windows-{arch_label}.zip"
        archive_path = DIST_DIR / archive_name
        print(f"[*] Creating release archive {archive_path.name}...")
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(binary, arcname=binary.name)
            for extra in ("README.md", "pyproject.toml"):
                extra_path = ROOT / extra
                if extra_path.exists():
                    zf.write(extra_path, arcname=extra)
    else:
        archive_name = f"panther-{system}-{arch_label}.tar.gz"
        archive_path = DIST_DIR / archive_name
        print(f"[*] Creating release archive {archive_path.name}...")
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(binary, arcname=binary.name)
            for extra in ("README.md", "pyproject.toml"):
                extra_path = ROOT / extra
                if extra_path.exists():
                    tf.add(extra_path, arcname=extra)

    print(f"[+] Release archive ready: {archive_path} ({archive_path.stat().st_size:,} bytes)")
    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Panther standalone executable")
    parser.add_argument("--no-archive", action="store_true", help="Skip creating release zip/tar.gz")
    args = parser.parse_args()

    binary = build_binary()
    if not args.no_archive:
        create_release_archive(binary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
