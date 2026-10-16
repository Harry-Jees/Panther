# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification file for Panther CLI.

Builds a standalone single-file binary for Windows (panther.exe)
and macOS (panther).
"""

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hidden_imports = [
    "keyring",
    "keyring.backends",
    "keyring.backends.Windows",
    "keyring.backends.macOS",
    "keyring.backends.SecretService",
    "dotenv",
    "prompt_toolkit",
    "prompt_toolkit.formatted_text",
    "prompt_toolkit.shortcuts",
    "prompt_toolkit.key_binding",
    "prompt_toolkit.output.win32",
    "prompt_toolkit.output.vt100",
] + collect_submodules("panther")

datas = [
    ("README.md", "."),
    ("pyproject.toml", "."),
]

a = Analysis(
    ["panther/__main__.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "scipy", "PIL", "matplotlib", "customtkinter", "PyQt6", "pandas", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="panther",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
