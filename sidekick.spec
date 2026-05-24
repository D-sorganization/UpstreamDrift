# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Sidekick standalone binary.
# Issue: #5985 (T7) — one-file binary for macOS / Linux / Windows.
#
# Build with:
#   python scripts/packaging/build_sidekick_binary.py
#
# Or directly:
#   pyinstaller sidekick.spec
#
# Size budget: MAX_MB = 250  (enforced by build_sidekick_binary.py)
# Growth > 10 % between releases requires PR justification.

MAX_MB = 250

import os
import sys
from pathlib import Path

root = Path(SPECPATH)  # noqa: F821  (defined by PyInstaller)

# ---------------------------------------------------------------------------
# Platform-specific icon
# ---------------------------------------------------------------------------
_icons = {
    "darwin": str(root / "src" / "shared" / "assets" / "sidekick.icns"),
    "win32": str(root / "src" / "shared" / "assets" / "sidekick.ico"),
}
icon = _icons.get(sys.platform)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(  # noqa: F821
    [str(root / "src" / "shared" / "python" / "sidekick" / "__main__.py")],
    pathex=[
        str(root / "src" / "shared" / "python"),
        str(root / "src"),
        str(root),
    ],
    binaries=[],
    datas=[
        # Theme assets
        (str(root / "src" / "shared" / "python" / "theme"), "theme"),
    ],
    hiddenimports=[
        "sidekick",
        "sidekick.standalone",
        "sidekick.standalone.runner",
        "sidekick.standalone.preferences",
        "sidekick.standalone.onboarding",
        "sidekick.calculators",
        "sidekick.process_calculators",
        "sidekick.utils",
        "sidekick.theme",
    ],
    excludes=[
        # Heavy physics engines — never ship in the Sidekick binary
        "pybullet",
        "mujoco",
        "pydrake",
        # Heavy ML/training deps
        "torch",
        "tensorflow",
        "stable_baselines3",
        "gymnasium",
        # Build/dev tooling
        "pytest",
        "ruff",
        "mypy",
        "IPython",
        # Rust extension (not needed for standalone GUI/CLI)
        "upstream_physics",
        "upstream_mocap_preproc",
        "upstream_mocap_io",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="sidekick",
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
    icon=icon,
    onefile=True,
)
