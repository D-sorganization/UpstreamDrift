# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Sidekick standalone binary.
# Issue: #5985 (T7) — one-file binary built separately on each native OS.
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

import sys
from pathlib import Path

root = Path(SPECPATH)  # noqa: F821  (defined by PyInstaller)
canonical_tools_python = (
    root / "vendor" / "ud-tools" / "src" / "shared" / "python"
)
canonical_tools_src = root / "vendor" / "ud-tools" / "src"
local_python = root / "src" / "shared" / "python"
canonical_sidekick_entrypoint = (
    canonical_tools_python / "sidekick" / "__main__.py"
)
binary_entrypoint = root / "scripts" / "packaging" / "sidekick_binary_entrypoint.py"
sys.path.insert(0, str(canonical_tools_python))

if not canonical_sidekick_entrypoint.is_file():
    raise FileNotFoundError(
        "Canonical Sidekick entrypoint is unavailable; initialize the pinned "
        "vendor/ud-tools submodule recursively before building"
    )
if not binary_entrypoint.is_file():
    raise FileNotFoundError(f"Sidekick binary adapter is unavailable: {binary_entrypoint}")

# ---------------------------------------------------------------------------
# Platform-specific icon
# ---------------------------------------------------------------------------
_icons = {
    "win32": str(root / "src" / "launchers" / "assets" / "golf_suite_unified.ico"),
}
icon = _icons.get(sys.platform)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(  # noqa: F821
    [str(binary_entrypoint)],
    pathex=[
        str(canonical_tools_python),
        str(canonical_tools_src),
        str(local_python),
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
        "sidekick.standalone.session_store",
        "sidekick.standalone.window",
        "sidekick.persistence",
        "sidekick.persistence.schema",
        "sidekick.persistence.state_profile",
        "sidekick.calculators",
        "sidekick.process_calculators",
        "sidekick.utils",
        "sidekick.theme",
    ],
    excludes=[
        # PyQt6 is canonical; PyInstaller cannot freeze multiple Qt bindings.
        "PyQt5",
        "PySide2",
        "PySide6",
        # Not imported by Sidekick; avoid collecting transitive tooling hooks.
        "docutils",
        "sklearn",
        "sphinx",
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
