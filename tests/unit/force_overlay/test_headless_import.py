"""Tests that force_overlay imports cleanly without GUI/engine dependencies (#11286)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_force_overlay_headless_import_purity() -> None:
    """force_overlay must not import PyQt6, matplotlib.pyplot, mujoco, pydrake, pinocchio, or opensim."""
    script = """
import sys

BANNED_PREFIXES = (
    "PyQt6",
    "matplotlib.pyplot",
    "mujoco",
    "pydrake",
    "pinocchio",
    "opensim",
)

class HeadlessImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        for banned in BANNED_PREFIXES:
            if fullname == banned or fullname.startswith(banned + "."):
                raise ImportError(f"Banned package imported: {fullname}")
        return None

sys.meta_path.insert(0, HeadlessImportGuard())

import src.shared.python.force_overlay as fo
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    ForceTorqueProvider,
    ForceTorqueSeries,
    OverlayWrench,
    WrenchKind,
    read_force_torque_frame,
)

# Exercise basic instantiation
w = OverlayWrench(
    kind=WrenchKind.CONTACT,
    label="contact:ground",
    body="foot",
    point_m=(0.0, 0.0, 0.0),
    force_n=(0.0, 0.0, 100.0),
    source="test",
)
frame = ForceTorqueFrame(
    time_s=0.0,
    engine="test_engine",
    wrenches=(w,),
)
series = ForceTorqueSeries(frames=(frame,))
assert series.frame_at(0.0) is frame

# Verify none of the banned modules are in sys.modules
for mod in sys.modules:
    for banned in BANNED_PREFIXES:
        assert not (mod == banned or mod.startswith(banned + ".")), f"Banned module in sys.modules: {mod}"

print("HEADLESS_IMPORT_OK")
"""
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Headless import failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "HEADLESS_IMPORT_OK" in result.stdout
