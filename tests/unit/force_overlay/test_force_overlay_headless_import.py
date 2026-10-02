"""Headless import verification for force_overlay package."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_force_overlay_package_headless_import():
    script = """
import sys
class NoRendering:
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('matplotlib', 'PyQt6', 'pyqtgraph', 'mujoco', 'pydrake', 'pinocchio', 'opensim')
        if fullname.split('.')[0] in blocked:
            raise ImportError('Heavy or rendering package imported: ' + fullname)
sys.meta_path.insert(0, NoRendering())
from src.shared.python.force_overlay import (
    WrenchKind,
    OverlayWrench,
    ForceTorqueFrame,
    ForceTorqueSeries,
    ForceTorqueProvider,
    read_force_torque_frame,
)
assert WrenchKind.JOINT_ACTUATOR.value == 'joint_actuator'
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Headless import failed: {result.stderr}"
