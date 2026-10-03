"""Force overlay contract remains importable without GUI or physics engine packages."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_force_overlay_does_not_import_gui_or_physics_packages():
    script = """
import sys
class NoForbiddenPackages:
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root in ('matplotlib', 'PyQt6', 'pyqtgraph', 'mujoco', 'pydrake', 'pinocchio', 'opensim'):
            raise ImportError(f'Forbidden package imported: {fullname}')
sys.meta_path.insert(0, NoForbiddenPackages())
from src.shared.python.force_overlay import (
    ArrowGlyph,
    ForceGlyphStyle,
    ForceTorqueFrame,
    ForceTorqueProvider,
    ForceTorqueSeries,
    GlyphSet,
    LegendSpec,
    OverlayWrench,
    TorqueArcGlyph,
    WrenchKind,
    build_glyphs,
    read_force_torque_frame,
    scale_for_view,
)
w = OverlayWrench(
    kind=WrenchKind.CONTACT,
    label="contact:ground",
    body="foot",
    point_m=(0.0, 0.0, 0.0),
    force_n=(0.0, 0.0, 100.0),
    torque_nm=None,
    source="headless_test",
)
f = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(w,))
assert f.engine == "test"
glyphs = build_glyphs(f)
assert len(glyphs.arrows) == 1
scaled = scale_for_view(glyphs, 1.5)
assert scaled.arrows[0].magnitude == glyphs.arrows[0].magnitude
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Import failed or imported forbidden packages:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
