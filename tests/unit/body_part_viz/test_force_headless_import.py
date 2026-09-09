"""Force data remain available without optional desktop rendering packages."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_force_contracts_do_not_import_rendering_packages():
    script = """
import sys
class NoRendering:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('matplotlib', 'PyQt6', 'pyqtgraph'):
            raise ImportError('Rendering package imported: ' + fullname)
sys.meta_path.insert(0, NoRendering())
from src.shared.python.body_part_viz import AxialLoadFrame, ForceColorScale
frame = AxialLoadFrame(0, {'link': 2}, 'Qualified headless source')
assert ForceColorScale(enabled=True, tension_limit_n=1).color(frame.values_n['link'], '') == '#0000ff'
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
