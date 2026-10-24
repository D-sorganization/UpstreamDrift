"""The force-overlay contracts import without matplotlib (#13450 drift gate).

The Drake URDF regeneration gate installs only PyYAML, numpy, defusedxml,
pandas and scipy. ``urdf_exporter`` imports ``grip_contact``, which reaches
``force_overlay`` through ``biomechanics.grip_wrench``; the glyph builders
(and their matplotlib-backed ``plot_style``) must load only on first use.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

_BLOCK_MATPLOTLIB = textwrap.dedent(
    """
    import sys

    class _Block:
        def find_spec(self, name, path=None, target=None):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ModuleNotFoundError("No module named 'matplotlib'")
            return None

    sys.meta_path.insert(0, _Block())
    sys.path.insert(0, "src")  # as scripts/build_humanoid_models.py does
    import src.shared.python.force_overlay as fo
    import tools.model_converter.urdf_exporter  # noqa: F401

    assert "src.shared.python.force_overlay.glyphs" not in sys.modules
    assert fo.OverlayWrench is not None
    try:
        fo.build_glyphs
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("glyphs must need matplotlib on first access")
    """
)


@pytest.mark.unit
def test_contracts_and_urdf_exporter_import_without_matplotlib() -> None:
    """Importing the exporter chain never loads the glyph module."""
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCK_MATPLOTLIB],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


@pytest.mark.unit
def test_glyph_exports_resolve_lazily() -> None:
    """Lazy exports are the same objects as the ``glyphs`` module's."""
    import src.shared.python.force_overlay as fo
    from src.shared.python.force_overlay import glyphs

    for name in ("ArrowGlyph", "build_glyphs", "scale_for_view", "LegendSpec"):
        assert getattr(fo, name) is getattr(glyphs, name)
        assert name in fo.__all__
    with pytest.raises(AttributeError):
        fo.no_such_export  # noqa: B018
