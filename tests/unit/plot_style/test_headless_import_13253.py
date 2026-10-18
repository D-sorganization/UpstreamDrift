"""Subprocess and contract tests for headless plot_style purity (#13253).

Validates that importing headless modules (plot_style, force_overlay,
loop_nullspace.plot_model) does not eagerly load PyQt6 or mpl_toolkits.mplot3d,
and confirms that all public plot_style symbols resolve via lazy __getattr__.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _run_python_script(script: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[3]
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_plot_style_headless_purity_subprocess() -> None:
    """Importing plot_style must not load PyQt6 or mpl_toolkits into sys.modules."""
    script = """
import sys
import src.shared.python.plot_style

pyqt_loaded = [m for m in sys.modules if m == "PyQt6" or m.startswith("PyQt6.")]
mpl_toolkits_loaded = [m for m in sys.modules if m == "mpl_toolkits" or m.startswith("mpl_toolkits.")]

assert not pyqt_loaded, f"PyQt6 modules eagerly loaded: {pyqt_loaded}"
assert not mpl_toolkits_loaded, f"mpl_toolkits modules eagerly loaded: {mpl_toolkits_loaded}"
print("PURITY_OK")
"""
    res = _run_python_script(script)
    assert res.returncode == 0, f"Script failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "PURITY_OK" in res.stdout


def test_force_overlay_headless_purity_subprocess() -> None:
    """Importing force_overlay must not load PyQt6 or mpl_toolkits into sys.modules."""
    script = """
import sys
import src.shared.python.force_overlay

pyqt_loaded = [m for m in sys.modules if m == "PyQt6" or m.startswith("PyQt6.")]
mpl_toolkits_loaded = [m for m in sys.modules if m == "mpl_toolkits" or m.startswith("mpl_toolkits.")]

assert not pyqt_loaded, f"PyQt6 modules eagerly loaded: {pyqt_loaded}"
assert not mpl_toolkits_loaded, f"mpl_toolkits modules eagerly loaded: {mpl_toolkits_loaded}"
print("PURITY_OK")
"""
    res = _run_python_script(script)
    assert res.returncode == 0, f"Script failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "PURITY_OK" in res.stdout


def test_plot_model_headless_purity_subprocess() -> None:
    """Importing loop_nullspace.plot_model must not load PyQt6 or mpl_toolkits."""
    script = """
import sys
import src.shared.python.loop_nullspace.plot_model

pyqt_loaded = [m for m in sys.modules if m == "PyQt6" or m.startswith("PyQt6.")]
mpl_toolkits_loaded = [m for m in sys.modules if m == "mpl_toolkits" or m.startswith("mpl_toolkits.")]

assert not pyqt_loaded, f"PyQt6 modules eagerly loaded: {pyqt_loaded}"
assert not mpl_toolkits_loaded, f"mpl_toolkits modules eagerly loaded: {mpl_toolkits_loaded}"
print("PURITY_OK")
"""
    res = _run_python_script(script)
    assert res.returncode == 0, f"Script failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "PURITY_OK" in res.stdout


def test_lazy_attribute_resolution_and_dir() -> None:
    """Public symbols resolve lazily and appear in dir()."""
    script = """
import sys
import src.shared.python.plot_style as ps

# Verify initial purity
assert not any(m.startswith("PyQt6") for m in sys.modules)
assert not any(m.startswith("mpl_toolkits") for m in sys.modules)

# Verify dir() includes all lazy exports
d = dir(ps)
for lazy_name in (
    "MatplotlibMarkerRenderer",
    "PyQtGLMarkerRenderer",
    "ColorPicker",
    "ColormapPicker",
    "DataChannelEditor",
    "MarkerStylePicker",
):
    assert lazy_name in d, f"{lazy_name} missing from dir(ps)"

# Verify MatplotlibMarkerRenderer resolves and loads mplot3d
renderer_cls = ps.MatplotlibMarkerRenderer
assert renderer_cls is not None
assert any(m.startswith("mpl_toolkits") for m in sys.modules)

# Verify Qt widgets resolve and load PyQt6
cp_cls = ps.ColorPicker
assert cp_cls is not None
assert any(m.startswith("PyQt6") for m in sys.modules)

# Non-existent attribute raises AttributeError
try:
    _ = ps.NonExistentSymbol
    raise AssertionError("NonExistentSymbol should have raised AttributeError")
except AttributeError:
    pass

print("LAZY_RESOLUTION_OK")
"""
    res = _run_python_script(script)
    assert res.returncode == 0, f"Script failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "LAZY_RESOLUTION_OK" in res.stdout


def test_every_public_name_in_all_resolves() -> None:
    """Every name declared in plot_style.__all__ must resolve without AttributeError."""
    import src.shared.python.plot_style as ps

    assert len(ps.__all__) > 0
    for name in ps.__all__:
        if name == "PyQtGLMarkerRenderer":
            # Optional pyqtgraph extra may not be installed; attribute access is tested in test_pyqtgl.py
            continue
        val = getattr(ps, name)
        assert val is not None, f"Resolved None for {name}"
