"""Headless shared modules must not load PyQt6 at import time.

AGENTS.md requires PyQt6 to stay optional and lazily imported so that
headless CI and the docs builder do not need the GUI extras. The
``loop_nullspace`` plot model reaches ``plot_style`` through
``force_overlay``; ``plot_style`` used to import its Qt widgets eagerly,
which dragged PyQt6 into every headless consumer.

The lazy re-exports must still resolve, so the public names keep working.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]

_HEADLESS_MODULES = (
    "src.shared.python.loop_nullspace.plot_model",
    "src.shared.python.loop_nullspace.contracts",
    "src.shared.python.force_overlay",
    "src.shared.python.plot_style",
)

# matplotlib itself is a real dependency of plot_style's color layer
# (``matplotlib.colors``); only Qt and the 3D renderer toolkit are optional.
_FORBIDDEN_PREFIXES = ("PyQt6", "mpl_toolkits")


def _loaded_forbidden_modules(module: str) -> list[str]:
    """Import *module* in a fresh interpreter; return forbidden modules loaded."""
    probe = (
        "import importlib, json, sys\n"
        f"importlib.import_module({module!r})\n"
        f"prefixes = {_FORBIDDEN_PREFIXES!r}\n"
        "print(json.dumps(sorted(\n"
        "    m for m in sys.modules if m.split('.')[0] in prefixes\n"
        ")))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", _HEADLESS_MODULES)
def test_headless_module_does_not_import_gui_stack(module: str) -> None:
    """Importing a headless module leaves PyQt6 and mplot3d unloaded."""
    loaded = _loaded_forbidden_modules(module)
    assert loaded == [], f"{module} eagerly imported {loaded}"


def test_plot_style_lazy_exports_resolve() -> None:
    """Lazily re-exported names resolve to the defining modules' objects."""
    plot_style = pytest.importorskip("src.shared.python.plot_style")
    from src.shared.python.plot_style.renderers.matplotlib import (
        MatplotlibMarkerRenderer,
    )

    assert plot_style.MatplotlibMarkerRenderer is MatplotlibMarkerRenderer
    assert "MatplotlibMarkerRenderer" in dir(plot_style)
    with pytest.raises(AttributeError):
        _ = plot_style.NoSuchPlotStyleName


def test_plot_style_widget_exports_resolve() -> None:
    """Widget names stay importable from the package root when Qt exists."""
    try:
        import PyQt6.QtWidgets  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    from src.shared.python.plot_style import ColorPicker, MarkerStylePicker
    from src.shared.python.plot_style.widgets import (
        ColorPicker as WidgetColorPicker,
    )
    from src.shared.python.plot_style.widgets import (
        MarkerStylePicker as WidgetMarkerStylePicker,
    )

    assert ColorPicker is WidgetColorPicker
    assert MarkerStylePicker is WidgetMarkerStylePicker


def test_plot_style_import_with_strict_meta_path_finder() -> None:
    """Importing plot_style does not invoke find_spec for PyQt6."""
    probe = (
        "import importlib.abc, sys\n"
        "class Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'PyQt6' or fullname.startswith('PyQt6.'):\n"
        "            raise AssertionError(f'native import: {fullname}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import src.shared.python.plot_style\n"
        "print('STRICT_BLOCKER_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "STRICT_BLOCKER_OK" in result.stdout
