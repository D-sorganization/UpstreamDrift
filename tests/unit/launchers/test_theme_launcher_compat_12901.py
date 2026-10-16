"""Issue #12901: the launcher theme imports must resolve in a clean process."""

from __future__ import annotations

import enum
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]


@pytest.mark.unit
def test_launcher_theme_imports_resolve_in_subprocess() -> None:
    code = (
        "from src.shared.python.theme.launcher_compat import "
        "ThemePreset, apply_golf_suite_style\n"
        "from src.shared.python.theme import ThemeManager, FontManager\n"
        "assert ThemePreset.DARK.value == 'Dark'\n"
        "apply_golf_suite_style()\n"
    )
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(_REPO)}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.unit
def test_theme_preset_values_match_theme_names() -> None:
    from src.shared.python.theme.launcher_compat import ThemePreset

    assert issubclass(ThemePreset, enum.Enum)
    assert [p.value for p in ThemePreset] == ["Dark", "Light", "High Contrast"]


@pytest.mark.unit
def test_apply_golf_suite_style_sets_matplotlib_defaults() -> None:
    import matplotlib as mpl

    from src.shared.python.theme.launcher_compat import apply_golf_suite_style

    apply_golf_suite_style()
    assert mpl.rcParams["axes.grid"] is True


@pytest.mark.unit
def test_launcher_theme_uses_concrete_compat_module() -> None:
    text = (_REPO / "src/launchers/launcher_theme.py").read_text(encoding="utf-8")
    assert "theme import ThemeManager, ThemePreset" not in text
    assert "theme.launcher_compat import" in text


@pytest.mark.unit
def test_draggable_tabs_imports_theme_manager_from_concrete_module() -> None:
    text = (_REPO / "src/shared/python/gui_pkg/draggable_tabs.py").read_text(
        encoding="utf-8"
    )
    assert "from src.shared.python.theme import get_theme_manager" not in text
    from src.shared.python.theme.theme_manager import get_theme_manager

    assert callable(get_theme_manager)
