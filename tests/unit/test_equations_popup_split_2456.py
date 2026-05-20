"""Contract tests for #2456: equations_popup.py split.

Tests run red before the split and green after.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
GUI_DIR = REPO / "src/shared/python/pendulum_simulator/gui"
LOC_BUDGET = 550


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestEquationsPopupSplitStructure:
    """Split modules must exist after refactor."""

    @pytest.mark.unit
    def test_css_module_exists(self) -> None:
        assert (GUI_DIR / "_equations_popup_css.py").exists()

    @pytest.mark.unit
    def test_dynamics_html_module_exists(self) -> None:
        assert (GUI_DIR / "_equations_popup_dynamics_html.py").exists()

    @pytest.mark.unit
    def test_jacobians_html_module_exists(self) -> None:
        assert (GUI_DIR / "_equations_popup_jacobians_html.py").exists()


class TestEquationsPopupFileSizes:
    """Each file must be under 550 LOC after split."""

    @pytest.mark.unit
    def test_equations_popup_split_2456_coordinator_loc(self) -> None:
        loc = _count_lines(GUI_DIR / "equations_popup.py")
        assert (
            loc <= LOC_BUDGET
        ), f"equations_popup.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_css_loc(self) -> None:
        loc = _count_lines(GUI_DIR / "_equations_popup_css.py")
        assert (
            loc <= LOC_BUDGET
        ), f"_equations_popup_css.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_dynamics_html_loc(self) -> None:
        loc = _count_lines(GUI_DIR / "_equations_popup_dynamics_html.py")
        assert (
            loc <= LOC_BUDGET
        ), f"_equations_popup_dynamics_html.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_jacobians_html_loc(self) -> None:
        loc = _count_lines(GUI_DIR / "_equations_popup_jacobians_html.py")
        assert (
            loc <= LOC_BUDGET
        ), f"_equations_popup_jacobians_html.py has {loc} LOC; budget {LOC_BUDGET}"


class TestEquationsPopupPublicAPI:
    """Public API must remain importable from equations_popup (backward compat)."""

    @pytest.mark.unit
    def test_import_equation_topic(self) -> None:
        from src.shared.python.pendulum_simulator.gui.equations_popup import (
            EquationTopic,
        )

        assert EquationTopic is not None

    @pytest.mark.unit
    def test_import_show_equations_popup(self) -> None:
        from src.shared.python.pendulum_simulator.gui.equations_popup import (
            show_equations_popup,
        )

        assert callable(show_equations_popup)
