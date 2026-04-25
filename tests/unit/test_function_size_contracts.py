"""
Contract tests: enforce <= 50-LOC function-size budget on nominated functions.

These tests use AST inspection so no imports of the target modules are needed.
They serve as regression gates: if a function grows beyond the budget the test
turns red immediately — before runtime or coverage issues surface.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent.parent


def _func_loc(filepath: Path, func_name: str) -> int | None:
    """Return the LOC of *func_name* in *filepath* via AST, or None if not found."""
    tree = ast.parse(filepath.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
        ):
            return node.end_lineno - node.lineno
    return None


LOC_BUDGET = 50


@pytest.mark.unit
class TestAnalyticalFkJacobiansJax:
    """analytical_fk_jacobians_jax must stay within the LOC budget."""

    _FILE = REPO / "src/shared/python/pendulum_simulator/physics_golfer_jax.py"
    _FUNC = "analytical_fk_jacobians_jax"

    def test_function_exists(self) -> None:
        loc = _func_loc(self._FILE, self._FUNC)
        assert loc is not None, f"{self._FUNC} not found in {self._FILE}"

    def test_loc_within_budget(self) -> None:
        loc = _func_loc(self._FILE, self._FUNC)
        assert loc is not None
        assert loc <= LOC_BUDGET, (
            f"{self._FUNC} is {loc} LOC — exceeds budget of {LOC_BUDGET}. "
            "Extract helper functions."
        )


@pytest.mark.unit
class TestBuildQtWindow:
    """_build_qt_window must stay within the LOC budget."""

    _FILE = REPO / "src/launchers/cross_engine_dashboard.py"
    _FUNC = "_build_qt_window"

    def test_function_exists(self) -> None:
        loc = _func_loc(self._FILE, self._FUNC)
        assert loc is not None, f"{self._FUNC} not found in {self._FILE}"

    def test_loc_within_budget(self) -> None:
        loc = _func_loc(self._FILE, self._FUNC)
        assert loc is not None
        assert loc <= LOC_BUDGET, (
            f"{self._FUNC} is {loc} LOC — exceeds budget of {LOC_BUDGET}. "
            "Extract helper functions."
        )


@pytest.mark.unit
class TestBuildTriplePanel:
    """build_triple_panel must stay within the LOC budget."""

    _FILE = REPO / "src/shared/python/pendulum_simulator/gui/panel_builders.py"
    _FUNC = "build_triple_panel"

    def test_function_exists(self) -> None:
        loc = _func_loc(self._FILE, self._FUNC)
        assert loc is not None, f"{self._FUNC} not found in {self._FILE}"

    def test_loc_within_budget(self) -> None:
        loc = _func_loc(self._FILE, self._FUNC)
        assert loc is not None
        assert loc <= LOC_BUDGET, (
            f"{self._FUNC} is {loc} LOC — exceeds budget of {LOC_BUDGET}. "
            "Extract helper class."
        )


@pytest.mark.unit
class TestBuildGolferPanel:
    """build_golfer_panel must stay within the LOC budget."""

    _FILE = REPO / "src/shared/python/pendulum_simulator/gui/panel_builders.py"
    _FUNC = "build_golfer_panel"

    def test_function_exists(self) -> None:
        loc = _func_loc(self._FILE, self._FUNC)
        assert loc is not None, f"{self._FUNC} not found in {self._FILE}"

    def test_loc_within_budget(self) -> None:
        loc = _func_loc(self._FILE, self._FUNC)
        assert loc is not None
        assert loc <= LOC_BUDGET, (
            f"{self._FUNC} is {loc} LOC — exceeds budget of {LOC_BUDGET}. "
            "Extract helper class."
        )


@pytest.mark.unit
class TestBuildOverlaySection:
    """_build_overlay_section must stay within the LOC budget."""

    _FILE = REPO / "src/shared/python/pendulum_simulator/gui/toolstrip_widget.py"
    _FUNC = "_build_overlay_section"

    def test_function_exists(self) -> None:
        loc = _func_loc(self._FILE, self._FUNC)
        assert loc is not None, f"{self._FUNC} not found in {self._FILE}"

    def test_loc_within_budget(self) -> None:
        loc = _func_loc(self._FILE, self._FUNC)
        assert loc is not None
        assert loc <= LOC_BUDGET, (
            f"{self._FUNC} is {loc} LOC — exceeds budget of {LOC_BUDGET}. "
            "Extract helper methods."
        )
