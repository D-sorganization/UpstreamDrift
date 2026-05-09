"""Contract tests for #2456: grip_modelling_tab.py split.

Tests run red before the split and green after.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
GRIP_DIR = REPO / "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf"
LOC_BUDGET_WIDGETS = 400
LOC_BUDGET_COORDINATOR = 900

_mujoco_available = importlib.util.find_spec("mujoco") is not None


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestGripModellingSplitStructure:
    """Split modules must exist after refactor."""

    @pytest.mark.unit
    def test_widgets_module_exists(self) -> None:
        assert (GRIP_DIR / "_grip_modelling_widgets.py").exists()


class TestGripModellingFileSizes:
    """Each file must be within LOC budget after split."""

    @pytest.mark.unit
    def test_coordinator_loc(self) -> None:
        loc = _count_lines(GRIP_DIR / "grip_modelling_tab.py")
        assert loc <= LOC_BUDGET_COORDINATOR, (
            f"grip_modelling_tab.py has {loc} LOC; budget {LOC_BUDGET_COORDINATOR}"
        )

    @pytest.mark.unit
    def test_widgets_loc(self) -> None:
        loc = _count_lines(GRIP_DIR / "_grip_modelling_widgets.py")
        assert loc <= LOC_BUDGET_WIDGETS, (
            f"_grip_modelling_widgets.py has {loc} LOC; budget {LOC_BUDGET_WIDGETS}"
        )
