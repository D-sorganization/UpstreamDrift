"""Contract tests for #2456: kinematic_forces.py split.

Tests run red before the split and green after.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
KF_DIR = REPO / "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf"
LOC_BUDGET_DATA = 400
LOC_BUDGET_COORDINATOR = 900

_mujoco_available = importlib.util.find_spec("mujoco") is not None


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestKinematicForcesSplitStructure:
    """Split modules must exist after refactor."""

    @pytest.mark.unit
    def test_data_module_exists(self) -> None:
        assert (KF_DIR / "_kinematic_force_data.py").exists()


class TestKinematicForcesFileSizes:
    """Each file must be within LOC budget after split."""

    @pytest.mark.unit
    def test_coordinator_loc(self) -> None:
        loc = _count_lines(KF_DIR / "kinematic_forces.py")
        assert loc <= LOC_BUDGET_COORDINATOR, (
            f"kinematic_forces.py has {loc} LOC; budget {LOC_BUDGET_COORDINATOR}"
        )

    @pytest.mark.unit
    def test_data_module_loc(self) -> None:
        loc = _count_lines(KF_DIR / "_kinematic_force_data.py")
        assert loc <= LOC_BUDGET_DATA, (
            f"_kinematic_force_data.py has {loc} LOC; budget {LOC_BUDGET_DATA}"
        )
