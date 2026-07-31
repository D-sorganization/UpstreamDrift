"""Contract tests for #2456: kinematic_forces split.

Originally these tests guarded a ``kinematic_forces.py`` module plus a set of
``_kfa_*`` mixins. That layout was later superseded by the
``kinematic_forces/`` *package*, and because a package shadows a same-named
module, the old files stopped executing entirely while these tests kept
asserting their existence (see #8021). The dead files have been removed; these
tests now guard the live package instead, preserving the original intent
(bounded module size + a stable public API).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
KF_DIR = REPO / "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf"
KF_PKG = KF_DIR / "kinematic_forces"
LOC_BUDGET_DATA = 400
LOC_BUDGET_COORDINATOR = 900

_mujoco_available = importlib.util.find_spec("mujoco") is not None


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestKinematicForcesSplitStructure:
    """Split modules must exist after refactor."""

    @pytest.mark.unit
    def test_package_is_a_package_not_a_module(self) -> None:
        assert KF_PKG.is_dir()
        assert (KF_PKG / "__init__.py").exists()

    @pytest.mark.unit
    def test_no_shadowed_sibling_module(self) -> None:
        """A same-named module beside the package would be dead code (#8021)."""
        shadowed = KF_DIR / "kinematic_forces.py"
        assert not shadowed.exists(), (
            f"{shadowed} is permanently shadowed by the kinematic_forces package "
            "and can never execute; do not re-add it."
        )

    @pytest.mark.unit
    def test_data_module_exists(self) -> None:
        assert (KF_PKG / "types.py").exists()
        assert (KF_PKG / "export.py").exists()


class TestKinematicForcesFileSizes:
    """Each file must be within LOC budget after split."""

    @pytest.mark.unit
    def test_kinematic_forces_split_2456_coordinator_loc(self) -> None:
        loc = _count_lines(KF_PKG / "analyzer.py")
        assert loc <= LOC_BUDGET_COORDINATOR, (
            f"kinematic_forces/analyzer.py has {loc} LOC; "
            f"budget {LOC_BUDGET_COORDINATOR}"
        )

    @pytest.mark.unit
    def test_data_module_loc(self) -> None:
        loc = _count_lines(KF_PKG / "types.py") + _count_lines(KF_PKG / "export.py")
        assert (
            loc <= LOC_BUDGET_DATA
        ), f"kinematic_forces data modules have {loc} LOC; budget {LOC_BUDGET_DATA}"


@pytest.mark.skipif(not _mujoco_available, reason="mujoco not installed")
class TestKinematicForcesPublicAPI:
    """Public API must remain importable from kinematic_forces (backward compat)."""

    @pytest.mark.unit
    def test_import_mj_data_context(self) -> None:
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.kinematic_forces import (
            MjDataContext,
        )

        assert MjDataContext is not None

    @pytest.mark.unit
    def test_import_kinematic_force_data(self) -> None:
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.kinematic_forces import (
            KinematicForceData,
        )

        assert KinematicForceData is not None

    @pytest.mark.unit
    def test_import_kinematic_force_analyzer(self) -> None:
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.kinematic_forces import (
            KinematicForceAnalyzer,
        )

        assert KinematicForceAnalyzer is not None
