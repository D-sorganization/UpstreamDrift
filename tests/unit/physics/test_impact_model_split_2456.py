"""Contract tests for #2456: impact_model.py split.

Tests run red before the split and green after.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).parents[3]
PHYSICS_DIR = REPO / "src/shared/python/physics"
LOC_BUDGET = 700


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestImpactModelSplitStructure:
    """Split modules must exist after refactor."""

    @pytest.mark.unit
    def test_impact_physics_module_exists(self) -> None:
        assert (PHYSICS_DIR / "_impact_physics.py").exists()

    @pytest.mark.unit
    def test_impact_recorder_module_exists(self) -> None:
        assert (PHYSICS_DIR / "_impact_recorder.py").exists()


class TestImpactModelFileSizes:
    """Each file must be under 700 LOC after split."""

    @pytest.mark.unit
    def test_impact_model_split_2456_coordinator_loc(self) -> None:
        loc = _count_lines(PHYSICS_DIR / "impact_model" / "solver.py") + _count_lines(
            PHYSICS_DIR / "impact_model" / "utils.py"
        )
        assert loc <= LOC_BUDGET, f"impact_model.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_impact_physics_loc(self) -> None:
        loc = _count_lines(PHYSICS_DIR / "_impact_physics.py")
        assert loc <= LOC_BUDGET, (
            f"_impact_physics.py has {loc} LOC; budget {LOC_BUDGET}"
        )

    @pytest.mark.unit
    def test_impact_recorder_loc(self) -> None:
        loc = _count_lines(PHYSICS_DIR / "_impact_recorder.py")
        assert loc <= LOC_BUDGET, (
            f"_impact_recorder.py has {loc} LOC; budget {LOC_BUDGET}"
        )


class TestImpactModelPublicAPI:
    """Public API must remain importable from impact_model (backward compat)."""

    @pytest.mark.unit
    def test_import_impact_model_type(self) -> None:
        from src.shared.python.physics.impact_model import ImpactModelType

        assert ImpactModelType is not None

    @pytest.mark.unit
    def test_import_pre_impact_state(self) -> None:
        from src.shared.python.physics.impact_model import PreImpactState

        assert PreImpactState is not None

    @pytest.mark.unit
    def test_import_post_impact_state(self) -> None:
        from src.shared.python.physics.impact_model import PostImpactState

        assert PostImpactState is not None

    @pytest.mark.unit
    def test_import_impact_parameters(self) -> None:
        from src.shared.python.physics.impact_model import ImpactParameters

        assert ImpactParameters is not None

    @pytest.mark.unit
    def test_import_rigid_body_impact_model(self) -> None:
        from src.shared.python.physics.impact_model import RigidBodyImpactModel

        assert RigidBodyImpactModel is not None

    @pytest.mark.unit
    def test_import_spring_damper_impact_model(self) -> None:
        from src.shared.python.physics.impact_model import SpringDamperImpactModel

        assert SpringDamperImpactModel is not None

    @pytest.mark.unit
    def test_import_finite_time_impact_model(self) -> None:
        from src.shared.python.physics.impact_model import FiniteTimeImpactModel

        assert FiniteTimeImpactModel is not None

    @pytest.mark.unit
    def test_import_compute_gear_effect_spin(self) -> None:
        from src.shared.python.physics.impact_model import compute_gear_effect_spin

        assert callable(compute_gear_effect_spin)

    @pytest.mark.unit
    def test_import_validate_energy_balance(self) -> None:
        from src.shared.python.physics.impact_model import validate_energy_balance

        assert callable(validate_energy_balance)

    @pytest.mark.unit
    def test_import_create_impact_model(self) -> None:
        from src.shared.python.physics.impact_model import create_impact_model

        assert callable(create_impact_model)

    @pytest.mark.unit
    def test_import_impact_event(self) -> None:
        from src.shared.python.physics.impact_model import ImpactEvent

        assert ImpactEvent is not None

    @pytest.mark.unit
    def test_import_impact_recorder(self) -> None:
        from src.shared.python.physics.impact_model import ImpactRecorder

        assert ImpactRecorder is not None

    @pytest.mark.unit
    def test_import_impact_solver_api(self) -> None:
        from src.shared.python.physics.impact_model import ImpactSolverAPI

        assert ImpactSolverAPI is not None
