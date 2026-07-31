"""Contract tests for #2456: data_fitting.py split.

Tests run red before the split and green after.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
VAL_DIR = REPO / "src/shared/python/validation_pkg"
LOC_BUDGET = 600


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestDataFittingSplitStructure:
    """Split modules must exist after refactor."""

    @pytest.mark.unit
    def test_data_fitting_split_2456_models_module_exists(self) -> None:
        assert (VAL_DIR / "_data_fitting_models.py").exists()

    @pytest.mark.unit
    def test_solvers_module_exists(self) -> None:
        assert (VAL_DIR / "_data_fitting_solvers.py").exists()


class TestDataFittingFileSizes:
    """Each file must be under 600 LOC after split."""

    @pytest.mark.unit
    def test_data_fitting_split_2456_coordinator_loc(self) -> None:
        loc = _count_lines(VAL_DIR / "data_fitting.py")
        assert loc <= LOC_BUDGET, f"data_fitting.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_data_fitting_split_2456_models_loc(self) -> None:
        loc = _count_lines(VAL_DIR / "_data_fitting_models.py")
        assert (
            loc <= LOC_BUDGET
        ), f"_data_fitting_models.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_solvers_loc(self) -> None:
        loc = _count_lines(VAL_DIR / "_data_fitting_solvers.py")
        assert (
            loc <= LOC_BUDGET
        ), f"_data_fitting_solvers.py has {loc} LOC; budget {LOC_BUDGET}"


class TestDataFittingPublicAPI:
    """Public API must remain importable from data_fitting (backward compat)."""

    @pytest.mark.unit
    def test_import_body_segment_params(self) -> None:
        from src.shared.python.validation_pkg.data_fitting import BodySegmentParams

        assert BodySegmentParams is not None

    @pytest.mark.unit
    def test_import_kinematic_state(self) -> None:
        from src.shared.python.validation_pkg.data_fitting import KinematicState

        assert KinematicState is not None

    @pytest.mark.unit
    def test_import_fit_result(self) -> None:
        from src.shared.python.validation_pkg.data_fitting import FitResult

        assert FitResult is not None

    @pytest.mark.unit
    def test_import_sensitivity_result(self) -> None:
        from src.shared.python.validation_pkg.data_fitting import SensitivityResult

        assert SensitivityResult is not None

    @pytest.mark.unit
    def test_import_parameter_estimation_report(self) -> None:
        from src.shared.python.validation_pkg.data_fitting import (
            ParameterEstimationReport,
        )

        assert ParameterEstimationReport is not None

    @pytest.mark.unit
    def test_import_inverse_kinematics_solver(self) -> None:
        from src.shared.python.validation_pkg.data_fitting import (
            InverseKinematicsSolver,
        )

        assert InverseKinematicsSolver is not None

    @pytest.mark.unit
    def test_import_parameter_estimator(self) -> None:
        from src.shared.python.validation_pkg.data_fitting import ParameterEstimator

        assert ParameterEstimator is not None

    @pytest.mark.unit
    def test_import_sensitivity_analyzer(self) -> None:
        from src.shared.python.validation_pkg.data_fitting import SensitivityAnalyzer

        assert SensitivityAnalyzer is not None

    @pytest.mark.unit
    def test_import_convert_poses_to_markers(self) -> None:
        from src.shared.python.validation_pkg.data_fitting import (
            convert_poses_to_markers,
        )

        assert callable(convert_poses_to_markers)

    @pytest.mark.unit
    def test_import_a3_fitting_pipeline(self) -> None:
        from src.shared.python.validation_pkg.data_fitting import A3FittingPipeline

        assert A3FittingPipeline is not None
