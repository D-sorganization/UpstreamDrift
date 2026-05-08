"""Unit tests for cross-engine stability validation matrix.

Validates:
    - ToleranceFramework tolerance computation and validation.
    - StabilityMatrix canonical test creation and management.
    - Cross-engine tolerance compliance (<2% error target).
    - Safe operating region documentation.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.motion_matching.stability_matrix import (
    CanonicalTestCase,
    InitialPose,
    StabilityBoundary,
    StabilityMatrix,
    ToleranceFramework,
    validate_cross_engine_stability,
)


class TestToleranceFramework:
    """Test cases for ToleranceFramework."""

    def test_tolerance_framework_initialization(self) -> None:
        """Test framework initializes with all engines."""
        fw = ToleranceFramework()
        assert len(fw.tolerances) == 4
        assert "drake" in fw.tolerances
        assert "opensim" in fw.tolerances
        assert "mujoco" in fw.tolerances
        assert "pinocchio" in fw.tolerances

    def test_get_tolerance_valid_engine(self) -> None:
        """Test retrieving tolerance for valid engine."""
        fw = ToleranceFramework()
        tol = fw.get_tolerance("drake")
        assert tol == 0.01  # 1% tolerance for Drake

    def test_get_tolerance_invalid_engine(self) -> None:
        """Test error on invalid engine."""
        fw = ToleranceFramework()
        with pytest.raises(ValueError, match="engine must be known"):
            fw.get_tolerance("invalid_engine")

    def test_relative_error_computation(self) -> None:
        """Test relative error computation."""
        fw = ToleranceFramework()
        # 10% error: (11 - 10) / 10 = 0.1
        rel_error = fw.compute_relative_error(10.0, 11.0)
        assert abs(rel_error - 0.1) < 1e-6

    def test_relative_error_with_array(self) -> None:
        """Test relative error with numpy arrays."""
        fw = ToleranceFramework()
        expected = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.05, 2.1, 3.15])
        rel_error = fw.compute_relative_error(expected, actual)
        # Max error: 2.1/2.0 - 1 = 0.05 (5%)
        assert rel_error == pytest.approx(0.05, abs=1e-6)

    def test_relative_error_nan_raises(self) -> None:
        """Test NaN in expected raises."""
        fw = ToleranceFramework()
        with pytest.raises(ValueError, match="contains NaN"):
            fw.compute_relative_error(np.array([np.nan]), np.array([1.0]))

    def test_relative_error_zero_expected_raises(self) -> None:
        """Test zero expected value raises."""
        fw = ToleranceFramework()
        with pytest.raises(ValueError, match="too close to zero"):
            fw.compute_relative_error(1e-20, 1.0)

    def test_is_within_tolerance_pass(self) -> None:
        """Test tolerance check passes for small error."""
        fw = ToleranceFramework()
        assert fw.is_within_tolerance("drake", 0.005) is True

    def test_is_within_tolerance_fail(self) -> None:
        """Test tolerance check fails for large error."""
        fw = ToleranceFramework()
        assert fw.is_within_tolerance("drake", 0.05) is False

    def test_is_within_tolerance_mujoco_looser(self) -> None:
        """Test MuJoCo has looser tolerance."""
        fw = ToleranceFramework()
        # MuJoCo allows 2% error
        assert fw.is_within_tolerance("mujoco", 0.015) is True
        assert fw.is_within_tolerance("mujoco", 0.025) is False


class TestCanonicalTestCase:
    """Test cases for CanonicalTestCase."""

    @staticmethod
    def create_valid_test_case() -> CanonicalTestCase:
        """Helper to create a valid test case."""
        return CanonicalTestCase(
            name="test",
            description="Test case",
            theta=np.zeros(23),
            pose=InitialPose(
                root_position=np.array([0.0, 0.0, 1.0]),
                root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                joint_angles=np.zeros(17),
            ),
            temperature_k=293.15,
            pressure_pa=101325.0,
        )

    def test_canonical_test_case_valid_creation(self) -> None:
        """Test valid test case creation."""
        case = self.create_valid_test_case()
        assert case.name == "test"
        assert case.theta.shape == (23,)
        assert case.temperature_k == 293.15

    def test_canonical_test_case_invalid_name(self) -> None:
        """Test invalid name raises."""
        with pytest.raises(ValueError, match="name must be non-empty"):
            CanonicalTestCase(
                name="",
                description="Test",
                theta=np.zeros(23),
                pose=InitialPose(
                    root_position=np.array([0.0, 0.0, 1.0]),
                    root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                    joint_angles=np.zeros(17),
                ),
                temperature_k=293.15,
                pressure_pa=101325.0,
            )

    def test_canonical_test_case_temp_out_of_range(self) -> None:
        """Test temperature out of range raises."""
        with pytest.raises(ValueError, match="temperature_k"):
            CanonicalTestCase(
                name="test",
                description="Test",
                theta=np.zeros(23),
                pose=InitialPose(
                    root_position=np.array([0.0, 0.0, 1.0]),
                    root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                    joint_angles=np.zeros(17),
                ),
                temperature_k=600.0,  # Too high
                pressure_pa=101325.0,
            )

    def test_canonical_test_case_pressure_out_of_range(self) -> None:
        """Test pressure out of range raises."""
        with pytest.raises(ValueError, match="pressure_pa"):
            CanonicalTestCase(
                name="test",
                description="Test",
                theta=np.zeros(23),
                pose=InitialPose(
                    root_position=np.array([0.0, 0.0, 1.0]),
                    root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                    joint_angles=np.zeros(17),
                ),
                temperature_k=293.15,
                pressure_pa=3.0e6,  # Too high
            )

    def test_canonical_test_case_nan_theta_raises(self) -> None:
        """Test NaN in theta raises."""
        with pytest.raises(ValueError, match="NaN"):
            CanonicalTestCase(
                name="test",
                description="Test",
                theta=np.array([1.0, np.nan] + [0.0] * 21),
                pose=InitialPose(
                    root_position=np.array([0.0, 0.0, 1.0]),
                    root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                    joint_angles=np.zeros(17),
                ),
                temperature_k=293.15,
                pressure_pa=101325.0,
            )


class TestStabilityMatrix:
    """Test cases for StabilityMatrix."""

    def test_stability_matrix_initialization(self) -> None:
        """Test matrix initializes with canonical tests."""
        matrix = StabilityMatrix()
        assert len(matrix.canonical_tests) >= 5
        assert "nominal" in matrix.canonical_tests
        assert "high_temperature" in matrix.canonical_tests
        assert "high_pressure" in matrix.canonical_tests

    def test_stability_matrix_boundaries_initialized(self) -> None:
        """Test stability boundaries are initialized."""
        matrix = StabilityMatrix()
        assert len(matrix.stability_boundaries) >= 3
        assert "nominal_region" in matrix.stability_boundaries

    def test_get_canonical_test_valid(self) -> None:
        """Test retrieving valid canonical test."""
        matrix = StabilityMatrix()
        case = matrix.get_canonical_test("nominal")
        assert case.name == "nominal"
        assert case.theta.shape == (23,)

    def test_get_canonical_test_invalid(self) -> None:
        """Test retrieving invalid test raises."""
        matrix = StabilityMatrix()
        from src.shared.python._contracts_exceptions import PreconditionError
        with pytest.raises(PreconditionError):
            matrix.get_canonical_test("nonexistent_test")

    def test_get_safe_operating_region(self) -> None:
        """Test safe operating region retrieval."""
        matrix = StabilityMatrix()
        region = matrix.get_safe_operating_region("drake")
        assert "engine" in region
        assert "temperature_range_k" in region
        assert "pressure_range_pa" in region
        assert region["engine"] == "drake"

    def test_record_test_result(self) -> None:
        """Test recording test result."""
        matrix = StabilityMatrix()
        matrix.record_test_result(
            engine="drake",
            test_name="nominal",
            passed=True,
            relative_error=0.005,
            metadata={"notes": "test passed"},
        )
        assert ("drake", "nominal") in matrix.test_results
        result = matrix.test_results[("drake", "nominal")]
        assert result["passed"] is True
        assert result["relative_error"] == 0.005

    def test_get_test_summary(self) -> None:
        """Test getting test summary."""
        matrix = StabilityMatrix()
        matrix.record_test_result(
            engine="drake",
            test_name="nominal",
            passed=True,
            relative_error=0.005,
        )
        matrix.record_test_result(
            engine="mujoco",
            test_name="nominal",
            passed=False,
            relative_error=0.03,
        )
        summary = matrix.get_test_summary()
        assert summary["total_tests"] == 2
        assert "drake_pass_rate" in summary


class TestValidateCrossEngineStability:
    """Test cases for validate_cross_engine_stability entry point."""

    def test_validate_with_valid_engines(self) -> None:
        """Test validation with valid engines."""
        matrix = validate_cross_engine_stability(
            engines=["drake", "opensim"],
            canonical_test_names=["nominal", "high_temperature"],
        )
        assert isinstance(matrix, StabilityMatrix)
        assert len(matrix.canonical_tests) >= 5

    def test_validate_with_invalid_engine_raises(self) -> None:
        """Test validation with invalid engine raises."""
        with pytest.raises(ValueError, match="Unknown engine"):
            validate_cross_engine_stability(engines=["invalid_engine"])

    def test_validate_with_all_engines(self) -> None:
        """Test validation with all engines."""
        matrix = validate_cross_engine_stability(
            engines=["drake", "opensim", "mujoco", "pinocchio"]
        )
        assert isinstance(matrix, StabilityMatrix)


class TestStabilityBoundary:
    """Test cases for StabilityBoundary."""

    def test_stability_boundary_creation(self) -> None:
        """Test creating stability boundary."""
        boundary = StabilityBoundary(
            name="test_region",
            temperature_k=300.0,
            pressure_pa=101325.0,
            composition_fraction=0.5,
            description="Test boundary",
        )
        assert boundary.name == "test_region"
        assert boundary.temperature_k == 300.0

    def test_stability_boundary_is_namedtuple(self) -> None:
        """Test StabilityBoundary is immutable namedtuple."""
        boundary = StabilityBoundary(
            name="test",
            temperature_k=300.0,
            pressure_pa=101325.0,
            composition_fraction=0.5,
            description="Test",
        )
        with pytest.raises(AttributeError):
            boundary.name = "changed"  # type: ignore
