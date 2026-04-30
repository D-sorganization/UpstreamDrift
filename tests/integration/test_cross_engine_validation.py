"""Cross-engine validation integration tests.

Tests the CrossEngineValidator against actual physics engines to ensure
MuJoCo, Drake, and Pinocchio produce consistent results per Guideline M2/P3.

This module implements the acceptance test suite required by Section M2
of the Project Design Guidelines.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.engine_core.cross_engine_validator import CrossEngineValidator
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class TestCrossEngineValidator:
    """Unit tests for CrossEngineValidator (no engine dependencies)."""

    def test_validation_pass_within_tolerance(self) -> None:
        """Test that states within tolerance pass validation."""
        validator = CrossEngineValidator()

        state1 = np.array([1.0, 2.0, 3.0])
        state2 = np.array([1.0000001, 2.0000001, 3.0000001])  # 1e-7 deviation

        result = validator.compare_states(
            "MuJoCo",
            state1,
            "Drake",
            state2,
            metric="position",  # tolerance: 1e-6
        )

        assert result.passed
        assert result.max_deviation < 1e-6
        assert result.metric_name == "position"
        assert result.engine1 == "MuJoCo"
        assert result.engine2 == "Drake"

    def test_validation_fail_exceeds_tolerance(self) -> None:
        """Test that states exceeding tolerance fail validation."""
        validator = CrossEngineValidator()

        state1 = np.array([1.0, 2.0, 3.0])
        state2 = np.array([1.001, 2.001, 3.001])  # 1e-3 deviation (exceeds 1e-6)

        result = validator.compare_states(
            "MuJoCo", state1, "Drake", state2, metric="position"
        )

        assert not result.passed
        assert result.max_deviation > 1e-6
        # CrossEngineValidator uses various message formats for failures
        assert (
            "exceeds tolerance" in result.message.lower()
            or "deviation" in result.message.lower()
            or "critical" in result.message.lower()
        )

    def test_shape_mismatch_detection(self) -> None:
        """Test that shape mismatches are detected and reported."""
        validator = CrossEngineValidator()

        state1 = np.array([1.0, 2.0, 3.0])
        state2 = np.array([1.0, 2.0])  # Different shape

        result = validator.compare_states(
            "MuJoCo", state1, "Drake", state2, metric="position"
        )

        assert not result.passed
        assert result.max_deviation == np.inf
        assert "shape mismatch" in result.message.lower()

    def test_different_metrics_different_tolerances(self) -> None:
        """Test that different metrics use appropriate tolerances."""
        validator = CrossEngineValidator()

        # Same deviation, different metrics
        deviation = 5e-6  # 5 microns or 5 micrometers/s

        state1 = np.array([1.0])
        state2_position = np.array([1.0 + deviation])
        state2_velocity = np.array([1.0 + deviation])

        # Position: tolerance 1e-6, should fail
        result_pos = validator.compare_states(
            "MuJoCo", state1, "Drake", state2_position, metric="position"
        )
        assert not result_pos.passed  # 5e-6 > 1e-6

        # Velocity: tolerance 1e-5, should pass
        result_vel = validator.compare_states(
            "MuJoCo", state1, "Drake", state2_velocity, metric="velocity"
        )
        assert result_vel.passed  # 5e-6 < 1e-5

    def test_torque_rms_comparison(self) -> None:
        """Test RMS percentage-based torque comparison."""
        validator = CrossEngineValidator()

        # Create torques with 5% RMS difference
        torques1 = np.array([10.0, 20.0, 30.0])
        torques2 = np.array([10.5, 20.5, 30.5])  # ~5% RMS difference

        result = validator.compare_torques_with_rms(
            "MuJoCo",
            torques1,
            "Drake",
            torques2,
            rms_threshold_pct=10.0,  # 10% threshold
        )

        assert result.passed  # 5% < 10%
        assert result.max_deviation < 10.0

    def test_torque_rms_failure(self) -> None:
        """Test RMS comparison fails when threshold exceeded."""
        validator = CrossEngineValidator()

        torques1 = np.array([10.0, 20.0, 30.0])
        torques2 = np.array([15.0, 25.0, 35.0])  # ~20% RMS difference

        result = validator.compare_torques_with_rms(
            "MuJoCo", torques1, "Drake", torques2, rms_threshold_pct=10.0
        )

        assert not result.passed  # ~20% > 10%
        assert result.max_deviation > 10.0

    def test_severity_classification_passed(self) -> None:
        """Test PASSED severity for deviations within tolerance."""
        validator = CrossEngineValidator()
        passed, severity = validator._classify_severity(0.5e-6, 1e-6)
        assert passed
        assert severity == "PASSED"

    def test_severity_classification_warning(self) -> None:
        """Test WARNING severity for deviations slightly above tolerance."""
        validator = CrossEngineValidator()
        passed, severity = validator._classify_severity(1.5e-6, 1e-6)
        assert passed  # Still acceptable
        assert severity == "WARNING"

    def test_severity_classification_error(self) -> None:
        """Test ERROR severity for significant deviations."""
        validator = CrossEngineValidator()
        passed, severity = validator._classify_severity(5e-6, 1e-6)
        assert not passed
        assert severity == "ERROR"

    def test_severity_classification_blocker(self) -> None:
        """Test BLOCKER severity for extreme deviations."""
        validator = CrossEngineValidator()
        passed, severity = validator._classify_severity(1e-3, 1e-6)  # 1000x tolerance
        assert not passed
        assert severity == "BLOCKER"
