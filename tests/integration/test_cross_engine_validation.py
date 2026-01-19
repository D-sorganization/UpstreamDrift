"""Cross-engine validation integration tests.

Tests the CrossEngineValidator against actual physics engines to ensure
MuJoCo, Drake, and Pinocchio produce consistent results per Guideline M2/P3.

This module implements the acceptance test suite required by Section M2
of the Project Design Guidelines.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from shared.python.cross_engine_validator import CrossEngineValidator
from tests.fixtures.fixtures_lib import (
    TOLERANCE_ACCELERATION_M_S2,
    TOLERANCE_JACOBIAN,
    compute_accelerations,
    set_identical_state,
    skip_if_insufficient_engines,
)

logger = logging.getLogger(__name__)


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


@pytest.mark.integration
class TestCrossEngineValidationIntegration:
    """Integration tests comparing actual physics engine outputs.

    These tests validate Guideline M2 requirement for cross-engine comparison.
    Tests automatically skip if required engines are not available.
    """

    def test_forward_dynamics_agreement(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Validate forward dynamics agree per Guideline P3.

        Sets identical initial conditions and compares accelerations.
        Tolerance: acceleration ±1e-4 m/s² per P3.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        validator = CrossEngineValidator()
        available_engines = [e for e in engines if e.available]

        # Set identical initial state: small angle (0.1 rad) from vertical
        q_init = np.array([0.1])  # Small angle for near-linear regime
        v_init = np.array([0.0])  # Starting from rest

        set_identical_state(available_engines, q_init, v_init)

        # Compute accelerations
        accelerations = compute_accelerations(available_engines)

        # Pairwise comparison
        engine_names = list(accelerations.keys())
        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i + 1 :]:
                result = validator.compare_states(
                    name1,
                    accelerations[name1],
                    name2,
                    accelerations[name2],
                    metric="acceleration",
                )
                logger.info(
                    f"Forward dynamics {name1} vs {name2}: "
                    f"deviation={result.max_deviation:.2e}, "
                    f"tolerance={TOLERANCE_ACCELERATION_M_S2:.2e}, "
                    f"severity={result.severity}"
                )
                # Allow WARNING severity (up to 2x tolerance) for cross-engine
                assert result.severity in ["PASSED", "WARNING"], (
                    f"Forward dynamics mismatch between {name1} and {name2}: "
                    f"{result.message}"
                )

    def test_inverse_dynamics_agreement(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Validate inverse dynamics agree per Guideline P3.

        Computes torques for a given motion and compares.
        Tolerance: RMS < 10% per P3.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        validator = CrossEngineValidator()
        available_engines = [e for e in engines if e.available]

        # Set state
        q_init = np.array([0.2])  # 0.2 rad
        v_init = np.array([0.5])  # 0.5 rad/s

        set_identical_state(available_engines, q_init, v_init)

        # Desired acceleration for ID computation
        qacc_desired = np.array([1.0])  # 1 rad/s² desired

        # Compute inverse dynamics torques
        torques: dict[str, np.ndarray] = {}
        for eng in available_engines:
            if eng.engine is not None:
                tau = eng.engine.compute_inverse_dynamics(qacc_desired)
                if tau.size > 0:
                    torques[eng.name] = tau

        if len(torques) < 2:
            pytest.skip("Inverse dynamics not available in enough engines")

        # Pairwise RMS comparison
        engine_names = list(torques.keys())
        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i + 1 :]:
                result = validator.compare_torques_with_rms(
                    name1, torques[name1], name2, torques[name2], rms_threshold_pct=10.0
                )
                logger.info(
                    f"Inverse dynamics {name1} vs {name2}: "
                    f"RMS deviation={result.max_deviation:.2f}%"
                )
                assert result.passed, (
                    f"Inverse dynamics mismatch between {name1} and {name2}: "
                    f"{result.message}"
                )

    def test_jacobian_agreement(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Validate Jacobians agree per Guideline P3.

        Computes spatial Jacobians for end-effector and compares.
        Tolerance: ±1e-8 element-wise per P3.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        validator = CrossEngineValidator()
        available_engines = [e for e in engines if e.available]

        # Set specific configuration
        q_init = np.array([0.3])  # 0.3 rad
        v_init = np.array([0.0])  # Static

        set_identical_state(available_engines, q_init, v_init)

        # Compute Jacobians for end effector/pendulum link
        jacobians: dict[str, np.ndarray] = {}
        for eng in available_engines:
            if eng.engine is None:
                continue

            # Prefer engine-provided body names when available to avoid
            # hardcoding URDF-specific names. Fall back to common names.
            candidate_names: list[str]
            if hasattr(eng.engine, "get_body_names"):
                candidate_names = list(eng.engine.get_body_names())
            elif hasattr(eng.engine, "body_names"):
                candidate_names = list(eng.engine.body_names)
            else:
                # Fallback for engines without body name API
                candidate_names = ["end_effector", "pendulum_link", "lower_link"]

            for body_name in candidate_names:
                jac = eng.engine.compute_jacobian(body_name)
                if jac is not None and "spatial" in jac:
                    jacobians[eng.name] = jac["spatial"]
                    break

        if len(jacobians) < 2:
            pytest.skip("Jacobian computation not available in enough engines")

        # Pairwise comparison
        engine_names = list(jacobians.keys())
        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i + 1 :]:
                result = validator.compare_states(
                    name1,
                    jacobians[name1].flatten(),
                    name2,
                    jacobians[name2].flatten(),
                    metric="jacobian",
                )
                logger.info(
                    f"Jacobian {name1} vs {name2}: "
                    f"deviation={result.max_deviation:.2e}, "
                    f"tolerance={TOLERANCE_JACOBIAN:.2e}"
                )
                # Allow some tolerance for numerical differences
                assert result.severity in ["PASSED", "WARNING"], (
                    f"Jacobian mismatch between {name1} and {name2}: {result.message}"
                )

    def test_ztcf_counterfactual_agreement(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Validate ZTCF (Zero-Torque Counterfactual) agrees across engines.

        Per Guideline G1: ZTCF isolates drift dynamics.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        validator = CrossEngineValidator()
        available_engines = [e for e in engines if e.available]

        # Test state
        q_test = np.array([0.2])
        v_test = np.array([0.3])

        # Compute ZTCF accelerations
        ztcf_accels: dict[str, np.ndarray] = {}
        for eng in available_engines:
            if eng.engine is not None:
                try:
                    a_ztcf = eng.engine.compute_ztcf(q_test, v_test)
                    if a_ztcf.size > 0:
                        ztcf_accels[eng.name] = a_ztcf
                except Exception as e:
                    logger.warning(f"ZTCF failed for {eng.name}: {e}")

        if len(ztcf_accels) < 2:
            pytest.skip("ZTCF not available in enough engines")

        # Pairwise comparison
        engine_names = list(ztcf_accels.keys())
        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i + 1 :]:
                result = validator.compare_states(
                    name1,
                    ztcf_accels[name1],
                    name2,
                    ztcf_accels[name2],
                    metric="acceleration",
                )
                logger.info(
                    f"ZTCF {name1} vs {name2}: deviation={result.max_deviation:.2e}"
                )
                assert result.severity in [
                    "PASSED",
                    "WARNING",
                ], f"ZTCF mismatch between {name1} and {name2}: {result.message}"

    def test_zvcf_counterfactual_agreement(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Validate ZVCF (Zero-Velocity Counterfactual) agrees across engines.

        Per Guideline G2: ZVCF isolates configuration-dependent dynamics.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        validator = CrossEngineValidator()
        available_engines = [e for e in engines if e.available]

        # Test configuration
        q_test = np.array([0.4])

        # Compute ZVCF accelerations
        zvcf_accels: dict[str, np.ndarray] = {}
        for eng in available_engines:
            if eng.engine is not None:
                try:
                    a_zvcf = eng.engine.compute_zvcf(q_test)
                    if a_zvcf.size > 0:
                        zvcf_accels[eng.name] = a_zvcf
                except Exception as e:
                    logger.warning(f"ZVCF failed for {eng.name}: {e}")

        if len(zvcf_accels) < 2:
            pytest.skip("ZVCF not available in enough engines")

        # Pairwise comparison
        engine_names = list(zvcf_accels.keys())
        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i + 1 :]:
                result = validator.compare_states(
                    name1,
                    zvcf_accels[name1],
                    name2,
                    zvcf_accels[name2],
                    metric="acceleration",
                )
                logger.info(
                    f"ZVCF {name1} vs {name2}: deviation={result.max_deviation:.2e}"
                )
                assert result.severity in [
                    "PASSED",
                    "WARNING",
                ], f"ZVCF mismatch between {name1} and {name2}: {result.message}"

    def test_mass_matrix_agreement(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Validate mass matrices agree across engines."""
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        validator = CrossEngineValidator()
        available_engines = [e for e in engines if e.available]

        # Set configuration
        q_init = np.array([0.25])
        v_init = np.array([0.0])
        set_identical_state(available_engines, q_init, v_init)

        # Compute mass matrices
        mass_matrices: dict[str, np.ndarray] = {}
        for eng in available_engines:
            if eng.engine is not None:
                M = eng.engine.compute_mass_matrix()
                if M.size > 0:
                    mass_matrices[eng.name] = M.flatten()

        if len(mass_matrices) < 2:
            pytest.skip("Mass matrix not available in enough engines")

        # Pairwise comparison (mass matrix should be very close)
        engine_names = list(mass_matrices.keys())
        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i + 1 :]:
                # Use position tolerance for mass matrix comparison
                result = validator.compare_states(
                    name1,
                    mass_matrices[name1],
                    name2,
                    mass_matrices[name2],
                    metric="position",  # Tight tolerance
                )
                logger.info(
                    f"Mass matrix {name1} vs {name2}: "
                    f"deviation={result.max_deviation:.2e}"
                )
                assert result.severity in ["PASSED", "WARNING"], (
                    f"Mass matrix mismatch between {name1} and {name2}: "
                    f"{result.message}"
                )
