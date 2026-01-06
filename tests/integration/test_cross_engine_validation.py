"""Cross-engine validation integration tests.

Tests the CrossEngineValidator against actual physics engines to ensure
MuJoCo, Drake, and Pinocchio produce consistent results per Guideline M2/P3.
"""

import pytest
import numpy as np
from shared.python.cross_engine_validator import CrossEngineValidator, ValidationResult


class TestCrossEngineValidator:
    """Unit tests for CrossEngineValidator (no engine dependencies)."""
    
    def test_validation_pass_within_tolerance(self):
        """Test that states within tolerance pass validation."""
        validator = CrossEngineValidator()
        
        state1 = np.array([1.0, 2.0, 3.0])
        state2 = np.array([1.0000001, 2.0000001, 3.0000001])  # 1e-7 deviation
        
        result = validator.compare_states(
            "MuJoCo", state1,
            "Drake", state2,
            metric="position"  # tolerance: 1e-6
        )
        
        assert result.passed
        assert result.max_deviation < 1e-6
        assert result.metric_name == "position"
        assert result.engine1 == "MuJoCo"
        assert result.engine2 == "Drake"
    
    def test_validation_fail_exceeds_tolerance(self):
        """Test that states exceeding tolerance fail validation."""
        validator = CrossEngineValidator()
        
        state1 = np.array([1.0, 2.0, 3.0])
        state2 = np.array([1.001, 2.001, 3.001])  # 1e-3 deviation (exceeds 1e-6)
        
        result = validator.compare_states(
            "MuJoCo", state1,
            "Drake", state2,
            metric="position"
        )
        
        assert not result.passed
        assert result.max_deviation > 1e-6
        assert "exceeds tolerance" in result.message.lower()
    
    def test_shape_mismatch_detection(self):
        """Test that shape mismatches are detected and reported."""
        validator = CrossEngineValidator()
        
        state1 = np.array([1.0, 2.0, 3.0])
        state2 = np.array([1.0, 2.0])  # Different shape
        
        result = validator.compare_states(
            "MuJoCo", state1,
            "Drake", state2,
            metric="position"
        )
        
        assert not result.passed
        assert result.max_deviation == np.inf
        assert "shape mismatch" in result.message.lower()
    
    def test_different_metrics_different_tolerances(self):
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
    
    def test_torque_rms_comparison(self):
        """Test RMS percentage-based torque comparison."""
        validator = CrossEngineValidator()
        
        # Create torques with 5% RMS difference
        torques1 = np.array([10.0, 20.0, 30.0])
        torques2 = np.array([10.5, 20.5, 30.5])  # ~5% RMS difference
        
        result = validator.compare_torques_with_rms(
            "MuJoCo", torques1,
            "Drake", torques2,
            rms_threshold_pct=10.0  # 10% threshold
        )
        
        assert result.passed  # 5% < 10%
        assert result.max_deviation < 10.0
    
    def test_torque_rms_failure(self):
        """Test RMS comparison fails when threshold exceeded."""
        validator = CrossEngineValidator()
        
        torques1 = np.array([10.0, 20.0, 30.0])
        torques2 = np.array([15.0, 25.0, 35.0])  # ~20% RMS difference
        
        result = validator.compare_torques_with_rms(
            "MuJoCo", torques1,
            "Drake", torques2,
            rms_threshold_pct=10.0
        )
        
        assert not result.passed  # ~20% > 10%
        assert result.max_deviation > 10.0


@pytest.mark.integration
@pytest.mark.mujoco
@pytest.mark.drake
@pytest.mark.skipif(
    not all([
        pytest.importorskip("mujoco", reason="MuJoCo not installed"),
        pytest.importorskip("pydrake", reason="Drake not installed"),
    ]),
    reason="Requires both MuJoCo and Drake"
)
class TestCrossEngineValidationIntegration:
    """Integration tests comparing actual MuJoCo and Drake outputs.
    
    These tests validate Guideline M2 requirement for cross-engine comparison.
    """
    
    def test_forward_dynamics_mujoco_drake_agreement(self):
        """Validate MuJoCo and Drake forward dynamics agree per Guideline P3."""
        pytest.skip("Requires model loading infrastructure - implement in follow-up")
        
        # TODO: Implementation:
        # 1. Load same URDF in both engines
        # 2. Set identical initial conditions
        # 3. Run forward dynamics for 5 seconds
        # 4. Compare positions (tolerance: ±1e-6m)
        # 5. Compare velocities (tolerance: ±1e-5m/s)
        # 6. Use CrossEngineValidator for comparison
    
    def test_inverse_dynamics_mujoco_drake_agreement(self):
        """Validate MuJoCo and Drake inverse dynamics agree per Guideline P3."""
        pytest.skip("Requires model loading infrastructure - implement in follow-up")
        
        # TODO: Implementation:
        # 1. Load same URDF in both engines
        # 2. Set joint positions, velocities, accelerations
        # 3. Compute inverse dynamics torques
        # 4. Compare torques (RMS < 10%)
        # 5. Use CrossEngineValidator.compare_torques_with_rms()
    
    def test_jacobian_mujoco_drake_agreement(self):
        """Validate MuJoCo and Drake Jacobians agree per Guideline P3."""
        pytest.skip("Requires model loading infrastructure - implement in follow-up")
        
        # TODO: Implementation:
        # 1. Load same URDF in both engines
        # 2. Set joint configuration
        # 3. Compute Jacobians for end-effector
        # 4. Compare element-wise (tolerance: ±1e-8)
        # 5. Use CrossEngineValidator with metric="jacobian"


@pytest.mark.integration
@pytest.mark.pinocchio
@pytest.mark.skipif(
    not pytest.importorskip("pinocchio", reason="Pinocchio not installed"),
    reason="Requires Pinocchio"
)
class TestCrossEngineValidationPinocchio:
    """Integration tests including Pinocchio engine."""
    
    def test_three_way_validation_mujoco_drake_pinocchio(self):
        """Validate all three engines agree in pairwise comparisons."""
        pytest.skip("Requires model loading infrastructure - implement in follow-up")
        
        # TODO: Implementation:
        # 1. Load same model in MuJoCo, Drake, Pinocchio
        # 2. Run same simulation
        # 3. Pairwise validation:
        #    - MuJoCo vs Drake
        #    - MuJoCo vs Pinocchio
        #    - Drake vs Pinocchio
        # 4. All pairs must pass Guideline P3 tolerances
