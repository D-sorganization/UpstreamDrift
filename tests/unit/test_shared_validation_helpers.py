import numpy as np
import pytest

from shared.python.validation_helpers import (
    MAX_CARTESIAN_ACCELERATION_M_S2,
    MAX_CARTESIAN_VELOCITY_M_S,
    MAX_JOINT_VELOCITY_RAD_S,
    PhysicsValidationError,
    ValidationLevel,
    validate_cartesian_state,
    validate_finite,
    validate_joint_state,
    validate_magnitude,
    validate_model_parameters,
)


class TestValidationHelpers:
    """Tests for physics validation helpers."""

    def test_validate_finite(self):
        """Test validate_finite function."""
        valid_array = np.array([1.0, 2.0, 3.0])
        nan_array = np.array([1.0, np.nan, 3.0])
        inf_array = np.array([1.0, np.inf, 3.0])

        # Valid case should pass
        validate_finite(valid_array, "test", ValidationLevel.STRICT)

        # STRICT raises error
        with pytest.raises(PhysicsValidationError, match="contains NaN or Inf"):
            validate_finite(nan_array, "test", ValidationLevel.STRICT)
        with pytest.raises(PhysicsValidationError, match="contains NaN or Inf"):
            validate_finite(inf_array, "test", ValidationLevel.STRICT)

        # STANDARD raises error
        with pytest.raises(PhysicsValidationError, match="contains NaN or Inf"):
            validate_finite(nan_array, "test", ValidationLevel.STANDARD)

        # PERMISSIVE warns
        with pytest.warns(UserWarning, match="contains NaN or Inf"):
            validate_finite(nan_array, "test", ValidationLevel.PERMISSIVE)

    def test_validate_magnitude(self):
        """Test validate_magnitude function."""
        small_array = np.array([1.0, -1.0])
        large_array = np.array([100.0, -100.0])
        threshold = 10.0

        # Valid case
        validate_magnitude(
            small_array, "test", threshold, "units", ValidationLevel.STRICT
        )

        # STRICT raises error
        with pytest.raises(PhysicsValidationError, match="implausibly large values"):
            validate_magnitude(
                large_array, "test", threshold, "units", ValidationLevel.STRICT
            )

        # STANDARD warns
        with pytest.warns(UserWarning, match="implausibly large values"):
            validate_magnitude(
                large_array, "test", threshold, "units", ValidationLevel.STANDARD
            )

        # PERMISSIVE warns
        with pytest.warns(UserWarning, match="implausibly large values"):
            validate_magnitude(
                large_array, "test", threshold, "units", ValidationLevel.PERMISSIVE
            )

    def test_validate_joint_state(self):
        """Test validate_joint_state function."""
        qpos = np.zeros(3)
        qvel = np.zeros(3)
        qacc = np.zeros(3)

        # Valid case
        validate_joint_state(qpos, qvel, qacc)

        # Dimension mismatch
        with pytest.raises(PhysicsValidationError, match="Dimension mismatch"):
            validate_joint_state(qpos, np.zeros(2))

        with pytest.raises(PhysicsValidationError, match="Dimension mismatch"):
            validate_joint_state(qpos, qvel, np.zeros(2))

        # NaN checks
        validate_joint_state(np.array([np.nan, 0, 0]), level=ValidationLevel.PERMISSIVE)
        with pytest.raises(PhysicsValidationError):
            validate_joint_state(np.array([np.nan, 0, 0]), level=ValidationLevel.STRICT)

        # Magnitude checks
        huge_vel = np.array([MAX_JOINT_VELOCITY_RAD_S * 2, 0, 0])
        with pytest.warns(UserWarning):
            validate_joint_state(qpos, huge_vel, level=ValidationLevel.STANDARD)

        with pytest.raises(PhysicsValidationError):
            validate_joint_state(qpos, huge_vel, level=ValidationLevel.STRICT)

    def test_validate_cartesian_state(self):
        """Test validate_cartesian_state function."""
        pos = np.zeros(3)
        vel = np.zeros(3)
        acc = np.zeros(3)

        # Valid
        validate_cartesian_state(pos, vel, acc)

        # High velocity
        huge_vel = np.array([MAX_CARTESIAN_VELOCITY_M_S * 2, 0, 0])
        with pytest.warns(UserWarning, match="implausibly large values"):
            validate_cartesian_state(velocity=huge_vel, level=ValidationLevel.STANDARD)

        # High acceleration
        huge_acc = np.array([MAX_CARTESIAN_ACCELERATION_M_S2 * 2, 0, 0])
        with pytest.raises(PhysicsValidationError):
            validate_cartesian_state(
                acceleration=huge_acc, level=ValidationLevel.STRICT
            )

    def test_validate_model_parameters(self):
        """Test validate_model_parameters function."""
        valid_masses = np.array([10.0, 50.0])

        validate_model_parameters(valid_masses)

        # Negative mass
        with pytest.raises(PhysicsValidationError, match="must be positive"):
            validate_model_parameters(np.array([-1.0, 10.0]))

        # Zero mass
        with pytest.raises(PhysicsValidationError, match="must be positive"):
            validate_model_parameters(np.array([0.0, 10.0]))

        # Implausible total mass
        tiny_masses = np.ones(15) * 0.1  # Total 1.5kg
        with pytest.warns(UserWarning, match="Total model mass"):
            validate_model_parameters(tiny_masses)
