"""Tests for digital twin state estimator."""

import numpy as np
import pytest

from src.deployment.digital_twin.estimator import EstimatorConfig, StateEstimator
from src.deployment.realtime.state import RobotState


@pytest.fixture
def empty_robot_state() -> RobotState:
    """Fixture for an empty 7-DOF robot state."""
    return RobotState(
        timestamp=0.0,
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        joint_torques=np.zeros(7),
    )


def test_estimator_config_defaults() -> None:
    """Test estimator configuration default values."""
    config = EstimatorConfig()
    assert config.process_noise == 0.001
    assert config.measurement_noise == 0.01
    assert config.use_velocity_filter is True
    assert config.velocity_filter_alpha == 0.3
    assert config.outlier_threshold == 3.0


def test_estimator_initialization() -> None:
    """Test estimator initializes correctly."""
    estimator = StateEstimator(n_dof=7)
    assert estimator.n_dof == 7
    assert estimator._state_dim == 21

    pos = estimator.get_position()
    assert pos.shape == (7,)
    assert np.allclose(pos, np.zeros(7))

    vel = estimator.get_velocity()
    assert vel.shape == (7,)
    assert np.allclose(vel, np.zeros(7))

    acc = estimator.get_acceleration()
    assert acc.shape == (7,)
    assert np.allclose(acc, np.zeros(7))

    cov = estimator.get_covariance()
    assert cov.shape == (21, 21)


def test_estimator_reset() -> None:
    """Test estimator reset."""
    estimator = StateEstimator(n_dof=7)

    # Update state artificially
    estimator._state[0] = 1.0

    # Reset
    estimator.reset(position=np.ones(7), velocity=np.ones(7) * 2)

    pos = estimator.get_position()
    assert np.allclose(pos, np.ones(7))

    vel = estimator.get_velocity()
    assert np.allclose(vel, np.ones(7) * 2)


def test_estimator_update(empty_robot_state: RobotState) -> None:
    """Test estimator update step."""
    estimator = StateEstimator(n_dof=7)

    # First update
    res1 = estimator.update(empty_robot_state, dt=0.01)

    # Check returns
    assert "position" in res1
    assert "velocity" in res1
    assert "acceleration" in res1

    # State moves slightly from zero, but since measurement is zero it should remain close
    assert np.allclose(res1["position"], np.zeros(7), atol=1e-3)

    # Update with non-zero but small enough to not be rejected as outlier (cov is 0.1 -> std is ~0.3 -> 3 sigma is 0.9)
    new_state = RobotState(
        timestamp=0.1,
        joint_positions=np.ones(7) * 0.5,
        joint_velocities=np.ones(7) * 0.2,
        joint_torques=np.zeros(7),
    )
    res2 = estimator.update(new_state)

    assert not np.allclose(res2["position"], np.zeros(7))


def test_estimator_uncertainty() -> None:
    """Test extracting uncertainty."""
    estimator = StateEstimator(n_dof=3)

    pos_unc = estimator.get_position_uncertainty()
    assert pos_unc.shape == (3,)

    vel_unc = estimator.get_velocity_uncertainty()
    assert vel_unc.shape == (3,)


def test_estimator_predict() -> None:
    """Test estimator prediction without update."""
    estimator = StateEstimator(n_dof=7)
    estimator.reset(position=np.ones(7), velocity=np.ones(7))

    prediction = estimator.predict(dt=0.1)

    # pos = p0 + v0*dt
    # Expected pos: 1.0 + 1.0 * 0.1 = 1.1
    assert np.allclose(prediction["position"], np.ones(7) * 1.1)

    control = np.ones(7) * 0.5
    prediction_with_control = estimator.predict(dt=0.1, control=control)
    assert np.allclose(prediction_with_control["acceleration"], control)


def test_estimator_outlier_rejection() -> None:
    """Test outlier rejection during update."""
    config = EstimatorConfig(outlier_threshold=1.0)
    estimator = StateEstimator(n_dof=3, config=config)

    # Provide a massive outlier measurement
    outlier_state = RobotState(
        timestamp=1.0,
        joint_positions=np.ones(3) * 1000.0,  # massive outlier
        joint_velocities=np.zeros(3),
        joint_torques=np.zeros(3),
    )

    res = estimator.update(outlier_state, dt=0.1)

    # The outlier rejection should clip it back towards the prediction (which is near 0)
    assert np.all(np.abs(res["position"]) < 500.0)
