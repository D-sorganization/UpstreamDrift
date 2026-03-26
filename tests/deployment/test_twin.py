"""Tests for digital twin module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.deployment.digital_twin.twin import AnomalyReport, AnomalyType, DigitalTwin
from src.deployment.realtime.state import RobotState


@pytest.fixture
def mock_sim() -> MagicMock:
    """Mock simulation engine."""
    sim = MagicMock()

    # Setup mock attributes
    sim.get_joint_positions.return_value = np.zeros(7)
    sim.get_joint_velocities.return_value = np.zeros(7)
    sim.get_joint_torques.return_value = np.zeros(7)

    return sim


@pytest.fixture
def mock_real() -> MagicMock:
    """Mock real-time controller."""
    real = MagicMock()

    # Setup default state
    state = RobotState(
        timestamp=1.0,
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        joint_torques=np.zeros(7),
    )
    real.get_last_state.return_value = state

    return real


def test_anomaly_report_validation() -> None:
    """Test validation of AnomalyReport."""
    with pytest.raises(ValueError, match="severity must be between 0 and 1"):
        AnomalyReport(
            timestamp=0.0,
            anomaly_type=AnomalyType.COLLISION,
            severity=1.5,
            affected_joints=[0],
            description="Test",
            recommended_action="Stop",
        )

    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        AnomalyReport(
            timestamp=0.0,
            anomaly_type=AnomalyType.COLLISION,
            severity=0.5,
            affected_joints=[0],
            description="Test",
            recommended_action="Stop",
            confidence=2.0,
        )


def test_digital_twin_synchronize(mock_sim: MagicMock, mock_real: MagicMock) -> None:
    """Test synchronizing digital twin."""
    twin = DigitalTwin(mock_sim, mock_real)

    # Real state changes
    new_state = RobotState(
        timestamp=2.0,
        joint_positions=np.ones(7),
        joint_velocities=np.ones(7) * 0.5,
        joint_torques=np.zeros(7),
    )
    mock_real.get_last_state.return_value = new_state

    # Synchronize
    err = twin.synchronize()

    # error = norm(pos_diff) + 0.1 * norm(vel_diff)
    # pos_diff = ones - zeros -> norm is sqrt(7) = 2.645
    # vel_diff = 0.5*ones - zeros -> norm is sqrt(7)*0.5 = 1.322
    # expected err = 2.645 + 0.1 * 1.322 = 2.777

    assert err > 0.0
    mock_sim.set_joint_positions.assert_called_once()
    mock_sim.set_joint_velocities.assert_called_once()


def test_digital_twin_predict(mock_sim: MagicMock, mock_real: MagicMock) -> None:
    """Test prediction over horizon."""
    twin = DigitalTwin(mock_sim, mock_real)

    controls = np.zeros((10, 7))
    traj = twin.predict(horizon=0.01, control_sequence=controls, dt=0.001)

    assert traj.shape == (
        11,
        14,
    )  # 10 steps + 1 initial, 14 dims per state (7 pos + 7 vel)
    assert mock_sim.step.call_count == 10
    assert mock_sim.set_joint_torques.call_count == 10


def test_digital_twin_detect_anomaly(mock_sim: MagicMock, mock_real: MagicMock) -> None:
    """Test detecting anomalies."""
    twin = DigitalTwin(mock_sim, mock_real)

    # Set anomaly threshold low
    twin.set_anomaly_threshold(0.05)

    # Real positions drift from simulation
    new_state = RobotState(
        timestamp=3.0,
        joint_positions=np.ones(7) * 0.1,  # larger than threshold 0.05
        joint_velocities=np.zeros(7),
        joint_torques=np.zeros(7),
    )
    mock_real.get_last_state.return_value = new_state

    anomaly = twin.detect_anomaly()
    assert anomaly is not None
    assert anomaly.anomaly_type == AnomalyType.MODEL_MISMATCH
    assert 0 in anomaly.affected_joints

    # Test torque spike
    new_state2 = RobotState(
        timestamp=4.0,
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        joint_torques=np.ones(7) * 20.0,  # Spike
    )
    mock_real.get_last_state.return_value = new_state2

    anomaly2 = twin.detect_anomaly()
    assert anomaly2 is not None
    assert anomaly2.anomaly_type == AnomalyType.COLLISION


def test_digital_twin_estimated_contacts(mock_sim: MagicMock, mock_real: MagicMock) -> None:
    """Test contact estimation."""
    twin = DigitalTwin(mock_sim, mock_real)

    state_with_contacts = RobotState(
        timestamp=1.0,
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        joint_torques=np.zeros(7),
        ft_wrenches={"wrist": np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
        contact_states=[True, False],
    )
    mock_real.get_last_state.return_value = state_with_contacts

    contacts = twin.get_estimated_contacts()
    assert len(contacts) == 2
    assert contacts[0]["sensor"] == "wrist"
    assert contacts[0]["magnitude"] == 10.0
    assert contacts[1]["contact_id"] == 0


def test_digital_twin_anomaly_history(mock_sim: MagicMock, mock_real: MagicMock) -> None:
    """Test anomaly history tracking."""
    twin = DigitalTwin(mock_sim, mock_real)

    # Add fake anomalies
    twin._anomaly_history.append(
        AnomalyReport(
            timestamp=1.0,
            anomaly_type=AnomalyType.SLIP,
            severity=1.0,
            affected_joints=[],
            description="",
            recommended_action="Fix",
        )
    )
    twin._anomaly_history.append(
        AnomalyReport(
            timestamp=5.0,
            anomaly_type=AnomalyType.SLIP,
            severity=1.0,
            affected_joints=[],
            description="",
            recommended_action="Fix",
        )
    )

    # Current time mocked via real_state timestamp = 1.0 (defaults from fixture)
    history = twin.get_anomaly_history()
    assert len(history) == 2

    twin.clear_anomaly_history()
    assert len(twin.get_anomaly_history()) == 0


def test_compute_virtual_forces(mock_sim: MagicMock, mock_real: MagicMock) -> None:
    """Test virtual force computation."""
    twin = DigitalTwin(mock_sim, mock_real)

    # Real torque has some values
    state = RobotState(
        timestamp=1.0,
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        joint_torques=np.ones(7) * 5.0,
    )
    mock_real.get_last_state.return_value = state

    # Sim torque is zero
    mock_sim.get_joint_torques.return_value = np.zeros(7)

    v_forces = twin.compute_virtual_forces()
    assert len(v_forces) == 6
    assert np.allclose(v_forces, np.ones(6) * 5.0)
