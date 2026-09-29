"""Comprehensive tests for deployment.digital_twin."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.deployment.digital_twin.estimator import (
    EstimatorConfig,
    StateEstimator,
)
from src.deployment.digital_twin.twin import (
    AnomalyReport,
    AnomalyType,
    DigitalTwin,
)
from src.deployment.realtime import RobotState


def _state(n: int = 3, **kw) -> RobotState:
    d = {
        "timestamp": 0.0,
        "joint_positions": np.zeros(n),
        "joint_velocities": np.zeros(n),
        "joint_torques": np.zeros(n),
    }
    d.update(kw)
    return RobotState(**d)


class TestAnomalyReport:
    def test_valid(self) -> None:
        r = AnomalyReport(
            timestamp=0.0,
            anomaly_type=AnomalyType.COLLISION,
            severity=0.5,
            affected_joints=[0],
            description="x",
            recommended_action="stop",
        )
        assert r.confidence == 0.9

    def test_bad_severity(self) -> None:
        with pytest.raises(ValueError, match="severity"):
            AnomalyReport(
                timestamp=0.0,
                anomaly_type=AnomalyType.COLLISION,
                severity=2.0,
                affected_joints=[],
                description="x",
                recommended_action="y",
            )

    def test_bad_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            AnomalyReport(
                timestamp=0.0,
                anomaly_type=AnomalyType.COLLISION,
                severity=0.5,
                affected_joints=[],
                description="x",
                recommended_action="y",
                confidence=2.0,
            )


class TestStateEstimator:
    def test_update_and_estimates(self) -> None:
        est = StateEstimator(n_dof=3)
        s = _state(n=3, timestamp=0.01)
        out = est.update(s, dt=0.01)
        assert "position" in out
        assert out["position"].shape == (3,)

    def test_update_auto_dt(self) -> None:
        est = StateEstimator(n_dof=3)
        # First call: dt computed as timestamp - 0
        est.update(_state(n=3, timestamp=0.01))
        # Second call same timestamp -> dt<=0 fallback
        est.update(_state(n=3, timestamp=0.01))

    def test_get_velocity_with_filter(self) -> None:
        est = StateEstimator(n_dof=3, config=EstimatorConfig(use_velocity_filter=True))
        v = est.get_velocity()
        assert v.shape == (3,)

    def test_get_velocity_no_filter(self) -> None:
        est = StateEstimator(n_dof=3, config=EstimatorConfig(use_velocity_filter=False))
        est.update(_state(n=3, joint_velocities=np.ones(3), timestamp=0.01), dt=0.01)
        v = est.get_velocity()
        assert v.shape == (3,)

    def test_get_acceleration(self) -> None:
        est = StateEstimator(n_dof=3)
        assert est.get_acceleration().shape == (3,)

    def test_get_covariance(self) -> None:
        est = StateEstimator(n_dof=3)
        cov = est.get_covariance()
        assert cov.shape == (9, 9)

    def test_get_uncertainties(self) -> None:
        est = StateEstimator(n_dof=3)
        assert est.get_position_uncertainty().shape == (3,)
        assert est.get_velocity_uncertainty().shape == (3,)

    def test_reset(self) -> None:
        est = StateEstimator(n_dof=3)
        est.reset(position=np.ones(3), velocity=np.ones(3) * 2)
        np.testing.assert_array_equal(est.get_position(), np.ones(3))

    def test_reset_defaults(self) -> None:
        est = StateEstimator(n_dof=3)
        est.reset()
        assert np.all(est.get_position() == 0)

    def test_predict_no_control(self) -> None:
        est = StateEstimator(n_dof=3)
        out = est.predict(0.1)
        assert out["position"].shape == (3,)

    def test_predict_with_control(self) -> None:
        est = StateEstimator(n_dof=3)
        out = est.predict(0.1, control=np.ones(3))
        np.testing.assert_array_equal(out["acceleration"], np.ones(3))


class TestDigitalTwin:
    def _twin(
        self, has_methods: bool = True
    ) -> tuple[DigitalTwin, MagicMock, MagicMock]:
        sim = MagicMock()
        if has_methods:
            sim.get_joint_positions.return_value = np.zeros(3)
            sim.get_joint_velocities.return_value = np.zeros(3)
            sim.get_joint_torques.return_value = np.zeros(3)
        else:
            del sim.get_joint_positions
            del sim.get_joint_velocities
            del sim.get_joint_torques
            del sim.set_joint_positions
            del sim.set_joint_velocities
            del sim.set_joint_torques
            del sim.step
        real = MagicMock()
        return DigitalTwin(sim, real), sim, real

    def test_sync_error_property(self) -> None:
        t, _, _ = self._twin()
        assert t.sync_error == 0.0

    def test_synchronize_no_state(self) -> None:
        t, _, real = self._twin()
        real.get_last_state.return_value = None
        assert t.synchronize() == 0.0

    def test_synchronize_with_state(self) -> None:
        t, sim, real = self._twin()
        real.get_last_state.return_value = _state(
            n=3, joint_positions=np.ones(3), joint_velocities=np.ones(3) * 0.5
        )
        err = t.synchronize()
        assert err > 0

    def test_synchronize_without_sim_methods(self) -> None:
        t, _, real = self._twin(has_methods=False)
        real.get_last_state.return_value = _state(n=3)
        err = t.synchronize()
        assert err >= 0

    def test_predict(self) -> None:
        t, sim, _ = self._twin()
        sim.get_joint_positions.return_value = np.zeros(3)
        sim.get_joint_velocities.return_value = np.zeros(3)
        controls = np.zeros((10, 3))
        traj = t.predict(0.01, controls, dt=0.001)
        assert traj.shape == (11, 6)

    def test_predict_without_methods(self) -> None:
        t, _, _ = self._twin(has_methods=False)
        traj = t.predict(0.005, np.zeros((5, 7)), dt=0.001)
        assert traj.shape[0] == 6

    def test_detect_anomaly_no_state(self) -> None:
        t, _, real = self._twin()
        real.get_last_state.return_value = None
        assert t.detect_anomaly() is None

    def test_detect_position_mismatch(self) -> None:
        t, sim, real = self._twin()
        sim.get_joint_positions.return_value = np.zeros(3)
        sim.get_joint_velocities.return_value = np.zeros(3)
        sim.get_joint_torques.return_value = np.zeros(3)
        real.get_last_state.return_value = _state(
            n=3, joint_positions=np.array([0.5, 0.0, 0.0])
        )
        anomaly = t.detect_anomaly()
        assert anomaly is not None
        assert anomaly.anomaly_type in (
            AnomalyType.MODEL_MISMATCH,
            AnomalyType.COLLISION,
        )

    def test_detect_torque_spike(self) -> None:
        t, sim, real = self._twin()
        sim.get_joint_positions.return_value = np.zeros(3)
        sim.get_joint_velocities.return_value = np.zeros(3)
        sim.get_joint_torques.return_value = np.zeros(3)
        real.get_last_state.return_value = _state(
            n=3, joint_torques=np.array([20.0, 0.0, 0.0])
        )
        anomaly = t.detect_anomaly()
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.COLLISION

    def test_detect_no_sim_methods(self) -> None:
        sim = MagicMock()
        del sim.get_joint_positions
        real = MagicMock()
        real.get_last_state.return_value = _state(n=3)
        t = DigitalTwin(sim, real)
        assert t.detect_anomaly() is None

    def test_detect_no_velocity_method(self) -> None:
        sim = MagicMock()
        sim.get_joint_positions.return_value = np.zeros(3)
        del sim.get_joint_velocities
        real = MagicMock()
        real.get_last_state.return_value = _state(n=3)
        t = DigitalTwin(sim, real)
        assert t.detect_anomaly() is None

    def test_get_estimated_contacts_no_state(self) -> None:
        t, _, real = self._twin()
        real.get_last_state.return_value = None
        assert t.get_estimated_contacts() == []

    def test_get_estimated_contacts_ft(self) -> None:
        t, _, real = self._twin()
        real.get_last_state.return_value = _state(
            n=3, ft_wrenches={"wrist": np.array([5.0, 0, 0, 0, 0, 0])}
        )
        contacts = t.get_estimated_contacts()
        assert len(contacts) == 1
        assert contacts[0]["sensor"] == "wrist"

    def test_get_estimated_contacts_below_threshold(self) -> None:
        t, _, real = self._twin()
        real.get_last_state.return_value = _state(
            n=3, ft_wrenches={"wrist": np.array([0.1, 0, 0, 0, 0, 0])}
        )
        assert t.get_estimated_contacts() == []

    def test_get_estimated_contacts_states(self) -> None:
        t, _, real = self._twin()
        real.get_last_state.return_value = _state(n=3, contact_states=[True, False])
        contacts = t.get_estimated_contacts()
        assert len(contacts) == 1

    def test_compute_virtual_forces_no_state(self) -> None:
        t, _, real = self._twin()
        real.get_last_state.return_value = None
        assert np.all(t.compute_virtual_forces() == 0)

    def test_compute_virtual_forces_with_torques(self) -> None:
        t, sim, real = self._twin()
        sim.get_joint_torques.return_value = np.zeros(7)
        real.get_last_state.return_value = _state(n=7, joint_torques=np.ones(7) * 2.0)
        f = t.compute_virtual_forces()
        assert f.shape == (6,)
        assert f[0] == 2.0

    def test_compute_virtual_forces_short(self) -> None:
        t, sim, real = self._twin()
        sim.get_joint_torques.return_value = np.zeros(3)
        real.get_last_state.return_value = _state(n=3)
        f = t.compute_virtual_forces()
        assert np.all(f == 0)

    def test_compute_virtual_forces_no_method(self) -> None:
        sim = MagicMock()
        del sim.get_joint_torques
        real = MagicMock()
        real.get_last_state.return_value = _state(n=3)
        t = DigitalTwin(sim, real)
        assert np.all(t.compute_virtual_forces() == 0)

    def test_anomaly_history(self) -> None:
        t, sim, real = self._twin()
        sim.get_joint_positions.return_value = np.zeros(3)
        sim.get_joint_velocities.return_value = np.zeros(3)
        sim.get_joint_torques.return_value = np.zeros(3)
        real.get_last_state.return_value = _state(
            n=3, joint_positions=np.array([0.5, 0, 0]), timestamp=10.0
        )
        t.detect_anomaly()
        assert len(t.get_anomaly_history()) >= 1
        # max_age filter
        real.get_last_state.return_value = _state(n=3, timestamp=10.0)
        h = t.get_anomaly_history(max_age=100.0)
        assert len(h) >= 1
        # No state
        real.get_last_state.return_value = None
        h = t.get_anomaly_history(max_age=1.0)
        assert isinstance(h, list)
        t.clear_anomaly_history()
        assert t.get_anomaly_history() == []

    def test_set_anomaly_threshold(self) -> None:
        t, _, _ = self._twin()
        t.set_anomaly_threshold(0.5)
        assert t._anomaly_threshold == 0.5
