"""Wave 6 coverage: src.learning.sim2real.system_identification."""

from __future__ import annotations

import numpy as np
import pytest

from src.learning.imitation.dataset import Demonstration
from src.learning.sim2real.system_identification import (
    IdentificationResult,
    SystemIdentifier,
)


class FakeModel:
    """Minimal physics model that emulates a passive integrator."""

    def __init__(self, n: int = 2) -> None:
        self.n = n
        self.masses = np.ones(n)
        self.damping = np.ones(n) * 0.1
        self.friction = np.ones(n) * 0.5
        self.motor = np.ones(n)
        self._q = np.zeros(n)
        self._v = np.zeros(n)
        self._torques = np.zeros(n)

    def get_link_masses(self) -> np.ndarray:
        return self.masses

    def set_link_masses(self, v: np.ndarray) -> None:
        self.masses = v

    def get_joint_damping(self) -> np.ndarray:
        return self.damping

    def set_joint_damping(self, v: np.ndarray) -> None:
        self.damping = v

    def get_friction_coefficients(self) -> np.ndarray:
        return self.friction

    def set_friction_coefficients(self, v: np.ndarray) -> None:
        self.friction = v

    def get_motor_strength(self) -> np.ndarray:
        return self.motor

    def set_motor_strength(self, v: np.ndarray) -> None:
        self.motor = v

    def set_joint_positions(self, q: np.ndarray) -> None:
        self._q = q.copy()

    def set_joint_velocities(self, v: np.ndarray) -> None:
        self._v = v.copy()

    def set_joint_torques(self, t: np.ndarray) -> None:
        self._torques = t

    def step(self, dt: float) -> None:
        # Simple second-order integration with motor scaling.
        a = self._torques / self.masses
        self._v = self._v + a * dt
        self._q = self._q + self._v * dt

    def get_joint_positions(self) -> np.ndarray:
        return self._q.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self._v.copy()


def _make_demo(n_frames: int = 6, n_joints: int = 2) -> Demonstration:
    return Demonstration(
        timestamps=np.arange(n_frames, dtype=float) * 0.01,
        joint_positions=np.linspace(0, 1, n_frames * n_joints).reshape(
            n_frames, n_joints
        ),
        joint_velocities=np.ones((n_frames, n_joints)) * 0.1,
        actions=np.zeros((n_frames, n_joints)),
    )


class TestSystemIdentifier:
    def test_init_none_raises(self) -> None:
        with pytest.raises(ValueError):
            SystemIdentifier(None)  # type: ignore[arg-type]

    def test_default_bounds(self) -> None:
        ident = SystemIdentifier(FakeModel())
        bounds = ident.param_bounds
        assert "mass_scale" in bounds
        assert bounds["mass_scale"] == (0.5, 2.0)

    def test_custom_bounds(self) -> None:
        ident = SystemIdentifier(FakeModel(), param_bounds={"mass_scale": (0.9, 1.1)})
        assert ident.param_bounds == {"mass_scale": (0.9, 1.1)}

    def test_apply_params_modifies_model(self) -> None:
        m = FakeModel()
        ident = SystemIdentifier(m)
        ident._apply_params(np.array([2.0, 1.0, 1.0, 1.0, 0, 0, 0]))
        np.testing.assert_array_equal(m.masses, np.ones(2) * 2.0)

    def test_apply_params_none_raises(self) -> None:
        ident = SystemIdentifier(FakeModel())
        with pytest.raises(ValueError):
            ident._apply_params(None)  # type: ignore[arg-type]

    def test_simulate_trajectory_shape(self) -> None:
        m = FakeModel()
        ident = SystemIdentifier(m)
        state0 = np.zeros(4)
        actions = np.zeros((3, 2))
        traj = ident._simulate_trajectory(state0, actions, dt=0.01)
        # initial + 3 steps
        assert traj.shape == (4, 4)

    def test_compute_error(self) -> None:
        ident = SystemIdentifier(FakeModel())
        sim = np.zeros((3, 4))
        real = np.ones((3, 4))
        err = ident._compute_trajectory_error(sim, real)
        assert err == pytest.approx(1.0)

    def test_compute_error_with_weights(self) -> None:
        ident = SystemIdentifier(FakeModel())
        sim = np.zeros((2, 4))
        real = np.ones((2, 4))
        weights = np.array([0.5, 0.5, 0.0, 0.0])
        err = ident._compute_trajectory_error(sim, real, weights)
        # weighted diff = 0.5 for half the entries, 0 for rest
        assert err < 1.0

    def test_identify_returns_result(self) -> None:
        ident = SystemIdentifier(FakeModel(), param_bounds={"mass_scale": (0.5, 2.0)})
        demos = [_make_demo()]
        result = ident.identify_from_trajectories(
            demos, params_to_identify=["mass_scale"], max_iterations=3
        )
        assert isinstance(result, IdentificationResult)
        assert "mass_scale" in result.identified_params
        assert result.iterations >= 1

    def test_compute_reality_gap(self) -> None:
        ident = SystemIdentifier(FakeModel())
        sim = np.zeros((5, 4))
        real = np.ones((5, 4))
        gap = ident.compute_reality_gap(sim, real)
        assert gap["total_mse"] == pytest.approx(1.0)
        assert gap["max_position_error"] == pytest.approx(1.0)
        assert gap["trajectory_length"] == 5
        assert "joint_0_position_mse" in gap

    def test_validate_identification(self) -> None:
        ident = SystemIdentifier(FakeModel())
        demos = [_make_demo(), _make_demo()]
        params = {
            "mass_scale": 1.0,
            "friction_scale": 1.0,
            "damping_scale": 1.0,
            "motor_scale": 1.0,
        }
        metrics = ident.validate_identification(demos, params)
        assert metrics["n_trajectories"] == 2
        assert "mean_error" in metrics
