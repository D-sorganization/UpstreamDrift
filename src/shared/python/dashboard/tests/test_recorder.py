"""Tests for the GenericPhysicsRecorder."""

from __future__ import annotations

import numpy as np

from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
from src.shared.python.engine_core.interfaces import PhysicsEngine


class MockPhysicsEngine(PhysicsEngine):
    """Minimal PhysicsEngine implementation for recorder tests."""

    def __init__(self) -> None:
        """Initialize with zero state."""
        self.q = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0.0

    @property
    def model_name(self) -> str:
        """Return the mock model name."""
        return "MockModel"

    def load_from_path(self, path: str) -> None:
        """No-op model loading from path."""

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """No-op model loading from string."""

    def reset(self) -> None:
        """No-op reset."""

    def step(self, dt: float | None = None) -> None:
        """Advance mock simulation by one timestep."""
        self.t += 0.01
        self.v += 0.1
        self.q += self.v * 0.01

    def forward(self) -> None:
        """No-op forward kinematics."""

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current position and velocity arrays."""
        return self.q, self.v

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set position and velocity arrays."""
        self.q = q
        self.v = v

    def set_control(self, u: np.ndarray) -> None:
        """No-op control setter."""

    def get_time(self) -> float:
        """Return current simulation time."""
        return self.t

    def compute_mass_matrix(self) -> np.ndarray:
        """Return identity mass matrix."""
        return np.eye(2)

    def compute_bias_forces(self) -> np.ndarray:
        """Return zero bias forces."""
        return np.zeros(2)

    def compute_gravity_forces(self) -> np.ndarray:
        """Return zero gravity forces."""
        return np.zeros(2)

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Return zero inverse dynamics torques."""
        return np.zeros(2)

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Return None (no Jacobian available)."""
        return None

    def compute_drift_acceleration(self) -> np.ndarray:
        """Return zero drift acceleration."""
        return np.zeros(2)

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Return zero control acceleration."""
        return np.zeros(2)

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return zero zero-torque counterfactual acceleration."""
        return np.zeros(2)

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Return zero zero-velocity counterfactual torque."""
        return np.zeros(2)


def test_recorder_basic() -> None:
    """Test basic recording and retrieval of time series data."""
    engine = MockPhysicsEngine()
    recorder = GenericPhysicsRecorder(engine)

    recorder.start()
    engine.step()
    recorder.record_step()
    engine.step()
    recorder.record_step()
    recorder.stop()

    times, positions = recorder.get_time_series("joint_positions")
    assert len(times) == 2
    assert len(positions) == 2

    data = recorder.get_data_dict()
    assert "times" in data
    assert "joint_positions" in data
    assert data["model_name"] == "MockModel"


def test_recorder_analysis() -> None:
    """Test post-hoc analysis and counterfactual series retrieval."""
    engine = MockPhysicsEngine()
    recorder = GenericPhysicsRecorder(engine)

    recorder.start()
    recorder.record_step()
    recorder.stop()

    recorder.compute_analysis_post_hoc()

    times, ztcf = recorder.get_counterfactual_series("ztcf_accel")
    assert len(times) == 1
    assert ztcf.shape == (1, 2)
