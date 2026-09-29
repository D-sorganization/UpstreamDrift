"""Tests for the GenericPhysicsRecorder."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
from src.shared.python.engine_core.checkpoint import StateCheckpoint
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

    @property
    def engine_type(self) -> str:
        """Return the engine type identifier."""
        return "mock"

    def save_checkpoint(self) -> StateCheckpoint:
        """Save current state as a checkpoint."""
        return StateCheckpoint(
            id="mock_cp",
            timestamp=self.t,
            wall_time=0.0,
            engine_type=self.engine_type,
            engine_state={"q": self.q.tolist(), "v": self.v.tolist()},
            q=tuple(self.q.tolist()),
            v=tuple(self.v.tolist()),
        )

    def restore_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        """Restore state from a checkpoint."""
        return


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


@pytest.mark.unit
def test_ensure_capacity_boundary_below_max_samples_growth() -> None:
    """Regression test for #8933 item 4.

    ``_ensure_capacity`` must only grow buffers whose current shape is
    smaller than ``new_capacity``. Several buffers (e.g. ``com_position``,
    ``ground_forces``, ``ground_moments``) are allocated directly at
    ``max_samples`` rather than the growable ``current_capacity``. When a
    growth step lands strictly between ``current_capacity`` and
    ``max_samples`` (i.e. ``current_capacity < new_capacity < max_samples``),
    those already-max-sized buffers must be left untouched. The old
    implementation instead resized every ndarray in ``self.data`` down to
    ``new_capacity`` and then tried to copy the (larger) old array into a
    (smaller) destination slice, raising a ``ValueError`` on shape mismatch.
    """
    engine = MockPhysicsEngine()
    # growth sequence with growth_factor=1.5 (default): 2 -> 3 -> 4 -> 5.
    # max_samples=5 means new_capacity=3 and new_capacity=4 both land
    # strictly below max_samples, exercising the boundary bug.
    recorder = GenericPhysicsRecorder(engine, max_samples=5, initial_capacity=2)

    recorder.start()
    for _ in range(4):
        engine.step()
        recorder.record_step()

    assert recorder.is_recording is True
    assert recorder.current_idx == 4
    assert recorder.current_capacity >= 4

    # Buffers allocated directly at max_samples must retain that full
    # allocation regardless of how current_capacity has grown.
    assert recorder.data["com_position"].shape[0] == recorder.max_samples
    assert recorder.data["ground_forces"].shape[0] == recorder.max_samples
    assert recorder.data["ground_moments"].shape[0] == recorder.max_samples

    # Growable buffers must have grown to at least current_capacity, and
    # recorded data must be preserved through the resize(s).
    assert recorder.data["joint_positions"].shape[0] == recorder.current_capacity
    assert recorder.data["times"].shape[0] == recorder.current_capacity

    recorder.stop()
    times, positions = recorder.get_time_series("joint_positions")
    assert len(times) == 4
    assert len(positions) == 4
