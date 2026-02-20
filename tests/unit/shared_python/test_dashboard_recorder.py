from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
from src.shared.python.engine_core.checkpoint import StateCheckpoint
from src.shared.python.engine_core.interfaces import PhysicsEngine


class MockPhysicsEngine(PhysicsEngine):
    """Mock physics engine for recorder tests."""

    def __init__(self) -> None:
        # self.model_name = "MockEngine" # Handled by property
        self._time = 0.0
        self._q = np.array([0.0, 0.0])
        self._v = np.array([0.0, 0.0])

    def get_time(self) -> float:
        """Return current simulation time."""
        return self._time

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current state as (positions, velocities)."""
        return self._q, self._v

    def set_state(self, q, v) -> None:
        """Set the engine state."""
        self._q = q
        self._v = v

    def forward(self) -> None:
        """Advance the engine forward."""

    def set_control(self, u) -> None:
        """Set control input."""

    def compute_mass_matrix(self) -> np.ndarray:
        """Return the identity mass matrix."""
        return np.eye(2)

    def compute_ztcf(self, q, v) -> np.ndarray:
        """Return mock zero-torque counterfactual values."""
        return np.array([0.1, 0.2])

    def compute_zvcf(self, q) -> np.ndarray:
        """Return mock zero-velocity counterfactual values."""
        return np.array([0.3, 0.4])

    def compute_drift_acceleration(self) -> np.ndarray:
        """Return mock drift acceleration."""
        return np.array([0.01, 0.02])

    def compute_control_acceleration(self, tau) -> np.ndarray:
        """Return control acceleration equal to input torque."""
        return tau  # Identity for simplicity

    # Implement other required abstract methods with dummy implementations
    def compute_gravity_forces(self) -> np.ndarray:
        """Return zero gravity forces."""
        return np.zeros(2)

    def compute_jacobian(self, body_name) -> dict:
        """Return empty Jacobian dictionary."""
        return {}

    def compute_coriolis_centrifugal_forces(self) -> np.ndarray:
        """Return zero Coriolis/centrifugal forces."""
        return np.zeros(2)

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Return zero inverse dynamics."""
        return np.zeros(2)

    def compute_forward_dynamics(self, q, v, tau) -> np.ndarray:
        """Return zero forward dynamics."""
        return np.zeros(2)

    def compute_bias_forces(self) -> np.ndarray:
        """Return zero bias forces."""
        return np.zeros(2)

    def load_from_path(self, path) -> None:
        """Load model from path (no-op)."""

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load model from string (no-op)."""

    @property
    def model_name(self) -> str:
        """Return mock engine name."""
        return "MockEngine"

    def step(self, dt: float | None = None) -> None:
        """Advance simulation by one step."""
        if dt is not None:
            self._time += dt

    def reset(self) -> None:
        """Reset engine to initial state."""
        self._time = 0.0

    def get_joint_names(self) -> list[str]:
        """Return joint names."""
        return ["j1", "j2"]

    def get_actuator_names(self) -> list[str]:
        """Return actuator names."""
        return ["a1", "a2"]

    def get_body_names(self) -> list[str]:
        """Return body names."""
        return ["b1"]

    def get_body_mass(self, body_name) -> float:
        """Return unit body mass."""
        return 1.0

    def get_body_inertia(self, body_name) -> np.ndarray:
        """Return identity inertia matrix."""
        return np.eye(3)

    def get_body_position(self, body_name) -> np.ndarray:
        """Return zero body position."""
        return np.zeros(3)

    def get_body_rotation(self, body_name) -> np.ndarray:
        """Return identity rotation matrix."""
        return np.eye(3)

    def get_body_velocity(self, body_name) -> np.ndarray:
        """Return zero body velocity."""
        return np.zeros(6)

    def get_contact_forces(self) -> list:
        """Return empty contact forces."""
        return []

    def get_joint_limits(self) -> np.ndarray:
        """Return zero joint limits."""
        return np.zeros((2, 2))

    def get_actuator_limits(self) -> np.ndarray:
        """Return zero actuator limits."""
        return np.zeros((2, 2))

    def get_dof(self) -> int:
        """Return number of degrees of freedom."""
        return 2

    def get_num_actuators(self) -> int:
        """Return number of actuators."""
        return 2

    def get_num_bodies(self) -> int:
        """Return number of bodies."""
        return 1

    @property
    def engine_type(self) -> str:
        """Return the engine type identifier."""
        return "mock"

    def save_checkpoint(self) -> StateCheckpoint:
        """Save current state as a checkpoint."""
        return StateCheckpoint(
            id="mock_cp",
            timestamp=self._time,
            wall_time=0.0,
            engine_type=self.engine_type,
            engine_state={"q": self._q.tolist(), "v": self._v.tolist()},
            q=tuple(self._q.tolist()),
            v=tuple(self._v.tolist()),
        )

    def restore_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        """Restore state from a checkpoint."""
        return


class TestGenericPhysicsRecorder:
    """Tests for GenericPhysicsRecorder."""

    @pytest.fixture
    def engine(self) -> MockPhysicsEngine:
        """Create a MockPhysicsEngine instance."""
        return MockPhysicsEngine()

    @pytest.fixture
    def recorder(self, engine) -> GenericPhysicsRecorder:
        """Create a GenericPhysicsRecorder instance."""
        return GenericPhysicsRecorder(engine, max_samples=100)

    def test_initialization(self, recorder) -> None:
        """Test recorder initializes with correct defaults."""
        assert recorder.current_idx == 0
        assert not recorder.is_recording
        assert not recorder._buffers_initialized
        assert "times" in recorder.data

    def test_recording_cycle(self, recorder, engine) -> None:
        """Test full recording start-record-stop cycle."""
        recorder.start()
        assert recorder.is_recording

        # Record a few steps
        for i in range(10):
            engine._time = i * 0.1
            engine._q = np.array([float(i), 0.0])
            engine._v = np.array([1.0, 1.0])
            recorder.record_step(control_input=np.array([0.5, 0.5]))

        recorder.stop()
        assert not recorder.is_recording
        assert recorder.current_idx == 10
        assert recorder._buffers_initialized

        # Check data
        times, pos = recorder.get_time_series("joint_positions")
        assert len(times) == 10
        assert len(pos) == 10
        assert pos[9, 0] == 9.0

    def test_buffer_allocation_on_first_step(self, recorder, engine) -> None:
        """Test buffer allocation happens on first recorded step."""
        recorder.start()
        recorder.record_step()

        assert recorder.data["joint_positions"] is not None
        # Dynamic buffer sizing: starts at 1000 (or max_samples if smaller)
        assert recorder.data["joint_positions"].shape == (1000, 2)

    def test_buffer_full(self, engine) -> None:
        """Test recorder behavior when buffer is full."""
        # Test with initial_capacity=5 to match max_samples
        recorder = GenericPhysicsRecorder(engine, max_samples=5, initial_capacity=5)
        recorder.start()

        for _i in range(10):
            recorder.record_step()

        # With max_samples=5 and initial_capacity=5, buffer stops at 5
        # After 5 records, buffer is full and can't grow (already at max_samples)
        # Recording stops at frame 5
        assert recorder.current_idx == 5
        assert not recorder.is_recording  # Auto-stopped when buffer full

    def test_analysis_config_allocation(self, recorder, engine) -> None:
        """Test analysis config enables additional buffer allocation."""
        recorder.start()
        recorder.record_step()  # Initialize buffers

        # Enable ZTCF
        recorder.set_analysis_config({"ztcf": True})
        assert recorder.data["ztcf_accel"] is not None
        assert recorder.data["ztcf_accel"].shape == (100, 2)

        # Record with ZTCF enabled
        recorder.record_step()
        assert recorder.data["ztcf_accel"][1, 0] == 0.1  # Mock value

    def test_post_hoc_analysis(self, recorder, engine) -> None:
        """Test post-hoc analysis computation."""
        recorder.start()
        for _i in range(5):
            recorder.record_step(control_input=np.array([1.0, 1.0]))
        recorder.stop()

        recorder.compute_analysis_post_hoc()

        assert "ztcf" in recorder.data["counterfactuals"]
        times, vals = recorder.data["counterfactuals"]["ztcf"]
        assert len(times) == 5
        assert len(vals) == 5

    def test_get_data_dict(self, recorder, engine) -> None:
        """Test get_data_dict returns correct structure."""
        recorder.start()
        recorder.record_step()

        data = recorder.get_data_dict()
        assert data["num_frames"] == 1
        assert data["times"].shape == (1,)  # Sliced to actual length
