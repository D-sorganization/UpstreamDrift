import numpy as np
import pytest

from shared.python.dashboard.recorder import GenericPhysicsRecorder
from shared.python.interfaces import PhysicsEngine


class MockPhysicsEngine(PhysicsEngine):
    def __init__(self):
        # self.model_name = "MockEngine" # Handled by property
        self._time = 0.0
        self._q = np.array([0.0, 0.0])
        self._v = np.array([0.0, 0.0])

    def get_time(self) -> float:
        return self._time

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q, self._v

    def set_state(self, q, v):
        self._q = q
        self._v = v

    def forward(self):
        pass

    def set_control(self, u):
        pass

    def compute_mass_matrix(self):
        return np.eye(2)

    def compute_ztcf(self, q, v):
        return np.array([0.1, 0.2])

    def compute_zvcf(self, q):
        return np.array([0.3, 0.4])

    def compute_drift_acceleration(self):
        return np.array([0.01, 0.02])

    def compute_control_acceleration(self, tau):
        return tau  # Identity for simplicity

    # Implement other required abstract methods with dummy implementations
    def compute_gravity_forces(self):
        return np.zeros(2)

    def compute_jacobian(self, body_name):
        return {}

    def compute_coriolis_centrifugal_forces(self):
        return np.zeros(2)

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        return np.zeros(2)

    def compute_forward_dynamics(self, q, v, tau):
        return np.zeros(2)

    def compute_bias_forces(self):
        return np.zeros(2)

    def load_from_path(self, path):
        pass

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        pass

    @property
    def model_name(self):
        return "MockEngine"

    def step(self, dt: float | None = None) -> None:
        if dt is not None:
            self._time += dt

    def reset(self):
        self._time = 0.0

    def get_joint_names(self):
        return ["j1", "j2"]

    def get_actuator_names(self):
        return ["a1", "a2"]

    def get_body_names(self):
        return ["b1"]

    def get_body_mass(self, body_name):
        return 1.0

    def get_body_inertia(self, body_name):
        return np.eye(3)

    def get_body_position(self, body_name):
        return np.zeros(3)

    def get_body_rotation(self, body_name):
        return np.eye(3)

    def get_body_velocity(self, body_name):
        return np.zeros(6)

    def get_contact_forces(self):
        return []

    def get_joint_limits(self):
        return np.zeros((2, 2))

    def get_actuator_limits(self):
        return np.zeros((2, 2))

    def get_dof(self):
        return 2

    def get_num_actuators(self):
        return 2

    def get_num_bodies(self):
        return 1


class TestGenericPhysicsRecorder:
    @pytest.fixture
    def engine(self):
        return MockPhysicsEngine()

    @pytest.fixture
    def recorder(self, engine):
        return GenericPhysicsRecorder(engine, max_samples=100)

    def test_initialization(self, recorder):
        assert recorder.current_idx == 0
        assert not recorder.is_recording
        assert not recorder._buffers_initialized
        assert "times" in recorder.data

    def test_recording_cycle(self, recorder, engine):
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

    def test_buffer_allocation_on_first_step(self, recorder, engine):
        recorder.start()
        recorder.record_step()

        assert recorder.data["joint_positions"] is not None
        assert recorder.data["joint_positions"].shape == (100, 2)

    def test_buffer_full(self, engine):
        recorder = GenericPhysicsRecorder(engine, max_samples=5)
        recorder.start()

        for _i in range(10):
            recorder.record_step()

        assert recorder.current_idx == 5
        assert not recorder.is_recording  # Auto-stopped

    def test_analysis_config_allocation(self, recorder, engine):
        recorder.start()
        recorder.record_step()  # Initialize buffers

        # Enable ZTCF
        recorder.set_analysis_config({"ztcf": True})
        assert recorder.data["ztcf_accel"] is not None
        assert recorder.data["ztcf_accel"].shape == (100, 2)

        # Record with ZTCF enabled
        recorder.record_step()
        assert recorder.data["ztcf_accel"][1, 0] == 0.1  # Mock value

    def test_post_hoc_analysis(self, recorder, engine):
        recorder.start()
        for _i in range(5):
            recorder.record_step(control_input=np.array([1.0, 1.0]))
        recorder.stop()

        recorder.compute_analysis_post_hoc()

        assert "ztcf" in recorder.data["counterfactuals"]
        times, vals = recorder.data["counterfactuals"]["ztcf"]
        assert len(times) == 5
        assert len(vals) == 5

    def test_get_data_dict(self, recorder, engine):
        recorder.start()
        recorder.record_step()

        data = recorder.get_data_dict()
        assert data["num_frames"] == 1
        assert data["times"].shape == (1,)  # Sliced to actual length
