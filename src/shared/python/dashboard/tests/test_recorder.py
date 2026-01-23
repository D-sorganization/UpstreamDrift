import numpy as np

from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
from src.shared.python.interfaces import PhysicsEngine


class MockPhysicsEngine(PhysicsEngine):
    def __init__(self) -> None:
        self.q = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0.0

    @property
    def model_name(self) -> str:
        return "MockModel"

    def load_from_path(self, path: str) -> None:
        pass

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        pass

    def reset(self) -> None:
        pass

    def step(self, dt: float | None = None) -> None:
        self.t += 0.01
        self.v += 0.1
        self.q += self.v * 0.01

    def forward(self) -> None:
        pass

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.q, self.v

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        self.q = q
        self.v = v

    def set_control(self, u: np.ndarray) -> None:
        pass

    def get_time(self) -> float:
        return self.t

    def compute_mass_matrix(self) -> np.ndarray:
        return np.eye(2)

    def compute_bias_forces(self) -> np.ndarray:
        return np.zeros(2)

    def compute_gravity_forces(self) -> np.ndarray:
        return np.zeros(2)

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        return np.zeros(2)

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        return None

    def compute_drift_acceleration(self) -> np.ndarray:
        return np.zeros(2)

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        return np.zeros(2)

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.zeros(2)

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        return np.zeros(2)


def test_recorder_basic() -> None:
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
    engine = MockPhysicsEngine()
    recorder = GenericPhysicsRecorder(engine)

    recorder.start()
    recorder.record_step()
    recorder.stop()

    recorder.compute_analysis_post_hoc()

    times, ztcf = recorder.get_counterfactual_series("ztcf_accel")
    assert len(times) == 1
    assert ztcf.shape == (1, 2)
