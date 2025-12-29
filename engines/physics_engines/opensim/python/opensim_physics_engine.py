"""OpenSim Physics Engine stub implementation."""

from __future__ import annotations

import numpy as np

from shared.python.interfaces import PhysicsEngine


class OpenSimPhysicsEngine(PhysicsEngine):
    """Stub for OpenSim Physics Engine."""

    @property
    def model_name(self) -> str:
        return "OpenSimStub"

    def load_from_path(self, path: str) -> None:
        raise NotImplementedError("OpenSim engine not yet implemented")

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        raise NotImplementedError("OpenSim engine not yet implemented")

    def reset(self) -> None:
        pass

    def step(self, dt: float | None = None) -> None:
        pass

    def forward(self) -> None:
        pass

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([]), np.array([])

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        pass

    def set_control(self, u: np.ndarray) -> None:
        pass

    def get_time(self) -> float:
        return 0.0

    def compute_mass_matrix(self) -> np.ndarray:
        return np.array([])

    def compute_bias_forces(self) -> np.ndarray:
        return np.array([])

    def compute_gravity_forces(self) -> np.ndarray:
        return np.array([])

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        return np.array([])

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        return None
