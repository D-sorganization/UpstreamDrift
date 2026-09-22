"""Mock physics engine for testing and light installation.

This module provides a mock implementation of the PhysicsEngine protocol
that can be used when heavy physics dependencies (MuJoCo, Drake, etc.)
are not available.

Usage:
    # In test fixtures
    from src.shared.python.engine_core.mock_engine import MockPhysicsEngine
    engine = MockPhysicsEngine()

    # For light development (set environment variable)
    export GOLF_USE_MOCK_ENGINE=1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.python.core.constants import GRAVITY
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MockPhysicsEngine:
    """Mock physics engine implementing PhysicsEngine protocol.

    ``num_q`` / ``num_v`` / ``num_u`` default to ``num_joints`` but may differ
    (quaternion layouts, under-actuation). Optional ``control_limits`` saturate
    applied controls while preserving the requested command.
    """

    num_joints: int = 7
    num_q: int | None = None
    num_v: int | None = None
    num_u: int | None = None
    control_limits: tuple[float, float] | None = None
    timestep: float = 0.001
    model_name: str = "mock_golfer"
    damping: float = 0.1

    _time: float = field(default=0.0, init=False)
    _positions: np.ndarray = field(default_factory=lambda: np.array([]))
    _velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    _accelerations: np.ndarray = field(default_factory=lambda: np.array([]))
    _torques: np.ndarray = field(default_factory=lambda: np.array([]))
    _requested_torques: np.ndarray = field(default_factory=lambda: np.array([]))
    _is_loaded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        n_q = self.num_joints if self.num_q is None else int(self.num_q)
        n_v = self.num_joints if self.num_v is None else int(self.num_v)
        n_u = self.num_joints if self.num_u is None else int(self.num_u)
        if n_q <= 0 or n_v <= 0 or n_u <= 0:
            raise ValueError("num_q, num_v and num_u must be > 0")
        object.__setattr__(self, "num_q", n_q)
        object.__setattr__(self, "num_v", n_v)
        object.__setattr__(self, "num_u", n_u)
        self._positions = np.zeros(n_q)
        self._velocities = np.zeros(n_v)
        self._accelerations = np.zeros(n_v)
        self._torques = np.zeros(n_u)
        self._requested_torques = np.zeros(n_u)
        logger.info("MockPhysicsEngine initialized n_q=%d n_v=%d n_u=%d", n_q, n_v, n_u)

    def load_model(self, model_path: str) -> None:
        if model_path is None:
            raise ValueError("model_path must be provided")
        logger.info("MockPhysicsEngine: Loading model from %s", model_path)
        self._is_loaded = True
        self.model_name = model_path

    def load_from_path(self, path: str) -> None:
        self.load_model(path)

    def step(self, dt: float | None = None) -> None:
        if dt is None:
            dt = self.timestep
        self.forward()
        self._velocities = self._velocities + self._accelerations * dt
        if self._positions.shape[0] == self._velocities.shape[0]:
            self._positions = self._positions + self._velocities * dt
        else:
            n = min(self._positions.shape[0], self._velocities.shape[0])
            self._positions[:n] = self._positions[:n] + self._velocities[:n] * dt
        self._time += dt

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self._positions.copy(), self._velocities.copy()

    def get_state_dict(self) -> dict[str, Any]:
        return {
            "time": self._time,
            "positions": self._positions.copy(),
            "velocities": self._velocities.copy(),
            "accelerations": self._accelerations.copy(),
            "torques": self._torques.copy(),
            "is_loaded": self._is_loaded,
        }

    def set_state(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        if positions is None:
            raise ValueError("positions must be provided")
        self._positions = np.array(positions, dtype=float)
        self._velocities = np.array(velocities, dtype=float)

    def set_joint_positions(self, positions: np.ndarray) -> None:
        self._positions = np.array(positions, dtype=float)

    def set_joint_velocities(self, velocities: np.ndarray) -> None:
        self._velocities = np.array(velocities, dtype=float)

    def apply_torque(self, joint_name: str, torque: float) -> None:
        try:
            if joint_name.startswith("joint_"):
                idx = int(joint_name.split("_")[1])
            else:
                idx = hash(joint_name) % int(self.num_u or self.num_joints)
            self._torques[idx] = torque
            self._requested_torques[idx] = torque
        except (ValueError, IndexError) as e:
            logger.warning("Failed to apply torque to %s: %s", joint_name, e)

    def set_control(self, torques: list[float] | np.ndarray) -> None:
        if torques is None:
            raise ValueError("torques must be provided")
        n_u = int(self.num_u) if self.num_u is not None else self.num_joints
        requested = np.asarray(torques, dtype=float).reshape(-1)
        if requested.size < n_u:
            requested = np.pad(requested, (0, n_u - requested.size), mode="constant")
        requested = requested[:n_u].copy()
        self._requested_torques = requested
        applied = requested.copy()
        if self.control_limits is not None:
            lo, hi = self.control_limits
            applied = np.clip(applied, lo, hi)
        self._torques = applied

    def get_applied_control(self) -> np.ndarray:
        return self._torques.copy()

    def get_requested_control(self) -> np.ndarray:
        return self._requested_torques.copy()

    def get_control_dim(self) -> int:
        return int(self.num_u) if self.num_u is not None else self.num_joints

    def reset(self) -> None:
        n_q = int(self.num_q) if self.num_q is not None else self.num_joints
        n_v = int(self.num_v) if self.num_v is not None else self.num_joints
        n_u = int(self.num_u) if self.num_u is not None else self.num_joints
        self._time = 0.0
        self._positions = np.zeros(n_q)
        self._velocities = np.zeros(n_v)
        self._accelerations = np.zeros(n_v)
        self._torques = np.zeros(n_u)
        self._requested_torques = np.zeros(n_u)
        logger.info("MockPhysicsEngine reset")

    def get_joint_names(self) -> list[str]:
        n_q = int(self.num_q) if self.num_q is not None else self.num_joints
        return [f"joint_{i}" for i in range(n_q)]

    def get_joint_positions(self) -> np.ndarray:
        return self._positions.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self._velocities.copy()

    def get_joint_accelerations(self) -> np.ndarray:
        return self._accelerations.copy()

    def get_simulation_time(self) -> float:
        return self._time

    def get_timestep(self) -> float:
        return self.timestep

    def get_time(self) -> float:
        return self._time

    def set_time(self, time: float) -> None:
        if time is None or not np.isfinite(time):
            raise ValueError(f"time must be finite; got {time!r}")
        self._time = float(time)

    def get_full_state(self) -> dict[str, Any]:
        n_v = int(self.num_v) if self.num_v is not None else self.num_joints
        return {
            "q": self._positions.copy(),
            "v": self._velocities.copy(),
            "t": self._time,
            "M": np.eye(n_v),
        }

    def forward(self) -> None:
        n_v = int(self.num_v) if self.num_v is not None else self.num_joints
        n_u = int(self.num_u) if self.num_u is not None else self.num_joints
        tau = np.zeros(n_v)
        tau[: min(n_u, n_v)] = self._torques[: min(n_u, n_v)]
        bias = self.compute_bias_forces()
        self._accelerations = tau - bias

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        if content is None:
            raise ValueError("content must be provided")
        self._is_loaded = True
        self.model_name = "mock_model"

    def compute_mass_matrix(self) -> np.ndarray:
        n_v = int(self.num_v) if self.num_v is not None else self.num_joints
        return np.eye(n_v)

    def compute_bias_forces(self) -> np.ndarray:
        return self.damping * self._velocities

    def compute_gravity_forces(self) -> np.ndarray:
        n_v = int(self.num_v) if self.num_v is not None else self.num_joints
        g = np.zeros(n_v)
        g[0] = -GRAVITY
        return g

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        if qacc is None:
            raise ValueError("qacc must be provided")
        return self.compute_mass_matrix() @ qacc + self.compute_bias_forces()

    def compute_drift_acceleration(self) -> np.ndarray:
        m = self.compute_mass_matrix()
        return np.linalg.solve(m, -self.compute_bias_forces())

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        if tau is None:
            raise ValueError("tau must be provided")
        m = self.compute_mass_matrix()
        n_v = m.shape[0]
        mapped = np.zeros(n_v)
        n = min(n_v, np.asarray(tau).size)
        mapped[:n] = np.asarray(tau, dtype=float).reshape(-1)[:n]
        return np.linalg.solve(m, mapped)

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.compute_drift_acceleration()

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        if q is None:
            raise ValueError("q must be provided")
        return np.zeros(self.compute_mass_matrix().shape[0])

    def compute_contact_forces(self) -> np.ndarray:
        return np.zeros(3)

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        n_v = int(self.num_v) if self.num_v is not None else self.num_joints
        return {
            "linear": np.zeros((3, n_v)),
            "angular": np.zeros((3, n_v)),
        }

    def get_body_position(self, body_name: str) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    def get_body_velocity(self, body_name: str) -> np.ndarray:
        return np.zeros(6)


def get_mock_engine() -> MockPhysicsEngine:
    """Factory function to create a mock engine."""
    return MockPhysicsEngine()
