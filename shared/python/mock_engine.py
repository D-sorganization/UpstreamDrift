"""Mock physics engine for testing and light installation.

This module provides a mock implementation of the PhysicsEngine protocol
that can be used when heavy physics dependencies (MuJoCo, Drake, etc.)
are not available.

Usage:
    # In test fixtures
    from shared.python.mock_engine import MockPhysicsEngine
    engine = MockPhysicsEngine()

    # For light development (set environment variable)
    export GOLF_USE_MOCK_ENGINE=1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class MockPhysicsEngine:
    """Mock physics engine implementing PhysicsEngine protocol.

    Provides deterministic, predictable behavior for:
    - Unit tests without heavy dependencies
    - UI development without physics engines
    - CI environments with limited resources
    """

    # Configuration
    num_joints: int = 7
    timestep: float = 0.001
    model_name: str = "mock_golfer"

    # State
    _time: float = field(default=0.0, init=False)
    _positions: np.ndarray = field(default_factory=lambda: np.array([]))
    _velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    _accelerations: np.ndarray = field(default_factory=lambda: np.array([]))
    _torques: np.ndarray = field(default_factory=lambda: np.array([]))
    _is_loaded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Initialize state arrays."""
        self._positions = np.zeros(self.num_joints)
        self._velocities = np.zeros(self.num_joints)
        self._accelerations = np.zeros(self.num_joints)
        self._torques = np.zeros(self.num_joints)
        LOGGER.info("MockPhysicsEngine initialized with %d joints", self.num_joints)

    # =========================================================================
    # PhysicsEngine Protocol Implementation
    # =========================================================================

    def load_model(self, model_path: str) -> None:
        """Load a model (mock implementation accepts any path).

        Args:
            model_path: Path to model file (ignored in mock)
        """
        LOGGER.info("MockPhysicsEngine: Loading model from %s", model_path)
        self._is_loaded = True
        self.model_name = model_path

    def load_from_path(self, path: str) -> None:
        """Alias for load_model for compatibility.

        Args:
            path: Path to model file
        """
        self.load_model(path)

    def step(self, dt: float | None = None) -> None:
        """Advance simulation by dt seconds.

        Uses simple Euler integration for mock physics.

        Args:
            dt: Timestep (uses default if not provided)
        """
        if dt is None:
            dt = self.timestep

        # Simple physics: F=ma with damping
        damping = 0.1
        mass = 1.0

        # Compute acceleration from torques
        self._accelerations = (self._torques - damping * self._velocities) / mass

        # Euler integration
        self._velocities = self._velocities + self._accelerations * dt
        self._positions = self._positions + self._velocities * dt

        self._time += dt

    def get_state(self) -> dict[str, Any]:
        """Get current simulation state.

        Returns:
            Dictionary containing positions, velocities, time, etc.
        """
        return {
            "time": self._time,
            "positions": self._positions.copy(),
            "velocities": self._velocities.copy(),
            "accelerations": self._accelerations.copy(),
            "torques": self._torques.copy(),
            "is_loaded": self._is_loaded,
        }

    def set_state(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Set simulation state.

        Args:
            positions: Joint positions array
            velocities: Joint velocities array
        """
        self._positions = np.array(positions)
        self._velocities = np.array(velocities)
        LOGGER.debug("State set: pos=%s, vel=%s", positions, velocities)

    def set_joint_positions(self, positions: np.ndarray) -> None:
        """Set joint positions.

        Args:
            positions: Joint positions array
        """
        self._positions = np.array(positions)

    def set_joint_velocities(self, velocities: np.ndarray) -> None:
        """Set joint velocities.

        Args:
            velocities: Joint velocities array
        """
        self._velocities = np.array(velocities)

    def apply_torque(self, joint_name: str, torque: float) -> None:
        """Apply torque to a joint.

        Args:
            joint_name: Name of joint (maps to index)
            torque: Torque value
        """
        # Map joint name to index (simple numeric mapping for mock)
        try:
            if joint_name.startswith("joint_"):
                idx = int(joint_name.split("_")[1])
            else:
                idx = hash(joint_name) % self.num_joints
            self._torques[idx] = torque
        except (ValueError, IndexError) as e:
            LOGGER.warning("Failed to apply torque to %s: %s", joint_name, e)

    def set_control(self, torques: list[float] | np.ndarray) -> None:
        """Set control torques for all joints.

        Args:
            torques: Array of torque values
        """
        self._torques = np.array(torques)[: self.num_joints]
        # Pad with zeros if not enough values
        if len(self._torques) < self.num_joints:
            self._torques = np.pad(
                self._torques,
                (0, self.num_joints - len(self._torques)),
                mode="constant",
            )

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self._time = 0.0
        self._positions = np.zeros(self.num_joints)
        self._velocities = np.zeros(self.num_joints)
        self._accelerations = np.zeros(self.num_joints)
        self._torques = np.zeros(self.num_joints)
        LOGGER.info("MockPhysicsEngine reset")

    # =========================================================================
    # Additional Methods for Compatibility
    # =========================================================================

    def get_joint_names(self) -> list[str]:
        """Get list of joint names.

        Returns:
            List of joint name strings
        """
        return [f"joint_{i}" for i in range(self.num_joints)]

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions.

        Returns:
            Array of joint positions
        """
        return self._positions.copy()

    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities.

        Returns:
            Array of joint velocities
        """
        return self._velocities.copy()

    def get_joint_accelerations(self) -> np.ndarray:
        """Get current joint accelerations.

        Returns:
            Array of joint accelerations
        """
        return self._accelerations.copy()

    def get_simulation_time(self) -> float:
        """Get current simulation time.

        Returns:
            Current time in seconds
        """
        return self._time

    def get_timestep(self) -> float:
        """Get simulation timestep.

        Returns:
            Timestep in seconds
        """
        return self.timestep

    # =========================================================================
    # Biomechanics Methods (Stubs for Protocol Compliance)
    # =========================================================================

    def compute_mass_matrix(self) -> np.ndarray:
        """Compute mass matrix (returns identity for mock).

        Returns:
            Mass matrix (n x n)
        """
        return np.eye(self.num_joints)

    def compute_jacobian(self, body_name: str) -> np.ndarray:
        """Compute Jacobian for a body (returns zeros for mock).

        Args:
            body_name: Name of body

        Returns:
            Jacobian matrix (6 x n)
        """
        return np.zeros((6, self.num_joints))

    def get_body_position(self, body_name: str) -> np.ndarray:
        """Get position of a body.

        Args:
            body_name: Name of body

        Returns:
            Position vector (3,)
        """
        # Return a position based on simple kinematics
        return np.array([0.0, 0.0, 1.0])

    def get_body_velocity(self, body_name: str) -> np.ndarray:
        """Get velocity of a body.

        Args:
            body_name: Name of body

        Returns:
            Velocity vector (6,) - linear and angular
        """
        return np.zeros(6)


def get_mock_engine() -> MockPhysicsEngine:
    """Factory function to create a mock engine.

    Returns:
        Configured MockPhysicsEngine instance
    """
    return MockPhysicsEngine()


# Note: MockPhysicsEngine is a partial implementation of PhysicsEngine protocol.
# It implements the core methods needed for testing but not all biomechanics methods.
# For full protocol compliance, see the real engine implementations in engines/.
