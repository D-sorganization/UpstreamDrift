"""Runtime engine protocols used by deployment, research, and RL code."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class PhysicsEngineProtocol(Protocol):
    """Structural contract for physics engines used outside engine_core."""

    n_q: int
    n_v: int

    def step(self) -> None:
        """Advance simulation by one integration step."""
        ...

    def reset(self) -> None:
        """Reset simulation state."""
        ...

    def get_joint_positions(self) -> NDArray[np.floating]:
        """Return generalized joint positions."""
        ...

    def get_joint_velocities(self) -> NDArray[np.floating]:
        """Return generalized joint velocities."""
        ...

    def get_joint_torques(self) -> NDArray[np.floating]:
        """Return current joint torques."""
        ...

    def set_joint_torques(self, torques: NDArray[np.floating]) -> None:
        """Apply joint torque commands."""
        ...

    def get_base_position(self) -> NDArray[np.floating]:
        """Return base position as xyz."""
        ...

    def get_base_velocity(self) -> NDArray[np.floating]:
        """Return base linear velocity as xyz."""
        ...

    def get_base_orientation(self) -> NDArray[np.floating]:
        """Return base orientation as quaternion."""
        ...

    def get_imu_data(self) -> NDArray[np.floating]:
        """Return IMU channels used by RL observations."""
        ...

    def get_contact_forces(self) -> NDArray[np.floating]:
        """Return contact force channels."""
        ...

    def render(self) -> NDArray[np.uint8]:
        """Render an RGB frame."""
        ...

    def close(self) -> None:
        """Release engine resources."""
        ...

    def __getattr__(self, name: str) -> Any:
        """Allow subsystem-specific optional engine extensions."""
        ...
