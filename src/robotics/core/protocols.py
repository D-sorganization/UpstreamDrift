"""Protocol definitions for robotics module.

This module defines the interfaces (protocols) that physics engines must
implement to support robotics functionality. Using protocols enables
engine-agnostic code that works with any compatible backend.

Design Principles:
    - Protocol-based polymorphism for engine agnosticism
    - Small, focused protocols (Interface Segregation)
    - Runtime checkable for validation

Example:
    >>> from src.robotics.core.protocols import HumanoidCapable
    >>>
    >>> def compute_balance(engine: HumanoidCapable) -> bool:
    ...     com = engine.get_com_position()
    ...     support = engine.get_support_polygon()
    ...     return point_in_polygon(com[:2], support)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class RoboticsCapable(Protocol):
    """Minimum protocol for robotics functionality.

    Any physics engine implementing this protocol can be used with
    the basic robotics components (contact, control, etc.).

    Design by Contract:
        Implementations must satisfy PhysicsEngine invariants plus:
        - compute_mass_matrix returns SPD matrix
        - compute_jacobian returns finite values
        - State (q, v) dimensions are consistent
    """

    def get_state(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get current state (positions, velocities).

        Returns:
            Tuple of (q, v) numpy arrays.
        """
        ...

    def set_state(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> None:
        """Set the current state.

        Args:
            q: Generalized coordinates.
            v: Generalized velocities.
        """
        ...

    def compute_mass_matrix(self) -> NDArray[np.float64]:
        """Compute the mass matrix M(q).

        Returns:
            Symmetric positive definite mass matrix (n_v, n_v).
        """
        ...

    def compute_bias_forces(self) -> NDArray[np.float64]:
        """Compute bias forces C(q,v) + g(q).

        Returns:
            Bias force vector (n_v,).
        """
        ...

    def compute_gravity_forces(self) -> NDArray[np.float64]:
        """Compute gravity forces g(q).

        Returns:
            Gravity vector (n_v,).
        """
        ...

    def compute_jacobian(
        self,
        body_name: str,
    ) -> dict[str, NDArray[np.float64]] | None:
        """Compute spatial Jacobian for a body.

        Args:
            body_name: Name of the body frame.

        Returns:
            Dictionary with 'linear' (3, n_v) and 'angular' (3, n_v),
            or None if body not found.
        """
        ...

    def get_time(self) -> float:
        """Get current simulation time.

        Returns:
            Simulation time in seconds.
        """
        ...


@runtime_checkable
class ContactCapable(Protocol):
    """Protocol for engines with contact detection support.

    Engines implementing this protocol can be used with the
    ContactManager for multi-contact scenarios.
    """

    def get_contact_count(self) -> int:
        """Get number of active contacts.

        Returns:
            Number of contacts currently detected.
        """
        ...

    def get_contact_info(self, contact_idx: int) -> dict[str, Any]:
        """Get information about a specific contact.

        Args:
            contact_idx: Contact index (0 to contact_count-1).

        Returns:
            Dictionary containing:
                - 'body_a': str, first body name
                - 'body_b': str, second body name
                - 'position': (3,) contact position
                - 'normal': (3,) contact normal
                - 'penetration': float, penetration depth
                - 'force': (3,) contact force (if available)
        """
        ...

    def get_contact_jacobian(
        self,
        contact_idx: int,
    ) -> NDArray[np.float64] | None:
        """Get Jacobian for a contact point.

        Args:
            contact_idx: Contact index.

        Returns:
            Contact Jacobian (3, n_v) or (6, n_v) for wrench,
            or None if not available.
        """
        ...


@runtime_checkable
class HumanoidCapable(RoboticsCapable, Protocol):
    """Protocol for humanoid robot capabilities.

    Extends RoboticsCapable with centroidal dynamics and
    foot contact information required for bipedal locomotion.
    """

    def get_com_position(self) -> NDArray[np.float64]:
        """Get center of mass position in world frame.

        Returns:
            CoM position (3,) [m].
        """
        ...

    def get_com_velocity(self) -> NDArray[np.float64]:
        """Get center of mass velocity in world frame.

        Returns:
            CoM velocity (3,) [m/s].
        """
        ...

    def get_total_mass(self) -> float:
        """Get total robot mass.

        Returns:
            Total mass [kg].
        """
        ...

    def compute_centroidal_momentum(self) -> NDArray[np.float64]:
        """Compute 6D centroidal momentum.

        Returns:
            Centroidal momentum [linear(3), angular(3)] = (6,).
            Linear in [kg*m/s], angular in [kg*m^2/s].
        """
        ...

    def compute_centroidal_momentum_matrix(self) -> NDArray[np.float64]:
        """Compute centroidal momentum matrix A_G(q).

        The centroidal momentum h = A_G(q) @ v.

        Returns:
            Centroidal momentum matrix (6, n_v).
        """
        ...

    def get_foot_position(self, foot: str) -> NDArray[np.float64]:
        """Get foot position in world frame.

        Args:
            foot: 'left' or 'right'.

        Returns:
            Foot position (3,) [m].
        """
        ...

    def get_foot_velocity(self, foot: str) -> NDArray[np.float64]:
        """Get foot velocity in world frame.

        Args:
            foot: 'left' or 'right'.

        Returns:
            Foot velocity (3,) [m/s].
        """
        ...

    def get_foot_jacobian(self, foot: str) -> NDArray[np.float64]:
        """Get foot Jacobian.

        Args:
            foot: 'left' or 'right'.

        Returns:
            Foot Jacobian (6, n_v) for position and orientation.
        """
        ...


@runtime_checkable
class ManipulationCapable(RoboticsCapable, Protocol):
    """Protocol for manipulation robot capabilities.

    Extends RoboticsCapable with end-effector and gripper
    functionality for manipulation tasks.
    """

    def get_end_effector_pose(
        self,
        ee_name: str,
    ) -> NDArray[np.float64]:
        """Get end-effector pose in world frame.

        Args:
            ee_name: End-effector name/identifier.

        Returns:
            Pose as 7D vector [x, y, z, qw, qx, qy, qz].
        """
        ...

    def get_end_effector_velocity(
        self,
        ee_name: str,
    ) -> NDArray[np.float64]:
        """Get end-effector velocity (twist) in world frame.

        Args:
            ee_name: End-effector name/identifier.

        Returns:
            Twist as 6D vector [vx, vy, vz, wx, wy, wz].
        """
        ...

    def get_end_effector_jacobian(
        self,
        ee_name: str,
    ) -> NDArray[np.float64]:
        """Get end-effector Jacobian.

        Args:
            ee_name: End-effector name/identifier.

        Returns:
            Jacobian (6, n_v).
        """
        ...

    def get_gripper_state(
        self,
        gripper_name: str,
    ) -> dict[str, Any]:
        """Get gripper state.

        Args:
            gripper_name: Gripper identifier.

        Returns:
            Dictionary containing:
                - 'position': float, gripper opening [m]
                - 'velocity': float, gripper velocity [m/s]
                - 'force': float, grip force [N]
                - 'is_grasping': bool, object detected
        """
        ...

    def set_gripper_command(
        self,
        gripper_name: str,
        command: float,
    ) -> None:
        """Set gripper command.

        Args:
            gripper_name: Gripper identifier.
            command: Position command [m] or force command [N].
        """
        ...


@runtime_checkable
class DynamicsComputable(Protocol):
    """Protocol for advanced dynamics computations.

    Provides inverse dynamics and forward dynamics capabilities
    needed for model-based control.
    """

    def compute_inverse_dynamics(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        a: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute inverse dynamics: tau = ID(q, v, a).

        Args:
            q: Joint positions.
            v: Joint velocities.
            a: Joint accelerations.

        Returns:
            Required joint torques (n_v,).
        """
        ...

    def compute_forward_dynamics(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        tau: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute forward dynamics: a = FD(q, v, tau).

        Args:
            q: Joint positions.
            v: Joint velocities.
            tau: Joint torques.

        Returns:
            Joint accelerations (n_v,).
        """
        ...

    def compute_aba(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        tau: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute forward dynamics using Articulated Body Algorithm.

        More efficient than explicit M^{-1} computation.

        Args:
            q: Joint positions.
            v: Joint velocities.
            tau: Joint torques.

        Returns:
            Joint accelerations (n_v,).
        """
        ...


@runtime_checkable
class Simulatable(Protocol):
    """Protocol for simulatable engines.

    Provides simulation stepping and reset capabilities.
    """

    def step(self, dt: float | None = None) -> None:
        """Advance simulation by one timestep.

        Args:
            dt: Timestep. Uses default if None.
        """
        ...

    def reset(self) -> None:
        """Reset simulation to initial state."""
        ...

    def forward(self) -> None:
        """Compute forward kinematics without advancing time."""
        ...


def is_robotics_capable(engine: Any) -> bool:
    """Check if engine implements RoboticsCapable protocol.

    Args:
        engine: Object to check.

    Returns:
        True if engine implements required methods.
    """
    return isinstance(engine, RoboticsCapable)


def is_humanoid_capable(engine: Any) -> bool:
    """Check if engine implements HumanoidCapable protocol.

    Args:
        engine: Object to check.

    Returns:
        True if engine implements required methods.
    """
    return isinstance(engine, HumanoidCapable)


def is_manipulation_capable(engine: Any) -> bool:
    """Check if engine implements ManipulationCapable protocol.

    Args:
        engine: Object to check.

    Returns:
        True if engine implements required methods.
    """
    return isinstance(engine, ManipulationCapable)
