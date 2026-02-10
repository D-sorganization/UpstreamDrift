"""Indexed Acceleration Analysis Utilities.

Section H2 Implementation: Closure-verified acceleration decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from shared.python.interfaces import PhysicsEngine  # noqa: F401


@dataclass
class IndexedAcceleration:
    """Section H2: Labeled acceleration components indexed by physical cause.

    The CLOSURE REQUIREMENT (non-negotiable): All components MUST sum to the
    total acceleration within tolerance. If closure fails, the physics model is invalid.

    Attributes:
        gravity: Acceleration due to gravity only [rad/s² or m/s²]
        coriolis: Acceleration due to velocity-dependent forces
        centrifugal: Acceleration due to centrifugal effects (can be merged with coriolis)
        applied_torque: Acceleration from control inputs (muscles or torques)
        constraint: Acceleration from constraint reactions (loop closures, contacts)
        external: Acceleration from external forces (ground reaction, aerodynamic)
    """

    gravity: np.ndarray
    coriolis: np.ndarray
    applied_torque: np.ndarray
    constraint: np.ndarray
    external: np.ndarray
    centrifugal: np.ndarray | None = None  # Optional: can be merged into coriolis

    @property
    def total(self) -> np.ndarray:
        """Section H2: Sum of all indexed components.

        This MUST equal the measured/simulated acceleration from forward dynamics.
        """
        components = [
            self.gravity,
            self.coriolis,
            self.applied_torque,
            self.constraint,
            self.external,
        ]

        if self.centrifugal is not None:
            components.append(self.centrifugal)

        return sum(components)  # type: ignore

    def assert_closure(
        self,
        measured_acceleration: np.ndarray,
        atol_joint_space: float = 1e-6,
        atol_task_space: float = 1e-4,
    ) -> None:
        """Section H2: Verify summation requirement (CLOSURE TEST).

        The indexed components MUST sum to the total acceleration within tolerance.
        If this assertion fails, the acceleration decomposition is INVALID.

        Args:
            measured_acceleration: Total acceleration from forward dynamics
            atol_joint_space: Tolerance for joint space [rad/s²] (default: 1e-6)
            atol_task_space: Tolerance for task space [m/s²] (default: 1e-4)

        Raises:
            AccelerationClosureError: If residual exceeds tolerance

        Example:
            >>> engine.set_state(q, v)
            >>> a_total = engine.compute_forward_dynamics()
            >>> indexed = compute_indexed_acceleration(engine, tau)
            >>> indexed.assert_closure(a_total)  # Raises if closure fails
        """
        residual = measured_acceleration - self.total
        max_error = np.max(np.abs(residual))

        # Choose tolerance based on magnitude (heuristic for joint vs task space)
        tolerance = (
            atol_task_space
            if np.mean(np.abs(measured_acceleration)) > 1.0
            else atol_joint_space
        )

        if max_error > tolerance:
            raise AccelerationClosureError(
                f"Indexed acceleration closure failed!\n"
                f"  Max residual: {max_error:.2e} (tolerance: {tolerance:.2e})\n"
                f"  Residual: {residual}\n"
                f"  Total reconstructed: {self.total}\n"
                f"  Total measured: {measured_acceleration}\n"
                f"\nPossible causes:\n"
                f"  1. Missing force component in decomposition\n"
                f"  2. Incorrect M⁻¹ computation\n"
                f"  3. Numerical instability in physics engine"
            )

    def get_contribution_percentages(self) -> dict[str, float]:
        """Calculate percentage contribution of each component to total acceleration.

        Returns:
            Dictionary mapping component names to percentage contributions.

        Example:
            >>> percentages = indexed.get_contribution_percentages()
            >>> print(f"Gravity contributed {percentages['gravity']:.1f}% to elbow acceleration")
        """
        total_magnitude = np.linalg.norm(self.total)

        if total_magnitude < 1e-12:
            # Near-zero acceleration - percentages undefined
            return dict.fromkeys(
                [
                    "gravity",
                    "coriolis",
                    "applied_torque",
                    "constraint",
                    "external",
                ],
                0.0,
            )

        return {
            "gravity": float(100.0 * np.linalg.norm(self.gravity) / total_magnitude),
            "coriolis": float(100.0 * np.linalg.norm(self.coriolis) / total_magnitude),
            "applied_torque": float(
                100.0 * np.linalg.norm(self.applied_torque) / total_magnitude
            ),
            "constraint": float(
                100.0 * np.linalg.norm(self.constraint) / total_magnitude
            ),
            "external": float(100.0 * np.linalg.norm(self.external) / total_magnitude),
        }


class AccelerationClosureError(Exception):
    """Raised when indexed acceleration components do not sum to total (closure failure)."""


def compute_indexed_acceleration_from_engine(
    engine: Any,  # PhysicsEngine protocol (avoid circular import)
    tau: np.ndarray,
) -> IndexedAcceleration:
    """Compute indexed acceleration using drift-control decomposition.

    Section H2 + F: Default implementation using engine's drift-control methods.

    Args:
        engine: Physics engine implementing PhysicsEngine protocol
        tau: Applied generalized forces [N·m or N]

    Returns:
        IndexedAcceleration with components populated

    Example:
        >>> indexed = compute_indexed_acceleration_from_engine(mujoco_engine, tau)
        >>> indexed.assert_closure(mujoco_engine.compute_forward_dynamics())
    """
    # Get drift and control components
    a_drift = engine.compute_drift_acceleration()
    a_control = engine.compute_control_acceleration(tau)

    # Further decompose drift into gravity and Coriolis
    M = engine.compute_mass_matrix()
    M_inv = np.linalg.inv(M)

    g_forces = engine.compute_gravity_forces()
    a_gravity = M_inv @ g_forces

    # Coriolis/centrifugal = drift - gravity
    a_coriolis = a_drift - a_gravity

    # Constraint and external forces (default to zero unless engine provides them)
    n_v = len(a_drift)
    a_constraint = np.zeros(n_v)
    a_external = np.zeros(n_v)

    return IndexedAcceleration(
        gravity=a_gravity,
        coriolis=a_coriolis,
        applied_torque=a_control,
        constraint=a_constraint,
        external=a_external,
    )
