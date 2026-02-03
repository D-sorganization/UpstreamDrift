"""Friction cone utilities for contact dynamics.

This module provides friction cone representations and linearization
methods for use in optimization-based control.

Design by Contract:
    All functions validate inputs and guarantee valid outputs.
    Friction cones are always represented in consistent frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition


@dataclass(frozen=True)
class FrictionCone:
    """Friction cone representation.

    A friction cone constrains contact forces such that:
        ||f_t|| <= mu * f_n

    where f_t is tangential friction force and f_n is normal force.

    Attributes:
        mu: Friction coefficient (Coulomb friction).
        normal: Contact normal direction (unit vector).
        num_sides: Number of sides for linearized approximation.
    """

    mu: float
    normal: NDArray[np.float64]
    num_sides: int = 8

    def __post_init__(self) -> None:
        """Validate friction cone parameters."""
        if self.mu < 0:
            raise ValueError(f"Friction coefficient must be >= 0, got {self.mu}")
        if self.num_sides < 3:
            raise ValueError(f"num_sides must be >= 3, got {self.num_sides}")

        # Ensure normal is unit vector
        object.__setattr__(
            self, "normal", np.asarray(self.normal, dtype=np.float64)
        )
        norm = np.linalg.norm(self.normal)
        if norm < 1e-10:
            raise ValueError("Normal vector cannot be zero")
        object.__setattr__(self, "normal", self.normal / norm)

    def contains(
        self,
        force: NDArray[np.float64],
        tolerance: float = 1e-6,
    ) -> bool:
        """Check if force is inside friction cone.

        Args:
            force: Force vector (3,) to check.
            tolerance: Numerical tolerance.

        Returns:
            True if force satisfies friction constraint.
        """
        force = np.asarray(force, dtype=np.float64)
        f_n = float(np.dot(force, self.normal))

        if f_n < -tolerance:
            return False  # Pulling force (tensile)

        f_t = force - f_n * self.normal
        f_t_mag = float(np.linalg.norm(f_t))

        return f_t_mag <= self.mu * f_n + tolerance

    def get_generators(self) -> NDArray[np.float64]:
        """Get friction cone generators (edge directions).

        Returns generators for a linearized friction cone pyramid.

        Returns:
            Generator matrix (3, num_sides), each column is a generator.
        """
        return _compute_cone_generators(self.normal, self.mu, self.num_sides)


@precondition(
    lambda normal, mu, num_sides: mu >= 0,
    "Friction coefficient must be non-negative",
)
@precondition(
    lambda normal, mu, num_sides: num_sides >= 3,
    "Number of sides must be at least 3",
)
@postcondition(
    lambda result: result.shape[1] >= 3,
    "Must have at least 3 generators",
)
def _compute_cone_generators(
    normal: NDArray[np.float64],
    mu: float,
    num_sides: int,
) -> NDArray[np.float64]:
    """Compute friction cone generators.

    Creates a polyhedral approximation of the friction cone
    with num_sides generators arranged in a circle.

    Args:
        normal: Contact normal (3,).
        mu: Friction coefficient.
        num_sides: Number of generator directions.

    Returns:
        Generator matrix (3, num_sides).
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    # Find two vectors perpendicular to normal
    t1, t2 = _get_tangent_vectors(normal)

    # Generate directions around the cone
    generators = np.zeros((3, num_sides))
    for i in range(num_sides):
        angle = 2 * np.pi * i / num_sides
        tangent = np.cos(angle) * t1 + np.sin(angle) * t2
        # Generator: normal + mu * tangent
        generators[:, i] = normal + mu * tangent

    return generators


def _get_tangent_vectors(
    normal: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get two orthogonal tangent vectors for a normal.

    Args:
        normal: Unit normal vector (3,).

    Returns:
        Tuple of two orthogonal unit tangent vectors.
    """
    # Choose initial vector not parallel to normal
    if abs(normal[0]) < 0.9:
        init = np.array([1.0, 0.0, 0.0])
    else:
        init = np.array([0.0, 1.0, 0.0])

    # Gram-Schmidt
    t1 = init - np.dot(init, normal) * normal
    t1 = t1 / np.linalg.norm(t1)

    t2 = np.cross(normal, t1)
    t2 = t2 / np.linalg.norm(t2)

    return t1, t2


@precondition(
    lambda mu, normal, num_faces: mu >= 0,
    "Friction coefficient must be non-negative",
)
def linearize_friction_cone(
    mu: float,
    normal: NDArray[np.float64],
    num_faces: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Linearize friction cone as polyhedral constraint A @ f <= b.

    Approximates the second-order cone constraint ||f_t|| <= mu * f_n
    with linear inequalities.

    Args:
        mu: Friction coefficient.
        normal: Contact normal direction (3,).
        num_faces: Number of faces in polyhedral approximation.

    Returns:
        Tuple of (A, b) such that A @ f <= b enforces friction cone.
        A has shape (num_faces, 3), b has shape (num_faces,).
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    t1, t2 = _get_tangent_vectors(normal)

    # Each face: -n^T @ f + (1/mu) * d_i^T @ f <= 0
    # where d_i is the i-th tangent direction
    A = np.zeros((num_faces, 3))
    b = np.zeros(num_faces)

    for i in range(num_faces):
        angle = 2 * np.pi * i / num_faces
        d = np.cos(angle) * t1 + np.sin(angle) * t2

        # Constraint: d^T @ f <= mu * n^T @ f
        # Rewrite: d^T @ f - mu * n^T @ f <= 0
        A[i] = d - mu * normal
        b[i] = 0.0

    return A, b


@precondition(
    lambda contact_normal, contact_position, friction_coeff, num_faces: (
        friction_coeff >= 0
    ),
    "Friction coefficient must be non-negative",
)
def compute_friction_cone_constraint(
    contact_normal: NDArray[np.float64],
    contact_position: NDArray[np.float64],
    friction_coeff: float,
    num_faces: int = 8,
) -> dict[str, NDArray[np.float64]]:
    """Compute friction cone constraint for optimization.

    Provides constraint matrices for QP-based controllers.

    Args:
        contact_normal: Contact normal (3,).
        contact_position: Contact position (3,), for wrench computation.
        friction_coeff: Friction coefficient.
        num_faces: Number of linearization faces.

    Returns:
        Dictionary containing:
            - 'A': Inequality matrix (num_faces+1, 3)
            - 'b': Inequality bound (num_faces+1,)
            - 'normal': Contact normal (3,)
            - 'generators': Cone generators (3, num_faces)
    """
    contact_normal = np.asarray(contact_normal, dtype=np.float64)
    contact_normal = contact_normal / np.linalg.norm(contact_normal)

    # Linearize friction cone
    A_friction, b_friction = linearize_friction_cone(
        friction_coeff, contact_normal, num_faces
    )

    # Add non-negative normal force constraint: -n^T @ f <= 0
    A_normal = -contact_normal.reshape(1, 3)
    b_normal = np.zeros(1)

    A = np.vstack([A_friction, A_normal])
    b = np.concatenate([b_friction, b_normal])

    # Compute generators for force parameterization
    cone = FrictionCone(friction_coeff, contact_normal, num_faces)
    generators = cone.get_generators()

    return {
        "A": A,
        "b": b,
        "normal": contact_normal,
        "generators": generators,
    }


def project_to_friction_cone(
    force: NDArray[np.float64],
    cone: FrictionCone,
) -> NDArray[np.float64]:
    """Project force vector onto friction cone.

    Finds the closest force inside the friction cone.

    Args:
        force: Force vector (3,) to project.
        cone: Friction cone to project onto.

    Returns:
        Projected force (3,) inside the cone.
    """
    force = np.asarray(force, dtype=np.float64)

    # Decompose into normal and tangential
    f_n = float(np.dot(force, cone.normal))
    f_t = force - f_n * cone.normal
    f_t_mag = float(np.linalg.norm(f_t))

    # Case 1: Force is inside cone
    if cone.contains(force):
        return force.copy()

    # Case 2: Normal force is negative (pulling)
    if f_n < 0:
        # Project to origin (no force)
        if f_t_mag <= -cone.mu * f_n:
            return np.zeros(3)
        # Project to cone surface
        else:
            return _project_to_cone_surface(f_n, f_t, f_t_mag, cone)

    # Case 3: Tangential force exceeds limit
    if f_t_mag > cone.mu * f_n:
        # Scale tangential to friction limit
        f_t_proj = f_t * (cone.mu * f_n / f_t_mag) if f_t_mag > 1e-10 else f_t
        return f_n * cone.normal + f_t_proj

    return force.copy()


def _project_to_cone_surface(
    f_n: float,
    f_t: NDArray[np.float64],
    f_t_mag: float,
    cone: FrictionCone,
) -> NDArray[np.float64]:
    """Project to cone surface when normal is negative.

    This handles the edge case where force points away from surface.

    Args:
        f_n: Normal force component (negative).
        f_t: Tangential force vector.
        f_t_mag: Magnitude of tangential force.
        cone: Friction cone.

    Returns:
        Projected force on cone surface.
    """
    # Find point on cone edge that minimizes distance
    # The cone edge is at angle arctan(mu) from normal
    if f_t_mag < 1e-10:
        return np.zeros(3)

    t_dir = f_t / f_t_mag

    # Project onto the line from origin along cone edge
    cone_angle = np.arctan(cone.mu)
    edge_dir = np.cos(cone_angle) * cone.normal + np.sin(cone_angle) * t_dir

    # Project force onto this direction
    proj = np.dot(np.array([f_n, f_t_mag]), np.array([np.cos(cone_angle), np.sin(cone_angle)]))

    if proj <= 0:
        return np.zeros(3)

    return proj * edge_dir
