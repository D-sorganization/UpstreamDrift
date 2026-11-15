"""Friction cone utilities for contact dynamics.

This module provides friction cone representations and linearization
methods for use in optimization-based control.

Design by Contract:
    All functions validate inputs and guarantee valid outputs.
    Friction cones are always represented in consistent frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts import postcondition, precondition


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
        object.__setattr__(self, "normal", np.asarray(self.normal, dtype=np.float64))
        norm = math.hypot(*self.normal)
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
        if force is None:
            raise ValueError("force must be provided")
        force = np.asarray(force, dtype=np.float64)
        f_n = float(np.dot(force, self.normal))

        if f_n < -tolerance:
            return False  # Pulling force (tensile)

        f_t = force - f_n * self.normal
        f_t_mag = math.hypot(*f_t)

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
    if normal is None:
        raise ValueError("normal must be provided")
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / math.hypot(*normal)

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
    t1 = t1 / math.hypot(*t1)

    t2 = np.cross(normal, t1)
    t2 = t2 / math.hypot(*t2)

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
    """Inscribed (inner) polyhedral approximation A @ f <= b of the friction cone.

    Approximates the second-order cone constraint ||f_t|| <= mu * f_n using an
    inscribed ``num_faces``-gon whose vertices lie on the exact circular cone
    ||f_t|| = mu * f_n, so every admissible force strictly satisfies Coulomb friction:
    d_i^T @ f - mu * cos(pi / num_faces) * n^T @ f <= 0.

    Args:
        mu: Friction coefficient.
        normal: Contact normal direction (3,).
        num_faces: Number of faces in polyhedral approximation (must be >= 3).

    Returns:
        Tuple of (A, b) such that A @ f <= b enforces the inscribed cone.
        A has shape (num_faces, 3), b has shape (num_faces,).
    """
    if mu is None:
        raise ValueError("mu must be provided")
    if num_faces < 3:
        raise ValueError(f"num_faces must be >= 3, got {num_faces}")
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / math.hypot(*normal)

    t1, t2 = _get_tangent_vectors(normal)
    mu_inscribed = mu * math.cos(math.pi / num_faces)

    # Inscribed polygon face i has unit normal d_i bisecting adjacent vertices;
    # distance to origin in f_t plane is mu * cos(pi / num_faces) * f_n.
    A = np.zeros((num_faces, 3))
    b = np.zeros(num_faces)

    for i in range(num_faces):
        angle = 2 * np.pi * i / num_faces
        d = np.cos(angle) * t1 + np.sin(angle) * t2
        A[i] = d - mu_inscribed * normal
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
    if contact_normal is None:
        raise ValueError("contact_normal must be provided")
    contact_normal = np.asarray(contact_normal, dtype=np.float64)
    contact_normal = contact_normal / math.hypot(*contact_normal)

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
    """Compute the exact Euclidean projection of ``force`` onto ``cone``.

    Projects onto the second-order Coulomb Lorentz cone
    ``{f : ||f_t|| <= cone.mu * f_n, f_n >= 0}``.

    Args:
        force: Force vector (3,) to project.
        cone: Friction cone to project onto.

    Returns:
        Euclidean closest force vector (3,) inside ``cone``.
    """
    if force is None:
        raise ValueError("force must be provided")
    force = np.asarray(force, dtype=np.float64)

    f_n = float(np.dot(force, cone.normal))
    f_t = force - f_n * cone.normal
    f_t_mag = math.hypot(*f_t)

    # Case 1: Frictionless limit reduces to half-space projection on normal.
    if cone.mu == 0.0:
        return max(f_n, 0.0) * cone.normal

    # Case 2: Force is already inside the second-order cone.
    if f_n >= 0.0 and f_t_mag <= cone.mu * f_n:
        return force.copy()

    # Case 3: Inside the polar cone (||f_t|| <= -f_n / mu); closest point is 0.
    if f_n + cone.mu * f_t_mag <= 0.0:
        return np.zeros(3, dtype=np.float64)

    # Case 4: Exterior region projecting orthogonally onto the lateral cone surface.
    return _project_to_cone_surface(f_n, f_t, f_t_mag, cone)


def _project_to_cone_surface(
    f_n: float,
    f_t: NDArray[np.float64],
    f_t_mag: float,
    cone: FrictionCone,
) -> NDArray[np.float64]:
    """Euclidean projection onto the lateral boundary of the second-order cone."""
    if f_n is None:
        raise ValueError("f_n must be provided")
    if f_t_mag < 1e-12:
        return max(f_n, 0.0) * cone.normal

    t_dir = f_t / f_t_mag
    alpha = (f_n + cone.mu * f_t_mag) / (1.0 + cone.mu * cone.mu)
    if alpha <= 0.0:
        return np.zeros(3, dtype=np.float64)
    return alpha * (cone.normal + cone.mu * t_dir)
