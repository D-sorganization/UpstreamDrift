"""Grasp analysis utilities for manipulation.

This module provides functions for analyzing grasp quality,
force closure, and grasp matrices for multi-fingered grasping.

Design by Contract:
    All analysis functions validate inputs and return meaningful results.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.robotics.contact.friction_cone import FrictionCone
from src.robotics.core.types import ContactState


def compute_grasp_matrix(
    contacts: list[ContactState],
    object_frame: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute grasp matrix mapping contact forces to object wrench.

    The grasp matrix G maps contact forces f to object wrench w:
        w = G @ f

    For point contacts, each contact contributes 3 columns (force only).

    Design by Contract:
        Preconditions:
            - len(contacts) >= 1

        Postconditions:
            - result.shape == (6, 3 * len(contacts))

    Args:
        contacts: List of contact states.
        object_frame: Object center position (3,). Uses centroid if None.

    Returns:
        Grasp matrix (6, 3*n_contacts).

    Raises:
        ValueError: If contacts list is empty.
    """
    if len(contacts) < 1:
        raise ValueError("At least one contact required")

    n_contacts = len(contacts)

    if object_frame is None:
        # Use centroid of contact points
        positions = np.array([c.position for c in contacts])
        object_frame = positions.mean(axis=0)

    object_frame = np.asarray(object_frame, dtype=np.float64)

    # Build grasp matrix
    # Each contact contributes: [I; r_x] where r_x is skew-symmetric
    G = np.zeros((6, 3 * n_contacts))

    for i, contact in enumerate(contacts):
        # Force transmission (identity)
        G[:3, 3 * i : 3 * i + 3] = np.eye(3)

        # Torque from force at contact point
        r = contact.position - object_frame
        r_skew = _skew_symmetric(r)
        G[3:6, 3 * i : 3 * i + 3] = r_skew

    return G


def _skew_symmetric(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Create skew-symmetric matrix from 3D vector.

    Args:
        v: Vector (3,).

    Returns:
        Skew-symmetric matrix (3, 3) such that skew(v) @ u = v x u.
    """
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def check_force_closure(
    contacts: list[ContactState],
    num_cone_faces: int = 8,
) -> tuple[bool, float]:
    """Check if grasp has force closure.

    A grasp has force closure if it can resist arbitrary wrenches.
    This is checked by verifying the origin is strictly inside
    the convex hull of the grasp wrench space.

    Design by Contract:
        Preconditions:
            - len(contacts) >= 2

    Args:
        contacts: List of contact states with friction.
        num_cone_faces: Number of faces for friction cone linearization.

    Returns:
        Tuple of (has_force_closure, quality_margin).
        Quality margin is the distance from origin to wrench space boundary.

    Raises:
        ValueError: If fewer than 2 contacts provided.
    """
    if len(contacts) < 2:
        raise ValueError("At least 2 contacts required for force closure")

    # Build grasp wrench space from friction cone generators
    wrench_generators = _build_wrench_generators(contacts, num_cone_faces)

    if wrench_generators.shape[1] < 6:
        # Not enough generators to span wrench space
        return False, 0.0

    # Check if origin is inside convex hull of generators
    # Using the criterion: origin inside iff we can find positive weights
    # summing to 1 that give zero wrench
    has_closure, margin = _check_origin_in_hull(wrench_generators)

    return has_closure, margin


def _build_wrench_generators(
    contacts: list[ContactState],
    num_cone_faces: int,
) -> NDArray[np.float64]:
    """Build wrench generators from contact friction cones.

    Args:
        contacts: Contact states.
        num_cone_faces: Friction cone faces per contact.

    Returns:
        Wrench generators (6, n_contacts * num_cone_faces).
    """
    # Compute object frame as centroid
    positions = np.array([c.position for c in contacts])
    object_center = positions.mean(axis=0)

    all_generators: list[NDArray[np.float64]] = []

    for contact in contacts:
        # Get friction cone generators
        cone = FrictionCone(
            contact.friction_coefficient,
            contact.normal,
            num_cone_faces,
        )
        force_generators = cone.get_generators()  # (3, num_faces)

        # Convert to wrench generators
        r = contact.position - object_center
        r_skew = _skew_symmetric(r)

        for j in range(force_generators.shape[1]):
            f = force_generators[:, j]
            tau = r_skew @ f
            wrench = np.concatenate([f, tau])
            all_generators.append(wrench)

    return np.column_stack(all_generators)


def _check_origin_in_hull(
    generators: NDArray[np.float64],
) -> tuple[bool, float]:
    """Check if origin is inside convex hull of generators.

    Uses linear programming: minimize c such that
        sum(alpha_i * g_i) = 0
        sum(alpha_i) = 1
        alpha_i >= 0

    If feasible, origin is inside the hull.

    Args:
        generators: Wrench generators (6, n_generators).

    Returns:
        Tuple of (inside, margin).
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        # Fallback: simple heuristic check
        return _heuristic_closure_check(generators)

    n_gen = generators.shape[1]

    # LP: find alpha >= 0, sum(alpha) = 1, G @ alpha = 0
    # We minimize a dummy objective
    c = np.zeros(n_gen)

    # Equality constraints: G @ alpha = 0, sum(alpha) = 1
    A_eq = np.vstack([generators, np.ones((1, n_gen))])
    b_eq = np.zeros(7)
    b_eq[-1] = 1.0

    # Bounds: alpha >= 0
    bounds = [(0, None) for _ in range(n_gen)]

    try:
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if result.success:
            # Compute quality margin as minimum weight
            margin = float(np.min(result.x))
            return True, margin
        else:
            return False, 0.0
    except Exception:
        return _heuristic_closure_check(generators)


def _heuristic_closure_check(
    generators: NDArray[np.float64],
) -> tuple[bool, float]:
    """Heuristic check for force closure.

    Uses SVD to check if wrench space spans R^6.

    Args:
        generators: Wrench generators (6, n_generators).

    Returns:
        Tuple of (likely_closure, heuristic_quality).
    """
    # Check if generators span R^6
    U, s, Vh = np.linalg.svd(generators)

    if len(s) >= 6 and s[5] > 1e-6:
        # Full rank - likely has force closure
        return True, float(s[5])
    else:
        return False, 0.0


def compute_grasp_quality(
    contacts: list[ContactState],
    metric: str = "min_singular_value",
) -> float:
    """Compute grasp quality metric.

    Available metrics:
        - 'min_singular_value': Smallest singular value of grasp matrix
        - 'volume': Product of singular values (grasp ellipsoid volume)
        - 'isotropy': Ratio of min to max singular value

    Design by Contract:
        Preconditions:
            - len(contacts) >= 1

    Args:
        contacts: Contact states defining the grasp.
        metric: Quality metric to compute.

    Returns:
        Quality metric value (higher is better, 0 if degenerate).

    Raises:
        ValueError: If no contacts provided or unknown metric.
    """
    if len(contacts) < 1:
        raise ValueError("At least one contact required")

    G = compute_grasp_matrix(contacts)
    U, s, Vh = np.linalg.svd(G)

    # Filter near-zero singular values
    s = s[s > 1e-10]

    if len(s) == 0:
        return 0.0

    if metric == "min_singular_value":
        return float(np.min(s))
    elif metric == "volume":
        return float(np.prod(s))
    elif metric == "isotropy":
        return float(np.min(s) / np.max(s)) if np.max(s) > 1e-10 else 0.0
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_contact_wrench_cone(
    contacts: list[ContactState],
    num_faces: int = 8,
) -> NDArray[np.float64]:
    """Compute the contact wrench cone generators.

    The contact wrench cone is the set of wrenches achievable
    by contact forces within their friction cones.

    Args:
        contacts: Contact states.
        num_faces: Friction cone linearization faces.

    Returns:
        Wrench generators (6, n_generators).
    """
    return _build_wrench_generators(contacts, num_faces)


def required_contact_forces(
    contacts: list[ContactState],
    desired_wrench: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Compute contact forces to achieve desired object wrench.

    Solves: G @ f = w, with f in friction cones.

    Args:
        contacts: Contact states.
        desired_wrench: Desired object wrench (6,).

    Returns:
        Contact forces (3*n_contacts,) or None if infeasible.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None

    len(contacts)
    G = compute_grasp_matrix(contacts)

    # Objective: minimize force magnitude
    def objective(f: NDArray[np.float64]) -> float:
        return float(np.sum(f**2))

    # Constraint: G @ f = w
    def wrench_constraint(f: NDArray[np.float64]) -> NDArray[np.float64]:
        return G @ f - desired_wrench

    # Initial guess: pseudoinverse solution
    f0 = np.linalg.lstsq(G, desired_wrench, rcond=None)[0]

    # Bounds: friction cone constraints (simplified as box for now)
    bounds = []
    for contact in contacts:
        max_force = contact.normal_force * (1 + contact.friction_coefficient)
        bounds.extend([(-max_force, max_force)] * 3)

    try:
        result = minimize(
            objective,
            f0,
            method="SLSQP",
            constraints={"type": "eq", "fun": wrench_constraint},
            bounds=bounds,
        )

        if result.success:
            return result.x
        else:
            return None
    except Exception:
        return None
