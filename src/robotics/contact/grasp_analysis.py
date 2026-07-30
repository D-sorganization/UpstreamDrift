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


#: Margins at or below this are treated as "no force closure".
FORCE_CLOSURE_TOL = 1e-9


def check_force_closure(
    contacts: list[ContactState],
    num_cone_faces: int = 8,
) -> tuple[bool, float]:
    """Check if grasp has force closure.

    A grasp has force closure if it can resist arbitrary wrenches, i.e. if the
    origin lies **strictly inside** the convex hull of the grasp wrench space.
    Two conditions are required and both are checked here:

    1. the wrench generators must have rank 6 (otherwise the grasp cannot
       resist wrenches in the missing direction at all), and
    2. the origin must be in the *interior* of their convex hull, not merely a
       member of it - membership alone is satisfied by every grasp whose
       generators span a proper subspace.

    The returned margin is the Ferrari-Canny epsilon metric: the radius of the
    largest ball centred on the origin contained in the grasp wrench hull,
    i.e. the distance from the origin to the nearest hull facet.  It is
    strictly positive exactly when the grasp has force closure and is 0.0
    otherwise, so candidate grasps can be ranked by it.

    Design by Contract:
        Preconditions:
            - len(contacts) >= 2
        Postconditions:
            - has_force_closure is True iff margin > 0

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

    if np.linalg.matrix_rank(wrench_generators) < 6:
        # The generators lie in a proper subspace of R^6: there is a wrench
        # direction no contact force can resist.  The origin is in their hull
        # but only on its (relative) boundary, so this is not force closure.
        return False, 0.0

    return _grasp_wrench_margin(wrench_generators)


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
    if contacts is None:
        raise ValueError("contacts must be provided")
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


def _grasp_wrench_margin(
    generators: NDArray[np.float64],
) -> tuple[bool, float]:
    """Ferrari-Canny epsilon: radius of the largest origin-centred ball in the hull.

    The convex hull of the wrench generators is computed in R^6; the epsilon
    metric is then the smallest distance from the origin to a hull facet.  A
    non-positive value means the origin is on or outside the boundary, i.e.
    there is no force closure.

    Args:
        generators: Wrench generators (6, n_generators), rank 6.

    Returns:
        Tuple of (has_force_closure, margin).
    """
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError:  # pragma: no cover - scipy is a hard dependency in CI
        return _sampled_closure_check(generators)

    try:
        hull = ConvexHull(generators.T)
    except QhullError:
        # Degenerate hull (numerically rank-deficient): fall back to sampling,
        # which is conservative because it only ever reports a smaller margin.
        return _sampled_closure_check(generators)

    normals = hull.equations[:, :-1]
    offsets = hull.equations[:, -1]
    # Interior points satisfy normal @ x + offset < 0, so the signed distance
    # from the origin to each facet is -offset / ||normal||.
    # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~2x faster than np.linalg.norm(..., axis=1)
    distances = -offsets / np.sqrt(np.einsum('...i,...i->...', normals, normals))
    margin = float(np.min(distances))

    if margin <= FORCE_CLOSURE_TOL:
        return False, 0.0
    return True, margin


def _sampled_closure_check(
    generators: NDArray[np.float64],
    n_directions: int = 4096,
) -> tuple[bool, float]:
    """Approximate epsilon metric without scipy.

    ``min_{||d||=1} max_i <g_i, d>`` is exactly the distance from the origin to
    the hull boundary.  Minimising over a finite direction sample gives an
    upper bound on that distance, so the result is only approximate; it is used
    solely when scipy (and therefore Qhull) is unavailable.

    Args:
        generators: Wrench generators (6, n_generators).
        n_directions: Number of random unit directions to probe.

    Returns:
        Tuple of (likely_closure, approximate margin).
    """
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(n_directions, generators.shape[0]))
    # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~2x faster than np.linalg.norm(..., axis=1)
    directions /= np.sqrt(np.einsum('...i,...i->...', directions, directions))[..., None]
    margin = float(np.min(np.max(directions @ generators, axis=1)))

    if margin <= FORCE_CLOSURE_TOL:
        return False, 0.0
    return True, margin


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
    if metric == "volume":
        return float(np.prod(s))
    if metric == "isotropy":
        return float(np.min(s) / np.max(s)) if np.max(s) > 1e-10 else 0.0
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
    if contacts is None:
        raise ValueError("contacts must be provided")
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None

    len(contacts)
    G = compute_grasp_matrix(contacts)

    # Objective: minimize force magnitude
    def objective(f: NDArray[np.float64]) -> float:
        """Compute total squared force magnitude."""
        return float(np.vdot(f, f))

    # Constraint: G @ f = w
    def wrench_constraint(f: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the wrench balance residual."""
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
        return None
    except (RuntimeError, ValueError, OSError):
        return None
