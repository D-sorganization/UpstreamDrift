# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Initial guess generation strategies for multi-start trajectory optimisation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .tuning import PERTURBATION_SCALE


def build_initial_guess(
    q_start: NDArray,
    q_end: NDArray,
    n_waypoints: int,
    n_dof: int,
    q_via: NDArray | None = None,
) -> NDArray:
    """Linear interpolation between start/end (or start/via/end).

    Preconditions:
        q_start and q_end have length n_dof.
        n_waypoints >= 1.
        q_via, if given, has length n_dof.

    Returns waypoint array of shape (n_waypoints, n_dof).
    """
    if q_via is not None:
        n_half = n_waypoints // 2
        # np.linspace is vectorized, directly taking start and end arrays
        wp1 = np.linspace(q_start, q_via, n_half + 2)[1:-1]
        wp2 = np.linspace(q_via, q_end, n_waypoints - n_half + 2)[1:-1]
        return np.vstack((wp1, wp2))
    return np.linspace(q_start, q_end, n_waypoints + 2)[1:-1]


def build_perturbed_guess(
    q_start: NDArray,
    q_end: NDArray,
    q_bounds: NDArray,
    n_waypoints: int,
    n_dof: int,
    seed: int,
    q_via: NDArray | None = None,
) -> NDArray:
    """Generate a perturbed initial guess for multi-start optimisation.

    Preconditions:
        seed >= 0.
        q_bounds has shape (n_dof, 2).

    Seed 0 returns the unperturbed baseline.  Other seeds add smooth
    random perturbations scaled to PERTURBATION_SCALE of the joint range.

    Returns waypoint array of shape (n_waypoints, n_dof).
    """
    wp = build_initial_guess(q_start, q_end, n_waypoints, n_dof, q_via)
    if seed == 0:
        return wp

    rng = np.random.default_rng(seed * 42 + 7)
    joint_range = q_bounds[:, 1] - q_bounds[:, 0]
    noise = rng.normal(0, PERTURBATION_SCALE, wp.shape) * joint_range
    wp += noise

    # Clip to joint bounds (vectorized in-place clipping is faster than looping)
    np.clip(wp, q_bounds[:, 0], q_bounds[:, 1], out=wp)
    return wp


def build_bounds(
    q_bounds: NDArray,
    n_waypoints: int,
    n_dof: int,
) -> list[tuple[float, float]]:
    """Build the flat bounds list required by scipy's minimize.

    Preconditions:
        q_bounds has shape (n_dof, 2).
        n_waypoints >= 1.

    Returns list of (lo, hi) tuples with length n_waypoints * n_dof.
    """
    # Direct list multiplication is ~20x faster than double loops
    return [tuple(bound) for bound in q_bounds] * n_waypoints
