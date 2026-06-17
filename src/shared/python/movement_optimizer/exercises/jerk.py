# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Jerk exercise configuration.

The jerk drives the bar from front rack (shoulders) to overhead lockout.
The lifter dips into a quarter-squat, then explosively extends to press
the bar overhead and catches it with arms locked out.

Uses the same 3-link planar model and balance utilities as the existing
squat/deadlift factories in ``models.py``.
"""

from __future__ import annotations

from numpy.typing import NDArray

from ..models import BodyModel, LagrangianDynamics
from ._common import balance_config_pose, default_bounds_deg, pose_deg


def _jerk_start_angles() -> NDArray:
    """Front rack position: standing with bar at shoulders."""
    return pose_deg(5, -8, 5)


def _jerk_via_angles() -> NDArray:
    """Dip position: quarter-squat before the explosive drive phase."""
    return pose_deg(12, -40, 8)


def _jerk_end_angles() -> NDArray:
    """Overhead lockout: torso vertical, bar overhead with arms locked out.

    The bar is above the shoulders.  In our 3-link model the shoulder
    is at the top of the chain; a near-vertical torso (~2 deg) puts
    the shoulder as high as possible, representing overhead lockout.
    """
    return pose_deg(3, -5, 2)


def make_jerk_config(
    body: BodyModel, bar_mass: float
) -> tuple[LagrangianDynamics, NDArray, NDArray, NDArray, NDArray]:
    """Create dynamics and trajectory config for the jerk.

    The bar is on the shoulders (squat-style mass distribution: arms
    are part of the torso segment, bar at shoulder level).

    Returns:
        (dynamics, q_start, q_end, q_bounds, q_via)
    """
    dyn = LagrangianDynamics(body, body.m_squat.copy(), body.I_squat.copy(), bar_mass)

    q_start_raw = _jerk_start_angles()
    q_start = balance_config_pose(dyn, q_start_raw, "squat", bar_mass, adjust_joint=0)

    q_end_raw = _jerk_end_angles()
    q_end = balance_config_pose(dyn, q_end_raw, "squat", bar_mass, adjust_joint=0)

    q_via_raw = _jerk_via_angles()
    q_via = balance_config_pose(dyn, q_via_raw, "squat", bar_mass, adjust_joint=2)

    q_bounds = default_bounds_deg((-5, 30), (-50, 5), (-5, 40))

    return dyn, q_start, q_end, q_bounds, q_via
