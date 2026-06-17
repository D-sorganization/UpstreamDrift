# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Snatch exercise configuration.

The snatch lifts the bar from the floor to overhead in one continuous
motion.  The wide grip means a slightly different torso angle at the
start compared to the clean.  The catch is an overhead squat position.

Uses the same 3-link planar model and balance utilities as the existing
squat/deadlift factories in ``models.py``.
"""

from __future__ import annotations

from numpy.typing import NDArray

from ..models import BodyModel, LagrangianDynamics
from ._common import (
    balance_config_pose,
    default_bounds_deg,
    pose_deg,
    pull_start_angles,
)


def _snatch_via_angles() -> NDArray:
    """Overhead squat catch position.

    Deep squat with bar overhead: significant knee flexion, torso
    relatively upright to keep the bar balanced overhead.
    This is the unique snatch catch -- a deep squat with arms locked
    out above.
    """
    return pose_deg(25, -90, 10)


def _snatch_end_angles() -> NDArray:
    """Standing with bar overhead: torso vertical, arms locked out.

    Nearly identical to jerk end -- the bar is overhead with the
    torso as vertical as possible.
    """
    return pose_deg(3, -5, 2)


def make_snatch_config(
    body: BodyModel, bar_mass: float
) -> tuple[LagrangianDynamics, NDArray, NDArray, NDArray, NDArray]:
    """Create dynamics and trajectory config for the snatch.

    The start uses deadlift-style mass distribution (arms hanging to
    reach the bar), while the end uses squat-style (bar overhead on
    shoulders).  We use the deadlift model for the pull phase dynamics.

    Simplification: the same deadlift-style dynamics object is used for
    the entire movement, including the overhead catch phase.  Ideally the
    catch phase (bar overhead, deep squat) should transition to squat-style
    dynamics where arm mass is lumped into the torso segment.  This would
    require a dynamics-switching mechanism that is not yet implemented.

    Returns:
        (dynamics, q_start, q_end, q_bounds, q_via)
    """
    load = body.m_arms + bar_mass
    dyn = LagrangianDynamics(body, body.m_deadlift.copy(), body.I_deadlift.copy(), load)

    q_start_raw = pull_start_angles(body, q2_deg=48)
    q_start = balance_config_pose(dyn, q_start_raw, "deadlift", bar_mass, adjust_joint=0)

    # End: standing with bar overhead -- use squat-style COM for balance check
    # but keep the same dynamics object
    q_end_raw = _snatch_end_angles()
    q_end = balance_config_pose(dyn, q_end_raw, "squat", bar_mass, adjust_joint=0)

    q_via_raw = _snatch_via_angles()
    q_via = balance_config_pose(dyn, q_via_raw, "deadlift", bar_mass, adjust_joint=2)

    q_bounds = default_bounds_deg((-5, 35), (-110, 10), (-10, 80))

    return dyn, q_start, q_end, q_bounds, q_via
