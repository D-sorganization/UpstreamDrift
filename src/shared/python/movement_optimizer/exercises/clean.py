# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Clean exercise configuration.

The clean lifts the bar from the floor to front rack (shoulder height).
The motion is modelled as a deadlift-style pull followed by a catch
at the shoulders with a slight squat.

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


def _clean_end_angles(body: BodyModel) -> NDArray:
    """Compute end joint angles for front rack position.

    Front rack: standing with bar at shoulder (clavicle) height.
    Torso near vertical (~5 deg), knees nearly straight.
    The bar rests on the front of the shoulders -- NOT overhead.
    """
    del body
    return pose_deg(5, -8, 5)


def _clean_via_angles(body: BodyModel) -> NDArray:
    """Compute via-point at the power position (second pull).

    Hips extended, bar at mid-thigh level.  This is the transition
    between the pull phase and the catch phase.
    """
    del body
    return pose_deg(8, -15, 20)


def make_clean_config(
    body: BodyModel, bar_mass: float
) -> tuple[LagrangianDynamics, NDArray, NDArray, NDArray, NDArray]:
    """Create dynamics and trajectory config for the clean.

    Returns:
        (dynamics, q_start, q_end, q_bounds, q_via)

    The dynamics use deadlift-style mass distribution (arms hang, load
    is arm_mass + bar_mass at end effector).
    """
    load = body.m_arms + bar_mass
    dyn = LagrangianDynamics(body, body.m_deadlift.copy(), body.I_deadlift.copy(), load)

    q_start_raw = pull_start_angles(body, q2_deg=52)
    q_start = balance_config_pose(dyn, q_start_raw, "deadlift", bar_mass, adjust_joint=0)

    q_end_raw = _clean_end_angles(body)
    q_end = balance_config_pose(dyn, q_end_raw, "deadlift", bar_mass, adjust_joint=2)

    q_via_raw = _clean_via_angles(body)
    q_via = balance_config_pose(dyn, q_via_raw, "deadlift", bar_mass, adjust_joint=2)

    q_bounds = default_bounds_deg((-5, 35), (-110, 10), (-10, 80))

    return dyn, q_start, q_end, q_bounds, q_via
