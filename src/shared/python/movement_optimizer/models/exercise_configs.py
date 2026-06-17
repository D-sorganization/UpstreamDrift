# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Exercise configuration factories for squat and deadlift movements.

Contains ``make_squat_config``, ``make_full_squat_config``, and
``make_deadlift_config``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..constants import SQUAT_BOTTOM_DEG
from .body_model import BodyModel
from .lagrangian_dynamics import LagrangianDynamics, _standing_balanced, balance_pose

__all__ = [
    "make_deadlift_config",
    "make_full_squat_config",
    "make_squat_config",
]


def make_squat_config(
    body: BodyModel, bar_mass: float
) -> tuple[LagrangianDynamics, NDArray, NDArray, NDArray]:
    """Create dynamics and trajectory config for a standard back squat.

    Start position is the squat bottom (deep knee bend, torso adjusted for
    COM balance); end position is a balanced standing pose.

    Args:
        body: Anthropometric body model.
        bar_mass: Barbell load in kg.

    Returns:
        Tuple of (dynamics, q_start, q_end, q_bounds).
    """
    dyn = LagrangianDynamics(body, body.m_squat.copy(), body.I_squat.copy(), bar_mass)
    # Squat bottom: deep knee bend, torso adjusted for COM balance
    q_bottom_raw = np.array([np.radians(a) for a in SQUAT_BOTTOM_DEG])
    q_start = balance_pose(dyn, q_bottom_raw, "squat", bar_mass, adjust_joint=2)
    q_end = _standing_balanced(dyn, bar_mass, "squat")
    q_bounds = np.array(
        [
            [np.radians(-5), np.radians(40)],
            [np.radians(-95), np.radians(5)],
            [np.radians(-5), np.radians(75)],
        ]
    )
    return dyn, q_start, q_end, q_bounds


def make_full_squat_config(
    body: BodyModel, bar_mass: float
) -> tuple[LagrangianDynamics, NDArray, NDArray, NDArray, NDArray]:
    """Create dynamics and trajectory config for a full squat (stand-to-stand via bottom).

    Uses a via-point trajectory: standing -> squat bottom -> standing.
    Both start and end are balanced standing poses; the via point is the
    squat bottom with COM adjusted for inner-BOS balance.

    Args:
        body: Anthropometric body model.
        bar_mass: Barbell load in kg.

    Returns:
        Tuple of (dynamics, q_start, q_end, q_bounds, q_via).
    """
    dyn = LagrangianDynamics(body, body.m_squat.copy(), body.I_squat.copy(), bar_mass)
    q_stand = _standing_balanced(dyn, bar_mass, "full_squat")
    q_start = q_stand.copy()
    q_end = q_stand.copy()
    q_bottom_raw = np.array([np.radians(a) for a in SQUAT_BOTTOM_DEG])
    q_via = balance_pose(dyn, q_bottom_raw, "full_squat", bar_mass, adjust_joint=2)
    q_bounds = np.array(
        [
            [np.radians(-5), np.radians(40)],
            [np.radians(-95), np.radians(5)],
            [np.radians(-5), np.radians(75)],
        ]
    )
    return dyn, q_start, q_end, q_bounds, q_via


def make_deadlift_config(
    body: BodyModel, bar_mass: float
) -> tuple[LagrangianDynamics, NDArray, NDArray, NDArray]:
    """Create dynamics and trajectory config for a conventional deadlift.

    The arm mass is added to the bar load to model the combined pulling
    force.  Start is the pull position (bar off floor) with COM balanced;
    end is a balanced standing lockout.

    Args:
        body: Anthropometric body model.
        bar_mass: Barbell load in kg (arms mass is added internally).

    Returns:
        Tuple of (dynamics, q_start, q_end, q_bounds).
    """
    load = body.m_arms + bar_mass
    dyn = LagrangianDynamics(body, body.m_deadlift.copy(), body.I_deadlift.copy(), load)
    from ..exercises._common import pull_start_angles

    q_start_raw = pull_start_angles(body, q2_deg=52)
    q_start = balance_pose(dyn, q_start_raw, "deadlift", bar_mass, adjust_joint=0)
    q_end = _standing_balanced(dyn, bar_mass, "deadlift")
    q_bounds = np.array(
        [
            [np.radians(-5), np.radians(30)],
            [np.radians(-80), np.radians(5)],
            [np.radians(-5), np.radians(75)],
        ]
    )
    return dyn, q_start, q_end, q_bounds
