# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Shared helpers for planar exercise configuration factories."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..constants import PLATE_RADIUS_STD_M
from ..models import BodyModel, LagrangianDynamics, balance_pose


def pose_deg(ankle: float, knee: float, hip: float) -> NDArray:
    """Return a 3-angle pose vector built from degree inputs."""
    return np.radians(np.array([ankle, knee, hip], dtype=float))


def balance_config_pose(
    dynamics: LagrangianDynamics,
    raw_pose: NDArray,
    exercise_type: str,
    bar_mass: float,
    *,
    adjust_joint: int,
) -> NDArray:
    """Balance a raw pose using the shared planar balance helper."""
    return balance_pose(dynamics, raw_pose, exercise_type, bar_mass, adjust_joint=adjust_joint)


def default_bounds_deg(
    lower: tuple[float, float], middle: tuple[float, float], upper: tuple[float, float]
) -> NDArray:
    """Return a `(3, 2)` array of joint bounds built from degree tuples."""
    return np.radians(np.array([lower, middle, upper], dtype=float))


def pull_start_angles(body: BodyModel, q2_deg: float) -> NDArray:
    """Compute starting joint angles for a pulling exercise (bar at plate height).

    Shared by deadlift, clean, and snatch.  The hip-flexion angle *q2_deg*
    (in degrees) varies by exercise -- wider grips use a smaller value.
    """
    target_shoulder_h = PLATE_RADIUS_STD_M + body.L_arm
    q0 = np.radians(15)
    q2 = np.radians(q2_deg)
    needed = target_shoulder_h - body.L[0] * np.cos(q0) - body.L[2] * np.cos(q2)
    cos_q1 = np.clip(needed / body.L[1], -1, 1)
    q1 = -np.arccos(cos_q1)
    return np.array([q0, q1, q2])
