# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Gait (walking) exercise configuration and analysis.

Defines a full gait cycle from right heel strike to next right heel strike.
Uses Winter (2009) normative joint angles for self-selected walking speed.

Design Principles:
    DBC -- public functions validate inputs.
    DRY -- reuses the shared ``_common`` helpers and ``BodyModel`` API.
    LoD -- callers interact only through the public factory + ``GaitAnalyzer``.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from ..models import BodyModel, LagrangianDynamics

logger = logging.getLogger(__name__)


def _gait_via_points() -> list[tuple[float, float, float, float]]:
    """Winter normative sagittal-plane angles for one gait cycle.

    Convention: (fraction, ankle_angle, knee_angle, hip_angle) in radians.
    Angles follow the repo convention: ankle-knee-hip from distal to proximal.
    """
    return [
        # fraction  ankle              knee                hip
        (0.00, math.radians(0), math.radians(-5), math.radians(30)),  # heel strike
        (
            0.10,
            math.radians(-5),
            math.radians(-15),
            math.radians(25),
        ),  # loading response
        (0.30, math.radians(5), math.radians(-5), math.radians(0)),  # midstance
        (
            0.50,
            math.radians(-15),
            math.radians(0),
            math.radians(-10),
        ),  # terminal stance + push-off
        (0.60, math.radians(5), math.radians(-40), math.radians(-5)),  # pre-swing
        (0.70, math.radians(0), math.radians(-60), math.radians(15)),  # initial swing
        (0.85, math.radians(0), math.radians(-30), math.radians(25)),  # mid swing
        (
            1.00,
            math.radians(0),
            math.radians(-5),
            math.radians(30),
        ),  # terminal swing = heel strike
    ]


def make_gait_config(
    body: BodyModel,
    stride_length: float = 0.7,
    cycle_duration: float = 1.0,
) -> tuple[
    LagrangianDynamics,
    NDArray,
    NDArray,
    NDArray,
    list[tuple[float, float, float, float]],
]:
    """Create gait cycle configuration.

    Returns:
        (dynamics, q_start, q_end, q_bounds, via_points)

    Preconditions:
        stride_length > 0
        cycle_duration > 0
    """
    if stride_length <= 0:
        raise ValueError("stride_length must be positive")
    if cycle_duration <= 0:
        raise ValueError("cycle_duration must be positive")

    via_points = _gait_via_points()

    # No external load during walking
    dyn = LagrangianDynamics(body, body.m_squat.copy(), body.I_squat.copy(), 0.0)

    q_start = np.array([via_points[0][1], via_points[0][2], via_points[0][3]])
    q_end = np.array([via_points[-1][1], via_points[-1][2], via_points[-1][3]])

    q_bounds = np.array(
        [
            [np.radians(-20), np.radians(20)],  # ankle
            [np.radians(-65), np.radians(5)],  # knee
            [np.radians(-15), np.radians(40)],  # hip
        ]
    )

    return dyn, q_start, q_end, q_bounds, via_points


class GaitAnalyzer:
    """Analyze gait cycle kinematics and kinetics.

    Computes:
    - Spatiotemporal parameters (cadence, stride length, speed)
    - Stance/swing phase timing
    - Symmetry indices
    """

    def __init__(self, model: BodyModel) -> None:
        if not isinstance(model, BodyModel):
            raise TypeError("model must be a BodyModel instance")
        self.model = model

    def compute_spatiotemporal(
        self, q: NDArray, t: NDArray, stride_length: float
    ) -> dict[str, float]:
        """Compute gait spatiotemporal parameters.

        Preconditions:
            q.ndim in (1, 2) and len(q) == len(t)
            stride_length > 0
        """
        if stride_length <= 0:
            raise ValueError("stride_length must be positive")
        if len(t) < 2:
            raise ValueError("need at least 2 time samples")

        duration = float(t[-1] - t[0])
        cadence = 60.0 / duration  # cycles / min
        speed = stride_length / duration

        # Estimate stance/swing from knee flexion
        knee_angles = q[:, 1] if q.ndim > 1 else q
        swing_mask = np.abs(knee_angles) > math.radians(20)
        stance_pct = 1.0 - float(np.mean(swing_mask))

        return {
            "cadence_steps_per_min": cadence * 2,
            "stride_length_m": stride_length,
            "walking_speed_m_s": speed,
            "stance_phase_pct": stance_pct * 100,
            "swing_phase_pct": (1 - stance_pct) * 100,
            "cycle_duration_s": duration,
        }

    def compute_symmetry_index(self, left_angles: NDArray, right_angles: NDArray) -> float:
        """Robinson symmetry index: SI = |L-R| / max(L,R) * 100.

        Preconditions:
            left_angles and right_angles are 1-D arrays of the same length.
        """
        l_range = float(np.ptp(left_angles))
        r_range = float(np.ptp(right_angles))
        if max(l_range, r_range) < 1e-10:
            return 0.0
        return abs(l_range - r_range) / max(l_range, r_range) * 100
