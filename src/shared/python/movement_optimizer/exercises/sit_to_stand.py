# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Sit-to-stand exercise configuration.

Phases: seated -> forward lean (momentum) -> seat-off -> rising -> standing.
The torso forward lean is critical for generating momentum to leave the chair.

Design Principles:
    DBC -- factory validates inputs.
    DRY -- reuses BodyModel and LagrangianDynamics.
    LoD -- callers get a standard config tuple.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from ..models import BodyModel, LagrangianDynamics

logger = logging.getLogger(__name__)


def _sts_via_points(q_start: NDArray, q_end: NDArray) -> list[tuple[float, float, float, float]]:
    """Via-points for sit-to-stand motion."""
    return [
        (0.00, float(q_start[0]), float(q_start[1]), float(q_start[2])),  # seated
        (0.20, math.radians(15), math.radians(-90), math.radians(100)),  # forward lean
        (
            0.35,
            math.radians(20),
            math.radians(-85),
            math.radians(90),
        ),  # max lean + seat off
        (0.50, math.radians(15), math.radians(-60), math.radians(60)),  # rising phase 1
        (0.75, math.radians(5), math.radians(-30), math.radians(30)),  # rising phase 2
        (1.00, float(q_end[0]), float(q_end[1]), float(q_end[2])),  # standing
    ]


def make_sit_to_stand_config(
    body: BodyModel,
    seat_height: float = 0.45,
) -> tuple[
    LagrangianDynamics,
    NDArray,
    NDArray,
    NDArray,
    list[tuple[float, float, float, float]],
]:
    """Create sit-to-stand configuration.

    Returns:
        (dynamics, q_start, q_end, q_bounds, via_points)

    Preconditions:
        seat_height > 0
    """
    if seat_height <= 0:
        raise ValueError("seat_height must be positive")

    # Seated angles: ankle dorsiflexion, knee ~90 deg flexion, hip ~80 deg flexion
    q_start = np.array(
        [
            math.radians(10),  # ankle: slight dorsiflexion
            math.radians(-90),  # knee: ~90 deg flexion (seated)
            math.radians(80),  # hip: ~80 deg flexion (seated)
        ]
    )

    # Standing: near neutral
    q_end = np.array(
        [
            math.radians(0),  # ankle: neutral
            math.radians(0),  # knee: straight
            math.radians(0),  # hip: upright
        ]
    )

    via_points = _sts_via_points(q_start, q_end)

    # No external load
    dyn = LagrangianDynamics(body, body.m_squat.copy(), body.I_squat.copy(), 0.0)

    q_bounds = np.array(
        [
            [np.radians(-5), np.radians(25)],  # ankle
            [np.radians(-95), np.radians(5)],  # knee
            [np.radians(-5), np.radians(110)],  # hip (allows forward lean)
        ]
    )

    return dyn, q_start, q_end, q_bounds, via_points
