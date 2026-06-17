# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Bench press anthropometric model and exercise configuration factory.

Contains ``BenchPressModel`` and ``make_bench_press_config``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..constants import (
    ARM_RADIUS_OF_GYRATION_FRAC,
    BENCH_FOREARM_FRAC,
    BENCH_PRESS_JOINT_LIMITS,
    BENCH_UPPER_ARM_FRAC,
    MASS_FRAC,
    WRIST_SEGMENT_FRAC,
)
from .body_model import BodyModel, ChainGeometry
from .lagrangian_dynamics import LagrangianDynamics

__all__ = [
    "BenchPressModel",
    "make_bench_press_config",
]


class BenchPressModel:
    """Anthropometric model for the bench press arm chain.

    Maps the 3-link model to: shoulder, elbow, wrist.
    Segment lengths are derived from the full body arm length.

    Preconditions:
        body is a valid BodyModel
    """

    def __init__(self, body: BodyModel) -> None:
        arm_len = body.L_arm
        self.L = np.array(
            [
                BENCH_UPPER_ARM_FRAC * arm_len,
                BENCH_FOREARM_FRAC * arm_len,
                WRIST_SEGMENT_FRAC * arm_len,  # wrist/hand — scales with body height
            ]
        )
        self.body_mass = body.body_mass
        # Arm segment masses: split arm mass into upper arm, forearm, hand
        # Fractions per Winter (2009): upper arm 56%, forearm 32%, hand 12%
        arm_mass = MASS_FRAC["arms"] * body.body_mass
        self.m = np.array(
            [
                0.56 * arm_mass,  # upper arm (Winter 2009)
                0.32 * arm_mass,  # forearm (Winter 2009)
                0.12 * arm_mass,  # hand/wrist (Winter 2009)
            ]
        )
        self.d = np.array(
            [
                0.436 * self.L[0],  # upper arm COM
                0.430 * self.L[1],  # forearm COM
                0.500 * self.L[2],  # hand COM
            ]
        )
        # Centroidal segment inertia (about each segment COM), matching the
        # convention BodyModel/LagrangianDynamics expect: I_com = m*(rho*L)**2.
        # LagrangianDynamics adds the parallel-axis term itself, so do NOT
        # pre-add it here. Previously used the uniform-rod (1/12)*m*L**2
        # COM-frame value with a generic rod rho; now uses arm-segment
        # radius-of-gyration fractions from Winter (2009) (issue #490).
        rho_arm = np.array(
            [
                ARM_RADIUS_OF_GYRATION_FRAC["upper_arm"],
                ARM_RADIUS_OF_GYRATION_FRAC["forearm"],
                ARM_RADIUS_OF_GYRATION_FRAC["hand"],
            ]
        )
        self.I = self.m * (rho_arm * self.L) ** 2
        self.g = body.g
        # NOTE: BOS bounds are copied for API compatibility but are NOT
        # physically meaningful for bench press.  The lifter is supine on
        # a bench, so the standing base-of-support constraint does not
        # apply.  The optimizer skips COM-in-BOS checks for bench_press.
        self.inner_heel = body.inner_heel
        self.inner_toe = body.inner_toe
        self.inner_center = body.inner_center
        self.height = body.height


def make_bench_press_config(
    body: BodyModel, bar_mass: float
) -> tuple[LagrangianDynamics, NDArray, NDArray, NDArray, NDArray]:
    """Create dynamics and trajectory config for bench press.

    The bench press is modelled as a supine press: gravity acts along
    the vertical axis while the lifter pushes the bar upward from chest
    level.  The 3-link chain represents shoulder, elbow, wrist.

    Full rep: lockout (arms straight) -> chest (arms flexed) -> lockout.
    This uses a via-point trajectory just like full_squat.

    Returns:
        (dynamics, q_start, q_end, q_bounds, q_via)
    """
    bp = BenchPressModel(body)

    # Create a dynamics object using arm segment properties.
    # ChainGeometry provides typed arm geometry (L, d) for all
    # coupling-coefficient calculations — no post-hoc attribute surgery needed.
    dyn = LagrangianDynamics(
        body,
        bp.m.copy(),
        bp.I.copy(),
        bar_mass,
        chain_geometry=ChainGeometry(
            L=bp.L,
            d=bp.d,
            joint_names=["shoulder", "elbow", "wrist", "hand"],
        ),
        supine=True,
    )

    # Start: lockout (arms straight up, perpendicular to supine body)
    q_start = np.array([np.radians(0), np.radians(0), np.radians(0)])
    # Via: bar at chest (upper arm ~horizontal, elbow bent ~90 degrees)
    q_via = np.array([np.radians(80), np.radians(-100), np.radians(0)])
    # End: lockout again (full rep)
    q_end = np.array([np.radians(0), np.radians(0), np.radians(0)])

    q_bounds = np.array(
        [
            [
                BENCH_PRESS_JOINT_LIMITS["shoulder"][0],
                BENCH_PRESS_JOINT_LIMITS["shoulder"][1],
            ],
            [
                BENCH_PRESS_JOINT_LIMITS["elbow"][0],
                BENCH_PRESS_JOINT_LIMITS["elbow"][1],
            ],
            [
                BENCH_PRESS_JOINT_LIMITS["wrist"][0],
                BENCH_PRESS_JOINT_LIMITS["wrist"][1],
            ],
        ]
    )

    return dyn, q_start, q_end, q_bounds, q_via
