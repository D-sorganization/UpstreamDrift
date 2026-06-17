# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Anthropometric body model and related utilities.

Contains ``BodyModel``, ``ChainGeometry``, and joint-angle helper functions.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

from ..constants import (
    BOS_INNER_FRACTION,
    COM_FRAC,
    JOINT_LIMITS,
    JOINT_NAMES,
    LENGTH_FRAC,
    MASS_FRAC,
    RADIUS_OF_GYRATION_FRAC,
)

__all__ = [
    "BodyModel",
    "ChainGeometry",
    "clamp_joint_angles",
    "joint_angles_within_limits",
]


@dataclasses.dataclass(frozen=True)
class ChainGeometry:
    """Explicit geometry for a non-default kinematic chain.

    Used to configure LagrangianDynamics for chains other than the default
    leg model (e.g. the arm chain for bench press).  Replaces the old
    dict-based ``body_override`` parameter with a typed, self-documenting
    structure.

    Attributes:
        L: Segment lengths (length-3 array).
        d: Segment COM distances from proximal joint (length-3 array).
        joint_names: Human-readable names for the 3 joints + tip.
    """

    L: NDArray
    d: NDArray
    joint_names: list[str] = dataclasses.field(
        default_factory=lambda: ["link0", "link1", "link2", "link3"]
    )


class BodyModel:
    """Anthropometric model storing segment lengths and masses.

    Preconditions:
        body_mass > 0
        height > 0
        seg_multipliers values in [0.5, 2.0]

    The base-of-support (BOS) is split into an *inner zone* (middle 60%
    of the foot) and an *outer zone*.  The optimizer uses ``inner_heel``
    and ``inner_toe`` as the hard constraint boundary for COM placement.
    """

    def __init__(
        self,
        body_mass: float = 75.0,
        height: float = 1.75,
        seg_multipliers: dict[str, float] | None = None,
        abduction_angle: float = 0.0,
        arm_angle: float = 0.0,
        squat_bar_depth: float = 0.0,
        squat_bar_height: float = 0.0,
    ) -> None:
        """Initialize body model with anthropometric parameters.

        Args:
            body_mass: Total body mass in kg (must be positive).
            height: Standing height in meters (must be positive).
            seg_multipliers: Optional per-segment length scale factors keyed
                by ``"lower_leg"``, ``"upper_leg"``, ``"torso"``. Each value
                must be in [0.5, 2.0]. Defaults to 1.0 for all segments.
            abduction_angle: Hip abduction angle in degrees used to project
                leg lengths into the sagittal plane (default 0).
            arm_angle: Arm elevation angle in degrees used for bench press
                sagittal projection (default 0).
            squat_bar_depth: Horizontal offset (m) of the bar behind the
                shoulder for low-bar squat geometry (must be >= 0).
            squat_bar_height: Vertical offset (m) of the bar below the
                top of the torso (must be >= 0).

        Raises:
            ValueError: If body_mass or height are not positive, any
                segment multiplier is outside [0.5, 2.0], or either
                squat-bar offset is negative.
        """
        if body_mass <= 0:
            raise ValueError("body_mass must be positive")
        if height <= 0:
            raise ValueError("height must be positive")
        if squat_bar_depth < 0:
            raise ValueError("squat_bar_depth must be non-negative")
        if squat_bar_height < 0:
            raise ValueError("squat_bar_height must be non-negative")

        self.body_mass = body_mass
        self.height = height
        self.g = 9.81
        self.abduction_angle = abduction_angle
        self.arm_angle = arm_angle
        self.squat_bar_depth = squat_bar_depth
        self.squat_bar_height = squat_bar_height

        mults = self._validated_multipliers(seg_multipliers)
        self._compute_lengths(height, mults)
        self._compute_base_of_support()
        self._compute_masses(body_mass)
        self._compute_com_distances()
        self._compute_inertias()

    # -- private helpers ----------------------------------------

    @staticmethod
    def _validated_multipliers(
        raw: dict[str, float] | None,
    ) -> dict[str, float]:
        """Validate and normalise segment length multipliers.

        Args:
            raw: Raw multiplier dict or None (uses defaults of 1.0).

        Returns:
            Dict with validated multipliers for ``lower_leg``, ``upper_leg``,
            and ``torso``.

        Raises:
            ValueError: If any multiplier is outside [0.5, 2.0].
        """
        defaults = {"lower_leg": 1.0, "upper_leg": 1.0, "torso": 1.0}
        if raw is None:
            return defaults
        out: dict[str, float] = {}
        for k, v in defaults.items():
            val = raw.get(k, v)
            if not (0.5 <= val <= 2.0):
                raise ValueError(f"{k} multiplier out of range")
            out[k] = val
        return out

    def _compute_lengths(self, height: float, mults: dict[str, float]) -> None:
        """Compute segment lengths and sagittal-plane projections.

        Sets ``self.L``, ``self.L_eff``, ``self.L_arm``, ``self.foot_length``,
        and ``self.L_arm_eff`` based on Winter (2009) fractions scaled by
        height and segment multipliers.

        Args:
            height: Standing height in meters.
            mults: Validated per-segment length multipliers.
        """
        self.L = np.array(
            [
                LENGTH_FRAC["lower_leg"] * height * mults["lower_leg"],
                LENGTH_FRAC["upper_leg"] * height * mults["upper_leg"],
                LENGTH_FRAC["torso"] * height * mults["torso"],
            ]
        )
        self.L_arm = LENGTH_FRAC["arm"] * height
        self.foot_length = LENGTH_FRAC["foot"] * height

        # Leg abduction correction: project leg lengths into sagittal plane
        abduction_rad = np.radians(self.abduction_angle)
        correction = np.cos(abduction_rad)
        self.L_eff = self.L.copy()
        self.L_eff[0] *= correction  # lower leg projected length
        self.L_eff[1] *= correction  # upper leg projected length
        # torso (index 2) is NOT corrected - it stays in sagittal plane

        # Arm angle projection for bench press side-view
        self.L_arm_eff = self.L_arm * np.sin(np.radians(self.arm_angle))

    def _compute_base_of_support(self) -> None:
        """Compute full and inner (constrained) base-of-support bounds.

        The ankle sits at x=0.  The heel is slightly behind, the toe
        extends forward.  The *inner* zone is the middle BOS_INNER_FRACTION
        of the foot -- the outer 20% on each end is out-of-bounds for the
        COM constraint.
        """
        self.heel_x = -0.05 * self.foot_length
        self.toe_x = 0.95 * self.foot_length

        foot_span = self.toe_x - self.heel_x
        outer_margin = (1.0 - BOS_INNER_FRACTION) / 2.0 * foot_span
        self.inner_heel = self.heel_x + outer_margin
        self.inner_toe = self.toe_x - outer_margin

        # Centers for soft preference penalties
        self.bos_center = 0.5 * (self.heel_x + self.toe_x)
        self.inner_center = 0.5 * (self.inner_heel + self.inner_toe)

    def _compute_masses(self, bm: float) -> None:
        """Compute per-segment masses from body mass fractions (Winter 2009).

        Sets ``self.m_feet``, ``self.m_squat``, ``self.m_deadlift``, and
        related mass arrays for each exercise configuration.

        Args:
            bm: Total body mass in kg.
        """
        self.m_feet = MASS_FRAC["feet"] * bm
        self.foot_com_x = self.bos_center
        self.foot_com_y = 0.0

        self.m_squat = np.array(
            [
                MASS_FRAC["lower_legs"] * bm,
                MASS_FRAC["upper_legs"] * bm,
                (MASS_FRAC["trunk_head"] + MASS_FRAC["arms"]) * bm,
            ]
        )
        self.m_deadlift = np.array(
            [
                MASS_FRAC["lower_legs"] * bm,
                MASS_FRAC["upper_legs"] * bm,
                MASS_FRAC["trunk_head"] * bm,
            ]
        )
        self.m_arms = MASS_FRAC["arms"] * bm

    def _compute_com_distances(self) -> None:
        """Compute segment COM distances from proximal joint (Winter 2009 fractions).

        Sets ``self.d`` (actual segment COM distances) and ``self.d_eff``
        (sagittal-plane projected COM distances) using COM_FRAC constants.
        """
        self.d = np.array(
            [
                COM_FRAC["lower_leg"] * self.L[0],
                COM_FRAC["upper_leg"] * self.L[1],
                COM_FRAC["torso"] * self.L[2],
            ]
        )
        # Projected COM distances for kinematic calculations
        self.d_eff = np.array(
            [
                COM_FRAC["lower_leg"] * self.L_eff[0],
                COM_FRAC["upper_leg"] * self.L_eff[1],
                COM_FRAC["torso"] * self.L_eff[2],
            ]
        )

    def _compute_inertias(self) -> None:
        """Compute **centroidal** segment inertias (about each segment COM).

        I_com = m * (rho * L)^2          (about segment COM)

        ``LagrangianDynamics`` adds the parallel-axis term ``m * d_com^2``
        itself when assembling the diagonal mass matrix, so the values stored
        here must be the centroidal inertia only. Pre-adding the parallel-axis
        term here double-counts it and overestimates joint inertia (issue
        #490).

        Radius-of-gyration fractions (rho) from Winter (2009), Table 3.1.
        """
        rho = np.array(
            [
                RADIUS_OF_GYRATION_FRAC["lower_leg"],
                RADIUS_OF_GYRATION_FRAC["upper_leg"],
                RADIUS_OF_GYRATION_FRAC["trunk"],
            ]
        )
        self.I_squat = self.m_squat * (rho * self.L) ** 2
        self.I_deadlift = self.m_deadlift * (rho * self.L) ** 2


def clamp_joint_angles(
    q: NDArray,
    joint_limits: dict[str, tuple[float, float]] | None = None,
    joint_names: tuple[str, ...] | None = None,
) -> NDArray:
    """Clamp joint angles to their anatomical limits.

    Returns a copy with each angle clamped to [lo, hi].
    Handles the math gracefully: no NaN, no inf, no out-of-range.

    Preconditions:
        q is a 1-D array of length len(joint_names)
        joint_limits keys must include all joint_names
    """
    limits = joint_limits or JOINT_LIMITS
    names = joint_names or JOINT_NAMES
    if len(q) != len(names):
        raise ValueError(f"q length {len(q)} != {len(names)} joint names")
    q_clamped = q.copy()
    for i, name in enumerate(names):
        lo, hi = limits[name]
        q_clamped[i] = np.clip(q_clamped[i], lo, hi)
    return q_clamped


def joint_angles_within_limits(
    q: NDArray,
    joint_limits: dict[str, tuple[float, float]] | None = None,
    joint_names: tuple[str, ...] | None = None,
    tol: float = 1e-6,
) -> bool:
    """Check whether all joint angles are within anatomical limits.

    Preconditions:
        q is a 1-D array of length len(joint_names)
    """
    limits = joint_limits or JOINT_LIMITS
    names = joint_names or JOINT_NAMES
    if len(q) != len(names):
        raise ValueError(f"q length {len(q)} != {len(names)} joint names")
    for i, name in enumerate(names):
        lo, hi = limits[name]
        if q[i] < lo - tol or q[i] > hi + tol:
            return False
    return True
