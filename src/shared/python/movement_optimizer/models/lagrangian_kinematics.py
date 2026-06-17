# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Kinematic output methods for the Lagrangian planar-chain model.

``LagrangianKinematicsMixin`` provides forward kinematics, bar-position
lookup, batch COM x-coordinate, and full COM position for the 3-link
sagittal chain.  It is mixed into ``LagrangianDynamics`` so that the core
physics class stays focused on mass-matrix and torque computation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .body_model import BodyModel


class LagrangianKinematicsMixin:
    """Kinematics helpers for a 3-link sagittal chain.

    Assumes the owning class has:
        self.body      -- BodyModel
        self.m         -- (3,) segment masses
        self.L_eff     -- (3,) effective segment lengths
        self.d_eff     -- (3,) effective COM distances
        self.joint_names -- list[str] of joint names
    """

    # Declare the expected attributes from the owning class (LagrangianDynamics)
    # so type checkers can verify access without requiring a full inheritance
    # relationship in the mixin itself.
    body: BodyModel
    m: NDArray
    L_eff: NDArray
    d_eff: NDArray
    joint_names: list[str]

    def com_x_batch(
        self,
        q: NDArray,
        exercise_type: str = "squat",
        bar_mass: float = 0.0,
    ) -> NDArray:
        """Vectorised batch COM x-coordinate.

        Parameters:
            q: shape (N, 3)
        Returns:
            com_x: shape (N,)

        Complexity:
            O(N) time and O(N) memory for ``N`` trajectory samples.
        """
        b = self.body
        sq = np.sin(q)
        L = self.L_eff
        d = self.d_eff

        knee_x = L[0] * sq[:, 0]
        hip_x = knee_x + L[1] * sq[:, 1]
        shoulder_x = hip_x + L[2] * sq[:, 2]

        c1x = d[0] * sq[:, 0]
        c2x = knee_x + d[1] * sq[:, 1]
        c3x = hip_x + d[2] * sq[:, 2]

        total_mass = b.body_mass + bar_mass
        numerator = b.m_feet * b.foot_com_x + self.m[0] * c1x + self.m[1] * c2x + self.m[2] * c3x

        if exercise_type in ("squat", "full_squat"):
            if hasattr(b, "squat_bar_depth") and (
                b.squat_bar_depth != 0.0 or b.squat_bar_height != 0.0
            ):
                bar_x = (
                    shoulder_x - b.squat_bar_height * sq[:, 2] - b.squat_bar_depth * np.cos(q[:, 2])
                )
            else:
                bar_x = shoulder_x
            numerator += bar_mass * bar_x
        else:
            numerator += b.m_arms * shoulder_x + bar_mass * shoulder_x

        return numerator / total_mass

    def forward_kinematics(self, q: NDArray) -> dict[str, NDArray]:
        """Compute joint positions for all joints in the chain.

        Complexity:
            O(1) time and memory for the fixed 3-link model.
        """
        L = self.L_eff
        names = self.joint_names

        # Performance optimization: Fully unroll scalar components and only instantiate
        # the final return vectors to prevent massive memory allocation overhead from
        # intermediate np.array combinations.
        q0, q1, q2 = q
        sq0, sq1, sq2 = math.sin(q0), math.sin(q1), math.sin(q2)
        cq0, cq1, cq2 = math.cos(q0), math.cos(q1), math.cos(q2)

        p1_x = L[0] * sq0
        p1_y = L[0] * cq0

        p2_x = p1_x + L[1] * sq1
        p2_y = p1_y + L[1] * cq1

        p3_x = p2_x + L[2] * sq2
        p3_y = p2_y + L[2] * cq2

        return {
            names[0]: np.array([0.0, 0.0]),
            names[1]: np.array([p1_x, p1_y]),
            names[2]: np.array([p2_x, p2_y]),
            names[3]: np.array([p3_x, p3_y]),
        }

    def bar_position(self, q: NDArray, exercise_type: str) -> NDArray:
        """Return the barbell position vector for a given joint configuration.

        Complexity:
            O(1) time and memory for the fixed 3-link model.
        """
        L = self.L_eff

        # Performance optimization: Calculate shoulder position directly and unroll scalar
        # components for exercise-specific logic to avoid intermediate array allocations.
        q0, q1, q2 = q
        sq0, sq1, sq2 = math.sin(q0), math.sin(q1), math.sin(q2)
        cq0, cq1, cq2 = math.cos(q0), math.cos(q1), math.cos(q2)

        p1_x = L[0] * sq0
        p1_y = L[0] * cq0

        p2_x = p1_x + L[1] * sq1
        p2_y = p1_y + L[1] * cq1

        s_x = p2_x + L[2] * sq2
        s_y = p2_y + L[2] * cq2

        if exercise_type in ("squat", "full_squat"):
            b = self.body
            if hasattr(b, "squat_bar_depth") and (
                b.squat_bar_depth != 0.0 or b.squat_bar_height != 0.0
            ):
                return np.array(
                    [
                        s_x - b.squat_bar_height * sq2 - b.squat_bar_depth * cq2,
                        s_y - b.squat_bar_height * cq2 + b.squat_bar_depth * sq2,
                    ]
                )
            return np.array([s_x, s_y])
        if exercise_type == "deadlift":
            # Bar hangs from hands: arm-length below shoulder
            return np.array([s_x, s_y - self.body.L_arm])
        if exercise_type in ("clean", "clean_and_jerk"):
            # Front rack: bar sits at shoulder height
            return np.array([s_x, s_y])
        if exercise_type in ("snatch", "jerk"):
            # Overhead: bar is arm-length above shoulder
            return np.array([s_x, s_y + self.body.L_arm])
        return np.array([s_x, s_y])

    def com_position(
        self,
        q: NDArray,
        exercise_type: str = "squat",
        bar_mass: float = 0.0,
    ) -> NDArray:
        """Return the whole-body COM position vector.

        Complexity:
            O(1) time and memory for the fixed 3-link model.
        """
        from ..constants import COM_FRAC

        b = self.body
        L = self.L_eff
        d = self.d_eff

        # Performance optimization: Fully unroll scalar components to avoid multiple
        # intermediate array allocations and vector math overhead.
        q0, q1, q2 = q
        sq0, sq1, sq2 = math.sin(q0), math.sin(q1), math.sin(q2)
        cq0, cq1, cq2 = math.cos(q0), math.cos(q1), math.cos(q2)

        knee_x = L[0] * sq0
        knee_y = L[0] * cq0

        hip_x = knee_x + L[1] * sq1
        hip_y = knee_y + L[1] * cq1

        shoulder_x = hip_x + L[2] * sq2
        shoulder_y = hip_y + L[2] * cq2

        c1_x = d[0] * sq0
        c1_y = d[0] * cq0

        c2_x = knee_x + d[1] * sq1
        c2_y = knee_y + d[1] * cq1

        c3_x = hip_x + d[2] * sq2
        c3_y = hip_y + d[2] * cq2

        total_mass = b.body_mass + bar_mass

        num_x = b.m_feet * b.foot_com_x + self.m[0] * c1_x + self.m[1] * c2_x + self.m[2] * c3_x
        num_y = b.m_feet * b.foot_com_y + self.m[0] * c1_y + self.m[1] * c2_y + self.m[2] * c3_y

        if exercise_type in ("squat", "full_squat"):
            bar_pos = self.bar_position(q, exercise_type)
            num_x += bar_mass * bar_pos[0]
            num_y += bar_mass * bar_pos[1]
        else:
            bar_pos = self.bar_position(q, exercise_type)
            # Arm vector from shoulder to bar
            arm_vec_x = bar_pos[0] - shoulder_x
            arm_vec_y = bar_pos[1] - shoulder_y
            arm_com_x = shoulder_x + COM_FRAC["arm"] * arm_vec_x
            arm_com_y = shoulder_y + COM_FRAC["arm"] * arm_vec_y
            num_x += b.m_arms * arm_com_x + bar_mass * bar_pos[0]
            num_y += b.m_arms * arm_com_y + bar_mass * bar_pos[1]

        return np.array([num_x / total_mass, num_y / total_mass])
