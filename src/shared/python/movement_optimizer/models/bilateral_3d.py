# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""3D Bilateral forward-kinematics model.

This module provides a minimal-but-correct 3D kinematic model of a human
squatter with two independent legs joined at the pelvis and a single
torso above.  It is the first step toward a true 3D Bilateral physics
backend.  For now it exposes **forward kinematics only** -- the existing
2D sagittal ``LagrangianDynamics`` continues to own inverse dynamics.

Design
------
Representation
    * Two legs (left / right) share anthropometry.  Each leg has three
      angles that follow the 2D convention of the existing sagittal
      model: they are **absolute** angles of each segment from vertical,
      positive = flexed forward.  The three angles per leg correspond to
      the shin, thigh, and torso orientations; the third angle is shared
      with the single torso segment so a symmetric 3D pose reduces to
      the 2D pose exactly.
    * A static **stance width** separates the two feet on the
      mediolateral axis so the 3D pose is visually honest (not merely a
      stack of 2D chains at the origin).

Coordinate frame
    Right-handed, **z-up**:
        ``+x`` -- forward (anterior)
        ``+y`` -- left
        ``+z`` -- up
    This matches common biomechanics conventions (e.g. OpenSim, ISB).

Forward kinematics
    A plain 4x4 homogeneous transform chain implemented with NumPy.
    Each segment contributes a rotation about the mediolateral (``y``)
    axis by its absolute angle, followed by a translation of its length
    along the segment's local z-axis.

Parity with 2D
    When the two legs are given the same joint angles, the sagittal
    (``x``/``z``) projection of the 3D model reproduces the 2D
    ``LagrangianDynamics.forward_kinematics`` output exactly (modulo the
    mediolateral offset of each foot).  This is enforced by a unit test.

Scope
    This is a *kinematics-only* MVP.  Inverse dynamics, 3D torques, and
    optimisation support are intentionally deferred to follow-up PRs.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

from .body_model import BodyModel

__all__ = [
    "Bilateral3DModel",
    "Bilateral3DPose",
]


# ----------------------------------------------------------------------
# Pose dataclass
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Bilateral3DPose:
    """Joint configuration for the 3D Bilateral model.

    All angles are in radians and follow the "0 = vertical, positive =
    sagittal flexion" convention of the existing 2D model.

    Attributes:
        left_leg:  ``(ankle, knee, hip)`` absolute segment angles from
            vertical for the left shin, left thigh, and torso as seen
            from the left leg's frame.
        right_leg: as ``left_leg`` but for the right side.
        torso: absolute torso angle from vertical (authoritative torso
            orientation; the leg-level torso entry is ignored when the
            two legs disagree).
    """

    left_leg: tuple[float, float, float]
    right_leg: tuple[float, float, float]
    torso: float

    @classmethod
    def from_sagittal(cls, q: NDArray | tuple[float, float, float]) -> Bilateral3DPose:
        """Build a symmetric pose from a 2D ``(ankle, knee, hip)`` vector.

        This is the canonical way to lift a 2D joint configuration into
        the 3D Bilateral representation: both legs receive the same
        joint angles and the torso angle matches ``q[2]`` (hip flexion).
        """
        q = np.asarray(q, dtype=float)
        if q.shape != (3,):
            raise ValueError(f"q must have shape (3,), got {q.shape}")
        legs = (float(q[0]), float(q[1]), float(q[2]))
        return cls(left_leg=legs, right_leg=legs, torso=float(q[2]))


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------


class Bilateral3DModel:
    """3D Bilateral kinematic model (two legs + torso).

    Preconditions:
        body is a valid ``BodyModel``
        stance_width_m >= 0  (distance between foot centres in y)

    The model holds no state beyond geometry -- each call to
    :meth:`forward_kinematics` is a pure function of the input pose.

    Joint naming::

        left_ankle, left_knee, left_hip,
        right_ankle, right_knee, right_hip,
        pelvis, shoulder

    ``shoulder`` is the top of the torso; the 2D model uses the same
    name for the same landmark.
    """

    JOINT_NAMES: tuple[str, ...] = (
        "left_ankle",
        "left_knee",
        "left_hip",
        "right_ankle",
        "right_knee",
        "right_hip",
        "pelvis",
        "shoulder",
    )

    def __init__(
        self,
        body: BodyModel,
        stance_width_m: float | None = None,
    ) -> None:
        if not isinstance(body, BodyModel):
            raise TypeError("body must be a BodyModel instance")
        if stance_width_m is not None and stance_width_m < 0:
            raise ValueError("stance_width_m must be non-negative")

        self.body = body

        # Segment lengths.  We use the sagittal-effective lengths so that
        # the projection onto the x/z plane matches the 2D model exactly.
        self.L_shin = float(body.L_eff[0])
        self.L_thigh = float(body.L_eff[1])
        self.L_torso = float(body.L_eff[2])

        # Default stance width: use twice the sagittal-plane chord
        # produced by hip abduction, falling back to a small visible
        # default so the two legs are distinguishable in the renderer.
        if stance_width_m is None:
            abduction_chord = 2.0 * body.L[1] * np.sin(np.radians(body.abduction_angle))
            stance_width_m = max(abduction_chord, 0.20)
        self.stance_width_m = float(stance_width_m)

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    @staticmethod
    def _sagittal_step(
        origin_xz: NDArray,
        length: float,
        angle: float,
    ) -> NDArray:
        """Return ``origin_xz + length * (sin(angle), cos(angle))``.

        Matches the 2D model's segment step exactly.  ``origin_xz`` is a
        2-vector in the (x, z) sagittal plane.
        """
        # Performance optimization: Skip intermediate array allocation
        return np.array(
            [origin_xz[0] + length * np.sin(angle), origin_xz[1] + length * np.cos(angle)]
        )

    def forward_kinematics(self, pose: Bilateral3DPose) -> dict[str, NDArray]:
        """Return joint positions in the world frame as a ``{name: xyz}`` dict.

        The world origin is the midpoint between the two ankles on the
        ground plane.  See the module docstring for the coordinate-frame
        convention.  Each returned value is a length-3 numpy array.
        """
        if not isinstance(pose, Bilateral3DPose):
            raise TypeError("pose must be a Bilateral3DPose")

        half_w = 0.5 * self.stance_width_m
        out: dict[str, NDArray] = {}

        # Each leg is a pure sagittal chain offset in +/-y.  Using the
        # same 2D step function guarantees sagittal-projection parity.
        for side, y_off, angles in (
            ("left", +half_w, pose.left_leg),
            ("right", -half_w, pose.right_leg),
        ):
            q_a, q_k, _q_h = angles
            ankle_xz = np.array([0.0, 0.0])
            knee_xz = self._sagittal_step(ankle_xz, self.L_shin, q_a)
            hip_xz = self._sagittal_step(knee_xz, self.L_thigh, q_k)

            out[f"{side}_ankle"] = np.array([ankle_xz[0], y_off, ankle_xz[1]])
            out[f"{side}_knee"] = np.array([knee_xz[0], y_off, knee_xz[1]])
            out[f"{side}_hip"] = np.array([hip_xz[0], y_off, hip_xz[1]])

        # Pelvis sits at the midpoint of the two hips.
        pelvis = 0.5 * (out["left_hip"] + out["right_hip"])
        out["pelvis"] = pelvis

        # Torso rises from the pelvis at the authoritative torso angle.
        # This keeps the 2D parity property: when both legs share the
        # same hip angle ``q_h`` and ``pose.torso == q_h``, the shoulder
        # lands at exactly ``hip + L_torso * (sin(q_h), cos(q_h))``
        # (projected into the sagittal plane).
        shoulder_xz = self._sagittal_step(
            np.array([pelvis[0], pelvis[2]]), self.L_torso, pose.torso
        )
        out["shoulder"] = np.array([shoulder_xz[0], pelvis[1], shoulder_xz[1]])

        return out

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def t_pose(self) -> Bilateral3DPose:
        """Return the zero-angle standing pose (all joints straight)."""
        return Bilateral3DPose(
            left_leg=(0.0, 0.0, 0.0),
            right_leg=(0.0, 0.0, 0.0),
            torso=0.0,
        )

    def segment_pairs(self) -> list[tuple[str, str]]:
        """Return the list of ``(proximal, distal)`` joint-name pairs.

        Used by the renderer to draw line segments between joints.
        """
        return [
            ("left_ankle", "left_knee"),
            ("left_knee", "left_hip"),
            ("right_ankle", "right_knee"),
            ("right_knee", "right_hip"),
            ("left_hip", "right_hip"),
            ("pelvis", "shoulder"),
        ]
