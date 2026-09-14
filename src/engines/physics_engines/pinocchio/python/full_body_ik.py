"""Pinocchio full-body inverse kinematics and marker tracking adapter (FB-4).

Supplies forward kinematics (``pose_fn``), dual-grip weld loop-closure residuals,
and least-squares inverse kinematics over the 41 full-body coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.pinocchio.python.native_model import (
    FullBodyPinocchioModel,
)
from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK
from src.shared.python.motion_matching.marker_calibration import Pose

Array: TypeAlias = NDArray[np.float64]


class PinocchioFullBodyIK(BaseFullBodyIK):
    """Full-body Pinocchio inverse kinematics adapter consuming a full-body spec."""

    def __init__(self, specification: Mapping[str, Any] | bytes | str) -> None:
        super().__init__(specification)
        self.model = FullBodyPinocchioModel(self.specification)
        self._pin_model = self.model.model
        self._pin_data = self.model.data
        self.coordinate_order: tuple[str, ...] = tuple(
            self.specification["coordinate_order"]
        )
        if len(self.coordinate_order) != 41:
            raise ValueError("Expected exactly 41 full-body coordinates")

        # Map marker bodies to Pinocchio frame ID or body tuple
        self._frame_ids: dict[str, int] = {}
        self._body_tuples: dict[str, tuple[int, Any]] = {}

        for b in self.marker_bodies:
            if b in self.model._frames:
                self._frame_ids[b] = self.model._frames[b]
            elif b in self.model._bodies:
                self._body_tuples[b] = self.model._bodies[b]
            else:
                raise ValueError(f"Body/frame {b} not found in Pinocchio model")

        closure = self.specification["closure"]
        ja, pa = self.model._bodies[closure["body_a"]]
        jb, pb = self.model._bodies[closure["body_b"]]
        self._closure_ja = ja
        self._closure_pa = pa * self.model._transform(closure["placement_a"])
        self._closure_jb = jb
        self._closure_pb = pb * self.model._transform(closure["placement_b"])

    def pose_fn(self, q: Array) -> dict[str, Pose]:
        """Compute world poses (R, t) for all bodies referenced by markers."""
        q_arr = np.asarray(q, dtype=float)
        if q_arr.size != len(self.coordinate_order):
            raise ValueError(f"Expected {len(self.coordinate_order)} coordinates")

        pin = self.model._pin
        pin.forwardKinematics(self._pin_model, self._pin_data, q_arr)
        pin.updateFramePlacements(self._pin_model, self._pin_data)

        poses: dict[str, Pose] = {}
        for b, fid in self._frame_ids.items():
            omf = self._pin_data.oMf[fid]
            poses[b] = (omf.rotation.copy(), omf.translation.copy())
        for b, (jid, b_placement) in self._body_tuples.items():
            omi = self._pin_data.oMi[jid] * b_placement
            poses[b] = (omi.rotation.copy(), omi.translation.copy())
        return poses

    def closure_residuals(self, q: Array) -> Array:
        """Evaluate position residual between dual-grip weld frames in world."""
        # forwardKinematics was called in pose_fn
        oma = self._pin_data.oMi[self._closure_ja] * self._closure_pa
        omb = self._pin_data.oMi[self._closure_jb] * self._closure_pb
        return np.asarray(oma.translation - omb.translation, dtype=float)
