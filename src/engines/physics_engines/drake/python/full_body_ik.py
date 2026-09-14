"""Drake full-body inverse kinematics and marker tracking adapter (FB-4).

Supplies forward kinematics (``pose_fn``), dual-grip weld loop-closure residuals,
and least-squares inverse kinematics over the 41 full-body coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.drake.python.full_body_model import (
    FullBodyDrakeModel,
)
from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK
from src.shared.python.motion_matching.marker_calibration import Pose

Array: TypeAlias = NDArray[np.float64]


class DrakeFullBodyIK(BaseFullBodyIK):
    """Full-body Drake inverse kinematics adapter consuming a full-body spec."""

    def __init__(self, specification: Mapping[str, Any] | bytes | str) -> None:
        super().__init__(specification)
        self.model = FullBodyDrakeModel(self.specification)
        self._plant = self.model.plant
        self._context = self.model.context
        self._metadata = self.model.metadata
        self._closure = self.model._closure
        self._world_frame = self._plant.world_frame()
        self.coordinate_order: tuple[str, ...] = tuple(self.model.names)
        if len(self.coordinate_order) != 41:
            raise ValueError("Expected exactly 41 full-body coordinates")

        # Map marker bodies to Drake Frame or Body
        self._frame_objects: dict[str, Any] = {}
        self._body_objects: dict[str, Any] = {}

        for b in self.marker_bodies:
            if b in self.model._frames:
                self._frame_objects[b] = self.model._frames[b]
            elif b in self._metadata["body_links"]:
                link_name = self._metadata["body_links"][b]
                self._body_objects[b] = self._plant.GetBodyByName(
                    link_name, self.model._instance
                )
            else:
                raise ValueError(f"Body/frame {b} not found in Drake model")

    def pose_fn(self, q: Array) -> dict[str, Pose]:
        """Compute world poses (R, t) for all bodies referenced by markers."""
        q_arr = np.asarray(q, dtype=float)
        if q_arr.size != len(self.coordinate_order):
            raise ValueError(f"Expected {len(self.coordinate_order)} coordinates")

        self._plant.SetPositions(self._context, q_arr)
        poses: dict[str, Pose] = {}

        for b, frame in self._frame_objects.items():
            x_wf = self._plant.CalcRelativeTransform(
                self._context, self._world_frame, frame
            )
            poses[b] = (x_wf.rotation().matrix().copy(), x_wf.translation().copy())

        for b, body in self._body_objects.items():
            x_wb = self._plant.EvalBodyPoseInWorld(self._context, body)
            poses[b] = (x_wb.rotation().matrix().copy(), x_wb.translation().copy())

        return poses

    def closure_residuals(self, q: Array) -> Array:
        """Evaluate position residual between dual-grip weld frames in world."""
        x_wa = self._plant.CalcRelativeTransform(
            self._context, self._world_frame, self._closure[0]
        )
        x_wb = self._plant.CalcRelativeTransform(
            self._context, self._world_frame, self._closure[1]
        )
        return np.asarray(x_wa.translation() - x_wb.translation(), dtype=float)
