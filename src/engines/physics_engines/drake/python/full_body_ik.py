"""Drake full-body inverse kinematics and marker tracking adapter (FB-4).

Supplies forward kinematics (``pose_fn``), dual-grip weld loop-closure residuals,
and least-squares inverse kinematics over the 41 full-body coordinates.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.drake.python.full_body_model import (
    FullBodyDrakeModel,
)
from src.shared.python.motion_matching.full_body_ik import (
    compute_marker_rms_trajectory,
    solve_full_body_ik_trajectory,
)
from src.shared.python.motion_matching.marker_calibration import Offsets, Pose
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

Array: TypeAlias = NDArray[np.float64]


class DrakeFullBodyIK:
    """Full-body Drake inverse kinematics adapter consuming a full-body spec."""

    def __init__(self, specification: Mapping[str, Any] | bytes | str) -> None:
        if isinstance(specification, bytes):
            spec_dict = json.loads(specification.decode("utf-8"))
        elif isinstance(specification, str):
            spec_dict = json.loads(specification)
        else:
            spec_dict = dict(specification)

        self.specification = spec_dict
        self.model = FullBodyDrakeModel(spec_dict)
        self._plant = self.model.plant
        self._context = self.model.context
        self._metadata = self.model.metadata
        self._closure = self.model._closure
        self._world_frame = self._plant.world_frame()
        self.coordinate_order: tuple[str, ...] = tuple(self.model.names)
        if len(self.coordinate_order) != 41:
            raise ValueError("Expected exactly 41 full-body coordinates")

        # Map marker bodies to Drake Frame or Body
        marker_attachments = self.specification.get("marker_attachments", {})
        bodies = sorted(
            set(v["body"] for v in marker_attachments.values() if "body" in v)
        )
        self._frame_objects: dict[str, Any] = {}
        self._body_objects: dict[str, Any] = {}

        for b in bodies:
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

    def ik_fn(
        self,
        offsets: Offsets,
        capture: TourCapture,
        initial_q: Array | None = None,
        *,
        closure_weight: float = 10.0,
        max_nfev: int = 50,
    ) -> Array:
        """Solve least-squares IK across all frames of the capture."""
        initial: Array
        if initial_q is None:
            initial = np.zeros(len(self.coordinate_order), dtype=float)
            waist_labels = ("WaistLeft", "WaistRight", "WaistLBack", "WaistRBack")
            if all(lbl in capture.labels for lbl in waist_labels):
                indices = [capture.index(lbl) for lbl in waist_labels]
                centroid = np.nanmean(capture.points_m[0, indices], axis=0)
                initial[0] = centroid[0]
                initial[1] = centroid[1]
                initial[2] = centroid[2]
        else:
            initial = np.asarray(initial_q, dtype=float)

        return solve_full_body_ik_trajectory(
            self.pose_fn,
            offsets,
            capture,
            initial,
            closure_fn=self.closure_residuals,
            closure_weight=closure_weight,
            max_nfev=max_nfev,
        )

    def evaluate_trajectory_rms(
        self,
        offsets: Offsets,
        capture: TourCapture,
        q_trajectory: Array,
    ) -> tuple[Array, dict[str, float], float]:
        """Compute per-frame, per-marker, and overall RMS errors."""
        return compute_marker_rms_trajectory(
            self.pose_fn, offsets, capture, q_trajectory
        )
