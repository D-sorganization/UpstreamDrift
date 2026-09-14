"""MuJoCo full-body inverse kinematics and marker tracking adapter (FB-4).

Supplies forward kinematics (``pose_fn``), dual-grip weld loop-closure residuals,
and least-squares inverse kinematics over the 41 full-body coordinates.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching.full_body_ik import (
    compute_marker_rms_trajectory,
    solve_full_body_ik_trajectory,
)
from src.shared.python.motion_matching.marker_calibration import Offsets, Pose
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

Array: TypeAlias = NDArray[np.float64]


class MujocoFullBodyIK:
    """Full-body MuJoCo inverse kinematics adapter consuming a full-body spec."""

    def __init__(self, specification: Mapping[str, Any] | bytes | str) -> None:
        if isinstance(specification, bytes):
            spec_bytes = specification
            spec_dict = json.loads(specification.decode("utf-8"))
        elif isinstance(specification, str):
            spec_bytes = specification.encode("utf-8")
            spec_dict = json.loads(specification)
        else:
            spec_dict = dict(specification)
            spec_bytes = json.dumps(spec_dict).encode("utf-8")

        self.specification = spec_dict
        self.model = NativeMujocoFullBodyModel(spec_bytes)
        self._mj_model = self.model.model
        self._mj_data = self.model.data
        self._metadata = self.model.metadata
        self.coordinate_order: tuple[str, ...] = tuple(self.model.coordinate_order)
        if len(self.coordinate_order) != 41:
            raise ValueError("Expected exactly 41 full-body coordinates")

        # Map marker bodies to MuJoCo site or body IDs
        marker_attachments = self.specification.get("marker_attachments", {})
        bodies = sorted(
            set(v["body"] for v in marker_attachments.values() if "body" in v)
        )
        self._site_ids: dict[str, int] = {}
        self._body_ids: dict[str, int] = {}
        frame_sites = self._metadata.get("frame_sites", {})

        for b in bodies:
            if b in frame_sites:
                self._site_ids[b] = self._mj_model.site(frame_sites[b]).id
            else:
                self._body_ids[b] = self._mj_model.body(b).id

        self._closure_a_id = self._mj_model.site("native_closure_a").id
        self._closure_b_id = self._mj_model.site("native_closure_b").id

    def pose_fn(self, q: Array) -> dict[str, Pose]:
        """Compute world poses (R, t) for all bodies referenced by markers."""
        import mujoco

        q_arr = np.asarray(q, dtype=float)
        if q_arr.size != len(self.coordinate_order):
            raise ValueError(f"Expected {len(self.coordinate_order)} coordinates")

        for i in range(len(q_arr)):
            self._mj_data.qpos[i] = q_arr[i]
        mujoco.mj_kinematics(self._mj_model, self._mj_data)

        poses: dict[str, Pose] = {}
        for b, sid in self._site_ids.items():
            r = self._mj_data.site_xmat[sid].reshape((3, 3)).copy()
            t = self._mj_data.site_xpos[sid].copy()
            poses[b] = (r, t)
        for b, bid in self._body_ids.items():
            r = self._mj_data.xmat[bid].reshape((3, 3)).copy()
            t = self._mj_data.xpos[bid].copy()
            poses[b] = (r, t)
        return poses

    def closure_residuals(self, q: Array) -> Array:
        """Evaluate position residual between dual-grip weld sites in world."""
        # mj_kinematics was executed in pose_fn
        pa = self._mj_data.site_xpos[self._closure_a_id]
        pb = self._mj_data.site_xpos[self._closure_b_id]
        return np.asarray(pa - pb, dtype=float)

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
            # Default translation from waist centroid if present
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
