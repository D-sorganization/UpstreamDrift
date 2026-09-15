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
from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK
from src.shared.python.motion_matching.marker_calibration import Pose

Array: TypeAlias = NDArray[np.float64]


class MujocoFullBodyIK(BaseFullBodyIK):
    """Full-body MuJoCo inverse kinematics adapter consuming a full-body spec."""

    def __init__(self, specification: Mapping[str, Any] | bytes | str) -> None:
        super().__init__(specification)
        spec_bytes = json.dumps(self.specification).encode("utf-8")
        self.model = NativeMujocoFullBodyModel(spec_bytes)
        self._mj_model = self.model.model
        self._mj_data = self.model.data
        self._metadata = self.model.metadata
        self.coordinate_order: tuple[str, ...] = tuple(self.model.coordinate_order)
        if len(self.coordinate_order) != 41:
            raise ValueError("Expected exactly 41 full-body coordinates")

        # Map marker bodies to MuJoCo site or body IDs
        self._site_ids: dict[str, int] = {}
        self._body_ids: dict[str, int] = {}
        frame_sites = self._metadata.get("frame_sites", {})

        for b in self.marker_bodies:
            if b in frame_sites:
                self._site_ids[b] = self._mj_model.site(frame_sites[b]).id
            else:
                self._body_ids[b] = self._mj_model.body(b).id

        self._closure_a_id = self._mj_model.site("native_closure_a").id
        self._closure_b_id = self._mj_model.site("native_closure_b").id

        # Map declared coordinate order to native qpos DOF addresses
        self._qpos_indices = np.array(
            [self.model._indices[name] for name in self.coordinate_order],
            dtype=np.int32,
        )

    def pose_fn(self, q: Array) -> dict[str, Pose]:
        """Compute world poses (R, t) for all bodies referenced by markers."""
        import mujoco

        q_arr = np.asarray(q, dtype=float)
        if q_arr.size != len(self.coordinate_order):
            raise ValueError(f"Expected {len(self.coordinate_order)} coordinates")

        self._mj_data.qpos[self._qpos_indices] = q_arr
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
        q_arr = np.asarray(q, dtype=float)
        if not np.array_equal(self._mj_data.qpos[self._qpos_indices], q_arr):
            import mujoco

            self._mj_data.qpos[self._qpos_indices] = q_arr
            mujoco.mj_kinematics(self._mj_model, self._mj_data)

        pa = self._mj_data.site_xpos[self._closure_a_id]
        pb = self._mj_data.site_xpos[self._closure_b_id]
        return np.asarray(pa - pb, dtype=float)
