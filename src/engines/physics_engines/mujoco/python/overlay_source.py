"""MuJoCo-backed contact and kinematics source for bundle overlays (NV-3, #11676).

The overlay wrenches (joint torques, ground reaction, weight) depend only on
the specification and a pose, so one MuJoCo evaluation of the specification
export serves every engine's native viewer; it is the same model the shared
contact law is validated on.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.mujoco_axial_loads import MujocoAxialLoadSource
from src.shared.python.force_overlay.bundle_provider import JointFrame

Vec3 = tuple[float, float, float]


class MujocoOverlaySource(NativeMujocoFullBodyModel):
    """Full-body MuJoCo adapter implementing ``ContactSource`` and ``KinematicsSource``."""

    def __init__(self, model_bytes: bytes) -> None:
        super().__init__(model_bytes)
        self.total_mass_kg: float = float(self.model.body_subtreemass[0])
        self._axial_source: MujocoAxialLoadSource | None = None

    def axial_load_frame(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        accelerations: Mapping[str, float],
        time_s: float,
    ) -> AxialLoadFrame | None:
        """Segment axial loads of a prescribed motion (inverse dynamics).

        Uses the ``kinematic`` axis policy (every body with a defined axis).
        Postcondition: tension positive, keyed by MuJoCo body name; ``None`` when
        the model has tendons, plugins or callbacks.
        """
        if self._axial_source is None:
            self._axial_source = MujocoAxialLoadSource(
                self.model, axis_policy="kinematic"
            )
        return self._axial_source.sample_prescribed(
            self._vector(coordinates),
            self._vector(rates),
            self._vector(accelerations),
            time_s,
        )

    def _forward(self, coordinates: Mapping[str, float]) -> None:
        self.data.qpos[:] = self._vector(coordinates)
        self._mj.mj_fwdPosition(self.model, self.data)

    def joint_frames(
        self, coordinates: Mapping[str, float]
    ) -> Mapping[str, JointFrame]:
        """World anchor and axis of every hinge coordinate at ``coordinates``."""
        self._forward(coordinates)
        mj: Any = self._mj
        out: dict[str, JointFrame] = {}
        for name in self.coordinate_order:
            j = self.model.joint(name).id
            if int(self.model.jnt_type[j]) != int(mj.mjtJoint.mjJNT_HINGE):
                continue
            body = mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_BODY, int(self.model.jnt_bodyid[j])
            )
            anchor = self.data.xanchor[j]
            axis = self.data.xaxis[j]
            out[name] = JointFrame(
                body or "body",
                (float(anchor[0]), float(anchor[1]), float(anchor[2])),
                (float(axis[0]), float(axis[1]), float(axis[2])),
            )
        return out

    def center_of_mass_m(self, coordinates: Mapping[str, float]) -> Vec3:
        """System centre of mass in the world frame."""
        self._forward(coordinates)
        com = np.asarray(self.data.subtree_com[0], dtype=float)
        return (float(com[0]), float(com[1]), float(com[2]))
