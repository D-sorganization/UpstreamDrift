"""MuJoCo force and torque overlay provider source (ADR-0052, FTO-9, #11294).

Emits world-frame OverlayWrenches for:
- JOINT_ACTUATOR: applied actuator torques/forces from qfrc_actuator
- JOINT_REACTION: parent-on-child internal reactions from cfrc_int
- CONTACT: active contact pair forces from mj_contactForce, plus the per-foot and
  net ground-reaction breakdown (GCV-2, #11708)
- EXTERNAL: applied external wrenches from xfrc_applied
- GRIP: per-hand wrench on the club from the grip weld efc_force (GCV-8, #11714)
- GRAVITY: optional gravitational body forces mass * g
Synchronizes with MujocoAxialLoadSource for rod tension/compression.
All calculations operate on an internal scratch MjData, preserving live simulation state.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from src.engines.physics_engines.mujoco.python.grip_efc import (
    DEFAULT_GRIP_WELDS,
    grip_analysis_from_efc,
)
from src.engines.physics_engines.mujoco.python.ground_contacts import (
    contact_wrenches,
    ground_reaction_wrenches,
)
from src.shared.python.biomechanics.grip_wrench import to_overlay_wrenches
from src.shared.python.body_part_viz.mujoco_axial_loads import MujocoAxialLoadSource
from src.shared.python.engine_core.mujoco_compat import copy_mjdata_state
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.conversions import joint_torque_wrench
from src.shared.python.motion_matching.force_torque import (
    SpatialWrench,
    transform_wrench,
)

__all__ = ["MujocoForceTorqueSource"]


def _vec3(seq: Any) -> tuple[float, float, float]:
    return (float(seq[0]), float(seq[1]), float(seq[2]))


class MujocoForceTorqueSource:
    """Extract instantaneous world-frame force/torque overlays from MuJoCo models."""

    ENGINE = "mujoco"

    def __init__(self, model: mujoco.MjModel) -> None:
        if not isinstance(model, mujoco.MjModel):
            raise TypeError("model must be an instance of mujoco.MjModel")
        self._model = model
        self._scratch = mujoco.MjData(model)
        self._axial_source = MujocoAxialLoadSource(model)

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    def _extract_actuators(self, scratch: mujoco.MjData) -> list[OverlayWrench]:
        model = self._model
        actuators: list[OverlayWrench] = []
        for j in range(model.njnt):
            jnt_type = model.jnt_type[j]
            dofadr = int(model.jnt_dofadr[j])
            body_id = int(model.jnt_bodyid[j])
            b_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                or f"body_{body_id}"
            )
            j_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
            )
            anchor = _vec3(scratch.xanchor[j])

            if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                axis = scratch.xaxis[j]
                norm = float(np.linalg.norm(axis))
                if norm > 1e-12:
                    tau = float(scratch.qfrc_actuator[dofadr])
                    actuators.append(
                        joint_torque_wrench(
                            f"actuator:{j_name}",
                            b_name,
                            tau,
                            axis / norm,
                            anchor,
                            self.ENGINE,
                        )
                    )
            elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
                axis = scratch.xaxis[j]
                norm = float(np.linalg.norm(axis))
                force_val = float(scratch.qfrc_actuator[dofadr])
                if norm > 1e-12 and abs(force_val) > 1e-9:
                    f_vec = _vec3(force_val * (axis / norm))
                    actuators.append(
                        OverlayWrench(
                            kind=WrenchKind.JOINT_ACTUATOR,
                            label=f"actuator:{j_name}",
                            body=b_name,
                            point_m=anchor,
                            force_n=f_vec,
                            torque_nm=None,
                            source=self.ENGINE,
                        )
                    )
            elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                tau_local = scratch.qfrc_actuator[dofadr : dofadr + 3]
                if float(np.linalg.norm(tau_local)) > 1e-9:
                    rot = scratch.xmat[body_id].reshape(3, 3)
                    t_world = _vec3(rot @ tau_local)
                    actuators.append(
                        OverlayWrench(
                            kind=WrenchKind.JOINT_ACTUATOR,
                            label=f"actuator:{j_name}",
                            body=b_name,
                            point_m=anchor,
                            force_n=None,
                            torque_nm=t_world,
                            source=self.ENGINE,
                        )
                    )
            elif jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                qfrc = scratch.qfrc_actuator[dofadr : dofadr + 6]
                if float(np.linalg.norm(qfrc)) > 1e-9:
                    pos = _vec3(scratch.xpos[body_id])
                    f_free = _vec3(qfrc[:3])
                    t_free = _vec3(qfrc[3:6])
                    actuators.append(
                        OverlayWrench(
                            kind=WrenchKind.EXTERNAL,
                            label=f"actuator:{j_name}",
                            body=b_name,
                            point_m=pos,
                            force_n=f_free,
                            torque_nm=t_free,
                            source=self.ENGINE,
                        )
                    )
        return actuators

    def _extract_reactions(self, scratch: mujoco.MjData) -> list[OverlayWrench]:
        model = self._model
        reactions: list[OverlayWrench] = []
        for b in range(1, model.nbody):
            b_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}"
            )
            root_id = int(model.body_rootid[b])
            com = scratch.subtree_com[root_id]

            j_start = int(model.body_jntadr[b])
            j_num = int(model.body_jntnum[b])
            anchor = scratch.xanchor[j_start] if j_num > 0 else scratch.xpos[b]
            # Label by joint (the cross-engine convention); a joint-less body
            # keeps its body name.
            j_label = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_start)
                if j_num > 0
                else None
            ) or b_name

            cfrc = scratch.cfrc_int[b]
            f_world = _vec3(cfrc[3:])
            t_com = _vec3(cfrc[:3])

            w_com = SpatialWrench("world", _vec3(com), f_world, t_com)
            anchor_pt = _vec3(anchor)
            w_anchor = transform_wrench(w_com, "world", anchor_pt)

            reactions.append(
                OverlayWrench(
                    kind=WrenchKind.JOINT_REACTION,
                    label=f"reaction:{j_label}",
                    body=b_name,
                    point_m=anchor_pt,
                    force_n=w_anchor.force_n,
                    torque_nm=w_anchor.torque_nm,
                    source=self.ENGINE,
                )
            )
        return reactions

    def _extract_contacts(self, scratch: mujoco.MjData) -> list[OverlayWrench]:
        return contact_wrenches(self._model, scratch, engine=self.ENGINE)

    def _extract_ground_reaction(self, scratch: mujoco.MjData) -> list[OverlayWrench]:
        """Per-foot and net GRF, CoP, free moment and moment about the CoM (GCV-2)."""
        return list(
            ground_reaction_wrenches(self._model, scratch, engine=self.ENGINE)
        )

    def _extract_externals(self, scratch: mujoco.MjData) -> list[OverlayWrench]:
        model = self._model
        externals: list[OverlayWrench] = []
        for b in range(1, model.nbody):
            xfrc = scratch.xfrc_applied[b]
            if float(np.linalg.norm(xfrc)) > 1e-9:
                b_name = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}"
                )
                pt = _vec3(scratch.xipos[b])
                f_val = (
                    _vec3(xfrc[:3]) if float(np.linalg.norm(xfrc[:3])) > 1e-9 else None
                )
                t_val = (
                    _vec3(xfrc[3:]) if float(np.linalg.norm(xfrc[3:])) > 1e-9 else None
                )
                if f_val is not None or t_val is not None:
                    externals.append(
                        OverlayWrench(
                            kind=WrenchKind.EXTERNAL,
                            label=f"external:{b_name}",
                            body=b_name,
                            point_m=pt,
                            force_n=f_val,
                            torque_nm=t_val,
                            source=self.ENGINE,
                        )
                    )
        return externals

    def _extract_grip(self, scratch: mujoco.MjData) -> list[OverlayWrench]:
        """Per-hand ``GRIP`` wrenches ON THE CLUB from the grip weld ``efc_force``.

        Models without the conventional ``grip_weld_l`` / ``grip_weld_r``
        equalities (the MyoSuite golfer scene has them) yield nothing.
        """
        model = self._model
        present = {
            side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name) >= 0
            for side, name in DEFAULT_GRIP_WELDS.items()
        }
        if not all(present.values()):
            return []
        analysis = grip_analysis_from_efc(model, scratch)
        return to_overlay_wrenches(analysis, source=self.ENGINE + ":efc_force")

    def _extract_gravity(self, scratch: mujoco.MjData) -> list[OverlayWrench]:
        model = self._model
        grav = model.opt.gravity
        if float(np.linalg.norm(grav)) <= 1e-9:
            return []
        gravity_wrenches: list[OverlayWrench] = []
        for b in range(1, model.nbody):
            mass = float(model.body_mass[b])
            if mass > 1e-9:
                b_name = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}"
                )
                f_grav = _vec3(mass * grav)
                gravity_wrenches.append(
                    OverlayWrench(
                        kind=WrenchKind.GRAVITY,
                        label=f"gravity:{b_name}",
                        body=b_name,
                        point_m=_vec3(scratch.xipos[b]),
                        force_n=f_grav,
                        torque_nm=None,
                        source=self.ENGINE,
                    )
                )
        return gravity_wrenches

    def sample(
        self, data: mujoco.MjData, *, include_gravity: bool = False
    ) -> ForceTorqueFrame:
        """Sample all world-frame forces and torques at the current MjData state.

        Copies data into an internal scratch buffer and calls mj_rnePostConstraint,
        ensuring the caller's MjData remains bit-identical.
        """
        if not isinstance(data, mujoco.MjData) or data.model is not self._model:
            raise TypeError("data must belong to this source's model")

        scratch = self._scratch
        copy_mjdata_state(scratch, data)
        mujoco.mj_forward(self._model, scratch)
        mujoco.mj_rnePostConstraint(self._model, scratch)

        wrenches: list[OverlayWrench] = []
        wrenches.extend(self._extract_actuators(scratch))
        wrenches.extend(self._extract_reactions(scratch))
        wrenches.extend(self._extract_contacts(scratch))
        wrenches.extend(self._extract_ground_reaction(scratch))
        wrenches.extend(self._extract_externals(scratch))
        wrenches.extend(self._extract_grip(scratch))
        if include_gravity:
            wrenches.extend(self._extract_gravity(scratch))

        axial_loads = self._axial_source.sample(data)

        return ForceTorqueFrame(
            time_s=float(data.time),
            engine=self.ENGINE,
            wrenches=tuple(wrenches),
            axial_loads=axial_loads,
            world_frame="world_Zup",
        )
