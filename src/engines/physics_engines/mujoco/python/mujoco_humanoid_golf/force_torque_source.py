"""MuJoCo instantaneous force and torque overlay provider (ADR-0052, #11294).

Samples actuator torques, joint reaction wrenches, contact forces, applied
external forces, and segment axial loads from MuJoCo physics state onto a
renderer-neutral ForceTorqueFrame.
"""

from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np

from src.shared.python.body_part_viz.axial_loads import (
    AxialLoadFrame,
    axial_force_from_proximal_reaction,
)
from src.shared.python.body_part_viz.mujoco_axial_loads import MujocoAxialLoadSource
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.conversions import move_wrench_point

logger = logging.getLogger(__name__)


class MujocoForceTorqueSource:
    """Extract instantaneous force/torque overlays from MuJoCo state.

    Operates on internal scratch MjData to avoid mutating caller data.
    Runs mj_forward and mj_rnePostConstraint to ensure constraint reactions
    and contact forces are fully populated.
    """

    def __init__(self, model: Any) -> None:
        """Initialize the force/torque provider for a MuJoCo model."""
        if not (
            isinstance(model, mujoco.MjModel)
            or hasattr(model, "nbody")
            or "Mock" in type(model).__name__
        ):
            raise TypeError("model must be an instance of mujoco.MjModel")
        self.model = model
        self._scratch = mujoco.MjData(model)
        self._axial_source = MujocoAxialLoadSource(model)

    def sample(self, data: Any, *, include_gravity: bool = False) -> ForceTorqueFrame:
        """Sample instantaneous force and torque overlays from MuJoCo data."""
        if not isinstance(data, mujoco.MjData) or data.model is not self.model:
            raise TypeError("data must belong to this source's model")

        scratch = self._scratch
        model = self.model

        # Copy caller dynamic data to scratch so live state is unmutated
        scratch.time = data.time
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = data.qvel
        scratch.ctrl[:] = data.ctrl
        scratch.qacc_warmstart[:] = data.qacc_warmstart
        scratch.qfrc_applied[:] = data.qfrc_applied
        scratch.xfrc_applied[:] = data.xfrc_applied
        if getattr(model, "na", 0):
            scratch.act[:] = data.act
        if getattr(model, "nmocap", 0):
            scratch.mocap_pos[:] = data.mocap_pos
            scratch.mocap_quat[:] = data.mocap_quat
        if getattr(model, "nuserdata", 0):
            scratch.userdata[:] = data.userdata

        mujoco.mj_forward(model, scratch)
        mujoco.mj_rnePostConstraint(model, scratch)

        wrenches: list[OverlayWrench] = []
        self._extract_actuators(scratch, wrenches)
        self._extract_joint_reactions(scratch, wrenches)
        self._extract_contacts(scratch, wrenches)
        self._extract_applied(scratch, wrenches)
        if include_gravity:
            self._extract_gravity(scratch, wrenches)

        axial_loads = self._axial_source.sample(scratch)
        if axial_loads is None:
            axial_loads = self._compute_fallback_axial_loads(scratch, float(data.time))

        return ForceTorqueFrame(
            time_s=float(data.time),
            engine="mujoco",
            wrenches=tuple(wrenches),
            axial_loads=axial_loads,
            world_frame="world_Zup",
        )

    def _body_name(self, body_id: int) -> str:
        return (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or f"body_{body_id}"
        )

    def _joint_name(self, jnt_id: int) -> str:
        return (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            or f"joint_{jnt_id}"
        )

    def _extract_actuators(
        self, scratch: mujoco.MjData, out: list[OverlayWrench]
    ) -> None:
        model = self.model
        for j in range(model.njnt):
            jnt_type = model.jnt_type[j]
            body_id = model.jnt_bodyid[j]
            dofadr = model.jnt_dofadr[j]
            j_name, b_name = self._joint_name(j), self._body_name(body_id)
            anchor = tuple(float(x) for x in scratch.xanchor[j])

            if jnt_type in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                val = tuple(
                    float(m) for m in (scratch.xaxis[j] * scratch.qfrc_actuator[dofadr])
                )
                is_hinge = jnt_type == mujoco.mjtJoint.mjJNT_HINGE
                out.append(
                    OverlayWrench(
                        kind=WrenchKind.JOINT_ACTUATOR,
                        label=f"actuator:{j_name}",
                        body=b_name,
                        point_m=anchor,
                        force_n=None if is_hinge else val,
                        torque_nm=val if is_hinge else None,
                        source="mujoco:qfrc_actuator",
                    )
                )
            elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                r_mat = scratch.xmat[body_id].reshape(3, 3)
                tau = tuple(
                    float(m)
                    for m in (r_mat @ scratch.qfrc_actuator[dofadr : dofadr + 3])
                )
                out.append(
                    OverlayWrench(
                        kind=WrenchKind.JOINT_ACTUATOR,
                        label=f"actuator:{j_name}",
                        body=b_name,
                        point_m=anchor,
                        torque_nm=tau,
                        source="mujoco:qfrc_actuator",
                    )
                )
            elif jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                qfrc = scratch.qfrc_actuator[dofadr : dofadr + 6]
                if np.any(np.abs(qfrc) > 1e-12):
                    r_mat = scratch.xmat[body_id].reshape(3, 3)
                    f_w = tuple(float(x) for x in (r_mat @ qfrc[:3]))
                    t_w = tuple(float(x) for x in (r_mat @ qfrc[3:]))
                    pos = tuple(float(x) for x in scratch.xpos[body_id])
                    out.append(
                        OverlayWrench(
                            kind=WrenchKind.EXTERNAL,
                            label=f"external:{j_name}",
                            body=b_name,
                            point_m=pos,
                            force_n=f_w,
                            torque_nm=t_w,
                            source="mujoco:qfrc_actuator",
                        )
                    )

    def _extract_joint_reactions(
        self, scratch: mujoco.MjData, out: list[OverlayWrench]
    ) -> None:
        model = self.model
        for b in range(1, model.nbody):
            b_name = self._body_name(b)
            start, count = model.body_jntadr[b], model.body_jntnum[b]
            joints = list(range(start, start + count))
            if any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in joints):
                continue

            cfrc = scratch.cfrc_int[b]
            t_com = (float(cfrc[0]), float(cfrc[1]), float(cfrc[2]))
            f_w = (float(cfrc[3]), float(cfrc[4]), float(cfrc[5]))
            p_com = tuple(float(x) for x in scratch.subtree_com[b])

            if count > 0:
                j_name = self._joint_name(joints[0])
                anchor = tuple(float(x) for x in scratch.xanchor[joints[0]])
            else:
                j_name, anchor = b_name, tuple(float(x) for x in scratch.xpos[b])

            w_com = OverlayWrench(
                kind=WrenchKind.JOINT_REACTION,
                label=f"joint_reaction:{j_name}",
                body=b_name,
                point_m=p_com,
                force_n=f_w,
                torque_nm=t_com,
                source="mujoco:cfrc_int",
            )
            out.append(move_wrench_point(w_com, anchor))

    def _extract_contacts(
        self, scratch: mujoco.MjData, out: list[OverlayWrench]
    ) -> None:
        model = self.model
        for i in range(scratch.ncon):
            con = scratch.contact[i]
            if con.geom1 < 0 or con.geom2 < 0:
                continue

            f_con = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, scratch, i, f_con)

            c_frame = con.frame.reshape(3, 3)
            f_w = c_frame.T @ f_con[:3]
            t_w = (c_frame.T @ f_con[3:]) if con.dim >= 4 else None
            point = tuple(float(x) for x in con.pos)

            g1_body, g2_body = (
                model.geom_bodyid[con.geom1],
                model.geom_bodyid[con.geom2],
            )
            for body_id, sign in ((g2_body, 1.0), (g1_body, -1.0)):
                if body_id == 0:
                    continue
                b_name = self._body_name(body_id)
                out.append(
                    OverlayWrench(
                        kind=WrenchKind.CONTACT,
                        label=f"contact:{b_name}:{len(out)}",
                        body=b_name,
                        point_m=point,
                        force_n=tuple(float(sign * x) for x in f_w),
                        torque_nm=tuple(float(sign * x) for x in t_w)
                        if t_w is not None
                        else None,
                        source="mujoco:contact",
                    )
                )

    def _extract_applied(
        self, scratch: mujoco.MjData, out: list[OverlayWrench]
    ) -> None:
        for b in range(1, self.model.nbody):
            xfrc = scratch.xfrc_applied[b]
            if np.any(np.abs(xfrc) > 1e-12):
                b_name = self._body_name(b)
                f = tuple(float(x) for x in xfrc[:3])
                t = tuple(float(x) for x in xfrc[3:])
                out.append(
                    OverlayWrench(
                        kind=WrenchKind.EXTERNAL,
                        label=f"external:{b_name}:{len(out)}",
                        body=b_name,
                        point_m=tuple(float(x) for x in scratch.xipos[b]),
                        force_n=f if np.any(np.abs(f) > 1e-12) else None,
                        torque_nm=t if np.any(np.abs(t) > 1e-12) else None,
                        source="mujoco:xfrc_applied",
                    )
                )

    def _extract_gravity(
        self, scratch: mujoco.MjData, out: list[OverlayWrench]
    ) -> None:
        model = self.model
        g = np.asarray(model.opt.gravity, dtype=np.float64)
        for b in range(1, model.nbody):
            mass = float(model.body_mass[b])
            if mass <= 0.0:
                continue
            out.append(
                OverlayWrench(
                    kind=WrenchKind.GRAVITY,
                    label=f"gravity:{self._body_name(b)}",
                    body=self._body_name(b),
                    point_m=tuple(float(x) for x in scratch.xipos[b]),
                    force_n=tuple(float(x) for x in (mass * g)),
                    source="mujoco:gravity",
                )
            )

    def _compute_fallback_axial_loads(
        self, scratch: mujoco.MjData, time_s: float
    ) -> AxialLoadFrame | None:
        model = self.model
        children: dict[int, list[int]] = {}
        for b in range(1, model.nbody):
            parent = model.body_parentid[b]
            if model.body_jntnum[b] > 0:
                children.setdefault(parent, []).append(model.body_jntadr[b])

        values: dict[str, float | None] = {}
        for b in range(1, model.nbody):
            if model.body_jntnum[b] != 1:
                continue
            j = model.body_jntadr[b]
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue

            proximal = scratch.xanchor[j]
            child_jnts = children.get(b, [])
            if len(child_jnts) == 1:
                distal = scratch.xanchor[child_jnts[0]]
            elif len(child_jnts) == 0:
                distal = scratch.xipos[b]
            else:
                continue

            if float(np.linalg.norm(distal - proximal)) < 1e-4:
                continue

            force = scratch.cfrc_int[b, 3:]
            val = axial_force_from_proximal_reaction(force, proximal, distal)
            values[self._body_name(b)] = float(val)

        if not values:
            return None

        return AxialLoadFrame(
            time_s=time_s,
            values_n=values,
            source="MuJoCo cfrc_int parent-on-body reaction; segment axis",
        )
