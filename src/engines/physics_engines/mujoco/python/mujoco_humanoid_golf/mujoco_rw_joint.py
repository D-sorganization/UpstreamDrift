"""Per-joint cfrc_int → Robertson–Winter endpoint power (#15390)."""

from __future__ import annotations

import numpy as np

from src.shared.python.biomechanics.robertson_winter_interface_power import (
    endpoint_interface_power_w,
)
from src.shared.python.motion_matching.force_torque import (
    SpatialWrench,
    transform_wrench,
)

__all__ = ["joint_interface_powers_w"]


def _vec3(seq: np.ndarray) -> tuple[float, float, float]:
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _spatial_twist_at_point(
    model,
    scratch,
    body_id: int,
    point: np.ndarray,
    qvel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, scratch, jacp, jacr, point, body_id)
    return jacp @ qvel, jacr @ qvel


def joint_interface_powers_w(
    model,
    scratch,
    joint_index: int,
    qvel: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """RW power on child and parent at one joint; returns child/parent/coord components."""

    child_id = int(model.jnt_bodyid[joint_index])
    parent_id = int(model.body_parentid[child_id])
    root_id = int(model.body_rootid[child_id])
    com = scratch.subtree_com[root_id]
    anchor = scratch.xanchor[joint_index]

    cfrc = scratch.cfrc_int[child_id]
    torque_com = np.asarray(cfrc[:3], dtype=np.float64)
    force_world = np.asarray(cfrc[3:], dtype=np.float64)
    wrench_com = SpatialWrench(
        "world", _vec3(com), _vec3(force_world), _vec3(torque_com)
    )
    wrench_anchor = transform_wrench(wrench_com, "world", _vec3(anchor))
    force_child = np.asarray(wrench_anchor.force_n, dtype=np.float64)
    moment_child = np.asarray(wrench_anchor.torque_nm, dtype=np.float64)

    v_child, w_child = _spatial_twist_at_point(model, scratch, child_id, anchor, qvel)
    v_parent, w_parent = _spatial_twist_at_point(
        model, scratch, parent_id, anchor, qvel
    )

    power_child = endpoint_interface_power_w(
        force_child, moment_child, v_child, w_child
    )
    # Parent sees equal-and-opposite wrench.
    force_parent = -force_child
    moment_parent = -moment_child
    power_parent = endpoint_interface_power_w(
        force_parent, moment_parent, v_parent, w_parent
    )

    dofadr = int(model.jnt_dofadr[joint_index])
    ndof = (
        int(model.jnt_dofadr[joint_index + 1] - dofadr)
        if joint_index + 1 < model.njnt
        else (int(model.nv - dofadr))
    )
    coord_power = 0.0
    if ndof == 1 and 0 <= dofadr < model.nv:
        coord_power = float(scratch.qfrc_inverse[dofadr] * qvel[dofadr])

    fp_child = float(np.dot(force_child, v_child))
    mp_child = float(np.dot(moment_child, w_child))
    return power_child, power_parent, fp_child, mp_child, coord_power
