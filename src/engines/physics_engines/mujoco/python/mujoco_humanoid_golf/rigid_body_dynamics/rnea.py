"""
Recursive Newton-Euler Algorithm (RNEA) for inverse dynamics.

Computes the joint forces/torques required to produce a given motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mujoco_humanoid_golf.rigid_body_dynamics.common import (
    DEFAULT_GRAVITY,
    NEG_DEFAULT_GRAVITY,
)
from mujoco_humanoid_golf.spatial_algebra import (
    cross_force_fast,
    cross_motion_axis,
    cross_motion_fast,
    jcalc,
)


@dataclass
class _RneaModelCache:
    parent: np.ndarray
    jtype: list
    xtree: list
    inertia: list

    @staticmethod
    def from_model(model: dict) -> _RneaModelCache:
        """Build a cache of model fields used during RNEA passes."""
        return _RneaModelCache(
            parent=model["parent"],
            jtype=model["jtype"],
            xtree=model["Xtree"],
            inertia=model["I"],
        )


@dataclass
class _RneaScratchBuffers:
    xj_buf: np.ndarray
    scratch_vec: np.ndarray
    i_v_buf: np.ndarray
    cross_buf: np.ndarray

    @staticmethod
    def create() -> _RneaScratchBuffers:
        """Allocate pre-sized scratch buffers for RNEA computation."""
        return _RneaScratchBuffers(
            xj_buf=np.empty((6, 6)),
            scratch_vec=np.empty(6),
            i_v_buf=np.empty(6),
            cross_buf=np.empty(6),
        )


def _rnea_validate_inputs(
    model: dict,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    q = np.asarray(q).ravel()
    qd = np.asarray(qd).ravel()
    qdd = np.asarray(qdd).ravel()

    nb = model["NB"]

    if len(q) != nb:
        msg = f"q must have length {nb}, got {len(q)}"
        raise ValueError(msg)
    if len(qd) != nb:
        msg = f"qd must have length {nb}, got {len(qd)}"
        raise ValueError(msg)
    if len(qdd) != nb:
        msg = f"qdd must have length {nb}, got {len(qdd)}"
        raise ValueError(msg)

    return q, qd, qdd, nb


def _rnea_forward_pass_body(
    i,
    q,
    qd,
    qdd,
    neg_a_grav,
    mdl: _RneaModelCache,
    xup,
    v,
    a,
    s_subspace_list,
    dof_indices,
    buf: _RneaScratchBuffers,
):
    xj_transform, s_subspace, dof_idx = jcalc(mdl.jtype[i], q[i], out=buf.xj_buf)
    s_subspace_list[i] = s_subspace
    dof_indices[i] = dof_idx

    if dof_idx != -1:
        buf.i_v_buf.fill(0)
        buf.i_v_buf[dof_idx] = qd[i]
    else:
        np.multiply(s_subspace, qd[i], out=buf.i_v_buf)
    vj_velocity = buf.i_v_buf

    if mdl.parent[i] == -1:
        v[:, i] = vj_velocity
        np.matmul(xj_transform, neg_a_grav, out=a[:, i])
        if dof_idx != -1:
            a[dof_idx, i] += qdd[i]
        else:
            np.multiply(s_subspace, qdd[i], out=buf.cross_buf)
            a[:, i] += buf.cross_buf
    else:
        p = mdl.parent[i]
        np.matmul(xj_transform, mdl.xtree[i], out=xup[i])
        np.matmul(xup[i], v[:, p], out=v[:, i])
        v[:, i] += vj_velocity
        np.matmul(xup[i], a[:, p], out=a[:, i])
        if dof_idx != -1:
            a[dof_idx, i] += qdd[i]
        else:
            np.multiply(s_subspace, qdd[i], out=buf.cross_buf)
            a[:, i] += buf.cross_buf
        if dof_idx != -1:
            cross_motion_axis(v[:, i], dof_idx, qd[i], out=buf.cross_buf)
        else:
            cross_motion_fast(v[:, i], vj_velocity, out=buf.cross_buf)
        a[:, i] += buf.cross_buf


def _rnea_backward_pass(
    nb,
    mdl: _RneaModelCache,
    v,
    a,
    f,
    tau,
    xup,
    s_subspace_list,
    dof_indices,
    f_ext,
    buf: _RneaScratchBuffers,
):
    for i in range(nb - 1, -1, -1):
        np.matmul(mdl.inertia[i], a[:, i], out=buf.scratch_vec)
        f_body = buf.scratch_vec

        np.matmul(mdl.inertia[i], v[:, i], out=buf.i_v_buf)

        cross_force_fast(v[:, i], buf.i_v_buf, out=buf.cross_buf)
        f_body += buf.cross_buf
        if f_ext is not None:
            f_body -= f_ext[:, i]

        f[:, i] += f_body

        s_subspace = s_subspace_list[i]
        dof_idx = dof_indices[i]

        if dof_idx != -1:
            tau[i] = f[dof_idx, i]
        else:
            tau[i] = s_subspace @ f[:, i]

        if mdl.parent[i] != -1:
            p = mdl.parent[i]
            np.matmul(xup[i].T, f[:, i], out=buf.scratch_vec)
            f[:, p] += buf.scratch_vec


def rnea(
    model: dict,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    f_ext: np.ndarray | None = None,
) -> np.ndarray:
    """
    Recursive Newton-Euler Algorithm for inverse dynamics.

    Computes the inverse dynamics of a kinematic tree. Given joint positions q,
    velocities qd, and accelerations qdd, this algorithm computes the joint
    forces/torques tau required to produce that motion.

    Args:
        model: Robot model dictionary with fields:
            NB: Number of bodies (int)
            parent: Parent body indices (array of length NB)
            jtype: Joint types (list of strings, length NB)
            Xtree: Joint transforms (NB-length list of 6x6 arrays)
            I: Spatial inertias (NB-length list of 6x6 arrays)
            gravity: 6x1 spatial gravity vector (optional)
        q: Joint positions (NB,)
        qd: Joint velocities (NB,)
        qdd: Joint accelerations (NB,)
        f_ext: External forces (6, NB) (optional)

    Returns:
        Joint forces/torques (NB,)

    Algorithm:
        Forward pass: compute velocities and accelerations
        Backward pass: compute forces and project to joint torques

    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
        Chapter 5: Independent Joint Equations of Motion, Algorithm 5.1

    Example:
        >>> model = create_robot_model()
        >>> q = np.array([0.5, -0.3])
        >>> qd = np.array([0.1, 0.2])
        >>> qdd = np.array([0.5, -0.2])
        >>> tau = rnea(model, q, qd, qdd)
    """
    q, qd, qdd, nb = _rnea_validate_inputs(model, q, qd, qdd)

    a_grav = model.get("gravity", DEFAULT_GRAVITY)
    neg_a_grav = NEG_DEFAULT_GRAVITY if a_grav is DEFAULT_GRAVITY else -a_grav

    v = np.empty((6, nb), order="F")
    a = np.empty((6, nb), order="F")
    f = np.zeros((6, nb), order="F")
    tau = np.empty(nb)
    xup = np.empty((nb, 6, 6))

    s_subspace_list: list[np.ndarray] = [None] * nb  # type: ignore[assignment, list-item]
    dof_indices: list[int] = [-1] * nb

    mdl = _RneaModelCache.from_model(model)
    buf = _RneaScratchBuffers.create()

    for i in range(nb):
        _rnea_forward_pass_body(
            i,
            q,
            qd,
            qdd,
            neg_a_grav,
            mdl,
            xup,
            v,
            a,
            s_subspace_list,
            dof_indices,
            buf,
        )

    _rnea_backward_pass(
        nb,
        mdl,
        v,
        a,
        f,
        tau,
        xup,
        s_subspace_list,
        dof_indices,
        f_ext,
        buf,
    )

    return tau
