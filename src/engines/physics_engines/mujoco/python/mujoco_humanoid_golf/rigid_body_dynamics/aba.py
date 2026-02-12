"""
Articulated Body Algorithm (ABA) for forward dynamics.

Computes joint accelerations given applied torques.
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
    cross_motion_fast,
    jcalc,
)

TOLERANCE = 1e-10  # Numerical tolerance to avoid division by zero


@dataclass
class _ModelCache:
    """Cached model dictionary lookups for tight-loop performance."""

    parent: np.ndarray
    jtype: list
    xtree: list
    inertia: list

    @staticmethod
    def from_model(model: dict) -> _ModelCache:
        """Extract and cache model fields to avoid repeated dict lookups."""
        return _ModelCache(
            parent=model["parent"],
            jtype=model["jtype"],
            xtree=model["Xtree"],
            inertia=model["I"],
        )


@dataclass
class _ScratchBuffers:
    """Pre-allocated scratch buffers to avoid allocations in tight loops."""

    xj_buf: np.ndarray  # (6, 6) - temporary joint transform / outer product
    cross_buf: np.ndarray  # (6,)   - cross product result
    scratch_vec: np.ndarray  # (6,)   - general scratch vector
    scratch_mat: np.ndarray  # (6, 6) - general scratch matrix
    i_v_buf: np.ndarray  # (6,)   - inertia * velocity product
    temp_vec: np.ndarray  # (6,)   - temporary vector

    @staticmethod
    def create() -> _ScratchBuffers:
        """Create a new set of scratch buffers."""
        return _ScratchBuffers(
            xj_buf=np.empty((6, 6)),
            cross_buf=np.empty(6),
            scratch_vec=np.empty(6),
            scratch_mat=np.empty((6, 6)),
            i_v_buf=np.empty(6),
            temp_vec=np.empty(6),
        )


def _aba_validate_inputs(
    model: dict,
    q: np.ndarray,
    qd: np.ndarray,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Validate and prepare ABA inputs.

    Returns:
        Tuple of (q, qd, tau, nb) with raveled arrays and body count.
    """
    q = np.asarray(q).ravel()
    qd = np.asarray(qd).ravel()
    tau = np.asarray(tau).ravel()

    nb = model["NB"]

    if len(q) != nb:
        msg = f"q must have length {nb}, got {len(q)}"
        raise ValueError(msg)
    if len(qd) != nb:
        msg = f"qd must have length {nb}, got {len(qd)}"
        raise ValueError(msg)
    if len(tau) != nb:
        msg = f"tau must have length {nb}, got {len(tau)}"
        raise ValueError(msg)

    return q, qd, tau, nb


def _aba_forward_kinematics(
    nb: int,
    q: np.ndarray,
    qd: np.ndarray,
    f_ext: np.ndarray | None,
    mdl: _ModelCache,
    xup: np.ndarray,
    s_subspace: list[np.ndarray],
    dof_indices: list[int],
    v: np.ndarray,
    c: np.ndarray,
    pa_bias: np.ndarray,
    buf: _ScratchBuffers,
) -> None:
    """Pass 1: Forward kinematics - compute velocities and bias accelerations."""
    for i in range(nb):
        # OPTIMIZATION: Use pre-allocated buffer for jcalc
        xj_transform, s_vec, dof_idx = jcalc(mdl.jtype[i], q[i], out=buf.xj_buf)
        s_subspace[i] = s_vec
        dof_indices[i] = dof_idx

        # OPTIMIZATION: Write directly to pre-allocated xup buffer
        np.matmul(xj_transform, mdl.xtree[i], out=xup[i])

        if dof_idx != -1:
            buf.temp_vec.fill(0)
            buf.temp_vec[dof_idx] = qd[i]
            vj_velocity = buf.temp_vec
        else:
            vj_velocity = s_subspace[i] * qd[i]

        if mdl.parent[i] == -1:
            v[:, i] = vj_velocity
            c[:, i] = np.zeros(6)  # No bias for base-connected bodies
        else:
            p = mdl.parent[i]
            # OPTIMIZATION: Use matmul with out directly to v[:, i]
            np.matmul(xup[i], v[:, p], out=v[:, i])
            v[:, i] += vj_velocity

            # OPTIMIZATION: Use pre-allocated buffer for cross product
            cross_motion_fast(v[:, i], vj_velocity, out=c[:, i])

        # Bias force: Coriolis + external forces
        np.matmul(mdl.inertia[i], v[:, i], out=buf.i_v_buf)
        cross_force_fast(v[:, i], buf.i_v_buf, out=buf.cross_buf)
        if f_ext is not None:
            np.subtract(buf.cross_buf, f_ext[:, i], out=pa_bias[:, i])
        else:
            pa_bias[:, i] = buf.cross_buf


def _aba_backward_body_inertia(
    i: int,
    dof_idx: int,
    model_parent_i: int,
    s_subspace_i: np.ndarray,
    tau_i: float,
    xup_i: np.ndarray,
    ia_articulated: np.ndarray,
    pa_bias: np.ndarray,
    u_force: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    c: np.ndarray,
    buf: _ScratchBuffers,
) -> None:
    """Compute articulated-body inertia update for a single body."""
    if dof_idx != -1:
        u_force[:, i] = ia_articulated[i][:, dof_idx]
    else:
        np.matmul(ia_articulated[i], s_subspace_i, out=u_force[:, i])

    # Joint-space inertia
    if dof_idx != -1:
        d[i] = u_force[dof_idx, i]
    else:
        d[i] = np.dot(s_subspace_i, u_force[:, i])

    # Bias torque
    if dof_idx != -1:
        u[i] = tau_i - pa_bias[dof_idx, i]
    else:
        u[i] = tau_i - np.dot(s_subspace_i, pa_bias[:, i])

    # Articulated-body inertia update for parent
    if model_parent_i != -1:
        _aba_propagate_to_parent(
            i,
            model_parent_i,
            xup_i,
            ia_articulated,
            pa_bias,
            u_force,
            d,
            u,
            c,
            buf,
        )


def _aba_propagate_to_parent(
    i: int,
    parent: int,
    xup_i: np.ndarray,
    ia_articulated: np.ndarray,
    pa_bias: np.ndarray,
    u_force: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    c: np.ndarray,
    buf: _ScratchBuffers,
) -> None:
    """Propagate articulated-body quantities from body i to its parent."""
    if abs(d[i]) < TOLERANCE:
        d[i] = np.sign(d[i]) * TOLERANCE if d[i] != 0 else TOLERANCE

    dinv = 1 / d[i]

    # OPTIMIZATION: Minimize allocations in the update
    np.matmul(ia_articulated[i], xup_i, out=buf.scratch_mat)
    np.matmul(xup_i.T, u_force[:, i], out=buf.scratch_vec)

    # Subtract rank-1 update from scratch_mat
    np.multiply(u_force[:, i], dinv, out=buf.temp_vec)
    np.outer(buf.temp_vec, buf.scratch_vec, out=buf.xj_buf)
    buf.scratch_mat -= buf.xj_buf

    # Add to parent articulated inertia
    np.matmul(xup_i.T, buf.scratch_mat, out=buf.xj_buf)
    ia_articulated[parent] += buf.xj_buf

    # Update bias force
    np.matmul(ia_articulated[i], c[:, i], out=buf.scratch_vec)
    buf.scratch_vec += pa_bias[:, i]
    scalar = dinv * u[i]
    np.multiply(u_force[:, i], scalar, out=buf.temp_vec)
    buf.scratch_vec += buf.temp_vec

    # Transform and add to parent
    np.matmul(xup_i.T, buf.scratch_vec, out=buf.cross_buf)
    pa_bias[:, parent] += buf.cross_buf


def _aba_backward_pass(
    nb: int,
    tau: np.ndarray,
    model_parent: np.ndarray,
    s_subspace: list[np.ndarray],
    dof_indices: list[int],
    xup: np.ndarray,
    ia_articulated: np.ndarray,
    pa_bias: np.ndarray,
    u_force: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    c: np.ndarray,
    buf: _ScratchBuffers,
) -> None:
    """Pass 2: Backward recursion - compute articulated-body inertias."""
    for i in range(nb - 1, -1, -1):
        _aba_backward_body_inertia(
            i,
            dof_indices[i],
            model_parent[i],
            s_subspace[i],
            tau[i],
            xup[i],
            ia_articulated,
            pa_bias,
            u_force,
            d,
            u,
            c,
            buf,
        )


def _aba_forward_accelerations(
    nb: int,
    model_parent: np.ndarray,
    s_subspace: list[np.ndarray],
    dof_indices: list[int],
    xup: np.ndarray,
    neg_a_grav: np.ndarray,
    c: np.ndarray,
    u_force: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    a: np.ndarray,
    qdd: np.ndarray,
) -> None:
    """Pass 3: Forward recursion - compute joint and spatial accelerations."""
    for i in range(nb):
        if model_parent[i] == -1:
            np.matmul(xup[i], neg_a_grav, out=a[:, i])
            a[:, i] += c[:, i]
        else:
            p = model_parent[i]
            np.matmul(xup[i], a[:, p], out=a[:, i])
            a[:, i] += c[:, i]

        qdd[i] = (u[i] - np.dot(u_force[:, i], a[:, i])) / d[i]

        dof_idx = dof_indices[i]
        if dof_idx != -1:
            a[dof_idx, i] += qdd[i]
        else:
            a[:, i] += s_subspace[i] * qdd[i]


def aba(
    model: dict,
    q: np.ndarray,
    qd: np.ndarray,
    tau: np.ndarray,
    f_ext: np.ndarray | None = None,
) -> np.ndarray:
    """
    Articulated Body Algorithm for forward dynamics.

    Computes the forward dynamics of a kinematic tree. Given joint positions q,
    velocities qd, and applied torques tau, this algorithm computes the
    resulting joint accelerations qdd.

    This is an O(n) algorithm that is much more efficient than inverting
    the mass matrix.

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
        tau: Joint forces/torques (NB,)
        f_ext: External forces (6, NB) (optional)

    Returns:
        Joint accelerations (NB,)

    Algorithm:
        Pass 1: Kinematics - compute velocities and bias accelerations
        Pass 2: Articulated bodies - compute articulated-body inertias
        Pass 3: Accelerations - compute joint and spatial accelerations

    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
        Chapter 7: Articulated-Body Algorithm, Algorithm 7.1

    Example:
        >>> model = create_robot_model()
        >>> q = np.array([0.5, -0.3])
        >>> qd = np.array([0.1, 0.2])
        >>> tau = np.array([1.5, 0.5])
        >>> qdd = aba(model, q, qd, tau)
    """
    q, qd, tau, nb = _aba_validate_inputs(model, q, qd, tau)

    # Get gravity vector
    a_grav = model.get("gravity", DEFAULT_GRAVITY)
    # OPTIMIZATION: Pre-compute negative gravity to avoid allocation in loop
    if a_grav is DEFAULT_GRAVITY:
        neg_a_grav = NEG_DEFAULT_GRAVITY
    else:
        neg_a_grav = -a_grav

    # Initialize arrays
    xup = np.empty((nb, 6, 6))
    s_subspace: list[np.ndarray] = [None] * nb  # type: ignore[assignment, list-item]
    dof_indices: list[int] = [-1] * nb

    v = np.empty((6, nb), order="F")
    c = np.empty((6, nb), order="F")
    ia_articulated = np.array(model["I"], dtype=float)
    pa_bias = np.zeros((6, nb), order="F")
    u_force = np.zeros((6, nb), order="F")
    d = np.zeros(nb)
    u = np.zeros(nb)
    a = np.zeros((6, nb), order="F")
    qdd = np.zeros(nb)

    # Temporary buffers
    buf = _ScratchBuffers.create()

    # OPTIMIZATION: Cache dictionary lookups to local variables
    mdl = _ModelCache.from_model(model)

    # --- Pass 1: Forward kinematics ---
    _aba_forward_kinematics(
        nb,
        q,
        qd,
        f_ext,
        mdl,
        xup,
        s_subspace,
        dof_indices,
        v,
        c,
        pa_bias,
        buf,
    )

    # --- Pass 2: Backward recursion (articulated-body inertias) ---
    _aba_backward_pass(
        nb,
        tau,
        mdl.parent,
        s_subspace,
        dof_indices,
        xup,
        ia_articulated,
        pa_bias,
        u_force,
        d,
        u,
        c,
        buf,
    )

    # --- Pass 3: Forward recursion (accelerations) ---
    _aba_forward_accelerations(
        nb,
        mdl.parent,
        s_subspace,
        dof_indices,
        xup,
        neg_a_grav,
        c,
        u_force,
        d,
        u,
        a,
        qdd,
    )

    return qdd
