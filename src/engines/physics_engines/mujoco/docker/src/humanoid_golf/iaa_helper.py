from typing import Any

import numpy as np


def compute_induced_accelerations(physics: Any) -> dict[str, Any]:
    """Compute induced accelerations (Gravity, Velocity, Control) for current state."""
    results: dict[str, Any] = {}
    try:
        from dm_control.mujoco.wrapper.mjbindings import mjlib
    except ImportError:
        return results

    # Ensure we are using compatible model/data
    # Use mjlib from dm_control for ctypes pointer compatibility
    model = physics.model
    data = physics.data

    # Backup State
    qpos_backup = data.qpos.copy()
    qvel_backup = data.qvel.copy()
    qacc_backup = data.qacc.copy()
    ctrl_backup = data.ctrl.copy()

    # We need M^-1 * (Force) for each component.
    # 1. Mass Matrix M is implicit in mj_solveM.

    # 2. Gravity Force (G)
    # mj_rne with v=0, a=0 returns G (as bias).
    data.qvel[:] = 0
    data.qacc[:] = 0
    # Note: mj_rne computes inverse dynamics: tau = M*a + C + G.
    # If a=0, v=0, then tau = G.
    # We want G vector.
    # Use the nv from the raw struct to ensure consistency
    nv = model.ptr.nv

    # Allocation of explicit buffers to ensure correct shape/type for raw bindings
    # mj_rne expects output buffer of size nv.
    g_force = np.zeros(nv, dtype=np.float64)
    mjlib.mj_rne(model.ptr, data.ptr, 0, g_force)

    # 3. Coriolis/Centrifugal Force (C)
    # Restore v, set a=0.
    data.qvel[:] = qvel_backup
    data.qacc[:] = 0

    # Needs separate buffer for the result of this call
    bias_force = np.zeros(nv, dtype=np.float64)
    mjlib.mj_rne(model.ptr, data.ptr, 0, bias_force)
    c_force = bias_force - g_force  # C(q, v)

    # 4. Control Force (from actuators)
    data.qpos[:] = qpos_backup
    data.qvel[:] = qvel_backup
    data.ctrl[:] = ctrl_backup
    mjlib.mj_fwdActuation(model.ptr, data.ptr)

    # Copy from data.qfrc_actuator (which is managed by mujoco)
    tau_control = data.qfrc_actuator.copy()
    # Check shape
    if tau_control.shape[0] != nv:
        print(f"WARNING: tau_control shape {tau_control.shape} != nv {nv}. Resizing.")
        tmp = np.zeros(nv, dtype=np.float64)
        tmp[: min(nv, tau_control.shape[0])] = tau_control[
            : min(nv, tau_control.shape[0])
        ]
        tau_control = tmp

    # Now solve M * a = F
    # Vectors to solve (overwritten by mj_solveM)
    acc_g = np.zeros(nv, dtype=np.float64)
    acc_c = np.zeros(nv, dtype=np.float64)
    acc_t = np.zeros(nv, dtype=np.float64)

    # Explicit input arrays
    neg_g_force = -g_force
    neg_c_force = -c_force

    _solve_m(model, data, acc_g, neg_g_force, mjlib)
    _solve_m(model, data, acc_c, neg_c_force, mjlib)
    _solve_m(model, data, acc_t, tau_control, mjlib)

    # Restore State fully
    data.qpos[:] = qpos_backup
    data.qvel[:] = qvel_backup
    data.qacc[:] = qacc_backup
    data.ctrl[:] = ctrl_backup

    return {"gravity": acc_g, "coriolis": acc_c, "control": acc_t}


def compute_counterfactuals(physics: Any) -> dict[str, Any]:
    """Compute Zero-Torque (ZTCF) and Zero-Velocity (ZVCF) accelerations."""
    results: dict[str, Any] = {}
    try:
        from dm_control.mujoco.wrapper.mjbindings import mjlib
    except ImportError:
        return results

    model = physics.model
    data = physics.data

    # Backup State
    qpos_backup = data.qpos.copy()
    qvel_backup = data.qvel.copy()
    qacc_backup = data.qacc.copy()
    ctrl_backup = data.ctrl.copy()

    # --- ZTCF: Zero Torque Counterfactual ---
    # a_ztcf = M^-1 * (-C - G) = Drift
    # This is effectively forward dynamics with ctrl=0.
    data.ctrl[:] = 0
    # We must call mj_forward to recompute everything (actuation=0, bias, etc)
    mjlib.mj_forward(model.ptr, data.ptr)
    ztcf_accel = data.qacc.copy()

    # Restore for ZVCF
    data.qpos[:] = qpos_backup
    data.qvel[:] = qvel_backup
    data.ctrl[:] = ctrl_backup

    # --- ZVCF: Zero Velocity Counterfactual ---
    # a_zvcf = M^-1 * (tau_v0 - G)
    # Set v=0
    data.qvel[:] = 0
    # Keep control inputs same (u), but recompute forces (if velocity dependent)
    mjlib.mj_forward(model.ptr, data.ptr)
    zvcf_accel = data.qacc.copy()

    # Restore State
    data.qpos[:] = qpos_backup
    data.qvel[:] = qvel_backup
    data.qacc[:] = qacc_backup
    data.ctrl[:] = ctrl_backup
    # Re-run forward to ensure data structure is consistent with current state
    # (Optional but good practice if caller expects consistency)
    mjlib.mj_forward(model.ptr, data.ptr)

    return {"ztcf": ztcf_accel, "zvcf": zvcf_accel}


def get_mass_matrix(physics: Any) -> Any:
    """Compute dense Mass Matrix M(q)."""
    try:
        from dm_control.mujoco.wrapper.mjbindings import mjlib
    except ImportError:
        return None

    model = physics.model
    data = physics.data
    nv = model.ptr.nv
    M = np.zeros((nv, nv), dtype=np.float64)

    # Ensure qM is updated (mj_forward usually does this, but explicit call is safer)
    # mj_fullM reads from data.qM
    mjlib.mj_fullM(model.ptr, M, data.ptr.qM)
    return M


def _solve_m(model: Any, data: Any, dst: Any, src: Any, mjlib: Any) -> None:
    """Helper to safely call mj_solveM with correct shapes."""
    # Clean inputs
    dst_clean = np.ascontiguousarray(dst, dtype=np.float64)
    src_clean = np.ascontiguousarray(src, dtype=np.float64)

    # Shapes to try: Flat, Column, Row
    shapes_to_try = [
        dst_clean.shape,  # (nv,)
        (dst_clean.shape[0], 1),  # (nv, 1)
        (1, dst_clean.shape[0]),  # (1, nv)
    ]

    last_err = None
    success = False

    for shape in shapes_to_try:
        try:
            # Reshape views (cheap)
            d_view = dst_clean.reshape(shape)
            s_view = src_clean.reshape(shape)

            # Attempt call
            mjlib.mj_solveM(model.ptr, data.ptr, d_view, s_view)

            # If successful, copy result back to original destination
            # (Handle flatten/shape mismatch by flat copy)
            dst[:] = d_view.flatten()
            success = True
            break
        except TypeError as e:
            last_err = e
        except Exception as e:
            last_err = e  # type: ignore[assignment]

    if not success and last_err is not None:
        raise last_err
