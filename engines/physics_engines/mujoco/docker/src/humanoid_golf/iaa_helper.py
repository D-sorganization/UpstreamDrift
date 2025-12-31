def compute_induced_accelerations(physics) -> dict:
    """Compute induced accelerations (Gravity, Velocity, Control) for current state."""
    results: dict = {}
    try:
        from dm_control.mujoco.wrapper.mjbindings import mjlib
        import numpy as np
        from ctypes import POINTER, c_double
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
    print(f"DEBUG: model.ptr.nv = {nv}", flush=True)

    # Allocation of explicit buffers to ensure correct shape/type for raw bindings
    # mj_rne expects output buffer of size nv.
    g_force = np.zeros(nv, dtype=np.float64)
    mjlib.mj_rne(model.ptr, data.ptr, 0, g_force)
    print(f"DEBUG: g_force shape = {g_force.shape}", flush=True)

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
        tmp[:min(nv, tau_control.shape[0])] = tau_control[:min(nv, tau_control.shape[0])]
        tau_control = tmp

    # Now solve M * a = F
    # Vectors to solve (overwritten by mj_solveM)
    acc_g = np.zeros(nv, dtype=np.float64)
    acc_c = np.zeros(nv, dtype=np.float64)
    acc_t = np.zeros(nv, dtype=np.float64)

    # Explicit input arrays
    neg_g_force = -g_force
    neg_c_force = -c_force

    def safe_solveM(m_ptr, d_ptr, dst, src):
        """Try calling mj_solveM with different array shapes to satisfy binding."""
        # Clean inputs
        dst_clean = np.ascontiguousarray(dst, dtype=np.float64)
        src_clean = np.ascontiguousarray(src, dtype=np.float64)
        
        # Shapes to try: Flat, Column, Row
        shapes_to_try = [
            dst_clean.shape,             # (nv,)
            (dst_clean.shape[0], 1),     # (nv, 1)
            (1, dst_clean.shape[0])      # (1, nv)
        ]
        
        last_err = None
        success = False
        
        for shape in shapes_to_try:
            try:
                # Reshape views (cheap)
                d_view = dst_clean.reshape(shape)
                s_view = src_clean.reshape(shape)
                
                # Attempt call
                mjlib.mj_solveM(m_ptr, d_ptr, d_view, s_view)
                
                # If successful, copy result back to original destination
                # (Handle flatten/shape mismatch by flat copy)
                dst[:] = d_view.flatten()
                success = True
                break
            except TypeError as e:
                last_err = e
            except Exception as e:
                last_err = e
        
        if not success:
            print(f"ERROR: safe_solveM failed all shapes: {shapes_to_try}", flush=True)
            if last_err:
                 print(f"Last Error: {last_err}", flush=True)
            raise last_err

    safe_solveM(model.ptr, data.ptr, acc_g, neg_g_force)
    safe_solveM(model.ptr, data.ptr, acc_c, neg_c_force)
    safe_solveM(model.ptr, data.ptr, acc_t, tau_control)

    # Restore State fully
    data.qpos[:] = qpos_backup
    data.qvel[:] = qvel_backup
    data.qacc[:] = qacc_backup
    data.ctrl[:] = ctrl_backup

    # Extract results for named joints
    # Convert vectors to dictionary mapping joint_name -> (g, c, t, total)
    # physics.named.data.qacc accesses by name, but we have raw arrays.
    # We need to map joint name to DOFindices.

    # Assuming standard joints (1 DOF).
    # Multi-DOF joints (root) have multiple qacc indices.

    # We'll just return the full arrays for the caller to parse,
    # OR parse them here using physics.model names.

    # Creating a dict of values for target joints only to save space
    # The caller (run_simulation) has TARGET_POSE keys.
    # We can access physics.model.jnt_dofadr to find address in qacc/qfrc.

    return {"gravity": acc_g, "coriolis": acc_c, "control": acc_t}
