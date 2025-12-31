def compute_induced_accelerations(physics) -> dict:
    """Compute induced accelerations (Gravity, Velocity, Control) for current state."""
    results: dict = {}
    try:
        import mujoco
        import numpy as np
    except ImportError:
        return results

    # Ensure we are using compatible model/data
    # If dm_control < 1.0, this might fail unless we extract pointers
    # But let's assume modern dm_control which uses 'mujoco' bindings.
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
    # Define nv explicitly
    nv = model.nv

    # Allocation of explicit buffers to ensure correct shape/type for raw bindings
    # mj_rne expects output buffer of size nv.
    g_force = np.zeros(nv, dtype=np.float64)
    mujoco.mj_rne(model.ptr, data.ptr, 0, g_force)

    # 3. Coriolis/Centrifugal Force (C)
    # Restore v, set a=0.
    data.qvel[:] = qvel_backup
    data.qacc[:] = 0
    
    # Needs separate buffer for the result of this call
    bias_force = np.zeros(nv, dtype=np.float64)
    mujoco.mj_rne(model.ptr, data.ptr, 0, bias_force)
    c_force = bias_force - g_force  # C(q, v)

    # 4. Control Force (from actuators)
    data.qpos[:] = qpos_backup
    data.qvel[:] = qvel_backup
    data.ctrl[:] = ctrl_backup
    mujoco.mj_fwdActuation(model.ptr, data.ptr)
    
    # Copy from data.qfrc_actuator (which is managed by mujoco)
    tau_control = data.qfrc_actuator.copy()

    # Now solve M * a = F
    # Vectors to solve (overwritten by mj_solveM)
    acc_g = np.zeros(nv, dtype=np.float64)
    acc_c = np.zeros(nv, dtype=np.float64)
    acc_t = np.zeros(nv, dtype=np.float64)

    # Solve M*a = F => a = M^-1 * F
    mujoco.mj_solveM(model.ptr, data.ptr, acc_g, -g_force)
    mujoco.mj_solveM(model.ptr, data.ptr, acc_c, -c_force)
    mujoco.mj_solveM(model.ptr, data.ptr, acc_t, tau_control)

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
