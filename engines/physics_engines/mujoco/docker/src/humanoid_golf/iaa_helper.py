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
    # But mj_rne outputs to data.qfrc_inverse.
    mujoco.mj_rne(model, data, 0, data.qfrc_inverse)
    g_force = data.qfrc_inverse.copy()  # This is G(q)

    # 3. Coriolis/Centrifugal Force (C)
    # Restore v, set a=0.
    # mj_rne(v, a=0) -> C + G.
    data.qvel[:] = qvel_backup
    data.qacc[:] = 0
    mujoco.mj_rne(model, data, 0, data.qfrc_inverse)
    bias_force = data.qfrc_inverse.copy()  # C + G
    c_force = bias_force - g_force  # C(q, v)

    # 4. Control Force (from actuators) is tricky in Inverse Dynamics.
    # Usually we go Forward: tau_ctrl is known.
    # In dm_control, physics.data.actuator_force contains forces?
    # Or qfrc_actuation after mj_fwdActuation?
    # We should run Forward logic to get qfrc_actuation.
    data.qpos[:] = qpos_backup  # Restore pos just in case
    data.qvel[:] = qvel_backup
    data.ctrl[:] = ctrl_backup
    mujoco.mj_fwdActuation(model, data)
    tau_control = data.qfrc_actuator.copy()

    # Now solve M * a = F
    # a_g = M^-1 * (-G)
    # a_c = M^-1 * (-C)
    # a_t = M^-1 * (tau_control)

    # Vectors to solve (overwritten by mj_solveM)
    # We need explicit output buffers for mj_solveM(m, d, out, in)
    acc_g = np.zeros_like(g_force)
    acc_c = np.zeros_like(c_force)
    acc_t = np.zeros_like(tau_control)

    # Solve M*a = F => a = M^-1 * F
    mujoco.mj_solveM(model, data, acc_g, -g_force)
    mujoco.mj_solveM(model, data, acc_c, -c_force)
    mujoco.mj_solveM(model, data, acc_t, tau_control)

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
