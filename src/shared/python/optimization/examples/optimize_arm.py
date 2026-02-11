import os
import sys

import numpy as np

try:
    import casadi as ca
    import pinocchio as pin
    import pinocchio.casadi as cpin
import logging


logger = logging.getLogger(__name__)
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEP_ERROR = e
    ca = None
    pin = None
    cpin = None


def main() -> None:
    """
    Example: Optimization of a 2-link arm swing using CasADi + Pinocchio.
    Objective: Swing from hanging down (0,0) to upright (pi, 0) with minimum effort.
    """
    if not DEPENDENCIES_AVAILABLE:
        logger.info(
            f"Skipping optimize_arm.py due to missing dependencies: {MISSING_DEP_ERROR}"
        )
        logger.info("Please install casadi and pinocchio:")
        logger.info("  pip install casadi")
        logger.info("  conda install pinocchio -c conda-forge")
        return

    logger.info("Setting up optimization problem...")

    # 1. Load Model
    # ----------------
    urdf_filename = "two_link_arm.urdf"
    urdf_path = os.path.join(os.path.dirname(__file__), urdf_filename)

    if not os.path.exists(urdf_path):
        logger.info(f"Error: URDF file not found at {urdf_path}")
        sys.exit(1)

    # Load Pinocchio model
    model = pin.buildModelFromUrdf(urdf_path)

    logger.info(f"Model loaded: {model.nq} DOFs, {model.nv} velocities")

    # 2. Initialize CasADi Model
    # ---------------------------
    # Create the CasADi-compatible model and data
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    # 3. Optimization Parameters
    # ---------------------------
    T = 2.0  # Total duration (s)
    N = 40  # Number of control intervals (timesteps)
    dt = T / N  # Time step duration

    # 4. Create Optimizer
    # -------------------
    opti = ca.Opti()

    # Decision Variables
    # Q: Joint positions [nq, N+1]
    # V: Joint velocities [nv, N+1]
    # U: Control torques [nv, N]

    nq = model.nq
    nv = model.nv
    nu = model.nv  # Fully actuated

    Q = opti.variable(nq, N + 1)
    V = opti.variable(nv, N + 1)
    U = opti.variable(nu, N)

    # 5. Define Constraints & Dynamics
    # --------------------------------

    # Initial State: Hanging down (all zeros)
    q0 = np.zeros(nq)
    v0 = np.zeros(nv)

    opti.subject_to(Q[:, 0] == q0)
    opti.subject_to(V[:, 0] == v0)

    # Target State: Upright (pi for first joint, 0 for second relative)
    # Note: Pinocchio's revolute joint (Y-axis) -> 0 is X-axis?
    # Let's assume standard intuitive angles:
    # If 0 is hanging down, we want to swing to pi (upright).
    # Check bounds or just specify end point.
    q_target = np.array([np.pi, 0.0])

    opti.subject_to(Q[:, -1] == q_target)
    opti.subject_to(V[:, -1] == np.zeros(nv))

    # Loop over time for Dynamics
    for k in range(N):
        q_k = Q[:, k]
        v_k = V[:, k]
        u_k = U[:, k]

        # Forward Dynamics (ABA) via CasADi
        # Computes acceleration ddq given (q, v, u)
        ddq = cpin.aba(cmodel, cdata, q_k, v_k, u_k)

        # Integration (Semi-Implicit Euler for stability)
        # v_{k+1} = v_k + a * dt
        # q_{k+1} = q_k + v_{k+1} * dt

        v_next = v_k + ddq * dt
        q_next = q_k + v_next * dt

        opti.subject_to(V[:, k + 1] == v_next)
        opti.subject_to(Q[:, k + 1] == q_next)

        # Torque Limits (optional)
        opti.subject_to(opti.bounded(-50, u_k, 50))

    # 6. Objective Function
    # ---------------------
    # Minimize sum of squared torques (Minimize Effort)
    # scale by dt to approximate integral
    cost = ca.sumsqr(U) * dt
    opti.minimize(cost)

    # 7. Solve
    # --------
    logger.info("Solving...")

    # Use IPOPT solver
    # Suppress output for cleanliness, or keep for debug
    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "print_level": 5}  # print_level 5 for progress
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()
        logger.info("\nSUCCESS: Optimal trajectory found!")

        # Retrieve results
        q_opt = sol.value(Q)
        v_opt = sol.value(V)
        u_opt = sol.value(U)
        cost_opt = sol.value(cost)

        logger.info(f"Final Cost: {cost_opt:.4f}")
        logger.info(f"Final Joint Angles: {q_opt[:, -1]}")

        # Save results
        out_file_q = "trajectory_q.csv"
        out_file_v = "trajectory_v.csv"
        out_file_u = "trajectory_u.csv"
        np.savetxt(out_file_q, q_opt.T, delimiter=",", header="q1,q2")
        np.savetxt(out_file_v, v_opt.T, delimiter=",", header="v1,v2")
        np.savetxt(out_file_u, u_opt.T, delimiter=",", header="u1,u2")

        logger.info(f"Trajectory saved to {out_file_q}, {out_file_v}, and {out_file_u}")
        logger.info("\nTest passed: CasADi + Pinocchio integration is working.")

    except Exception as e:  # noqa: BLE001 - CasADi solver may raise various errors
        logger.info("\nFAILURE: Optimization failed.")
        logger.info(e)
        # Debug info
        logger.info("Debug values (last iteration):")
        logger.info(opti.debug.value(Q[:, -1]))
        sys.exit(1)


if __name__ == "__main__":
    main()
