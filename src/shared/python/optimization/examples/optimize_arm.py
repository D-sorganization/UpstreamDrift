import logging
import os
import sys
from typing import Any

import numpy as np

try:
    import casadi as ca
    import pinocchio as pin
    import pinocchio.casadi as cpin

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEP_ERROR = e
    ca = None
    pin = None
    cpin = None


logger = logging.getLogger(__name__)


def _load_model(urdf_path: str) -> object:
    """Load the Pinocchio model from a URDF file."""
    if not os.path.exists(urdf_path):
        logger.error(f"Error: URDF file not found at {urdf_path}")
        sys.exit(1)

    model = pin.buildModelFromUrdf(urdf_path)
    logger.debug(f"Model loaded: {model.nq} DOFs, {model.nv} velocities")
    return model


def _setup_optimization(model: Any, cmodel: Any, cdata: Any) -> tuple:
    """Create optimizer, decision variables, and apply dynamics constraints.

    Returns:
        Tuple of (opti, Q, V, U, dt).
    """
    T = 2.0
    N = 40
    dt = T / N

    opti = ca.Opti()
    nq = model.nq
    nv = model.nv
    nu = model.nv

    Q = opti.variable(nq, N + 1)
    V = opti.variable(nv, N + 1)
    U = opti.variable(nu, N)

    # Boundary conditions
    opti.subject_to(Q[:, 0] == np.zeros(nq))
    opti.subject_to(V[:, 0] == np.zeros(nv))
    opti.subject_to(Q[:, -1] == np.array([np.pi, 0.0]))
    opti.subject_to(V[:, -1] == np.zeros(nv))

    # Dynamics via semi-implicit Euler
    for k in range(N):
        q_k, v_k, u_k = Q[:, k], V[:, k], U[:, k]
        ddq = cpin.aba(cmodel, cdata, q_k, v_k, u_k)
        v_next = v_k + ddq * dt
        q_next = q_k + v_next * dt
        opti.subject_to(V[:, k + 1] == v_next)
        opti.subject_to(Q[:, k + 1] == q_next)
        opti.subject_to(opti.bounded(-50, u_k, 50))

    # Objective: minimize effort
    cost = ca.sumsqr(U) * dt
    opti.minimize(cost)

    return opti, Q, V, U, dt


def _solve_and_export(opti: Any, Q: Any, V: Any, U: Any) -> None:
    """Solve the optimization and save trajectory results."""
    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "print_level": 5}
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()
        logger.info("\nSUCCESS: Optimal trajectory found!")

        q_opt = sol.value(Q)
        v_opt = sol.value(V)
        u_opt = sol.value(U)
        cost_opt = sol.value(opti.f)

        logger.info(f"Final Cost: {cost_opt:.4f}")
        logger.info(f"Final Joint Angles: {q_opt[:, -1]}")

        np.savetxt("trajectory_q.csv", q_opt.T, delimiter=",", header="q1,q2")
        np.savetxt("trajectory_v.csv", v_opt.T, delimiter=",", header="v1,v2")
        np.savetxt("trajectory_u.csv", u_opt.T, delimiter=",", header="u1,u2")

        logger.info(
            "Trajectory saved to trajectory_q.csv, trajectory_v.csv, and trajectory_u.csv"
        )
        logger.info("\nTest passed: CasADi + Pinocchio integration is working.")

    except Exception as e:  # noqa: BLE001 - CasADi solver may raise various errors
        logger.error("\nFAILURE: Optimization failed.")
        logger.info(e)
        logger.debug("Debug values (last iteration):")
        logger.debug(opti.debug.value(Q[:, -1]))
        sys.exit(1)


def main() -> None:
    """
    Example: Optimization of a 2-link arm swing using CasADi + Pinocchio.
    Objective: Swing from hanging down (0,0) to upright (pi, 0) with minimum effort.
    """
    if not DEPENDENCIES_AVAILABLE:
        logger.error(
            f"Skipping optimize_arm.py due to missing dependencies: {MISSING_DEP_ERROR}"
        )
        logger.info("Please install casadi and pinocchio:")
        logger.info("  pip install casadi")
        logger.info("  conda install pinocchio -c conda-forge")
        return

    logger.info("Setting up optimization problem...")

    urdf_path = os.path.join(os.path.dirname(__file__), "two_link_arm.urdf")
    model = _load_model(urdf_path)

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    opti, Q, V, U, dt = _setup_optimization(model, cmodel, cdata)

    logger.info("Solving...")
    _solve_and_export(opti, Q, V, U)


if __name__ == "__main__":
    main()
