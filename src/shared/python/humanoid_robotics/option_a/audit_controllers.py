"""Known-answer controllers for the controller synergy audit (#13884, slice 2).

Both act on the #13837 double pendulum linearised along its nominal reference
(the fixture's reference rollout, ``mjd_transitionFD`` per step):

* :func:`minimum_intervention_gains`: TVLQR with ``Q = 0`` and terminal weight
  ``C^T W C`` (NSX-93 :func:`minimum_intervention_design`), which corrects only
  task-relevant deviations;
* :func:`full_state_gains`: TVLQR with ``Q > 0``, which corrects every
  deviation.

The declared task is the absolute angle of the distal link at the terminal
step, ``y = q0 + q1`` (``TASK_ROWS``); ``STATE_SCALE`` puts rad and rad/s in
comparable units. Deviations evolve on the linear closed loop
``x+ = (A_t - B_t K_t) x`` from a seeded Gaussian perturbation grid.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.estimation.mosaic.local_policy import (
    LinearizedDynamics,
    tvlqr_gains,
)
from src.shared.python.loop_nullspace.controller_audit import ControllerAuditSpec
from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.minimum_intervention import (
    minimum_intervention_design,
)

from .fixture import HORIZON_S, _nominal_torque, _reference
from .plant import OPTION_A_DOUBLE_PENDULUM_SPEC, build_mujoco_model, linearize_step

TASK_ROWS = np.array([[1.0, 1.0, 0.0, 0.0]])
STATE_SCALE = np.array([1.0, 1.0, 10.0, 10.0])
COORDINATES = ("q0_rad", "q1_rad", "v0_rad_s", "v1_rad_s")
TASK_WEIGHT = 1e4
INPUT_WEIGHT = 1e-2
FULL_STATE_WEIGHT = 1e2
PERTURBATION_STD = np.array([0.05, 0.05, 0.5, 0.5])
AUDIT_SEED = 13884


def audit_spec() -> ControllerAuditSpec:
    """The pre-registered audit spec for this plant and task."""
    return ControllerAuditSpec(TASK_ROWS, STATE_SCALE, COORDINATES, seed=AUDIT_SEED)


def reference_linearization() -> LinearizedDynamics:
    """``(A_t, B_t)`` along the fixture's nominal reference, ``T = 350`` steps."""
    import mujoco as mj

    model = build_mujoco_model(OPTION_A_DOUBLE_PENDULUM_SPEC)
    dt = float(model.opt.timestep)
    steps = int(round(HORIZON_S / dt))
    q_ref, v_ref = _reference(model, steps, dt)
    data = mj.MjData(model)
    a_all, b_all = [], []
    for k in range(steps):
        data.qpos[:], data.qvel[:] = q_ref[k], v_ref[k]
        a, b = linearize_step(model, data, _nominal_torque(k * dt))
        a_all.append(a)
        b_all.append(b)
    return LinearizedDynamics(np.array(a_all), np.array(b_all))


def minimum_intervention_gains(plant: LinearizedDynamics) -> Array:
    """Q = 0 TVLQR gains ``(T, nu, nx)`` with terminal ``TASK_WEIGHT C^T C``."""
    n_u = plant.b.shape[2]
    weights = TASK_WEIGHT * np.eye(TASK_ROWS.shape[0])
    design = minimum_intervention_design(
        plant, TASK_ROWS, weights, INPUT_WEIGHT * np.eye(n_u)
    )
    return np.asarray(design.gains)


def full_state_gains(plant: LinearizedDynamics) -> Array:
    """Q = ``FULL_STATE_WEIGHT I`` TVLQR gains with the same terminal task weight."""
    n_u = plant.b.shape[2]
    q = FULL_STATE_WEIGHT * np.eye(plant.n_state)
    terminal = TASK_WEIGHT * TASK_ROWS.T @ TASK_ROWS
    return np.asarray(tvlqr_gains(plant, q, INPUT_WEIGHT * np.eye(n_u), terminal))


def perturbation_grid(n_trials: int, seed: int = AUDIT_SEED) -> Array:
    """Seeded initial deviations ``(n_trials, 4)``, std ``PERTURBATION_STD``."""
    if n_trials < 2:
        raise ValueError("n_trials must be >= 2")
    rng = np.random.default_rng(seed)
    return np.asarray(rng.normal(size=(n_trials, 4)) * PERTURBATION_STD)


def closed_loop_terminal_states(
    plant: LinearizedDynamics, gains: Array, initial: Array
) -> Array:
    """Terminal deviations of ``x+ = (A_t - B_t K_t) x`` from each initial row."""
    x = np.asarray(initial, dtype=np.float64)
    for t in range(plant.n_steps):
        x = x @ (plant.a[t] - plant.b[t] @ gains[t]).T
    return x
