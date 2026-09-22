"""Topology recording and feasibility diagnostics for the closed-loop upper-body golfer (TB-06 #10591).

The golfer model (`src.shared.python.pendulum_simulator.physics_golfer`) has a genuine
closed kinematic loop: both hands grip a shared club, giving 8 generalized coordinates
subject to 4 holonomic constraints whose Jacobian is rank-3 (per TB-00's registry entry
for `constrained_upper_body_golfer`), leaving 5 independent DOFs. This module never
substitutes a simpler open-chain model and never fakes loop closure by drawing the
second hand onto the club after the fact -- it reuses the existing Baumgarte-stabilized
constrained solver (`constraint_solver.py`) and Newton constraint projection
(`project_to_constraints` / `project_velocity`) that already enforce the loop exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.shared.python.pendulum_simulator.constraint_solver import (
    project_to_constraints,
    project_velocity,
)
from src.shared.python.pendulum_simulator.golfer_constraints import (
    analytical_constraint_jacobian,
    constraint_vector,
)
from src.shared.python.pendulum_simulator.golfer_kinematics import forward_kinematics
from src.shared.python.pendulum_simulator.physics_golfer import (
    N_CONSTRAINTS,
    N_DOF,
    GolferParams,
)
from src.shared.python.tour_baselines.registry import get_golf_model

# Model identity per TB-00's registry (#10585); reused, never re-derived.
MODEL_ID_GOLFER: str = "constrained_upper_body_golfer"

# Jacobian singular values below this magnitude are treated as numerically
# rank-deficient when auditing loop closure (distinct from a legitimate,
# well-conditioned rank-3 Jacobian that TB-00 already established).
_RANK_SINGULAR_VALUE_TOL: float = 1e-8


def describe_golfer_topology() -> dict[str, object]:
    """Return the TB-00-registered DOF/constraint/topology record for this model.

    Reuses the authoritative `GolfModelIdentity` entry rather than re-deriving
    independent-DOF counts from scratch, so this module and TB-00's registry
    cannot silently drift apart.
    """
    identity = get_golf_model(MODEL_ID_GOLFER)
    return {
        "model_id": identity.model_id,
        "n_generalized_coordinates": identity.dof,
        "n_constraints": identity.constraint_count,
        "constraint_jacobian_rank": identity.dof - identity.independent_dof,
        "n_independent_dof": identity.independent_dof,
        "has_hand_club_loop_closure": identity.constraint_count > 0,
        # Club DOF (theta_club) has no independent actuator; the remaining
        # 7 generalized coordinates (hub, both shoulders, both elbows, both
        # wrists) each carry an applied joint torque.
        "n_actuated_joints": N_DOF - 1,
    }


def create_calibrated_golfer_params(
    *,
    arm_length_scale: float = 1.0,
) -> GolferParams:
    """Return one fixed, physically plausible golfer geometry (bounded work item 2).

    Segment lengths follow representative adult male anthropometrics (arm/forearm
    proportions from the existing double/triple pendulum calibration targets used
    by TB-04/TB-05); this is a single frozen geometry, never re-derived per frame.
    """
    if not (arm_length_scale > 0.0):
        raise ValueError(f"arm_length_scale must be positive, got {arm_length_scale}")

    return GolferParams(
        m_hub=8.0,
        m_r_upper=2.1,
        m_r_fore=1.3,
        m_l_upper=2.1,
        m_l_fore=1.3,
        m_club=0.35,
        L_hub=0.15,
        L_r_upper=0.30 * arm_length_scale,
        L_r_fore=0.27 * arm_length_scale,
        L_l_upper=0.30 * arm_length_scale,
        L_l_fore=0.27 * arm_length_scale,
        L_club=1.02,
        d_rs=0.19,
        d_ls=0.19,
        grip_right=0.02,
        grip_left=0.12,
        m_clubhead=0.20,
        b_hub=0.5,
        b_rs=0.3,
        b_re=0.2,
        b_rh=0.1,
        b_ls=0.3,
        b_le=0.2,
        b_lh=0.1,
    )


@dataclass(frozen=True)
class GolferFeasibilityReport:
    """Diagnostic outcome of attempting to close the loop from a candidate state.

    `feasible=False` is a first-class, honest outcome (singular Jacobian or
    non-convergent Newton projection), never silently coerced to a fake success.
    """

    feasible: bool
    q0: np.ndarray
    v0: np.ndarray
    position_residual: float
    velocity_residual: float
    jacobian_singular_values: np.ndarray
    jacobian_rank: int
    reason: str = ""


def _jacobian_rank(q: np.ndarray, params: GolferParams) -> tuple[np.ndarray, int]:
    jac = analytical_constraint_jacobian(q, params)
    singular_values = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(singular_values > _RANK_SINGULAR_VALUE_TOL))
    return singular_values, rank


def solve_feasible_initial_state(
    q_guess: np.ndarray,
    v_guess: np.ndarray,
    params: GolferParams,
    *,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> GolferFeasibilityReport:
    """Project a candidate state onto the loop-closure manifold, or report why not.

    Reuses `project_to_constraints`/`project_velocity` (Newton iteration on the
    existing analytical Jacobian) rather than reimplementing constraint solving.
    Never substitutes a simpler open-chain model and never reports success on a
    projection that did not actually converge.
    """
    if not isinstance(q_guess, np.ndarray) or not isinstance(v_guess, np.ndarray):
        raise TypeError("q_guess and v_guess must be numpy ndarrays")
    if q_guess.shape != (N_DOF,):
        raise ValueError(f"q_guess must have shape ({N_DOF},), got {q_guess.shape}")
    if v_guess.shape != (N_DOF,):
        raise ValueError(f"v_guess must have shape ({N_DOF},), got {v_guess.shape}")
    if not isinstance(params, GolferParams):
        raise TypeError("params must be a GolferParams instance")

    try:
        q0 = project_to_constraints(q_guess, params, max_iter=max_iter, tol=tol)
    except RuntimeError as exc:
        singular_values, rank = _jacobian_rank(q_guess, params)
        return GolferFeasibilityReport(
            feasible=False,
            q0=q_guess.copy(),
            v0=v_guess.copy(),
            position_residual=float(
                math.sqrt(
                    float(
                        np.dot(
                            constraint_vector(q_guess, params),
                            constraint_vector(q_guess, params),
                        )
                    )
                )
            ),
            velocity_residual=float("nan"),
            jacobian_singular_values=singular_values,
            jacobian_rank=rank,
            reason=f"constraint projection did not converge: {exc}",
        )

    singular_values, rank = _jacobian_rank(q0, params)
    if rank < N_CONSTRAINTS - 1:
        # TB-00 established a genuine rank-3 Jacobian (redundant row) as expected;
        # anything lower indicates a true kinematic singularity at this pose.
        return GolferFeasibilityReport(
            feasible=False,
            q0=q0,
            v0=v_guess.copy(),
            position_residual=0.0,
            velocity_residual=float("nan"),
            jacobian_singular_values=singular_values,
            jacobian_rank=rank,
            reason=(
                f"constraint Jacobian rank {rank} < expected minimum "
                f"{N_CONSTRAINTS - 1} at the projected pose (kinematic singularity)"
            ),
        )

    v0 = project_velocity(q0, v_guess, params)
    jac = analytical_constraint_jacobian(q0, params)
    velocity_violation = jac @ v0
    phi = constraint_vector(q0, params)

    return GolferFeasibilityReport(
        feasible=True,
        q0=q0,
        v0=v0,
        position_residual=float(math.sqrt(float(np.dot(phi, phi)))),
        velocity_residual=float(
            math.sqrt(float(np.dot(velocity_violation, velocity_violation)))
        ),
        jacobian_singular_values=singular_values,
        jacobian_rank=rank,
        reason="",
    )


def grip_and_clubhead_positions(
    q: np.ndarray, params: GolferParams
) -> dict[str, tuple[float, float]]:
    """Return independently-tracked grip_right, grip_left and clubhead positions.

    Delegates to the existing forward kinematics (never infers one hand's
    position from the other): `grip_right` comes from the right arm chain,
    `grip_left` from the left arm chain and club geometry, and `clubhead`
    from the club tip.
    """
    fk = forward_kinematics(q, params)
    return {
        "grip_right": fk["grip_right"],
        "grip_left": fk["grip_left"],
        "clubhead": fk["club_tip"],
    }
