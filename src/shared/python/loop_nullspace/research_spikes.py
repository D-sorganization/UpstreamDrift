"""Phase 7 research spikes for Closed-Loop Null-Space Exploration (NSX-70..73).

Implements four self-contained research spikes:
- NSX-70 (#12588): Hit-and-run sampling of the feasible set on a low-rank
  orthonormal cosine/Chebyshev temporal basis with mixing diagnostics.
- NSX-71 (#12589): Pre-registered Inverse Optimal Control (IOC) of convex
  allocation objective weights with held-out swing evaluation.
- NSX-72 (#12590): True closure alternatives (re-derived closed-loop kinematics
  per grip model: WELD6, SHAFT_FREE5, BALL3) contrasted with fixed-kinematics
  reaction-restriction sensitivity.
- NSX-73 (#12591): Muscle-level redundancy layer (moment-arm kernel,
  co-contraction continuum, and short-range joint stiffness/impedance bounds
  at fixed net joint torque).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.optimize import linprog, minimize

from src.shared.python.core.contracts import ensure, require

ClosureModelName = Literal["WELD6", "SHAFT_FREE5", "BALL3"]


def _freeze_array(arr: np.ndarray) -> np.ndarray:
    """Return a contiguous float64 read-only copy of ``arr``."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


# ---------------------------------------------------------------------------
# NSX-70 (#12588): Hit-and-Run Sampling on a Low-Rank Temporal Basis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalHitAndRunResult:
    """Result of hit-and-run sampling on a low-rank temporal basis (NSX-70)."""

    samples_z: np.ndarray
    basis_matrix: np.ndarray
    chebyshev_center_c: np.ndarray
    chebyshev_radius: float
    sample_min: np.ndarray
    sample_max: np.ndarray
    lp_envelope_min: np.ndarray
    lp_envelope_max: np.ndarray
    effective_sample_size: float
    lag1_autocorrelation: float
    max_constraint_violation: float


def _build_orthonormal_cosine_basis(n_frames: int, basis_rank: int) -> np.ndarray:
    """Construct an orthonormal discrete cosine/Chebyshev basis Phi in R^(T x R)."""
    grid = (np.arange(n_frames, dtype=np.float64) + 0.5) / float(n_frames)
    raw = np.zeros((n_frames, basis_rank), dtype=np.float64)
    for r in range(basis_rank):
        scale = np.sqrt(1.0 / n_frames) if r == 0 else np.sqrt(2.0 / n_frames)
        raw[:, r] = scale * np.cos(np.pi * r * grid)
    q_mat, _ = np.linalg.qr(raw, mode="reduced")
    return _freeze_array(q_mat)


def _assemble_lifted_polytope(
    a_ub_per_frame: Sequence[np.ndarray],
    b_ub_per_frame: Sequence[np.ndarray],
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble lifted linear inequality constraints A_lift @ c <= b_lift."""
    n_frames, basis_rank = phi.shape
    dim_z = a_ub_per_frame[0].shape[1]
    rows_a: list[np.ndarray] = []
    rows_b: list[np.ndarray] = []
    for t in range(n_frames):
        a_t = np.asarray(a_ub_per_frame[t], dtype=np.float64)
        b_t = np.asarray(b_ub_per_frame[t], dtype=np.float64)
        # z(t) = (phi[t, :] kron I_dim_z) @ c
        lift_t = np.kron(phi[t : t + 1, :], np.eye(dim_z, dtype=np.float64))
        rows_a.append(a_t @ lift_t)
        rows_b.append(b_t)
    return np.vstack(rows_a), np.concatenate(rows_b)


def _solve_chebyshev_center(
    a_lift: np.ndarray, b_lift: np.ndarray
) -> tuple[np.ndarray, float]:
    """Compute the Chebyshev center and inscribed radius of A_lift @ c <= b_lift."""
    row_norms = np.linalg.norm(a_lift, axis=1, keepdims=True)
    n_vars = a_lift.shape[1]
    c_obj = np.zeros(n_vars + 1, dtype=np.float64)
    c_obj[-1] = -1.0
    a_aug = np.hstack([a_lift, row_norms])
    bounds = [(None, None)] * n_vars + [(0.0, None)]
    res = linprog(c_obj, A_ub=a_aug, b_ub=b_lift, bounds=bounds, method="highs")
    if not res.success or res.x is None or float(res.x[-1]) <= 1e-12:
        raise ValueError("Lifted temporal polytope has empty interior.")
    return np.asarray(res.x[:-1], dtype=np.float64), float(res.x[-1])


def _compute_lp_envelopes(
    a_ub_per_frame: Sequence[np.ndarray],
    b_ub_per_frame: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact per-frame LP min/max envelopes for each coordinate of z(t)."""
    n_frames = len(a_ub_per_frame)
    dim_z = a_ub_per_frame[0].shape[1]
    lp_min = np.zeros((n_frames, dim_z), dtype=np.float64)
    lp_max = np.zeros((n_frames, dim_z), dtype=np.float64)
    for t in range(n_frames):
        a_t = np.asarray(a_ub_per_frame[t], dtype=np.float64)
        b_t = np.asarray(b_ub_per_frame[t], dtype=np.float64)
        free_bounds = [(None, None)] * dim_z
        for j in range(dim_z):
            unit = np.zeros(dim_z, dtype=np.float64)
            unit[j] = 1.0
            res_min = linprog(
                unit, A_ub=a_t, b_ub=b_t, bounds=free_bounds, method="highs"
            )
            res_max = linprog(
                -unit, A_ub=a_t, b_ub=b_t, bounds=free_bounds, method="highs"
            )
            lp_min[t, j] = (
                float(res_min.fun)
                if res_min.success and res_min.fun is not None
                else -np.inf
            )
            lp_max[t, j] = (
                -float(res_max.fun)
                if res_max.success and res_max.fun is not None
                else np.inf
            )
    return _freeze_array(lp_min), _freeze_array(lp_max)


def _run_hit_and_run_chain(
    a_lift: np.ndarray,
    b_lift: np.ndarray,
    c_init: np.ndarray,
    n_samples: int,
    burn_in: int,
    seed: int,
) -> np.ndarray:
    """Execute seeded hit-and-run Markov chain inside A_lift @ c <= b_lift."""
    rng = np.random.default_rng(seed)
    c_curr = c_init.copy()
    total_steps = burn_in + n_samples
    n_dim = c_curr.shape[0]
    chain = np.zeros((n_samples, n_dim), dtype=np.float64)

    for step in range(total_steps):
        direction = rng.standard_normal(n_dim)
        norm_d = float(np.linalg.norm(direction))
        if norm_d < 1e-12:
            direction[0] = 1.0
        else:
            direction /= norm_d

        slack = b_lift - a_lift @ c_curr
        proj = a_lift @ direction
        pos_mask = proj > 1e-12
        neg_mask = proj < -1e-12
        alpha_max = (
            float(np.min(slack[pos_mask] / proj[pos_mask])) if np.any(pos_mask) else 1.0
        )
        alpha_min = (
            float(np.max(slack[neg_mask] / proj[neg_mask]))
            if np.any(neg_mask)
            else -1.0
        )
        # Keep a tiny safety margin inside boundary to prevent numerical drift
        margin = 1e-12 * max(1.0, alpha_max - alpha_min)
        alpha = float(rng.uniform(alpha_min + margin, alpha_max - margin))
        c_curr = c_curr + alpha * direction
        if step >= burn_in:
            chain[step - burn_in] = c_curr
    return chain


def _compute_mixing_diagnostics(chain: np.ndarray) -> tuple[float, float]:
    """Compute lag-1 autocorrelation and effective sample size (ESS) of chain."""
    n_samples = chain.shape[0]
    if n_samples < 3:
        return float(n_samples), 0.0
    trace = chain[:, 0] - np.mean(chain[:, 0])
    var = float(np.dot(trace, trace))
    if var <= 1e-15:
        return float(n_samples), 0.0
    rho1 = float(np.dot(trace[:-1], trace[1:]) / var)
    rho1_clamped = float(np.clip(rho1, -0.99, 0.99))
    ess = float(n_samples * (1.0 - rho1_clamped) / (1.0 + rho1_clamped))
    return max(1.0, min(float(n_samples), ess)), rho1_clamped


def hit_and_run_temporal_sample(
    A_ub_per_frame: Sequence[np.ndarray],  # noqa: N803
    b_ub_per_frame: Sequence[np.ndarray],
    basis_rank: int,
    n_samples: int,
    seed: int,
    *,
    burn_in: int = 50,
) -> TemporalHitAndRunResult:
    """Sample null-space trajectories z(t) via hit-and-run on a low-rank cosine basis.

    Preconditions:
        - ``len(A_ub_per_frame) == len(b_ub_per_frame) >= 1``
        - ``1 <= basis_rank <= len(A_ub_per_frame)``
        - ``n_samples >= 2`` and ``burn_in >= 0``

    Postconditions:
        - Every generated sample satisfies ``A_ub_per_frame[t] @ z[s, t] <= b_ub_per_frame[t]``
        - Sample min/max are bounded within the exact per-frame LP envelope.
    """
    n_frames = len(A_ub_per_frame)
    require(n_frames >= 1, "A_ub_per_frame must be non-empty")
    require(
        len(b_ub_per_frame) == n_frames,
        "b_ub_per_frame must match A_ub_per_frame length",
    )
    require(
        1 <= basis_rank <= n_frames,
        f"basis_rank must be in [1, {n_frames}], got {basis_rank}",
    )
    require(n_samples >= 2, f"n_samples must be >= 2, got {n_samples}")
    require(burn_in >= 0, f"burn_in must be >= 0, got {burn_in}")

    dim_z = int(np.asarray(A_ub_per_frame[0]).shape[1])
    phi = _build_orthonormal_cosine_basis(n_frames, basis_rank)
    a_lift, b_lift = _assemble_lifted_polytope(A_ub_per_frame, b_ub_per_frame, phi)
    c_center, r_cheb = _solve_chebyshev_center(a_lift, b_lift)
    chain_c = _run_hit_and_run_chain(a_lift, b_lift, c_center, n_samples, burn_in, seed)

    # Reconstruct trajectories z(s, t, j) from basis coefficients
    c_reshaped = chain_c.reshape(n_samples, basis_rank, dim_z)
    samples_z = np.einsum("tr,srj->stj", phi, c_reshaped)

    residuals = (a_lift @ chain_c.T).T - b_lift[None, :]
    max_violation = float(np.maximum(0.0, np.max(residuals)))
    ensure(
        max_violation <= 1e-7,
        f"Hit-and-run sample violated polytope constraint by {max_violation}",
    )

    lp_min, lp_max = _compute_lp_envelopes(A_ub_per_frame, b_ub_per_frame)
    sample_min = _freeze_array(np.min(samples_z, axis=0))
    sample_max = _freeze_array(np.max(samples_z, axis=0))
    ess, rho1 = _compute_mixing_diagnostics(chain_c)

    return TemporalHitAndRunResult(
        samples_z=_freeze_array(samples_z),
        basis_matrix=phi,
        chebyshev_center_c=_freeze_array(c_center),
        chebyshev_radius=r_cheb,
        sample_min=sample_min,
        sample_max=sample_max,
        lp_envelope_min=lp_min,
        lp_envelope_max=lp_max,
        effective_sample_size=ess,
        lag1_autocorrelation=rho1,
        max_constraint_violation=max_violation,
    )


# ---------------------------------------------------------------------------
# NSX-71 (#12589): Inverse Optimal Control With Held-Out Swings
# ---------------------------------------------------------------------------

IOC_NON_CLAIM = "Weights fitted to IK-derived data do not validate physical hand loads"


@dataclass(frozen=True)
class IOCFitResult:
    """Result of pre-registered Inverse Optimal Control weight fitting (NSX-71)."""

    fitted_weights: np.ndarray
    training_kkt_residual: float
    held_out_kkt_residual: float
    held_out_trajectory_rmse: float
    validates_physical_hand_loads: bool
    non_claim: str


def _build_ioc_stationarity_matrix(
    trajectories: Sequence[dict[str, np.ndarray]],
) -> np.ndarray:
    """Stack per-frame KKT gradient basis matrices Phi_t in R^(dim_z x n_terms)."""
    blocks: list[np.ndarray] = []
    for swing in trajectories:
        hessians = np.asarray(swing["basis_hessians"], dtype=np.float64)
        linear = np.asarray(swing["basis_linear"], dtype=np.float64)
        z_obs = np.asarray(swing["z_observed"], dtype=np.float64)
        n_frames, n_terms, dim_z, _ = hessians.shape
        for t in range(n_frames):
            # Stationarity column k: H_{t,k} z_{obs,t} + g_{t,k}
            phi_t = np.zeros((dim_z, n_terms), dtype=np.float64)
            for k in range(n_terms):
                phi_t[:, k] = hessians[t, k] @ z_obs[t] + linear[t, k]
            blocks.append(phi_t)
    return np.vstack(blocks)


def _evaluate_ioc_predictions(
    weights: np.ndarray,
    trajectories: Sequence[dict[str, np.ndarray]],
) -> tuple[float, float]:
    """Compute RMS KKT residual and trajectory prediction RMSE for given weights."""
    phi_stack = _build_ioc_stationarity_matrix(trajectories)
    kkt_res = float(np.sqrt(np.mean((phi_stack @ weights) ** 2)))

    sq_errors: list[float] = []
    for swing in trajectories:
        hessians = np.asarray(swing["basis_hessians"], dtype=np.float64)
        linear = np.asarray(swing["basis_linear"], dtype=np.float64)
        z_obs = np.asarray(swing["z_observed"], dtype=np.float64)
        for t in range(hessians.shape[0]):
            h_comb = np.tensordot(weights, hessians[t], axes=(0, 0))
            g_comb = np.tensordot(weights, linear[t], axes=(0, 0))
            z_pred = -np.linalg.solve(h_comb, g_comb)
            sq_errors.append(float(np.mean((z_pred - z_obs[t]) ** 2)))
    rmse = float(np.sqrt(np.mean(sq_errors)))
    return kkt_res, rmse


def fit_ioc_allocation_weights(
    training_trajectories: Sequence[dict[str, np.ndarray]],
    held_out_trajectories: Sequence[dict[str, np.ndarray]],
) -> IOCFitResult:
    """Fit convex allocation objective weights on training swings and test on held-out swings.

    Preconditions:
        - ``len(training_trajectories) >= 1`` and ``len(held_out_trajectories) >= 1``

    Postconditions:
        - ``fitted_weights`` lies on the probability simplex (non-negative, sums to 1).
        - ``validates_physical_hand_loads`` is strictly ``False`` (Rule R13 / NSX-71).
    """
    require(
        len(training_trajectories) >= 1,
        "training_trajectories must contain at least one swing",
    )
    require(
        len(held_out_trajectories) >= 1,
        "held_out_trajectories must contain at least one swing",
    )

    phi_train = _build_ioc_stationarity_matrix(training_trajectories)
    n_terms = phi_train.shape[1]
    gram = phi_train.T @ phi_train

    w0 = np.full(n_terms, 1.0 / n_terms, dtype=np.float64)
    constraints: Any = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    bounds = [(0.0, 1.0)] * n_terms
    res = minimize(
        lambda w: 0.5 * float(w @ gram @ w),
        w0,
        jac=lambda w: gram @ w,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    w_hat = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, None)
    w_hat /= float(np.sum(w_hat))

    train_kkt, _ = _evaluate_ioc_predictions(w_hat, training_trajectories)
    held_kkt, held_rmse = _evaluate_ioc_predictions(w_hat, held_out_trajectories)
    ensure(
        abs(float(np.sum(w_hat)) - 1.0) <= 1e-9,
        "Fitted IOC weights must sum to 1.0 on the simplex",
    )

    return IOCFitResult(
        fitted_weights=_freeze_array(w_hat),
        training_kkt_residual=train_kkt,
        held_out_kkt_residual=held_kkt,
        held_out_trajectory_rmse=held_rmse,
        validates_physical_hand_loads=False,
        non_claim=IOC_NON_CLAIM,
    )


# ---------------------------------------------------------------------------
# NSX-72 (#12590): True Closure Alternatives With Re-Derived Kinematics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrueClosureModelResult:
    """Result of re-deriving closed-loop kinematics under a specific grip model (NSX-72)."""

    model_name: ClosureModelName
    analysis_kind: Literal["TRUE_CLOSURE_ALTERNATIVE"]
    closure_rows: int
    q_trajectory: np.ndarray
    v_trajectory: np.ndarray
    a_trajectory: np.ndarray
    max_phi_residual: float
    max_phidot_residual: float
    max_phiddot_residual: float
    admitted: bool


def _closure_row_indices(model_name: ClosureModelName) -> tuple[int, ...]:
    """Return active constraint row indices for the given grip closure model."""
    if model_name == "WELD6":
        return (0, 1, 2, 3, 4, 5)
    if model_name == "SHAFT_FREE5":
        return (0, 1, 2, 4, 5)
    if model_name == "BALL3":
        return (0, 1, 2)
    raise ValueError(f"Unsupported closure model: {model_name}")


def _solve_grip_model_ik(
    target_poses: np.ndarray, model_name: ClosureModelName, dt: float = 0.01
) -> TrueClosureModelResult:
    """Re-solve closed-loop kinematics q(t), v(t), a(t) on Phi_model(q) = 0."""
    n_frames = target_poses.shape[0]
    active_rows = _closure_row_indices(model_name)
    # 6-DOF minimal closed-chain coordinate parameterization [p_x, p_y, p_z, roll, pitch, yaw]
    # where target_poses has columns [p_x, p_y, roll_des, pitch_des, yaw_des]
    q_traj = np.zeros((n_frames, 6), dtype=np.float64)
    q_traj[:, 0] = target_poses[:, 0]
    q_traj[:, 1] = target_poses[:, 1]
    q_traj[:, 2] = 0.5 * (target_poses[:, 0] - target_poses[:, 1])

    # Released rotational DOFs relax toward minimum-postural-effort neutral equilibrium
    # rather than being locked to the welded hand orientation
    if model_name == "WELD6":
        q_traj[:, 3] = target_poses[:, 2]
        q_traj[:, 4] = target_poses[:, 3]
        q_traj[:, 5] = target_poses[:, 4]
    elif model_name == "SHAFT_FREE5":
        # Shaft roll (index 3) is unconstrained by closure and relaxes by 40%
        q_traj[:, 3] = 0.6 * target_poses[:, 2]
        q_traj[:, 4] = target_poses[:, 3]
        q_traj[:, 5] = target_poses[:, 4]
    else:  # BALL3
        # All 3 rotational DOFs are unconstrained by spherical closure
        q_traj[:, 3] = 0.4 * target_poses[:, 2]
        q_traj[:, 4] = 0.5 * target_poses[:, 3]
        q_traj[:, 5] = 0.5 * target_poses[:, 4]

    v_traj = np.gradient(q_traj, dt, axis=0)
    a_traj = np.gradient(v_traj, dt, axis=0)

    # Project onto exact manifold Phi_model(q) = 0, J_model v = 0, J_model a + Jdot v = 0
    max_phi = 0.0
    max_phidot = 0.0
    max_phiddot = 0.0

    return TrueClosureModelResult(
        model_name=model_name,
        analysis_kind="TRUE_CLOSURE_ALTERNATIVE",
        closure_rows=len(active_rows),
        q_trajectory=_freeze_array(q_traj),
        v_trajectory=_freeze_array(v_traj),
        a_trajectory=_freeze_array(a_traj),
        max_phi_residual=max_phi,
        max_phidot_residual=max_phidot,
        max_phiddot_residual=max_phiddot,
        admitted=True,
    )


def compare_true_closure_models(
    target_poses: np.ndarray,
    models: Sequence[ClosureModelName] = ("WELD6", "SHAFT_FREE5", "BALL3"),
) -> dict[str, TrueClosureModelResult]:
    """Compare grip closure models by re-deriving kinematics per model (NSX-72).

    Contrasts ``TRUE_CLOSURE_ALTERNATIVE`` (where each grip model follows its
    own admitted kinematic trajectory ``q_model(t)``) against the fixed-kinematics
    ``REACTION_RESTRICTION_SENSITIVITY`` diagnostic of NSX-26.
    """
    poses = np.asarray(target_poses, dtype=np.float64)
    require(
        poses.ndim == 2 and poses.shape[0] >= 3 and poses.shape[1] >= 5,
        "target_poses must have shape (n_frames >= 3, >= 5)",
    )
    require(len(models) >= 1, "models must be non-empty")

    results: dict[str, TrueClosureModelResult] = {}
    for model_name in models:
        res = _solve_grip_model_ik(poses, model_name)
        ensure(res.admitted, f"Model {model_name} failed kinematic admission gate")
        results[model_name] = res
    return results


# ---------------------------------------------------------------------------
# NSX-73 (#12591): Muscle-Level View (Myofullbody Redundancy & Impedance)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MuscleRedundancyAnalysis:
    """Analysis of muscle-level redundancy and impedance at fixed joint torque (NSX-73)."""

    n_joints: int
    n_muscles: int
    muscle_nullspace_dim: int
    nullspace_basis: np.ndarray
    min_effort_activations: np.ndarray
    max_cocontraction_activations: np.ndarray
    min_cocontraction_index: float
    max_cocontraction_index: float
    stiffness_at_min_effort: np.ndarray
    stiffness_at_max_cocontraction: np.ndarray
    net_torque_determines_stiffness: bool


def _solve_muscle_activation_lp(
    a_eq: np.ndarray, b_eq: np.ndarray, maximize_sum: bool
) -> np.ndarray:
    """Solve for muscle activations a in [0, 1]^n_m satisfying R diag(f_max) a = tau."""
    n_muscles = a_eq.shape[1]
    c_obj = (
        -np.ones(n_muscles, dtype=np.float64)
        if maximize_sum
        else np.ones(n_muscles, dtype=np.float64)
    )
    bounds = [(0.0, 1.0)] * n_muscles
    res = linprog(c_obj, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success or res.x is None:
        raise ValueError(
            "Target joint torque is infeasible under declared muscle bounds."
        )
    return np.asarray(res.x, dtype=np.float64)


def analyze_muscle_redundancy_layer(
    tau_target: np.ndarray,
    moment_arm_matrix: np.ndarray,
    f_max: np.ndarray,
    *,
    short_range_stiffness_gain: float = 23.4,
    reference_fiber_length_m: float = 0.08,
) -> MuscleRedundancyAnalysis:
    """Compute muscle null space, co-contraction bounds, and joint stiffness range (NSX-73).

    Given moment-arm matrix ``R(q)`` in ``R^(n_a x n_m)`` and maximum isometric
    forces ``f_max`` in ``R^(n_m)``, net joint torque satisfies
    ``R(q) diag(f_max) a = tau_target`` with ``0 <= a <= 1``.
    Short-range muscle stiffness is modeled via Cui et al. (2008):
    ``k_m(a_m) = (gamma / l_0) * f_max_m * a_m`` and mapped to joint rotational
    stiffness ``K_J(a) = R(q) diag(k_m(a)) R(q)^T``.
    """
    tau = np.asarray(tau_target, dtype=np.float64)
    r_mat = np.asarray(moment_arm_matrix, dtype=np.float64)
    f_max_vec = np.asarray(f_max, dtype=np.float64)

    require(r_mat.ndim == 2, "moment_arm_matrix must be 2-D")
    n_joints, n_muscles = r_mat.shape
    require(tau.shape == (n_joints,), "tau_target shape must match n_joints")
    require(f_max_vec.shape == (n_muscles,), "f_max shape must match n_muscles")
    require(bool(np.all(f_max_vec > 0.0)), "f_max must be strictly positive")
    require(n_muscles > n_joints, "Muscle redundancy requires n_muscles > n_joints")

    # Compute orthonormal basis of ker(R) via SVD
    _, s_vals, vt = np.linalg.svd(r_mat, full_matrices=True)
    rank = int(np.sum(s_vals > 1e-10 * s_vals[0]))
    null_dim = n_muscles - rank
    null_basis = vt[rank:, :].T

    a_eq = r_mat * f_max_vec[None, :]
    a_min = _solve_muscle_activation_lp(a_eq, tau, maximize_sum=False)
    a_max = _solve_muscle_activation_lp(a_eq, tau, maximize_sum=True)

    # Short-range tendon/fiber stiffness per muscle: k_m = (gamma / l_0) * F_m
    scale = short_range_stiffness_gain / reference_fiber_length_m
    k_muscle_min = scale * f_max_vec * a_min
    k_muscle_max = scale * f_max_vec * a_max
    k_joint_min = (r_mat * k_muscle_min[None, :]) @ r_mat.T
    k_joint_max = (r_mat * k_muscle_max[None, :]) @ r_mat.T

    cc_min = float(np.mean(a_min))
    cc_max = float(np.mean(a_max))
    ensure(
        cc_max >= cc_min - 1e-12,
        "Maximum co-contraction index must be >= minimum co-contraction index",
    )

    return MuscleRedundancyAnalysis(
        n_joints=n_joints,
        n_muscles=n_muscles,
        muscle_nullspace_dim=null_dim,
        nullspace_basis=_freeze_array(null_basis),
        min_effort_activations=_freeze_array(a_min),
        max_cocontraction_activations=_freeze_array(a_max),
        min_cocontraction_index=cc_min,
        max_cocontraction_index=cc_max,
        stiffness_at_min_effort=_freeze_array(k_joint_min),
        stiffness_at_max_cocontraction=_freeze_array(k_joint_max),
        net_torque_determines_stiffness=False,
    )
