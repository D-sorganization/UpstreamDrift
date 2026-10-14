"""Leave-one-trial-out (LOTO) cross-validation study and qualification gates (MOSAIC-09, #11541).

This module provides:
1. :class:`LeaveOneTrialOutConfig`: reproducible study configuration with deterministic seeds,
   frozen protocol thresholds, and qualification bounds.
2. :class:`TrialFoldReceipt`: cryptographic and numerical receipts for a single evaluation fold,
   evaluating held-out tracking RMSE, inverse-dynamics residual, jerk-whiteness gate,
   and open-loop replay.
3. :class:`LeaveOneTrialOutReport`: aggregate multi-trial cross-validation report capturing
   shared-parameter spread, template explained variance, and gate conformance.
4. :func:`run_leave_one_trial_out_study`: executes the complete LOTO pipeline across M trials,
   fitting shared geometry and inertial parameters on M-1 trials and qualifying on the
   held-out trial, reporting any un-evaluatable gate explicitly as "unqualified".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import logging
import time
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import ensure, require
from src.shared.python.estimation.mosaic.benchmark_protocol import (
    BenchmarkProtocol,
    verify_protocol_integrity,
)
from src.shared.python.estimation.mosaic.ik_init import initialize_trajectory
from src.shared.python.estimation.mosaic.inner_solve import (
    InnerProblem,
    solve_inner,
)
from src.shared.python.estimation.mosaic.jerk_whiteness import (
    evaluate_whiteness_gate,
)
from src.shared.python.estimation.mosaic.local_policy import (
    ReplayReport,
    linearize_along_trajectory,
    replay_acceptance,
    tvlqr_gains,
)
from src.shared.python.estimation.mosaic.outer_solve import (
    RegressorModel,
    TrialData,
)
from src.shared.python.estimation.mosaic.qualification_gates import (
    GateName,
    GateResult,
    GateStatus,
    check_closed_loop_effort,
    check_cross_trial_consistency,
    check_declared_discrepancy,
    check_kinematic_whiteness,
    check_observability,
    check_open_loop_replay,
    check_physical_consistency,
)
from src.shared.python.estimation.mosaic.subject_fit import (
    ReplayConfig,
    SubjectFitProblem,
    SubjectFitReport,
    fit_subject,
)

logger = logging.getLogger(__name__)

FloatArray: TypeAlias = npt.NDArray[np.float64]

__all__ = [
    "LeaveOneTrialOutConfig",
    "LeaveOneTrialOutReport",
    "TrialFoldReceipt",
    "run_leave_one_trial_out_study",
]


@dataclass(frozen=True)
class LeaveOneTrialOutConfig:
    """Configuration for reproducible leave-one-trial-out evaluation."""

    n_phase_bins: int = 15
    whiteness_threshold: float = 1.5
    min_explained_variance: float = 0.95
    seed: int = 42
    min_structural_rank: int = 9
    max_replay_rms_rad: float = 0.035
    max_closed_loop_effort_rms: float = 0.05
    basis_acceleration_error: float | None = None
    mass_scale: float | None = None
    benchmark_protocol: BenchmarkProtocol | None = None
    replay_config: ReplayConfig | None = None

    def __post_init__(self) -> None:
        require(self.whiteness_threshold > 0.0, "whiteness_threshold must be positive")
        require(
            0.0 < self.min_explained_variance <= 1.0,
            "min_explained_variance must be in (0, 1]",
        )
        require(self.min_structural_rank >= 1, "min_structural_rank must be >= 1")
        require(self.max_replay_rms_rad > 0.0, "max_replay_rms_rad must be positive")
        require(
            self.max_closed_loop_effort_rms > 0.0,
            "max_closed_loop_effort_rms must be positive",
        )


@dataclass(frozen=True)
class TrialFoldReceipt:
    """Outcome and receipts from a single leave-one-trial-out fold."""

    held_out_trial_index: int
    training_trial_indices: tuple[int, ...]
    tracking_rmse: float
    dynamics_residual_rms: float
    gate_results: tuple[GateResult, ...]
    gate_summary: dict[str, str]
    all_gates_passed: bool
    held_out_whiteness_status: str
    input_hash: str
    fitted_parameters: tuple[float, ...]
    fitted_geometry: tuple[float, ...]
    timings_s: dict[str, float]
    replay_rms: float | None = None

    @property
    def has_unqualified(self) -> bool:
        """Report whether any qualification gate is unqualified."""
        return any(g.status == GateStatus.UNQUALIFIED for g in self.gate_results)

    @property
    def is_fully_qualified(self) -> bool:
        """Report whether all qualification gates evaluated without being unqualified."""
        return not self.has_unqualified


@dataclass(frozen=True)
class LeaveOneTrialOutReport:
    """Aggregate multi-trial leave-one-trial-out evaluation report."""

    config: LeaveOneTrialOutConfig
    n_trials: int
    fold_receipts: tuple[TrialFoldReceipt, ...]
    mean_tracking_rmse: float
    std_tracking_rmse: float
    mean_dynamics_residual_rms: float
    std_dynamics_residual_rms: float
    shared_parameter_spread: dict[str, float]
    template_explained_variance: float | None
    all_folds_qualified: bool
    all_folds_passed: bool
    study_hash: str
    timestamp: str

    @property
    def n_folds(self) -> int:
        return len(self.fold_receipts)


def _compute_fold_hash(
    fold_idx: int,
    train_indices: tuple[int, ...],
    held_trial: TrialData,
    seed: int,
) -> str:
    """Compute reproducible sha256 hash for a fold without exposing raw markers."""
    require(fold_idx >= 0, "fold_idx must be non-negative")
    hasher = hashlib.sha256()
    hasher.update(str(fold_idx).encode("utf-8"))
    hasher.update(str(train_indices).encode("utf-8"))
    hasher.update(str(seed).encode("utf-8"))
    hasher.update(np.ascontiguousarray(held_trial.marker_positions).tobytes())
    hasher.update(np.ascontiguousarray(held_trial.times).tobytes())
    return hasher.hexdigest()


def _compute_study_hash(
    fold_hashes: tuple[str, ...],
    seed: int,
    n_phase_bins: int,
) -> str:
    """Compute overall study hash from fold receipts and configuration."""
    require(len(fold_hashes) >= 1, "at least one fold hash required")
    hasher = hashlib.sha256()
    for fh in fold_hashes:
        hasher.update(fh.encode("utf-8"))
    hasher.update(str(seed).encode("utf-8"))
    hasher.update(str(n_phase_bins).encode("utf-8"))
    return hasher.hexdigest()


def _fit_training_subject(
    problem: SubjectFitProblem,
    train_indices: tuple[int, ...],
    config: LeaveOneTrialOutConfig,
    replay_config: ReplayConfig | None,
) -> SubjectFitReport:
    """Fit shared parameters on the M-1 training trials."""
    trials = tuple(problem.trials[i] for i in train_indices)
    first_frames = tuple(problem.first_frame_q[i] for i in train_indices)
    train_problem = SubjectFitProblem(
        factory=problem.factory,
        trials=trials,
        first_frame_q=first_frames,
        initial_geometry=problem.initial_geometry,
        initial_parameters=problem.initial_parameters,
        parameterization=problem.parameterization,
        priors=problem.priors,
        options=problem.options,
        activation_model=problem.activation_model,
    )
    return fit_subject(train_problem, replay_config, config.n_phase_bins)


def _evaluate_held_out_kinematics(
    model: RegressorModel,
    trial: TrialData,
    seed_q: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Evaluate IK initialization and marker predictions on held-out trial."""
    q_ik = initialize_trajectory(model, trial.marker_positions, trial.markers, seed_q)
    basis = trial.basis
    c_held = basis.fit_least_squares(q_ik)
    q_eval, v_eval, a_eval = basis.evaluate(c_held)
    pred_markers = model.marker_positions(q_eval, trial.markers)
    return c_held, q_eval, v_eval, a_eval, pred_markers


def _compute_dynamics_residual_rms(
    model: RegressorModel,
    trial: TrialData,
    kinematics: tuple[FloatArray, FloatArray, FloatArray],
    problem: SubjectFitProblem,
    fixed_parameters: FloatArray,
) -> tuple[float, FloatArray]:
    """Compute held-out inverse dynamics residual and fitted input torques."""
    q_eval, v_eval, a_eval = kinematics
    regressors = model.regressor(q_eval, v_eval, a_eval)
    priors = problem.priors
    inner_prob = InnerProblem(
        regressors=regressors,
        input_matrix=model.input_matrix,
        trial_index=np.zeros(trial.n_times, dtype=np.int64),
        phase_index=trial.phase_index,
        dynamics_weight=1.0 / priors.dynamics_sigma,
        parameter_prior_mean=priors.parameter_prior_mean,
        parameter_prior_weight=priors.parameter_prior_weight,
        input_effort_weight=priors.input_effort_weight,
        input_smoothness_weight=priors.input_smoothness_weight,
        template_weight=0.0,
        anchors=(),
        fixed_parameters=fixed_parameters,
    )
    inner_sol = solve_inner(inner_prob)
    tau_id = np.einsum("tvp,p->tv", regressors, fixed_parameters)
    tau_u = inner_sol.inputs @ model.input_matrix.T
    diff = tau_id - tau_u
    residual_rms = float(np.sqrt(np.mean(diff**2)))
    return residual_rms, inner_sol.inputs


def _evaluate_held_out_replay(
    trial: TrialData,
    coefficients: FloatArray,
    inputs: FloatArray,
    geometry: FloatArray,
    parameters: FloatArray,
    config: ReplayConfig | None,
) -> tuple[ReplayReport | None, float | None]:
    """Optionally simulate open-loop and closed-loop replay on the held-out trial."""
    if config is None:
        return None, None
    q, v, _ = trial.basis.evaluate(coefficients)
    reference = np.concatenate([q, v], axis=1)
    steps = np.diff(trial.times)
    dt = float(steps[0])
    step = config.step_factory(geometry, parameters, dt)
    nominal_inputs = inputs[:-1]
    linearization = linearize_along_trajectory(
        step, reference[:-1], nominal_inputs, 1e-6
    )
    gains = tvlqr_gains(linearization, config.state_weight, config.input_weight)
    report = replay_acceptance(
        step,
        reference,
        nominal_inputs,
        gains,
        config.initial_perturbation,
        config.position_tolerance,
        dt,
    )
    return report, float(report.open_loop_rms)


def _evaluate_fold_qualification_gates(
    whiteness_metric: float | None,
    config: LeaveOneTrialOutConfig,
    train_report: SubjectFitReport,
    replay_report: ReplayReport | None,
    dynamics_sigma_mean: float,
) -> list[GateResult]:
    """Evaluate the 7 MOSAIC qualification gates on the fold."""
    fit = train_report.fit
    g1 = check_kinematic_whiteness(
        whiteness_metric=whiteness_metric, threshold=config.whiteness_threshold
    )
    sigma_tau = (
        dynamics_sigma_mean if config.basis_acceleration_error is not None else None
    )
    g2 = check_declared_discrepancy(
        sigma_tau=sigma_tau,
        basis_accel_error=config.basis_acceleration_error,
        mass_scale=config.mass_scale,
    )
    g3 = check_physical_consistency(fit.parameters)
    g4 = check_observability(
        train_report.observability, min_rank=config.min_structural_rank
    )
    replays = [replay_report] if replay_report is not None else None
    g5 = check_open_loop_replay(replays, max_rms_rad=config.max_replay_rms_rad)
    g6 = check_closed_loop_effort(
        replays, max_effort_rms=config.max_closed_loop_effort_rms
    )
    g7 = check_cross_trial_consistency(
        template=train_report.template,
        modes=train_report.modes,
        min_explained_variance=config.min_explained_variance,
    )
    results = [g1, g2, g3, g4, g5, g6, g7]
    ensure(len(results) == 7, "All seven qualification gates must be evaluated")
    return results


def _evaluate_single_fold(
    problem: SubjectFitProblem,
    fold_idx: int,
    config: LeaveOneTrialOutConfig,
    replay_config: ReplayConfig | None,
    dynamics_sigma_mean: float,
) -> TrialFoldReceipt:
    """Execute training and held-out evaluation for one fold."""
    n_trials = len(problem.trials)
    train_indices = tuple(i for i in range(n_trials) if i != fold_idx)
    started = time.perf_counter()
    train_report = _fit_training_subject(problem, train_indices, config, replay_config)
    train_fit = train_report.fit

    held_trial = problem.trials[fold_idx]
    held_seed = problem.first_frame_q[fold_idx]
    model = problem.factory(train_fit.geometry)
    c_held, q_eval, v_eval, a_eval, pred_markers = _evaluate_held_out_kinematics(
        model, held_trial, held_seed
    )
    marker_diff = pred_markers - held_trial.marker_positions
    tracking_rmse = float(np.sqrt(np.mean(marker_diff**2)))

    whiteness_res = evaluate_whiteness_gate(
        held_trial.marker_positions, pred_markers, held_trial.marker_sigma
    )
    whiteness_metric = (
        abs(whiteness_res.autocorrelation_lag1)
        if whiteness_res.status in ("pass", "fail")
        else None
    )

    kinematics = (q_eval, v_eval, a_eval)
    dyn_rms, held_inputs = _compute_dynamics_residual_rms(
        model, held_trial, kinematics, problem, train_fit.parameters
    )

    effective_replay_cfg = replay_config or config.replay_config
    replay_report, replay_rms = _evaluate_held_out_replay(
        held_trial,
        c_held,
        held_inputs,
        train_fit.geometry,
        train_fit.parameters,
        effective_replay_cfg,
    )

    gate_results = _evaluate_fold_qualification_gates(
        whiteness_metric,
        config,
        train_report,
        replay_report,
        dynamics_sigma_mean,
    )
    gate_summary = {g.name.value: g.status.value for g in gate_results}
    all_passed = all(g.status == GateStatus.PASSED for g in gate_results)
    fold_hash = _compute_fold_hash(fold_idx, train_indices, held_trial, config.seed)
    timings = dict(train_report.timings_s)
    timings["fold_total"] = time.perf_counter() - started

    return TrialFoldReceipt(
        held_out_trial_index=fold_idx,
        training_trial_indices=train_indices,
        tracking_rmse=tracking_rmse,
        dynamics_residual_rms=dyn_rms,
        gate_results=tuple(gate_results),
        gate_summary=gate_summary,
        all_gates_passed=all_passed,
        held_out_whiteness_status=whiteness_res.status,
        input_hash=fold_hash,
        fitted_parameters=tuple(float(p) for p in train_fit.parameters),
        fitted_geometry=tuple(float(g) for g in train_fit.geometry),
        timings_s=timings,
        replay_rms=replay_rms,
    )


def _summarize_spread(fold_receipts: Sequence[TrialFoldReceipt]) -> dict[str, float]:
    """Compute spread statistics of shared parameters across folds."""
    if not fold_receipts:
        return {}
    pi_matrix = np.array([r.fitted_parameters for r in fold_receipts], dtype=np.float64)
    geom_matrix = np.array([r.fitted_geometry for r in fold_receipts], dtype=np.float64)
    pi_std = (
        np.std(pi_matrix, axis=0)
        if len(fold_receipts) > 1
        else np.zeros(pi_matrix.shape[1])
    )
    geom_std = (
        np.std(geom_matrix, axis=0)
        if len(fold_receipts) > 1
        else np.zeros(geom_matrix.shape[1])
    )
    return {
        "parameters_std_norm": float(np.linalg.norm(pi_std)),
        "geometry_std_norm": float(np.linalg.norm(geom_std)),
        "parameters_max_std": float(np.max(pi_std)),
        "geometry_max_std": float(np.max(geom_std)),
    }


def run_leave_one_trial_out_study(
    problem: SubjectFitProblem,
    config: LeaveOneTrialOutConfig,
    replay_config: ReplayConfig | None = None,
) -> LeaveOneTrialOutReport:
    """Run reproducible leave-one-trial-out study across all trials."""
    n_trials = len(problem.trials)
    require(n_trials >= 2, "leave-one-trial-out requires at least 2 trials", n_trials)
    require(
        len(problem.first_frame_q) == n_trials, "first_frame_q length matches trials"
    )

    if config.benchmark_protocol is not None:
        verify_protocol_integrity(config.benchmark_protocol)

    priors = problem.priors
    dynamics_sigma_mean = float(np.mean(priors.dynamics_sigma))
    effective_replay = replay_config or config.replay_config

    receipts: list[TrialFoldReceipt] = []
    for fold_idx in range(n_trials):
        receipt = _evaluate_single_fold(
            problem, fold_idx, config, effective_replay, dynamics_sigma_mean
        )
        receipts.append(receipt)

    tracking_errors = [r.tracking_rmse for r in receipts]
    dynamics_residuals = [r.dynamics_residual_rms for r in receipts]
    spread_stats = _summarize_spread(receipts)
    fold_hashes = tuple(r.input_hash for r in receipts)
    study_hash = _compute_study_hash(fold_hashes, config.seed, config.n_phase_bins)
    all_qualified = all(r.is_fully_qualified for r in receipts)
    all_passed = all(r.all_gates_passed for r in receipts)

    return LeaveOneTrialOutReport(
        config=config,
        n_trials=n_trials,
        fold_receipts=tuple(receipts),
        mean_tracking_rmse=float(np.mean(tracking_errors)),
        std_tracking_rmse=float(np.std(tracking_errors)),
        mean_dynamics_residual_rms=float(np.mean(dynamics_residuals)),
        std_dynamics_residual_rms=float(np.std(dynamics_residuals)),
        shared_parameter_spread=spread_stats,
        template_explained_variance=None,
        all_folds_qualified=all_qualified,
        all_folds_passed=all_passed,
        study_hash=study_hash,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
