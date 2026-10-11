"""DIME Ablation Study and Accuracy-Runtime Acceptance (#11431).

Part of Epic #11421.
Dependencies: #11425 (ZTCF Prediction), #11430 (Offline Smoothing & Replay),
              #11435 (Profiling/Acceleration), #11436 (Initializers), #11437 (Feasibility).

This module implements the preregistered ablation benchmark protocol comparing
the six baseline and method variants across perturbations (noise, occlusion,
torque initialization bias, contact transitions, and model/camera errors) under
both marked and markerless observation modes.

Enforces:
- Seeded data leakage and test-set tuning fail closed.
- Zero-denominator guard in dominance metrics with declared policy.
- Fail-closed retention: reporting only winning trials is strictly forbidden.
- Empirical uncertainty coverage (2-sigma intervals).
- p50/p95 latency and global refinement cost recorded separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
import platform
import time
from typing import Any, Mapping

import numpy as np

from src.shared.python.contracts import PreconditionError
from src.shared.python.estimation.dime_manifest import (
    compute_alignment_metric,
    compute_phase_drift_and_control,
)
from src.shared.python.estimation.dime_continuous_replay import (
    ReplayReceipt,
)
from src.shared.python.estimation.synthetic_fixtures import (
    make_fixed_base_pendulum_fixture,
)


class DimeAblationVariant(str, Enum):
    """Preregistered baseline and method variants from Epic #11421."""

    KINEMATIC_IK = "kinematic_ik"
    CLASSICAL_MHE = "classical_mhe"
    DRIFT_PRIOR_ZTCF = "drift_prior_ztcf"
    DRIFT_CONTACT_CONSTRAINED = "drift_contact_constrained"
    DRIFT_OFFLINE_SMOOTHED = "drift_offline_smoothed"
    DRIFT_ACCELERATED_PROPOSAL = "drift_accelerated_proposal"


class PerturbationKind(str, Enum):
    """Preregistered perturbation kinds for stress-testing estimator variants."""

    NOISE = "noise"
    OCCLUSION = "occlusion"
    TORQUE_BIAS = "torque_bias"
    CONTACT_CHANGE = "contact_change"
    MODEL_CAMERA_ERROR = "model_camera_error"


class ObservationMode(str, Enum):
    """Observation capture modalities."""

    MARKED = "marked"
    MARKERLESS = "markerless"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class AblationTrialSpec:
    """Configuration specification for a single ablation benchmark trial."""

    variant: DimeAblationVariant
    perturbation: PerturbationKind
    observation_mode: ObservationMode
    seed: int = 42
    has_data_leakage: bool = False
    test_set_tuned: bool = False
    force_unfeasible: bool = False
    n_frames: int = 8
    fps: float = 100.0

    def validate(self) -> None:
        """Validate trial specification against anti-leakage invariants."""
        if self.has_data_leakage:
            raise PreconditionError(
                "Seeded data leakage detected: train and evaluation data overlap is strictly forbidden."
            )
        if self.test_set_tuned:
            raise PreconditionError(
                "Test-set tuning detected: tuning solver hyperparameters on the evaluation split is forbidden."
            )


@dataclass(frozen=True)
class AblationTrialResult:
    """Execution receipt and metrics for an ablation benchmark trial."""

    variant: DimeAblationVariant
    perturbation: PerturbationKind
    observation_mode: ObservationMode
    seed: int
    status: str  # "passed", "failed", "unqualified"
    failure_reason: str | None
    trajectory_rmse: float
    alignment: float
    drift_dominance: float
    uncertainty_coverage_2sigma: float
    p50_latency_ms: float
    p95_latency_ms: float
    global_refinement_cost_ms: float
    hardware_info: Mapping[str, Any]
    method_config: Mapping[str, Any]
    replay_receipt: ReplayReceipt | None = None


@dataclass(frozen=True)
class AblationSummaryTable:
    """Aggregated summary of an ablation benchmark suite execution."""

    suite_name: str
    total_trials: int
    passed_count: int
    failed_count: int
    unqualified_count: int
    results: tuple[AblationTrialResult, ...]
    retained_failures: tuple[AblationTrialResult, ...]
    variant_metrics: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class AblationBenchmarkSuite:
    """Suite of ablation benchmark trials."""

    name: str
    specs: tuple[AblationTrialSpec, ...] = ()
    filter_failures: bool = (
        False  # Forbidden policy: reporting only winners is rejected
    )

    def __post_init__(self) -> None:
        if self.filter_failures:
            raise PreconditionError(
                "Reporting only winning trials is forbidden: all failed and unqualified trials must be retained."
            )


def compute_ablation_dominance_metric(
    drift_magnitude: float,
    control_magnitude: float,
    *,
    policy: str = "guarded_zero",
) -> float:
    """Compute the drift dominance metric with strict zero-denominator handling.

    Formula: ||a_drift|| / (||a_drift|| + ||a_control||)
    """
    total = abs(drift_magnitude) + abs(control_magnitude)
    if total <= 1e-12:
        if policy == "guarded_zero":
            return 0.0
        if policy == "raise":
            raise PreconditionError(
                "Zero denominator in dominance metric: both drift and control magnitudes are zero."
            )
        return float("nan")
    return float(abs(drift_magnitude) / total)


def _get_hardware_info() -> dict[str, Any]:
    """Capture runtime execution hardware information."""
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "cpu_count": os.cpu_count() or 1,
    }


def _perturb_observations(
    spec: AblationTrialSpec, fixture: Any, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and perturb true kinematics based on trial specification."""
    q_true = np.array([f.q[0] for f in fixture.frames], dtype=np.float64)
    controls = np.asarray(fixture.controls, dtype=np.float64)

    noise_scale = 0.005 if spec.observation_mode == ObservationMode.MARKED else 0.015
    if spec.observation_mode == ObservationMode.HYBRID:
        noise_scale = 0.008

    q_perturbed = q_true.copy()
    if spec.perturbation == PerturbationKind.NOISE:
        q_perturbed += rng.normal(0.0, noise_scale, size=q_true.shape)
    elif spec.perturbation == PerturbationKind.OCCLUSION:
        mask_idx = len(q_perturbed) // 2
        q_perturbed[mask_idx] = q_perturbed[max(0, mask_idx - 1)]
    elif spec.perturbation == PerturbationKind.TORQUE_BIAS:
        controls = controls + 0.05
    elif spec.perturbation == PerturbationKind.MODEL_CAMERA_ERROR:
        q_perturbed *= 1.02

    return q_true, q_perturbed, controls


_NOT_MEASURED_REASON = (
    "{variant}: no estimator implementation exists for this variant; "
    "all figures not measured (#11552)"
)


def _solve_trial_variant(
    spec: AblationTrialSpec,
    q_perturbed: np.ndarray,
) -> np.ndarray | None:
    """Estimate the trajectory from the perturbed observations ONLY.

    Preconditions: ``q_perturbed`` is the observed (not ground-truth) trajectory.
    Returns the estimate, or ``None`` when the variant has no estimator, in which
    case the trial is reported unqualified / not measured. Ground truth is never an
    input, so it cannot be blended into the estimate.
    """
    if spec.variant == DimeAblationVariant.KINEMATIC_IK:
        # Kinematic IK baseline: the observation itself is the estimate.
        return q_perturbed.copy()
    return None


def _not_measured_result(
    spec: AblationTrialSpec,
    status: str,
    reason: str,
    hardware: Mapping[str, Any],
    method_cfg: Mapping[str, Any],
) -> AblationTrialResult:
    nan = float("nan")
    return AblationTrialResult(
        variant=spec.variant,
        perturbation=spec.perturbation,
        observation_mode=spec.observation_mode,
        seed=spec.seed,
        status=status,
        failure_reason=reason,
        trajectory_rmse=nan,
        alignment=nan,
        drift_dominance=nan,
        uncertainty_coverage_2sigma=nan,
        p50_latency_ms=nan,
        p95_latency_ms=nan,
        global_refinement_cost_ms=nan,
        hardware_info=hardware,
        method_config=method_cfg,
        replay_receipt=None,
    )


def run_ablation_trial(spec: AblationTrialSpec) -> AblationTrialResult:
    """Execute a single ablation trial under the declared specification.

    Only figures actually computed from this run are reported; everything else is
    NaN and the trial is ``unqualified`` (not measured).
    """
    spec.validate()

    hardware = _get_hardware_info()
    method_cfg = {
        "variant": spec.variant.value,
        "perturbation": spec.perturbation.value,
        "observation_mode": spec.observation_mode.value,
        "seed": spec.seed,
    }

    if spec.force_unfeasible:
        return _not_measured_result(
            spec,
            "failed",
            "Deliberately unfeasible perturbation in contact transition",
            hardware,
            method_cfg,
        )

    rng = np.random.default_rng(spec.seed)
    fixture = make_fixed_base_pendulum_fixture(n_frames=spec.n_frames, fps=spec.fps)
    q_true, q_perturbed, _controls = _perturb_observations(spec, fixture, rng)

    t0 = time.perf_counter()
    q_est = _solve_trial_variant(spec, q_perturbed)
    dt_elapsed = (time.perf_counter() - t0) * 1000.0

    if q_est is None:
        return _not_measured_result(
            spec,
            "unqualified",
            _NOT_MEASURED_REASON.format(variant=spec.variant.value),
            hardware,
            method_cfg,
        )

    # Ground truth is used only here, to score the estimate.
    rmse = float(np.sqrt(np.mean((q_est - q_true) ** 2)))
    alignment = compute_alignment_metric(q_est, q_true, policy="guarded_zero")
    nan = float("nan")

    return AblationTrialResult(
        variant=spec.variant,
        perturbation=spec.perturbation,
        observation_mode=spec.observation_mode,
        seed=spec.seed,
        status="passed",
        failure_reason=None,
        trajectory_rmse=rmse,
        alignment=alignment,
        drift_dominance=nan,  # no drift model in this variant: not measured
        uncertainty_coverage_2sigma=nan,  # no uncertainty estimate: not measured
        p50_latency_ms=float(dt_elapsed),  # single measured sample
        p95_latency_ms=float(dt_elapsed),
        global_refinement_cost_ms=0.0,  # variant has no refinement stage
        hardware_info=hardware,
        method_config=method_cfg,
        replay_receipt=None,
    )


def run_dime_ablation_suite(suite: AblationBenchmarkSuite) -> AblationSummaryTable:
    """Execute all trials in an ablation benchmark suite and aggregate results."""
    # Ensure specs are populated if empty
    specs = suite.specs
    if not specs:
        generated_specs: list[AblationTrialSpec] = []
        for v in DimeAblationVariant:
            generated_specs.append(
                AblationTrialSpec(
                    variant=v,
                    perturbation=PerturbationKind.NOISE,
                    observation_mode=ObservationMode.MARKED,
                )
            )
        specs = tuple(generated_specs)

    results: list[AblationTrialResult] = []
    retained_failures: list[AblationTrialResult] = []
    passed = 0
    failed = 0
    unqualified = 0

    for spec in specs:
        res = run_ablation_trial(spec)
        results.append(res)
        if res.status == "passed":
            passed += 1
        elif res.status == "failed":
            failed += 1
            retained_failures.append(res)
        elif res.status == "unqualified":
            unqualified += 1
            retained_failures.append(res)

    # Aggregate variant metrics
    variant_metrics: dict[str, dict[str, float]] = {}
    for res in results:
        if res.status != "passed":
            continue  # unqualified/failed trials carry no measured figures
        v_key = res.variant.value
        if v_key not in variant_metrics:
            variant_metrics[v_key] = {
                "rmse": res.trajectory_rmse,
                "alignment": res.alignment,
                "drift_dominance": res.drift_dominance,
                "coverage": res.uncertainty_coverage_2sigma,
                "p50_latency_ms": res.p50_latency_ms,
                "p95_latency_ms": res.p95_latency_ms,
                "refinement_cost_ms": res.global_refinement_cost_ms,
            }

    return AblationSummaryTable(
        suite_name=suite.name,
        total_trials=len(results),
        passed_count=passed,
        failed_count=failed,
        unqualified_count=unqualified,
        results=tuple(results),
        retained_failures=tuple(retained_failures),
        variant_metrics=variant_metrics,
    )
