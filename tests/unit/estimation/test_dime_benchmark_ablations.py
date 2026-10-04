"""Focused behavioral tests for DIME ablation study and accuracy-runtime acceptance (#11431).

Parent: #11421
Dependencies: #11425, #11430, #11435, #11436, #11437

Enforces:
- RED:
  * Seeded data leakage fails closed (train/test overlap or explicit leak flag rejected).
  * Test-set tuning fails closed (tuning hyperparameters on evaluation split rejected).
  * Zero denominator in dominance metric guarded according to policy.
  * Reporting only winning trials fails closed (every trial/failure must be retained).
- GREEN:
  * Six baseline/method variants evaluated across perturbations:
    noise, occlusion, torque initialization bias, contact changes, model/camera error.
  * Marked and markerless observation modes supported.
  * Deterministic reproduction with recorded seeds.
  * All trials, failures, and unqualified statuses retained in summary tables.
  * Uncertainty coverage and error/runtime metrics generated.
  * p50/p95 latency and global-refinement cost recorded separately.
  * Headless import guard verified.
"""

from __future__ import annotations

import sys
import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.estimation.dime_benchmark_ablations import (
    AblationBenchmarkSuite,
    AblationSummaryTable,
    AblationTrialResult,
    AblationTrialSpec,
    DimeAblationVariant,
    ObservationMode,
    PerturbationKind,
    compute_ablation_dominance_metric,
    run_ablation_trial,
    run_dime_ablation_suite,
)
from src.shared.python.estimation.dime_manifest import (
    NumericAcceptanceThresholds,
)
from src.shared.python.estimation.synthetic_fixtures import (
    make_fixed_base_pendulum_fixture,
    make_underactuated_analytic_fixture,
)

pytestmark = pytest.mark.unit


# =============================================================================
# RED Cases: Contract Enforcement & Defensive Guards
# =============================================================================


def test_six_ablation_variants_defined_and_distinct() -> None:
    """The parent epic preregisters exactly 6 baseline/method variants."""
    expected_variants = {
        "kinematic_ik",
        "classical_mhe",
        "drift_prior_ztcf",
        "drift_contact_constrained",
        "drift_offline_smoothed",
        "drift_accelerated_proposal",
    }
    actual_variants = {v.value for v in DimeAblationVariant}
    assert actual_variants == expected_variants, (
        f"Mismatch in ablation variants: expected {expected_variants}, got {actual_variants}"
    )


def test_perturbation_and_observation_modes() -> None:
    """Verify preregistered perturbation kinds and observation modes."""
    expected_perturbations = {
        "noise",
        "occlusion",
        "torque_bias",
        "contact_change",
        "model_camera_error",
    }
    actual_perturbations = {p.value for p in PerturbationKind}
    assert actual_perturbations == expected_perturbations

    expected_modes = {"marked", "markerless", "hybrid"}
    actual_modes = {m.value for m in ObservationMode}
    assert actual_modes == expected_modes


def test_red_seeded_data_leakage_rejected() -> None:
    """A trial with data leakage must fail closed before execution."""
    spec = AblationTrialSpec(
        variant=DimeAblationVariant.DRIFT_PRIOR_ZTCF,
        perturbation=PerturbationKind.NOISE,
        observation_mode=ObservationMode.MARKED,
        has_data_leakage=True,
    )
    with pytest.raises(PreconditionError, match="(?i)data leakage"):
        run_ablation_trial(spec)


def test_red_test_set_tuning_rejected() -> None:
    """Hyperparameter tuning directly on test-set data must fail closed."""
    spec = AblationTrialSpec(
        variant=DimeAblationVariant.DRIFT_OFFLINE_SMOOTHED,
        perturbation=PerturbationKind.NOISE,
        observation_mode=ObservationMode.MARKED,
        test_set_tuned=True,
    )
    with pytest.raises(PreconditionError, match="(?i)test-set tuning"):
        run_ablation_trial(spec)


def test_red_zero_denominator_in_dominance_metric() -> None:
    """Zero denominator in dominance metric must follow the declared policy."""
    # When both drift and control are zero, guarded_zero must return 0.0 without ZeroDivisionError
    metric_val = compute_ablation_dominance_metric(
        drift_magnitude=0.0,
        control_magnitude=0.0,
        policy="guarded_zero",
    )
    assert metric_val == 0.0

    # Under 'raise' policy, zero denominator must raise PreconditionError
    with pytest.raises(PreconditionError, match="(?i)zero denominator"):
        compute_ablation_dominance_metric(
            drift_magnitude=0.0,
            control_magnitude=0.0,
            policy="raise",
        )


def test_red_reporting_only_winning_trials_rejected() -> None:
    """Filtering out failures or reporting only winning trials is forbidden."""
    with pytest.raises(PreconditionError, match="(?i)winning trials|retained"):
        AblationBenchmarkSuite(
            name="test_suite",
            filter_failures=True,  # Forbidden policy
        )


# =============================================================================
# GREEN Cases: Execution, Accuracy-Runtime Acceptance & Metrics
# =============================================================================


def test_green_deterministic_manifest_reproduction() -> None:
    """Ablation trials must reproduce identically across runs with identical seed."""
    spec = AblationTrialSpec(
        variant=DimeAblationVariant.DRIFT_PRIOR_ZTCF,
        perturbation=PerturbationKind.NOISE,
        observation_mode=ObservationMode.MARKED,
        seed=1337,
    )
    res1 = run_ablation_trial(spec)
    res2 = run_ablation_trial(spec)

    assert res1.status == "passed"
    assert res2.status == "passed"
    assert res1.trajectory_rmse == pytest.approx(res2.trajectory_rmse, rel=1e-12)
    assert res1.alignment == pytest.approx(res2.alignment, rel=1e-12)
    assert res1.drift_dominance == pytest.approx(res2.drift_dominance, rel=1e-12)


def test_green_all_trials_and_failures_retained() -> None:
    """Every trial and failure must be retained in the summary table with explicit reasons."""
    specs = [
        AblationTrialSpec(
            variant=DimeAblationVariant.KINEMATIC_IK,
            perturbation=PerturbationKind.NOISE,
            observation_mode=ObservationMode.MARKED,
            seed=42,
        ),
        # Deliberately unfeasible perturbation causing qualification failure
        AblationTrialSpec(
            variant=DimeAblationVariant.DRIFT_CONTACT_CONSTRAINED,
            perturbation=PerturbationKind.CONTACT_CHANGE,
            observation_mode=ObservationMode.MARKERLESS,
            force_unfeasible=True,
            seed=43,
        ),
    ]
    suite = AblationBenchmarkSuite(name="retention_suite", specs=tuple(specs))
    summary: AblationSummaryTable = run_dime_ablation_suite(suite)

    assert summary.total_trials == 2
    assert summary.passed_count == 1
    assert summary.failed_count == 1
    assert len(summary.retained_failures) == 1
    assert (
        summary.retained_failures[0].variant
        == DimeAblationVariant.DRIFT_CONTACT_CONSTRAINED
    )
    assert "unfeasible" in summary.retained_failures[0].failure_reason.lower()


def test_green_uncertainty_coverage_computed() -> None:
    """Uncertainty coverage fraction within declared confidence intervals."""
    spec = AblationTrialSpec(
        variant=DimeAblationVariant.DRIFT_PRIOR_ZTCF,
        perturbation=PerturbationKind.NOISE,
        observation_mode=ObservationMode.HYBRID,
        seed=42,
    )
    res = run_ablation_trial(spec)
    assert 0.0 <= res.uncertainty_coverage_2sigma <= 1.0
    assert res.uncertainty_coverage_2sigma >= 0.85  # Well-calibrated 2-sigma coverage


def test_green_latency_and_refinement_cost_recorded_separately() -> None:
    """p50/p95 latency and global refinement cost must be separately reported."""
    spec = AblationTrialSpec(
        variant=DimeAblationVariant.DRIFT_OFFLINE_SMOOTHED,
        perturbation=PerturbationKind.TORQUE_BIAS,
        observation_mode=ObservationMode.MARKED,
        seed=42,
    )
    res = run_ablation_trial(spec)
    assert res.p50_latency_ms > 0.0
    assert res.p95_latency_ms >= res.p50_latency_ms
    assert res.global_refinement_cost_ms >= 0.0


def test_green_six_variants_benchmark_execution() -> None:
    """All six variants must execute across all perturbation kinds and observation modes."""
    specs = []
    perturbations = list(PerturbationKind)
    modes = list(ObservationMode)
    for i, variant in enumerate(DimeAblationVariant):
        p = perturbations[i % len(perturbations)]
        m = modes[i % len(modes)]
        specs.append(
            AblationTrialSpec(
                variant=variant,
                perturbation=p,
                observation_mode=m,
                seed=100 + i,
            )
        )
    suite = AblationBenchmarkSuite(name="full_matrix", specs=tuple(specs))
    summary = run_dime_ablation_suite(suite)

    assert summary.total_trials == 6
    assert summary.passed_count == 6
    variants_evaluated = {r.variant for r in summary.results}
    assert variants_evaluated == set(DimeAblationVariant)


def test_headless_import_guard() -> None:
    """Module must be importable in headless environment without GUI dependencies."""
    import subprocess
    from pathlib import Path

    script = """
import sys

BANNED_PREFIXES = (
    "PyQt6",
    "matplotlib.pyplot",
    "mujoco",
    "pydrake",
    "pinocchio",
    "opensim",
)

class HeadlessImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        for banned in BANNED_PREFIXES:
            if fullname == banned or fullname.startswith(banned + "."):
                raise ImportError(f"Banned package imported: {fullname}")
        return None

sys.meta_path.insert(0, HeadlessImportGuard())

import src.shared.python.estimation.dime_benchmark_ablations as dba

for mod in sys.modules:
    for banned in BANNED_PREFIXES:
        assert not (mod == banned or mod.startswith(banned + ".")), f"Banned module in sys.modules: {mod}"

print("HEADLESS_IMPORT_OK")
"""
    import os

    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{repo_root / 'src'}"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"Headless import failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "HEADLESS_IMPORT_OK" in result.stdout
