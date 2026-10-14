"""Tests for MOSAIC leave-one-trial-out (LOTO) cross-validation study (MOSAIC-09, #11541)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.mosaic.benchmark_protocol import (
    BenchmarkProtocol,
    FixtureConfig,
    ToleranceConfig,
)
from src.shared.python.estimation.mosaic.inertial import (
    PlanarParameterization,
    project_planar_consistent,
)
from src.shared.python.estimation.mosaic.leave_one_trial_out import (
    LeaveOneTrialOutConfig,
    LeaveOneTrialOutReport,
    TrialFoldReceipt,
    run_leave_one_trial_out_study,
)
from src.shared.python.estimation.mosaic.outer_solve import OuterOptions
from src.shared.python.estimation.mosaic.qualification_gates import (
    GateName,
    GateStatus,
)
from src.shared.python.estimation.mosaic.subject_fit import (
    ReplayConfig,
    SubjectFitProblem,
)
from tests.unit.estimation.mosaic.test_outer_solve import (
    TRUE_LENGTHS,
    TRUE_PI,
    _factory,
    _priors,
    _synthetic_trials,
)

pytestmark = pytest.mark.unit


def _step_factory(geometry: np.ndarray, parameters: np.ndarray, dt: float):
    chain = _factory(geometry)

    def step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return chain.step(x, u, parameters, dt)

    return step


def _make_problem(
    n_trials: int = 2, n_nodes: int = 80, seed: int = 8
) -> SubjectFitProblem:
    trials, truth = _synthetic_trials(
        n_trials=n_trials, n_nodes=n_nodes, noise=5e-4, seed=seed
    )
    rng = np.random.default_rng(2)
    pi_prior = TRUE_PI * (1.0 + rng.uniform(-0.1, 0.1, size=TRUE_PI.size))
    pi_prior[2::4] = 0.0
    pi_prior = project_planar_consistent(
        pi_prior.reshape(-1, 4) * [1, 1, 1, 1.1]
    ).ravel()

    return SubjectFitProblem(
        factory=_factory,
        trials=tuple(trials),
        first_frame_q=tuple(t[0][0] + 0.1 for t in truth),
        initial_geometry=TRUE_LENGTHS * 1.05,
        initial_parameters=pi_prior,
        parameterization=PlanarParameterization(),
        priors=_priors(pi_prior, True),
        options=OuterOptions(
            continuation=(0.0, 0.01, 0.1), max_iterations_per_stage=15
        ),
    )


def test_leave_one_trial_out_preconditions_and_contracts() -> None:
    # Requires at least 2 trials for cross-validation
    problem_1_trial = _make_problem(n_trials=2, n_nodes=50)
    single_trial_problem = SubjectFitProblem(
        factory=problem_1_trial.factory,
        trials=(problem_1_trial.trials[0],),
        first_frame_q=(problem_1_trial.first_frame_q[0],),
        initial_geometry=problem_1_trial.initial_geometry,
        initial_parameters=problem_1_trial.initial_parameters,
        parameterization=problem_1_trial.parameterization,
        priors=problem_1_trial.priors,
        options=problem_1_trial.options,
    )
    config = LeaveOneTrialOutConfig()
    with pytest.raises(ContractViolationError):
        run_leave_one_trial_out_study(single_trial_problem, config)

    # Invalid config parameters raise contract violations
    with pytest.raises(ContractViolationError):
        LeaveOneTrialOutConfig(whiteness_threshold=-1.0)
    with pytest.raises(ContractViolationError):
        LeaveOneTrialOutConfig(min_explained_variance=0.0)


def test_leave_one_trial_out_synthetic_cross_validation() -> None:
    problem = _make_problem(n_trials=2, n_nodes=60, seed=8)
    config = LeaveOneTrialOutConfig(
        n_phase_bins=10,
        whiteness_threshold=2.0,
        min_explained_variance=0.90,
        seed=42,
    )

    report = run_leave_one_trial_out_study(problem, config)

    assert isinstance(report, LeaveOneTrialOutReport)
    assert report.n_trials == 2
    assert report.n_folds == 2
    assert len(report.fold_receipts) == 2

    fold_0 = report.fold_receipts[0]
    assert fold_0.held_out_trial_index == 0
    assert fold_0.training_trial_indices == (1,)
    assert fold_0.tracking_rmse > 0.0
    assert fold_0.dynamics_residual_rms > 0.0
    assert len(fold_0.gate_results) == 7
    assert fold_0.input_hash != ""

    fold_1 = report.fold_receipts[1]
    assert fold_1.held_out_trial_index == 1
    assert fold_1.training_trial_indices == (0,)
    assert fold_1.tracking_rmse > 0.0
    assert fold_1.dynamics_residual_rms > 0.0

    assert report.mean_tracking_rmse > 0.0
    assert report.mean_dynamics_residual_rms > 0.0
    assert "parameters_std_norm" in report.shared_parameter_spread
    assert "geometry_std_norm" in report.shared_parameter_spread
    assert report.study_hash != ""


def test_deterministic_seed_and_hash_receipts() -> None:
    problem = _make_problem(n_trials=2, n_nodes=50, seed=8)
    config1 = LeaveOneTrialOutConfig(seed=12345, n_phase_bins=8)
    config2 = LeaveOneTrialOutConfig(seed=12345, n_phase_bins=8)

    report1 = run_leave_one_trial_out_study(problem, config1)
    report2 = run_leave_one_trial_out_study(problem, config2)

    assert report1.study_hash == report2.study_hash
    for f1, f2 in zip(report1.fold_receipts, report2.fold_receipts, strict=True):
        assert f1.input_hash == f2.input_hash
        assert np.isclose(f1.tracking_rmse, f2.tracking_rmse)
        assert np.isclose(f1.dynamics_residual_rms, f2.dynamics_residual_rms)

    # Different seed produces different study hash
    config_diff_seed = LeaveOneTrialOutConfig(seed=99999, n_phase_bins=8)
    report_diff = run_leave_one_trial_out_study(problem, config_diff_seed)
    assert report_diff.study_hash != report1.study_hash


def test_unqualified_reporting_when_gate_cannot_be_evaluated() -> None:
    # When replay_config and basis_acceleration_error are None, corresponding gates MUST report unqualified
    problem = _make_problem(n_trials=2, n_nodes=60, seed=8)
    config = LeaveOneTrialOutConfig(
        basis_acceleration_error=None,  # Unqualified Gate 2
        mass_scale=None,
        replay_config=None,  # Unqualified Gate 5 & Gate 6
    )

    report = run_leave_one_trial_out_study(problem, config, replay_config=None)

    for fold in report.fold_receipts:
        assert fold.has_unqualified
        assert not fold.is_fully_qualified
        assert (
            fold.gate_summary[GateName.DECLARED_DISCREPANCY.value]
            == GateStatus.UNQUALIFIED.value
        )
        assert (
            fold.gate_summary[GateName.OPEN_LOOP_REPLAY.value]
            == GateStatus.UNQUALIFIED.value
        )
        assert fold.gate_summary[GateName.PHYSICAL_CONSISTENCY.value] in (
            GateStatus.PASSED.value,
            GateStatus.FAILED.value,
        )
        assert (
            fold.gate_summary[GateName.PHYSICAL_CONSISTENCY.value]
            != GateStatus.UNQUALIFIED.value
        )

    assert not report.all_folds_qualified
    assert not report.all_folds_passed


def test_full_qualification_with_replay_and_protocol() -> None:
    problem = _make_problem(n_trials=2, n_nodes=60, seed=8)
    replay_cfg = ReplayConfig(
        step_factory=_step_factory,
        state_weight=np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0]),
        input_weight=np.eye(2) * 0.01,
        position_tolerance=0.15,
        initial_perturbation=np.zeros(6),
    )
    protocol = BenchmarkProtocol(
        fixture=FixtureConfig(n_trials=2, n_nodes=60),
        tolerances=ToleranceConfig(),
        anchors={},
        promotion_metrics={},
    )
    config = LeaveOneTrialOutConfig(
        basis_acceleration_error=0.01,
        mass_scale=2.3,
        whiteness_threshold=2.5,
        min_structural_rank=4,
        max_replay_rms_rad=0.15,
        max_closed_loop_effort_rms=0.10,
        benchmark_protocol=protocol,
        replay_config=replay_cfg,
    )

    report = run_leave_one_trial_out_study(problem, config, replay_config=replay_cfg)

    fold = report.fold_receipts[0]
    assert (
        fold.gate_summary[GateName.PHYSICAL_CONSISTENCY.value]
        == GateStatus.PASSED.value
    )
    assert fold.gate_summary[GateName.OBSERVABILITY.value] == GateStatus.PASSED.value
    assert (
        fold.gate_summary[GateName.DECLARED_DISCREPANCY.value]
        == GateStatus.PASSED.value
    )
    assert (
        fold.gate_summary[GateName.OPEN_LOOP_REPLAY.value]
        != GateStatus.UNQUALIFIED.value
    )
    assert (
        fold.gate_summary[GateName.CLOSED_LOOP_EFFORT.value]
        != GateStatus.UNQUALIFIED.value
    )
    assert fold.replay_rms is not None
    assert fold.replay_rms > 0.0
    assert fold.is_fully_qualified


def test_benchmark_protocol_widened_tolerance_requires_evidence() -> None:
    problem = _make_problem(n_trials=2, n_nodes=50, seed=8)
    # Loosened tolerance without trailer
    loosened_tol = ToleranceConfig(open_loop_replay_rms_max_rad=0.05)
    bad_protocol = BenchmarkProtocol(
        fixture=FixtureConfig(),
        tolerances=loosened_tol,
        anchors={},
        promotion_metrics={},
        evidence_trailer=None,
    )
    config = LeaveOneTrialOutConfig(benchmark_protocol=bad_protocol)

    with pytest.raises(
        AssertionError, match="Tolerance-Change-Evidence trailer required"
    ):
        run_leave_one_trial_out_study(problem, config)
