"""F04 bounded loop tuning and descriptive coupling on known systems."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.motion_matching.distributed_feedback import (
    ActuatorMap,
    ControlTask,
    DirectBoundedAllocator,
    DistributedFeedbackController,
    EuclideanTangentModel,
    NominalTrajectory,
    TaskGroup,
    TaskKinematics,
)
from src.shared.python.motion_matching.control_loop_tuning import (
    LoopParameter,
    LoopTuningConfig,
    LoopTuningProblem,
    TuningEvaluation,
    TuningTrial,
    compare_perturbation_compensation,
    descriptive_phase_covariance,
    frozen_coupling_diagnostics,
    tune_control_loops,
)

pytestmark = pytest.mark.unit


def _quadratic_problem(*, coupled: bool = True) -> LoopTuningProblem:
    parameters = (
        LoopParameter("pelvis_gain", "pelvis", "gain", 0.2, 0.0, 4.0, 1.0),
        LoopParameter("club_gain", "club", "gain", 0.2, 0.0, 4.0, 1.0),
    )
    trials = (
        TuningTrial("train-a", "train"),
        TuningTrial("train-b", "train"),
        TuningTrial("holdout-a", "holdout"),
    )

    def evaluate(
        values: np.ndarray, trial: TuningTrial, active: tuple[str, ...]
    ) -> TuningEvaluation:
        a, b = values
        interaction = 0.4 * (a - 1.0) * (b - 1.5) if coupled else 0.0
        pelvis = (a - 1.0) ** 2 + interaction + 0.1
        club = (b - 1.5) ** 2 + interaction + 0.1
        if trial.split == "holdout":
            pelvis += 0.03 * (a - 1.2) ** 2
            club += 0.03 * (b - 1.3) ** 2
        losses = np.array([[pelvis, club], [pelvis + 0.02, club + 0.03]])
        if active != ("pelvis", "club"):
            losses[
                :,
                [i for i, name in enumerate(("pelvis", "club")) if name not in active],
            ] += 0.4
        return TuningEvaluation(
            losses,
            effort=0.01 * (a * a + b * b),
            robustness=0.0,
            constraint_violation=0.0,
            saturation_fraction=0.0,
        )

    return LoopTuningProblem(
        parameters=parameters,
        groups=("pelvis", "club"),
        phases=("backswing", "downswing"),
        trials=trials,
        evaluate=evaluate,
        effort_weight=0.01,
        robustness_weight=0.0,
        regularization_weight=0.001,
    )


def test_block_then_joint_refinement_is_bounded_and_reproducible() -> None:
    problem = _quadratic_problem()
    config = LoopTuningConfig(
        max_passes=3,
        block_max_evaluations=40,
        joint_max_evaluations=60,
        trust_radius_scaled=2.0,
        max_cross_group_regression=0.3,
        max_constraint_violation=0.0,
        seed=17,
    )
    first = tune_control_loops(problem, config)
    again = tune_control_loops(problem, config)
    assert first.objective < first.initial_objective
    assert all(0.0 <= value <= 4.0 for value in first.parameters)
    assert any(checkpoint.stage == "joint" for checkpoint in first.checkpoints)
    np.testing.assert_allclose(first.parameters, again.parameters, atol=1e-10)
    assert first.checkpoint_digest == again.checkpoint_digest
    assert first.generalization_status == "multi_trial_descriptive"
    assert first.phase_covariance is not None
    assert first.phase_covariance.causal_claim is False
    assert first.holdout_full.tracking_rmse < first.holdout_reduced[0].tracking_rmse
    assert first.holdout_full.wall_seconds > 0
    assert all(score.wall_seconds > 0 for score in first.holdout_reduced)


def test_cross_block_regression_and_saturation_are_reported() -> None:
    base = _quadratic_problem()

    def adversarial(
        values: np.ndarray, trial: TuningTrial, active: tuple[str, ...]
    ) -> TuningEvaluation:
        a, b = values
        return TuningEvaluation(
            np.array(
                [
                    [(a - 2.0) ** 2, a * a + (b - 0.2) ** 2],
                    [(a - 2.0) ** 2, a * a + (b - 0.2) ** 2],
                ]
            ),
            effort=0.0,
            robustness=0.0,
            constraint_violation=0.0,
            saturation_fraction=min(1.0, float(a) / 4.0),
        )

    problem = replace(base, evaluate=adversarial)
    result = tune_control_loops(
        problem,
        LoopTuningConfig(
            max_passes=1,
            block_max_evaluations=30,
            joint_max_evaluations=10,
            trust_radius_scaled=2.0,
            max_cross_group_regression=0.01,
            max_constraint_violation=0.0,
            seed=3,
        ),
    )
    assert any(
        checkpoint.reason == "cross_group_regression"
        for checkpoint in result.checkpoints
    )
    assert any(checkpoint.saturation_fraction > 0 for checkpoint in result.checkpoints)


def test_critical_phase_regression_blocks_aggregate_improvement() -> None:
    base = _quadratic_problem(coupled=False)

    def phase_tradeoff(
        values: np.ndarray, trial: TuningTrial, active: tuple[str, ...]
    ) -> TuningEvaluation:
        gain = values[0]
        return TuningEvaluation(
            np.array([[100.0 * (gain - 2.0) ** 2, 0.0], [10.0 * gain * gain, 0.0]]),
            effort=0.0,
            robustness=0.0,
            constraint_violation=0.0,
            saturation_fraction=0.0,
        )

    result = tune_control_loops(
        replace(base, evaluate=phase_tradeoff),
        LoopTuningConfig(
            max_passes=1,
            block_max_evaluations=30,
            joint_max_evaluations=30,
            trust_radius_scaled=2.0,
            max_cross_group_regression=100.0,
            max_phase_group_regression=0.1,
            max_constraint_violation=0.0,
        ),
    )
    assert any(item.reason == "phase_group_regression" for item in result.checkpoints)


def test_constraint_regression_and_nonconvergence_are_retained() -> None:
    base = _quadratic_problem(coupled=False)

    def bounded(
        values: np.ndarray, trial: TuningTrial, active: tuple[str, ...]
    ) -> TuningEvaluation:
        a, b = values
        losses = np.array([[(a - 1.0) ** 2, (b - 1.5) ** 2]] * 2)
        return TuningEvaluation(losses, 0.0, 0.0, max(0.0, float(a) - 0.2), 0.0)

    problem = replace(base, evaluate=bounded)
    constrained = tune_control_loops(
        problem,
        LoopTuningConfig(
            max_passes=1,
            block_max_evaluations=30,
            joint_max_evaluations=30,
            trust_radius_scaled=2.0,
            max_cross_group_regression=1.0,
            max_constraint_violation=0.0,
            seed=2,
        ),
    )
    assert any(
        item.reason == "constraint_regression" for item in constrained.checkpoints
    )
    assert all(
        item.constraint_violation == 0.0
        for item in (constrained.holdout_full, *constrained.holdout_reduced)
    )
    exhausted = tune_control_loops(
        base,
        LoopTuningConfig(
            max_passes=1,
            block_max_evaluations=1,
            joint_max_evaluations=1,
            trust_radius_scaled=2.0,
            max_cross_group_regression=1.0,
            max_constraint_violation=0.0,
            seed=2,
        ),
    )
    assert any(item.reason == "budget_exhausted" for item in exhausted.checkpoints)
    assert all(item.evaluations <= 1 for item in exhausted.checkpoints)


def test_scaled_frozen_cross_sensitivity_separates_coupled_and_uncoupled() -> None:
    coupled = frozen_coupling_diagnostics(_quadratic_problem(), np.array([1.3, 1.1]))
    uncoupled = frozen_coupling_diagnostics(
        _quadratic_problem(coupled=False), np.array([1.3, 1.1])
    )
    assert coupled.jacobian.shape == (2, 2, 2)
    assert abs(coupled.jacobian[0, 0, 1]) > 0.05
    assert abs(uncoupled.jacobian[0, 0, 1]) < 1e-5
    assert coupled.cross_hessian.shape == (2, 2)
    assert coupled.cross_hessian[0, 1] == pytest.approx(0.8, abs=1e-4)
    assert uncoupled.cross_hessian[0, 1] == pytest.approx(0.0, abs=1e-4)
    assert coupled.interpretation == "frozen_controller_association_not_causation"


def test_phase_confounded_covariance_does_not_claim_causality() -> None:
    # Within each phase the coordinates are uncorrelated; phase means shift both.
    observations = np.array(
        [
            [[-1.0, 1.0], [9.0, 11.0]],
            [[1.0, -1.0], [11.0, 9.0]],
            [[-1.0, -1.0], [9.0, 9.0]],
            [[1.0, 1.0], [11.0, 11.0]],
        ]
    )
    report = descriptive_phase_covariance(
        observations, ("backswing", "downswing"), ("pelvis", "club")
    )
    assert report.pooled_covariance[0, 1] > 10.0
    assert abs(report.within_phase_covariance[0, 1]) < 1e-12
    assert report.causal_claim is False


def test_single_training_trial_and_nonidentifiable_gains_are_disclosed() -> None:
    base = _quadratic_problem()
    trials = (base.trials[0], base.trials[-1])

    def confounded(
        values: np.ndarray, trial: TuningTrial, active: tuple[str, ...]
    ) -> TuningEvaluation:
        product = values[0] * values[1]
        loss = (product - 1.0) ** 2
        return TuningEvaluation(np.full((2, 2), loss), 0.0, 0.0, 0.0, 0.0)

    problem = replace(base, trials=trials, evaluate=confounded)
    result = tune_control_loops(
        problem,
        LoopTuningConfig(
            max_passes=1,
            block_max_evaluations=20,
            joint_max_evaluations=20,
            trust_radius_scaled=1.0,
            max_cross_group_regression=1.0,
            max_constraint_violation=0.0,
            seed=4,
        ),
    )
    assert result.generalization_status == "single_training_trial_insufficient"
    assert result.phase_covariance is None
    assert result.coupling.rank_deficient
    assert result.feedforward_policy == "frozen_not_jointly_identified"


def test_invalid_bounds_nonfinite_evaluation_and_missing_holdout_fail_closed() -> None:
    base = _quadratic_problem()
    with pytest.raises(ValueError, match="bounds"):
        replace(
            base,
            parameters=(replace(base.parameters[0], upper=0.0), base.parameters[1]),
        )
    with pytest.raises(ValueError, match="holdout"):
        replace(base, trials=base.trials[:2])

    def nonfinite(
        values: np.ndarray, trial: TuningTrial, active: tuple[str, ...]
    ) -> TuningEvaluation:
        return TuningEvaluation(np.full((2, 2), np.nan), 0.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="finite"):
        tune_control_loops(replace(base, evaluate=nonfinite), LoopTuningConfig())


class _LinearTask:
    def __init__(self, row: np.ndarray) -> None:
        self.row = row

    def sample(self, q: np.ndarray, v: np.ndarray) -> TaskKinematics:
        return TaskKinematics(self.row @ q, self.row @ v, self.row)

    def difference(self, target: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return target - actual


def _coupled_controller_evaluator(
    values: np.ndarray, trial: TuningTrial, active: tuple[str, ...]
) -> TuningEvaluation:
    """Drive an independently stepped two-DOF coupled plant through F02."""
    dt = 0.02
    steps = 40
    groups = ("pelvis", "club")
    rows = (np.array([[1.0, 0.45]]), np.array([[0.35, 1.0]]))
    actuators = ActuatorMap(("hip", "wrist"), (0, 1), ())
    schedule = NominalTrajectory(
        np.linspace(0.0, steps * dt, steps + 1),
        np.zeros((steps, 2)),
        np.zeros((steps, 2)),
        np.zeros((steps, 2)),
        np.zeros((steps, 2, 4)),
        ("backswing",) * 20 + ("downswing",) * 20,
        actuators.channel_ids,
    )
    tasks = tuple(
        ControlTask(
            name,
            TaskGroup.PELVIS_FEET if name == "pelvis" else TaskGroup.CLUB,
            0,
            ("backswing", "downswing"),
            _LinearTask(row),
            np.zeros((steps, 1)),
            np.zeros((steps, 1)),
            kp=float(values[i]),
            kd=0.4 * float(values[i]),
            frame_id="world",
            position_unit="rad",
        )
        for i, (name, row) in enumerate(zip(groups, rows, strict=True))
        if name in active
    )
    controller = DistributedFeedbackController(
        EuclideanTangentModel(2),
        actuators,
        schedule,
        DirectBoundedAllocator(
            actuators,
            np.full(2, -8.0),
            np.full(2, 8.0),
            np.full(2, 1000.0),
            contact_free=True,
        ),
        tasks=tasks,
    )
    initial = {
        "train-a": np.array([0.35, -0.25]),
        "train-b": np.array([0.45, -0.15]),
        "holdout-a": np.array([0.5, -0.3]),
    }
    q = initial[trial.trial_id].copy()
    v = np.zeros(2)
    inverse_mass = np.linalg.inv(np.array([[1.0, 0.3], [0.3, 1.2]]))
    losses = np.zeros((2, 2))
    effort = 0.0
    saturation = 0
    for index in range(steps):
        step = controller.command_for_step(index * dt, q, v, dt)
        assert step.information_pattern == "exact_simulated_state"
        phase = 0 if index < 20 else 1
        for group_index, row in enumerate(rows):
            losses[phase, group_index] += float((row @ q)[0] ** 2) / 20.0
        effort += float(step.applied @ step.applied) / steps
        saturation += int(np.any(np.abs(step.total_requested - step.applied) > 1e-12))
        disturbance = np.array([0.03, -0.02]) if phase == 1 else np.zeros(2)
        acceleration = inverse_mass @ (step.applied - 0.2 * v - 0.4 * q + disturbance)
        v = v + dt * acceleration
        q = q + dt * v
    return TuningEvaluation(
        losses, effort, float(np.linalg.norm(q)), 0.0, saturation / steps
    )


def test_actual_coupled_f02_controller_improves_heldout_tracking() -> None:
    problem = LoopTuningProblem(
        parameters=(
            LoopParameter("pelvis_kp", "pelvis", "gain", 0.2, 0.0, 5.0, 1.0),
            LoopParameter("club_kp", "club", "gain", 0.2, 0.0, 5.0, 1.0),
        ),
        groups=("pelvis", "club"),
        phases=("backswing", "downswing"),
        trials=(
            TuningTrial("train-a", "train"),
            TuningTrial("train-b", "train"),
            TuningTrial("holdout-a", "holdout"),
        ),
        evaluate=_coupled_controller_evaluator,
        effort_weight=0.001,
        robustness_weight=0.001,
        regularization_weight=0.0001,
    )
    result = tune_control_loops(
        problem,
        LoopTuningConfig(
            max_passes=2,
            block_max_evaluations=30,
            joint_max_evaluations=40,
            trust_radius_scaled=2.0,
            max_cross_group_regression=0.05,
            max_constraint_violation=0.0,
            seed=8,
        ),
    )
    assert result.objective < result.initial_objective
    assert (
        np.max(
            result.coupling.cross_jacobian_norms
            - np.diag(np.diag(result.coupling.cross_jacobian_norms))
        )
        > 1e-5
    )
    assert result.holdout_full.tracking_rmse < max(
        score.tracking_rmse for score in result.holdout_reduced
    )
    assert result.holdout_full.constraint_violation == 0.0


def test_frozen_perturbation_response_is_separate_from_refit_compensation() -> None:
    baseline_problem = _quadratic_problem(coupled=False)
    config = LoopTuningConfig(
        max_passes=2,
        block_max_evaluations=30,
        joint_max_evaluations=40,
        trust_radius_scaled=2.0,
        max_cross_group_regression=1.0,
        max_constraint_violation=0.0,
        seed=9,
    )
    baseline = tune_control_loops(baseline_problem, config)

    def perturbed(
        values: np.ndarray, trial: TuningTrial, active: tuple[str, ...]
    ) -> TuningEvaluation:
        a, b = values
        # An external phase-specific bias shifts the useful gains. The frozen
        # controller sees it before any parameter compensation is allowed.
        return TuningEvaluation(
            np.array(
                [
                    [(a - 1.7) ** 2 + 0.1, (b - 2.1) ** 2 + 0.1],
                    [(a - 1.9) ** 2 + 0.1, (b - 2.3) ** 2 + 0.1],
                ]
            ),
            effort=0.0,
            robustness=0.0,
            constraint_violation=0.0,
            saturation_fraction=0.0,
        )

    comparison = compare_perturbation_compensation(
        baseline, replace(baseline_problem, evaluate=perturbed), config
    )
    assert comparison.frozen_holdout.phase_group_losses.shape == (2, 2)
    assert (
        comparison.reoptimized_holdout.tracking_rmse
        < comparison.frozen_holdout.tracking_rmse
    )
    assert comparison.scaled_parameter_shift > 0.1
    assert (
        comparison.interpretation
        == "refit_compensation_not_frozen_controller_sensitivity"
    )
    with pytest.raises(ValueError, match="parameter identity"):
        compare_perturbation_compensation(
            baseline,
            replace(
                baseline_problem,
                parameters=tuple(reversed(baseline_problem.parameters)),
            ),
            config,
        )


def test_coupling_diagnostic_budget_fails_before_expensive_rollouts() -> None:
    problem = _quadratic_problem()
    with pytest.raises(ValueError, match="diagnostic budget"):
        frozen_coupling_diagnostics(problem, problem.initial, max_evaluations=12)


def test_report_arrays_are_immutable_evidence_snapshots() -> None:
    result = tune_control_loops(_quadratic_problem(), LoopTuningConfig())
    with pytest.raises(ValueError):
        result.parameters[0] = 99.0
    with pytest.raises(ValueError):
        result.holdout_full.group_losses[0] = 99.0
    with pytest.raises(ValueError):
        result.coupling.jacobian[0, 0, 0] = 99.0
    assert result.phase_covariance is not None
    with pytest.raises(ValueError):
        result.phase_covariance.pooled_covariance[0, 0] = 99.0
