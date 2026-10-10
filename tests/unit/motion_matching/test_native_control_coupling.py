"""F04 native train/holdout tuning and controlled plant interventions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pytest

from tests.unit.motion_matching.test_native_tangent_derivative import (
    _floating_two_hinge_xml,
    _initial,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _native_trial(tmp_path: Path) -> tuple[Path, object, np.ndarray, object, object]:
    mj = pytest.importorskip("mujoco")
    path = tmp_path / "coupled.xml"
    path.write_text(_floating_two_hinge_xml(), encoding="utf-8")
    model = mj.MjModel.from_xml_path(str(path))
    initial = _initial(model)
    steps = 12
    times = np.arange(steps + 1) * model.opt.timestep
    commands = np.column_stack(
        (
            0.25 + 0.1 * np.sin(np.arange(steps + 1)),
            -0.15 + 0.08 * np.cos(np.arange(steps + 1)),
        )
    )
    commands[-1] = commands[-2]
    from src.engines.physics_engines.mujoco.python.native_torque_replay import (
        build_native_torque_bundle,
    )
    from src.engines.physics_engines.mujoco.python.native_distributed_feedback import (
        build_native_nominal_policy_from_replay,
    )

    bundle = build_native_torque_bundle(
        path, initial, times, commands, experiment_id="f04-native-teacher"
    )
    schedule = build_native_nominal_policy_from_replay(
        path,
        bundle,
        state_weight=np.diag([1.0] * 6 + [30.0, 30.0] + [0.1] * 8),
        input_weight=np.eye(2) * 0.01,
        phase_names=("early",) * 6 + ("late",) * 6,
    )
    return path, model, initial, bundle, schedule


def test_native_tuning_uses_train_only_and_independent_holdout(
    tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    from src.engines.physics_engines.mujoco.python.native_control_coupling import (
        NativeCoupledLoopEvaluator,
    )
    from src.shared.python.motion_matching.control_loop_tuning import (
        LoopTuningConfig,
        tune_control_loops,
    )

    path, _model, initial, bundle, schedule = _native_trial(tmp_path)
    train_hip = initial.copy()
    train_hip[8] += 0.2
    train_knee = initial.copy()
    train_knee[9] -= 0.15
    holdout = initial.copy()
    holdout[8] += 0.25
    holdout[9] -= 0.1
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    evaluator = NativeCoupledLoopEvaluator(
        path,
        bundle,
        schedule,
        {"train-hip": train_hip, "train-knee": train_knee, "holdout": holdout},
        rate_limit_nm_s=np.array([400.0, 400.0]),
    )
    problem = evaluator.problem(
        train_ids=("train-hip", "train-knee"),
        holdout_ids=("holdout",),
        initial_scales=(0.3, 0.3),
    )
    result = tune_control_loops(
        problem,
        LoopTuningConfig(
            max_passes=1,
            block_max_evaluations=30,
            joint_max_evaluations=50,
            max_diagnostic_evaluations=28,
            max_cross_group_regression=1.0,
            max_phase_group_regression=1.0,
        ),
    )
    assert result.objective < result.initial_objective
    assert result.holdout_full.phase_group_losses.shape == (2, 2)
    assert result.holdout_full.tracking_rmse > 0
    assert result.phase_covariance is not None
    assert result.phase_covariance.causal_claim is False
    assert (
        result.coupling.interpretation == "frozen_controller_association_not_causation"
    )
    assert result.coupling.singular_values.shape == (2,)
    assert (
        evaluator.evaluations_by_split["train"]
        > evaluator.evaluations_by_split["holdout"]
    )
    assert all(
        np.allclose(values, result.parameters)
        for values in evaluator.parameters_by_split["holdout"]
    )
    # This diagnostic is evaluated only after training and frozen holdout scoring.
    baseline_holdout = problem.evaluate(
        problem.initial, problem.holdout_trials[0], problem.groups
    )
    initial_holdout_rmse = float(np.sqrt(np.mean(baseline_holdout.phase_losses)))
    assert result.holdout_full.tracking_rmse < initial_holdout_rmse
    assert (
        evaluator.uncertainty_status
        == "unavailable_no_resampling_or_capture_noise_model"
    )
    assert result.generalization_status == "multi_trial_descriptive"
    assert evaluator.max_full_state_replay_error <= 1e-12
    assert len(set(evaluator.applied_input_hashes)) > 1
    total_wall_s = time.perf_counter() - wall_started
    total_cpu_s = time.process_time() - cpu_started
    if os.environ.get("F04_NATIVE_RECEIPT") == "1":
        record_property(
            "f04_native_tuning_evidence",
            json.dumps(
                {
                    "source_model_sha256": bundle.model.source_model_sha256,
                    "initial_state_sha256": bundle.integrity.initial_state_sha256,
                    "policy_sha256": bundle.policy_sha256,
                    "time_grid_sha256": bundle.time_grid_sha256,
                    "teacher_input_sha256": bundle.applied_input_sha256,
                    "initial_objective": result.initial_objective,
                    "final_objective": result.objective,
                    "parameters": result.parameters.tolist(),
                    "holdout_phase_group_losses": result.holdout_full.phase_group_losses.tolist(),
                    "holdout_initial_rmse_rad": initial_holdout_rmse,
                    "holdout_final_rmse_rad": result.holdout_full.tracking_rmse,
                    "cross_jacobian_norms": result.coupling.cross_jacobian_norms.tolist(),
                    "singular_values": result.coupling.singular_values.tolist(),
                    "rank_deficient": bool(result.coupling.rank_deficient),
                    "uncertainty_status": evaluator.uncertainty_status,
                    "train_evaluations": evaluator.evaluations_by_split["train"],
                    "holdout_evaluations": evaluator.evaluations_by_split["holdout"],
                    "distinct_applied_inputs": len(set(evaluator.applied_input_hashes)),
                    "checkpoint_reasons": [item.reason for item in result.checkpoints],
                    "max_full_state_replay_error": evaluator.max_full_state_replay_error,
                    "total_wall_s": total_wall_s,
                    "total_cpu_s": total_cpu_s,
                },
                sort_keys=True,
            ),
        )


def test_paired_native_motor_intervention_is_distinct_from_residual_covariance(
    tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    from src.engines.physics_engines.mujoco.python.native_control_coupling import (
        paired_native_motor_interventions,
    )

    path, _model, _initial_state, bundle, _schedule = _native_trial(tmp_path)
    wall_started = time.perf_counter()
    evidence = paired_native_motor_interventions(path, bundle, step=0, delta_nm=0.2)
    wall_s = time.perf_counter() - wall_started
    assert evidence.response_rad_per_nm.shape == (2, 2)
    assert abs(evidence.response_rad_per_nm[0, 1]) > 1e-6
    assert abs(evidence.response_rad_per_nm[1, 0]) > 1e-6
    assert evidence.channel_ids == ("hip_torque", "knee_torque")
    assert evidence.response_joint_ids == ("hip", "knee")
    assert len(set(evidence.applied_input_sha256)) == 4
    assert (
        evidence.interpretation
        == "synthetic_plant_input_intervention_not_human_control"
    )
    with pytest.raises(ValueError, match="bound"):
        paired_native_motor_interventions(path, bundle, step=0, delta_nm=4.0)
    if os.environ.get("F04_NATIVE_RECEIPT") == "1":
        record_property(
            "f04_native_intervention_evidence",
            json.dumps(
                {
                    "response_rad_per_nm": evidence.response_rad_per_nm.tolist(),
                    "channel_ids": evidence.channel_ids,
                    "response_joint_ids": evidence.response_joint_ids,
                    "applied_input_sha256": evidence.applied_input_sha256,
                    "initial_state_sha256": evidence.initial_state_sha256,
                    "policy_sha256": evidence.policy_sha256,
                    "time_grid_sha256": evidence.time_grid_sha256,
                    "interpretation": evidence.interpretation,
                    "intervention_wall_s": wall_s,
                },
                sort_keys=True,
            ),
        )
