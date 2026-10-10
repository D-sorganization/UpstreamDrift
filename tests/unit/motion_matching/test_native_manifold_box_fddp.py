"""F05d native manifold action and accepted-input tests."""

from __future__ import annotations

import importlib
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
from src.engines.physics_engines.mujoco.python.native_torque_replay import (
    build_native_torque_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _providers() -> tuple[object, object, object]:
    mj = pytest.importorskip("mujoco")
    crocoddyl = pytest.importorskip("crocoddyl")
    module = importlib.import_module(
        "src.engines.physics_engines.mujoco.python.native_manifold_box_fddp"
    )
    return mj, crocoddyl, module


def _native_state(mj: object, model: object) -> np.ndarray:
    data = mj.MjData(model)
    mj.mj_setState(model, data, _initial(model), mj.mjtState.mjSTATE_INTEGRATION)
    return np.r_[data.qpos, data.qvel]


def _realized_tracking_metrics(
    controller: object, run: object, references: np.ndarray
) -> tuple[float, float, float]:
    tangent = np.array(
        [
            controller.state.diff(
                references[i], np.r_[run.native_qpos[i], run.native_qvel[i]]
            )
            for i in range(1, len(run.native_qpos))
        ]
    )
    hinge_rmse = float(np.sqrt(np.mean(tangent[:, 6:8] ** 2)))
    commands = np.asarray([command.applied for command in run.commands])
    effort = float(np.sum(commands**2))
    stage_cost = float(
        np.sum((tangent**2) * controller.state_weights)
        + np.sum((commands**2) * controller.input_weights)
    )
    terminal_cost = float(np.dot(controller.terminal_weights, tangent[-1] ** 2))
    return hinge_rmse, effort, stage_cost + terminal_cost


def _reachable_native_reference(
    mj: object, model: object, initial: np.ndarray
) -> np.ndarray:
    """Freeze a native-generated target including unactuated-root motion."""
    data = mj.MjData(model)
    mj.mj_setState(model, data, initial, mj.mjtState.mjSTATE_INTEGRATION)
    rows = [np.r_[data.qpos, data.qvel]]
    teacher = np.array(
        [
            [-1.0, 0.5],
            [-1.0, 0.5],
            [-0.8, 0.4],
            [-0.6, 0.3],
            [-0.4, 0.2],
            [-0.2, 0.1],
            [0.0, 0.0],
        ]
    )
    for command in teacher:
        data.ctrl[:] = command
        mj.mj_step(model, data)
        rows.append(np.r_[data.qpos, data.qvel])
    return np.asarray(rows)


def test_state_uses_native_quaternion_tangent_and_sign_equivalence() -> None:
    mj, crocoddyl, module = _providers()
    model = mj.MjModel.from_xml_string(_floating_two_hinge_xml())
    state = module.NativeManifoldState(model)
    assert isinstance(state, crocoddyl.StateAbstract)
    assert (state.nx, state.ndx, model.nq, model.nv) == (17, 16, 9, 8)
    original = _native_state(mj, model)
    delta = np.zeros(16)
    delta[0:3] = (0.01, -0.02, 0.03)
    delta[3:6] = (0.08, -0.05, 0.04)
    delta[6:8] = (0.1, -0.07)
    delta[8:] = np.linspace(-0.2, 0.2, 8)
    moved = state.integrate(original, delta)
    np.testing.assert_allclose(state.diff(original, moved), delta, atol=1e-12)
    assert np.isclose(np.linalg.norm(moved[3:7]), 1.0)
    equivalent = moved.copy()
    equivalent[3:7] *= -1
    np.testing.assert_allclose(state.diff(original, equivalent), delta, atol=1e-12)
    first, second = state.Jdiff(original, original, crocoddyl.Jcomponent.both)
    np.testing.assert_allclose(first, -np.eye(16), atol=1e-7)
    np.testing.assert_allclose(second, np.eye(16), atol=1e-7)
    for jacobian in state.Jintegrate(original, np.zeros(16), crocoddyl.Jcomponent.both):
        np.testing.assert_allclose(jacobian, np.eye(16), atol=1e-7)
    with pytest.raises(ValueError, match="normalized"):
        state.diff(original, np.r_[moved[:3], moved[3:7] * 1.1, moved[7:]])


def test_action_uses_native_discrete_transition_and_tangent_derivatives() -> None:
    mj, crocoddyl, module = _providers()
    model = mj.MjModel.from_xml_string(_floating_two_hinge_xml())
    state = module.NativeManifoldState(model)
    x0 = _native_state(mj, model)
    u = np.array([0.3, -0.2])
    reference = state.integrate(x0, np.r_[np.zeros(8), np.zeros(8)])
    action = module.NativeManifoldAction(
        state,
        reference,
        np.r_[np.ones(8), np.full(8, 0.1)],
        np.array([0.01, 0.01]),
    )
    data = action.createData()
    action.calc(data, x0, u)
    action.calcDiff(data, x0, u)
    native = mj.MjData(model)
    native.qpos[:] = x0[:9]
    native.qvel[:] = x0[9:]
    native.ctrl[:] = u
    mj.mj_forward(model, native)
    np.testing.assert_array_equal(native.qfrc_actuator[:6], np.zeros(6))
    mj.mj_step(model, native)
    np.testing.assert_allclose(data.xnext, np.r_[native.qpos, native.qvel], atol=1e-12)
    assert data.Fx.shape == (16, 16)
    assert data.Fu.shape == (16, 2)
    assert np.linalg.norm(data.Fu[6:8]) > 0
    assert np.isfinite(data.Lxx).all()
    assert action.u_lb.shape == (2,)
    assert action.u_ub.shape == (2,)
    assert np.all(action.u_lb < u) and np.all(u < action.u_ub)


def test_solver_rejects_stale_observation_and_bad_clock() -> None:
    mj, _, module = _providers()
    model = mj.MjModel.from_xml_string(_floating_two_hinge_xml())
    state = module.NativeManifoldState(model)
    initial = _native_state(mj, model)
    references = np.repeat(initial[None, :], 7, axis=0)
    references[:, 7] = np.linspace(0.4, 0.15, 7)
    controller = module.NativeManifoldBoxFDDP(
        model,
        references,
        horizon_steps=3,
        max_iterations=8,
        max_wall_s=0.2,
    )
    stale = controller.command_for_step(
        0, initial, observation_time_s=-0.01, current_time_s=0.0
    )
    assert stale.status == "fallback_stale_observation"
    np.testing.assert_array_equal(stale.applied, [0.0, 0.0])
    with pytest.raises(ValueError, match="clock"):
        controller.command_for_step(
            1, initial, observation_time_s=0.0, current_time_s=0.0
        )
    assert state.ndx == 16


def test_box_fddp_candidate_improves_exact_nonlinear_native_rollout() -> None:
    mj, _, module = _providers()
    model = mj.MjModel.from_xml_string(_floating_two_hinge_xml())
    initial = _native_state(mj, model)
    references = np.repeat(initial[None, :], 8, axis=0)
    references[:, 7] = np.linspace(0.4, 0.12, 8)
    references[:, 8] = np.linspace(-0.3, -0.18, 8)
    controller = module.NativeManifoldBoxFDDP(
        model,
        references,
        horizon_steps=4,
        max_iterations=15,
        max_wall_s=1.0,
    )
    receipt = controller.command_for_step(
        0, initial, observation_time_s=0.0, current_time_s=0.0
    )
    assert receipt.status == "optimized"
    assert receipt.objective is not None
    assert receipt.objective < receipt.fallback_objective
    assert receipt.elapsed_s >= receipt.solve_s >= 0.0
    assert np.all(receipt.applied >= model.actuator_ctrlrange[:, 0])
    assert np.all(receipt.applied <= model.actuator_ctrlrange[:, 1])


def test_native_multidof_control_exports_only_applied_torque_and_replays(
    tmp_path: Path,
) -> None:
    mj, _, module = _providers()
    path = tmp_path / "floating.xml"
    path.write_text(_floating_two_hinge_xml(), encoding="utf-8")
    model = mj.MjModel.from_xml_path(str(path))
    initial = _initial(model)
    start = _native_state(mj, model)
    references = np.repeat(start[None, :], 8, axis=0)
    references[:, 7] = np.linspace(0.4, 0.12, 8)
    references[:, 8] = np.linspace(-0.3, -0.18, 8)
    controller = module.NativeManifoldBoxFDDP(
        model,
        references,
        horizon_steps=4,
        max_iterations=12,
        max_wall_s=1.0,
    )
    run = module.run_native_manifold_box_fddp_tracking(
        path, initial, controller, steps=4, experiment_id="f05d-native-multidof"
    )
    assert len(run.commands) == 4
    assert run.native_qpos.shape == (5, 9)
    assert run.native_qvel.shape == (5, 8)
    assert all(
        command.input_boundary == "post_limit_actuator_command"
        for command in run.commands
    )
    np.testing.assert_allclose(
        run.native_integration_states, run.replay.integration_states, atol=1e-12
    )
    np.testing.assert_array_equal(
        run.replay.applied_actuator_torques,
        [command.applied for command in run.commands],
    )
    assert run.bundle.model.state_schema.components[0].dimension == 9
    assert run.bundle.model.state_schema.components[1].dimension == 8
    assert not run.bundle.policy.observation_access
    assert not run.bundle.policy.state_feedback_access


def test_native_controller_rejects_assistance() -> None:
    mj, _, module = _providers()
    altered = _floating_two_hinge_xml().replace(
        '<body name="pelvis">', '<body name="pelvis" gravcomp="0.1">'
    )
    model = mj.MjModel.from_xml_string(altered)
    references = np.repeat(_native_state(mj, model)[None, :], 7, axis=0)
    with pytest.raises(ValueError, match="unforced"):
        module.NativeManifoldBoxFDDP(
            model, references, horizon_steps=3, max_iterations=5, max_wall_s=0.2
        )


def test_execution_model_actuator_order_must_match_controller(tmp_path: Path) -> None:
    mj, _, module = _providers()
    source = _floating_two_hinge_xml()
    swapped = source.replace(
        '<motor name="hip_torque" joint="hip" gear="1" ctrllimited="true" ctrlrange="-2 2"/>\n'
        '    <motor name="knee_torque" joint="knee" gear="1" ctrllimited="true" ctrlrange="-1.5 1.5"/>',
        '<motor name="knee_torque" joint="knee" gear="1" ctrllimited="true" ctrlrange="-1.5 1.5"/>\n'
        '    <motor name="hip_torque" joint="hip" gear="1" ctrllimited="true" ctrlrange="-2 2"/>',
    )
    assert swapped != source
    path = tmp_path / "reordered.xml"
    path.write_text(swapped, encoding="utf-8")
    model = mj.MjModel.from_xml_string(source)
    initial = _initial(model)
    reference = _native_state(mj, model)
    controller = module.NativeManifoldBoxFDDP(
        model,
        np.repeat(reference[None, :], 7, axis=0),
        horizon_steps=3,
        max_iterations=5,
        max_wall_s=0.2,
    )
    with pytest.raises(ValueError, match="execution model identity differs"):
        module.run_native_manifold_box_fddp_tracking(
            path, initial, controller, steps=3, experiment_id="reordered"
        )


def test_hidden_native_callback_is_rejected() -> None:
    mj, _, module = _providers()
    model = mj.MjModel.from_xml_string(_floating_two_hinge_xml())
    initial = _native_state(mj, model)
    assert mj.get_mjcb_control() is None
    try:
        mj.set_mjcb_control(lambda _model, _data: None)
        with pytest.raises(ValueError, match="callbacks"):
            module.NativeManifoldBoxFDDP(
                model,
                np.repeat(initial[None, :], 7, axis=0),
                horizon_steps=3,
                max_iterations=5,
                max_wall_s=0.2,
            )
    finally:
        mj.set_mjcb_control(None)


def test_candidate_outside_native_motor_bounds_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mj, crocoddyl, module = _providers()
    model = mj.MjModel.from_xml_string(_floating_two_hinge_xml())
    initial = _native_state(mj, model)
    controller = module.NativeManifoldBoxFDDP(
        model,
        np.repeat(initial[None, :], 7, axis=0),
        horizon_steps=3,
        max_iterations=5,
        max_wall_s=1.0,
    )

    class FalseSuccess:
        def __init__(self, shooting: object) -> None:
            self.us = [np.array([20.0, 0.0])] * 3
            self.iter = 1

        def solve(self, *args: object) -> bool:
            return True

    monkeypatch.setattr(crocoddyl, "SolverBoxFDDP", FalseSuccess)
    receipt = controller.command_for_step(
        0, initial, observation_time_s=0.0, current_time_s=0.0
    )
    assert receipt.status == "fallback_infeasible"
    np.testing.assert_array_equal(receipt.applied, [0.0, 0.0])
    assert not receipt.warm_started


def test_loaded_native_model_mutation_is_rejected() -> None:
    mj, _, module = _providers()
    model = mj.MjModel.from_xml_string(_floating_two_hinge_xml())
    initial = _native_state(mj, model)
    controller = module.NativeManifoldBoxFDDP(
        model,
        np.repeat(initial[None, :], 7, axis=0),
        horizon_steps=3,
        max_iterations=5,
        max_wall_s=0.2,
    )
    model.body_gravcomp[1] = 0.1
    with pytest.raises(ValueError, match="identity changed"):
        controller.command_for_step(
            0, initial, observation_time_s=0.0, current_time_s=0.0
        )


@pytest.mark.parametrize("start_hip", (0.4, 0.7))
def test_matched_native_box_and_scipy_runs_keep_frozen_replay(
    tmp_path: Path,
    start_hip: float,
    record_property: Callable[[str, object], None],
) -> None:
    mj, _, module = _providers()
    path = tmp_path / "paired-floating.xml"
    path.write_text(_floating_two_hinge_xml(), encoding="utf-8")
    model = mj.MjModel.from_xml_path(str(path))
    data = mj.MjData(model)
    mj.mj_setState(model, data, _initial(model), mj.mjtState.mjSTATE_INTEGRATION)
    data.qpos[7] = start_hip
    initial = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(model, data, initial, mj.mjtState.mjSTATE_INTEGRATION)
    references = _reachable_native_reference(mj, model, initial)
    warm_started = time.perf_counter()
    preflight = build_native_torque_bundle(
        path,
        initial,
        np.arange(5) * model.opt.timestep,
        np.zeros((5, 2)),
        experiment_id=f"f05d-common-preflight-{start_hip}",
    )
    common_bootstrap_s = time.perf_counter() - warm_started
    results = {}
    variants = (
        ("box", module.NativeManifoldBoxFDDP),
        ("scipy", module.NativeManifoldSciPyShooting),
    )
    if start_hip > 0.5:
        variants = variants[::-1]
    for name, solver_type in variants:
        candidate_model = mj.MjModel.from_xml_path(str(path))
        started = time.perf_counter()
        controller = solver_type(
            candidate_model,
            references,
            horizon_steps=4,
            max_iterations=15,
            max_wall_s=0.2,
        )
        run = module.run_native_manifold_box_fddp_tracking(
            path,
            initial,
            controller,
            steps=4,
            experiment_id=f"f05d-{name}-{start_hip}",
        )
        total_wall_s = time.perf_counter() - started
        np.testing.assert_allclose(
            run.native_integration_states, run.replay.integration_states, atol=1e-12
        )
        results[name] = (controller, run, total_wall_s)
    box_controller, box, box_wall_s = results["box"]
    scipy_controller, scipy, scipy_wall_s = results["scipy"]
    assert (
        box.bundle.model.source_model_sha256 == scipy.bundle.model.source_model_sha256
    )
    assert (
        box.bundle.model.loaded_native_model_sha256
        == scipy.bundle.model.loaded_native_model_sha256
    )
    assert (
        box.bundle.integrity.initial_state_sha256
        == scipy.bundle.integrity.initial_state_sha256
    )
    assert box.bundle.policy_sha256 == scipy.bundle.policy_sha256
    assert box.bundle.time_grid_sha256 == scipy.bundle.time_grid_sha256
    assert box.bundle.state_schema_sha256 == scipy.bundle.state_schema_sha256
    assert (
        box.bundle.input_channel_schema_sha256
        == scipy.bundle.input_channel_schema_sha256
    )
    assert box.bundle.model == preflight.model
    assert len(box.commands) == len(scipy.commands) == 4
    for name, controller, run, total_wall_s in (
        ("box", box_controller, box, box_wall_s),
        ("scipy", scipy_controller, scipy, scipy_wall_s),
    ):
        assert all(command.elapsed_s >= 0 for command in run.commands)
        assert all(command.status != "pending" for command in run.commands)
        assert total_wall_s >= float(np.sum(run.control_wall_s))
        hinge_rmse, effort, native_cost = _realized_tracking_metrics(
            controller, run, references
        )
        if os.environ.get("F05D_BENCHMARK_RECEIPT") == "1":
            record_property(f"{name}_total_wall_s", total_wall_s + common_bootstrap_s)
            record_property(f"{name}_run_wall_s", total_wall_s)
            record_property(f"{name}_control_wall_s", float(np.sum(run.control_wall_s)))
            record_property(
                f"{name}_statuses", json.dumps([c.status for c in run.commands])
            )
            record_property(
                f"{name}_accepted_steps",
                sum(c.status == "optimized" for c in run.commands),
            )
            record_property(f"{name}_solver_id", controller.solver_id)
            record_property(f"{name}_applied_sha256", run.bundle.applied_input_sha256)
            record_property(
                f"{name}_qpos_final", json.dumps(run.native_qpos[-1].tolist())
            )
            record_property(f"{name}_hinge_rmse_rad", hinge_rmse)
            record_property(f"{name}_effort_sum_n2m2", effort)
            record_property(f"{name}_realized_native_cost", native_cost)
            record_property(
                f"{name}_call_p95_s", float(np.percentile(run.control_wall_s, 95))
            )
    if os.environ.get("F05D_BENCHMARK_RECEIPT") == "1":
        record_property("execution_order", json.dumps([name for name, _ in variants]))
        record_property("common_bootstrap_s", common_bootstrap_s)
        record_property("source_model_sha256", box.bundle.model.source_model_sha256)
        record_property(
            "loaded_model_sha256", box.bundle.model.loaded_native_model_sha256
        )
        record_property(
            "initial_state_sha256", box.bundle.integrity.initial_state_sha256
        )
        record_property("policy_sha256", box.bundle.policy_sha256)
        record_property("time_grid_sha256", box.bundle.time_grid_sha256)
        record_property("state_schema_sha256", box.bundle.state_schema_sha256)
        record_property(
            "input_channel_schema_sha256", box.bundle.input_channel_schema_sha256
        )
