"""Native derivative checks for the bounded BoxFDDP hinge candidate (F05b)."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.native_box_fddp_hinge import (
    HingeDiscreteModel,
    linearize_native_hinge,
)
from src.engines.physics_engines.mujoco.python.native_nmpc_tracking import (
    run_native_nmpc_tracking,
)
from src.engines.physics_engines.mujoco.python.native_torque_replay import _contracts
from src.engines.physics_engines.mujoco.python.box_fddp_tracking import (
    BoxFDDPConfig,
    BoxFDDPHingeController,
    run_native_box_fddp_tracking,
)
from src.shared.python.motion_matching.bounded_nmpc import (
    BoundedNMPC,
    MPCConfig,
    MPCProblem,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _model_xml(mass: float) -> str:
    return f"""<mujoco model="paired-nmpc"><option timestep="0.01" integrator="RK4" gravity="0 0 0"/>
    <worldbody><body name="arm"><joint name="hinge" axis="0 0 1" damping="0.05"/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="{mass}"/>
    </body></worldbody><actuator><motor name="hinge_torque" joint="hinge" gear="1"
    ctrllimited="true" ctrlrange="-2 2"/></actuator></mujoco>"""


def _problem(nominal: HingeDiscreteModel, perturbed: HingeDiscreteModel) -> MPCProblem:
    return MPCProblem(
        step=lambda x, u: nominal.A @ x + nominal.B @ u,
        scenario_steps=(lambda x, u: perturbed.A @ x + perturbed.B @ u,),
        time_step_s=0.01,
        target_states=np.zeros((13, 2)),
        state_weights=np.array([10.0, 0.1]),
        terminal_weights=np.array([30.0, 0.1]),
        input_weights=np.array([0.01]),
        input_lower=np.array([-2.0]),
        input_upper=np.array([2.0]),
        state_lower=np.array([-2.0, -20.0]),
        state_upper=np.array([2.0, 20.0]),
        state_component_ids=("hinge_q", "hinge_v"),
        state_units=("rad", "rad/s"),
        input_channel_ids=("hinge_torque",),
        input_units=("N*m",),
    )


@pytest.mark.parametrize("mass", (0.3, 0.42))
def test_discrete_derivatives_match_native_rk4_step(mass: float) -> None:
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("crocoddyl")
    model = mj.MjModel.from_xml_string(_model_xml(mass))
    discrete = linearize_native_hinge(model)

    def step(state: np.ndarray, torque: float) -> np.ndarray:
        data = mj.MjData(model)
        data.qpos[0], data.qvel[0] = state
        data.ctrl[0] = torque
        mj.mj_step(model, data)
        return np.array([data.qpos[0], data.qvel[0]])

    for state, torque in (
        (np.array([0.4, -1.2]), 0.7),
        (np.array([-0.7, 0.8]), -1.1),
    ):
        np.testing.assert_allclose(
            discrete.A @ state + discrete.B[:, 0] * torque,
            step(state, torque),
            atol=1e-11,
            rtol=0,
        )
        epsilon = 1e-6
        for column in range(2):
            delta = np.eye(2)[column] * epsilon
            measured = (step(state + delta, torque) - step(state - delta, torque)) / (
                2 * epsilon
            )
            np.testing.assert_allclose(discrete.A[:, column], measured, atol=1e-8)
        measured_u = (step(state, torque + epsilon) - step(state, torque - epsilon)) / (
            2 * epsilon
        )
        np.testing.assert_allclose(discrete.B[:, 0], measured_u, atol=1e-8)


def test_non_rk4_and_non_unit_motor_are_rejected() -> None:
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("crocoddyl")
    with pytest.raises(ValueError, match="RK4"):
        linearize_native_hinge(
            mj.MjModel.from_xml_string(
                _model_xml(0.3).replace('integrator="RK4"', 'integrator="Euler"')
            )
        )
    with pytest.raises(ValueError, match="unit"):
        linearize_native_hinge(
            mj.MjModel.from_xml_string(_model_xml(0.3).replace('gear="1"', 'gear="2"'))
        )
    with pytest.raises(ValueError, match="contact-free"):
        linearize_native_hinge(
            mj.MjModel.from_xml_string(
                _model_xml(0.3).replace('gravity="0 0 0"', 'gravity="0 0 -9.81"')
            )
        )
    with pytest.raises(ValueError, match="contact-free"):
        linearize_native_hinge(
            mj.MjModel.from_xml_string(
                _model_xml(0.3).replace(
                    '<body name="arm">', '<body name="arm" gravcomp="0.1">'
                )
            )
        )


def test_box_fddp_uses_bounded_native_derivatives_and_verified_first_input() -> None:
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("crocoddyl")
    nominal = linearize_native_hinge(mj.MjModel.from_xml_string(_model_xml(0.3)))
    perturbed = linearize_native_hinge(mj.MjModel.from_xml_string(_model_xml(0.42)))
    problem = _problem(nominal, perturbed)
    controller = BoxFDDPHingeController(
        problem,
        nominal,
        perturbed,
        BoxFDDPConfig(horizon_steps=4, max_iterations=20, max_wall_s=0.1),
        fallback=lambda x, t: np.array([0.0]),
    )
    receipt = controller.command_for_step(
        0, np.array([0.4, 0.0]), observation_time_s=0.0, current_time_s=0.0
    )
    assert receipt.status == "optimized"
    assert -2.0 <= receipt.applied[0] < 0.0
    assert receipt.objective < receipt.fallback_objective
    assert receipt.input_boundary == "post_limit_actuator_command"
    assert receipt.solver_iterations <= 20
    assert receipt.elapsed_s >= receipt.solve_s >= 0.0


def test_solver_exception_returns_verified_fallback_without_reusing_warm_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mj = pytest.importorskip("mujoco")
    crocoddyl = pytest.importorskip("crocoddyl")
    nominal = linearize_native_hinge(mj.MjModel.from_xml_string(_model_xml(0.3)))
    perturbed = linearize_native_hinge(mj.MjModel.from_xml_string(_model_xml(0.42)))
    controller = BoxFDDPHingeController(
        _problem(nominal, perturbed),
        nominal,
        perturbed,
        BoxFDDPConfig(horizon_steps=4, max_iterations=20, max_wall_s=0.1),
        fallback=lambda x, t: np.array([0.0]),
    )
    first = controller.command_for_step(
        0, np.array([0.4, 0.0]), observation_time_s=0.0, current_time_s=0.0
    )
    assert first.status == "optimized"

    class BrokenSolver:
        def __init__(self, shooting: object) -> None:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(crocoddyl, "SolverBoxFDDP", BrokenSolver)
    second = controller.command_for_step(
        1, np.array([0.35, 0.0]), observation_time_s=0.01, current_time_s=0.01
    )
    assert second.status == "fallback_solver_exception"
    np.testing.assert_array_equal(second.applied, [0.0])
    assert second.warm_started
    assert controller.applied_history == (first, second)


def test_derivative_identity_and_fallback_bounds_fail_closed() -> None:
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("crocoddyl")
    nominal = linearize_native_hinge(mj.MjModel.from_xml_string(_model_xml(0.3)))
    perturbed = linearize_native_hinge(mj.MjModel.from_xml_string(_model_xml(0.42)))
    problem = _problem(nominal, perturbed)
    config = BoxFDDPConfig(horizon_steps=4, max_iterations=20, max_wall_s=0.1)
    with pytest.raises(ValueError, match="callback differs"):
        BoxFDDPHingeController(
            replace(problem, step=lambda x, u: x),
            nominal,
            perturbed,
            config,
            fallback=lambda x, t: np.array([0.0]),
        )
    controller = BoxFDDPHingeController(
        problem,
        nominal,
        perturbed,
        config,
        fallback=lambda x, t: np.array([3.0]),
    )
    with pytest.raises(ValueError, match="fallback violates native torque limits"):
        controller.command_for_step(
            0, np.array([0.4, 0.0]), observation_time_s=0.0, current_time_s=0.0
        )
    assert controller.applied_history == ()


def test_native_run_rejects_execution_model_derivative_mismatch(tmp_path: Path) -> None:
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("crocoddyl")
    nominal_path = tmp_path / "nominal.xml"
    execution_path = tmp_path / "execution.xml"
    nominal_path.write_text(_model_xml(0.3), encoding="utf-8")
    execution_path.write_text(_model_xml(0.5), encoding="utf-8")
    nominal = linearize_native_hinge(mj.MjModel.from_xml_path(str(nominal_path)))
    declared_execution = linearize_native_hinge(
        mj.MjModel.from_xml_string(_model_xml(0.42))
    )
    controller = BoxFDDPHingeController(
        _problem(nominal, declared_execution),
        nominal,
        declared_execution,
        BoxFDDPConfig(horizon_steps=4, max_iterations=20, max_wall_s=0.1),
        fallback=lambda x, t: np.array([0.0]),
    )
    native_model = mj.MjModel.from_xml_path(str(execution_path))
    data = mj.MjData(native_model)
    initial = np.empty(mj.mj_stateSize(native_model, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(native_model, data, initial, mj.mjtState.mjSTATE_INTEGRATION)
    with pytest.raises(ValueError, match="derivative identity differs"):
        run_native_box_fddp_tracking(
            nominal_path,
            execution_path,
            initial,
            controller,
            steps=8,
            experiment_id="mismatch",
        )


@pytest.mark.parametrize("start_q", (0.4, 0.7))
def test_box_fddp_native_tracking_replays_frozen_applied_torques(
    tmp_path: Path,
    start_q: float,
    record_property: Callable[[str, object], None],
) -> None:
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("crocoddyl")
    nominal_path = tmp_path / "nominal.xml"
    execution_path = tmp_path / "perturbed.xml"
    nominal_path.write_text(_model_xml(0.3), encoding="utf-8")
    execution_path.write_text(_model_xml(0.42), encoding="utf-8")
    cold_started = time.perf_counter()
    _contracts()  # Common T01 provider bootstrap, charged equally to either method.
    common_contract_import_s = time.perf_counter() - cold_started
    execution_model = mj.MjModel.from_xml_path(str(execution_path))
    started = time.perf_counter()
    nominal = linearize_native_hinge(mj.MjModel.from_xml_path(str(nominal_path)))
    perturbed = linearize_native_hinge(execution_model)
    controller = BoxFDDPHingeController(
        _problem(nominal, perturbed),
        nominal,
        perturbed,
        BoxFDDPConfig(horizon_steps=4, max_iterations=20, max_wall_s=0.01),
        fallback=lambda x, t: np.array([0.0]),
    )
    setup_s = time.perf_counter() - started
    data = mj.MjData(execution_model)
    data.qpos[0] = start_q
    initial = np.empty(
        mj.mj_stateSize(execution_model, mj.mjtState.mjSTATE_INTEGRATION)
    )
    mj.mj_getState(execution_model, data, initial, mj.mjtState.mjSTATE_INTEGRATION)
    started = time.perf_counter()
    native = run_native_box_fddp_tracking(
        nominal_path,
        execution_path,
        initial,
        controller,
        steps=8,
        experiment_id=f"f05b-boxfddp-{start_q:.1f}",
    )
    controlled_plus_export_replay_s = time.perf_counter() - started
    assert native.bundle.input_history.input_kind.value == "actuator_torque"
    assert len(native.commands) == 8
    np.testing.assert_allclose(
        native.native_qpos, native.replay.qpos, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        native.native_qvel, native.replay.qvel, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        native.native_integration_states,
        native.replay.integration_states,
        atol=1e-12,
        rtol=0,
    )
    assert all(-2.0 <= row.applied[0] <= 2.0 for row in native.commands)
    assert all(
        row.status == "optimized" or row.status.startswith("fallback_")
        for row in native.commands
    )
    if os.environ.get("F05B_BENCHMARK_RECEIPT") == "1":
        scipy_controller = BoundedNMPC(
            _problem(nominal, perturbed),
            MPCConfig(4, 500, 1.0),
            fallback=lambda x, t: np.array([0.0]),
        )
        scipy_started = time.perf_counter()
        scipy_native = run_native_nmpc_tracking(
            execution_path,
            initial,
            scipy_controller,
            steps=8,
            experiment_id=f"f05b-scipy-reference-{start_q:.1f}",
        )
        scipy_controlled_export_replay_s = time.perf_counter() - scipy_started
        np.testing.assert_allclose(
            scipy_native.native_integration_states,
            scipy_native.replay.integration_states,
            atol=1e-12,
            rtol=0,
        )
        metrics = {
            "box_q_rmse_rad": float(np.sqrt(np.mean(native.native_qpos[:, 0] ** 2))),
            "box_effort_l1_nm": float(
                np.abs(native.replay.applied_actuator_torques).sum()
            ),
            "box_optimized_steps": sum(
                row.status == "optimized" for row in native.commands
            ),
            "box_statuses": ",".join(row.status for row in native.commands),
            "box_latency_p50_s": float(np.percentile(native.control_wall_s, 50)),
            "box_latency_p95_s": float(np.percentile(native.control_wall_s, 95)),
            "box_latency_worst_s": float(native.control_wall_s.max()),
            "box_setup_s": setup_s,
            "common_contract_import_s": common_contract_import_s,
            "box_controlled_export_replay_s": controlled_plus_export_replay_s,
            "box_iterations_total": sum(
                row.solver_iterations for row in native.commands
            ),
            "box_input_sha256": native.bundle.applied_input_sha256,
            "box_initial_state_sha256": native.bundle.integrity.initial_state_sha256,
            "box_policy_sha256": native.bundle.policy_sha256,
            "box_model_sha256": native.bundle.model.source_model_sha256,
            "scipy_q_rmse_rad": float(
                np.sqrt(np.mean(scipy_native.native_qpos[:, 0] ** 2))
            ),
            "scipy_optimized_steps": sum(
                row.status == "optimized" for row in scipy_native.commands
            ),
            "scipy_statuses": ",".join(row.status for row in scipy_native.commands),
            "scipy_latency_p50_s": float(
                np.percentile(scipy_native.control_wall_s, 50)
            ),
            "scipy_latency_p95_s": float(
                np.percentile(scipy_native.control_wall_s, 95)
            ),
            "scipy_latency_worst_s": float(scipy_native.control_wall_s.max()),
            "scipy_controlled_export_replay_s": scipy_controlled_export_replay_s,
            "scipy_evaluations_total": sum(
                row.evaluations for row in scipy_native.commands
            ),
            "scipy_input_sha256": scipy_native.bundle.applied_input_sha256,
        }
        for key, value in metrics.items():
            record_property(key, value)
