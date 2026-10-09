"""Paired native MuJoCo controls and independent held-input replay (F05)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.native_torque_replay import (
    build_native_torque_bundle,
    replay_native_torque_bundle,
)
from src.shared.python.estimation.mosaic.local_policy import (
    LinearizedDynamics,
    tvlqr_gains,
)
from src.shared.python.motion_matching.bounded_nmpc import (
    BoundedNMPC,
    MPCConfig,
    MPCProblem,
)
from src.shared.python.motion_matching.distributed_feedback import (
    ActuatorMap,
    DirectBoundedAllocator,
    DistributedFeedbackController,
    EuclideanTangentModel,
    NominalTrajectory,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
_DT = 0.01
_STEPS = 8
_HORIZON = 4


def _model_path(tmp_path: Path, mass: float, name: str) -> Path:
    path = tmp_path / name
    path.write_text(
        f"""<mujoco model="paired-nmpc"><option timestep="{_DT}" integrator="RK4" gravity="0 0 0"/>
    <worldbody><body name="arm"><joint name="hinge" axis="0 0 1" damping="0.05"/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="{mass}"/>
    </body></worldbody><actuator><motor name="hinge_torque" joint="hinge" gear="1"
    ctrllimited="true" ctrlrange="-2 2"/></actuator></mujoco>""",
        encoding="utf-8",
    )
    return path


def _predictor(model: object) -> object:
    import mujoco as mj

    def step(state: np.ndarray, effort: np.ndarray) -> np.ndarray:
        data = mj.MjData(model)
        data.qpos[0], data.qvel[0] = state
        data.ctrl[0] = effort[0]
        mj.mj_step(model, data)
        return np.array([data.qpos[0], data.qvel[0]])

    return step


def _tvlqr_controller(step: object) -> DistributedFeedbackController:
    state = np.zeros(2)
    zero = np.zeros(1)
    origin = step(state, zero)
    eps = 1e-5
    a = np.column_stack(
        [(step(state + eps * np.eye(2)[i], zero) - origin) / eps for i in range(2)]
    )
    b = ((step(state, np.array([eps])) - origin) / eps).reshape(2, 1)
    gains = tvlqr_gains(
        LinearizedDynamics(np.tile(a, (_STEPS, 1, 1)), np.tile(b, (_STEPS, 1, 1))),
        np.diag([10.0, 0.1]),
        np.array([[0.01]]),
    )
    actuators = ActuatorMap(("hinge_torque",), (0,), ())
    schedule = NominalTrajectory(
        times=np.arange(_STEPS + 1) * _DT,
        q=np.zeros((_STEPS, 1)),
        v=np.zeros((_STEPS, 1)),
        feedforward=np.zeros((_STEPS, 1)),
        gains=gains,
        phase_names=("downswing",) * _STEPS,
        channel_ids=actuators.channel_ids,
    )
    return DistributedFeedbackController(
        EuclideanTangentModel(1),
        actuators,
        schedule,
        DirectBoundedAllocator(
            actuators,
            np.array([-2.0]),
            np.array([2.0]),
            np.array([400.0]),
            contact_free=True,
        ),
    )


def test_native_robust_nmpc_and_tvlqr_replay_same_executed_inputs(
    tmp_path: Path,
) -> None:
    mj = pytest.importorskip("mujoco")
    nominal_path = _model_path(tmp_path, 0.3, "nominal.xml")
    perturbed_path = _model_path(tmp_path, 0.42, "perturbed.xml")
    nominal = mj.MjModel.from_xml_path(str(nominal_path))
    perturbed = mj.MjModel.from_xml_path(str(perturbed_path))
    predictor = _predictor(nominal)
    scenario = _predictor(perturbed)
    problem = MPCProblem(
        step=predictor,
        scenario_steps=(scenario,),
        time_step_s=_DT,
        target_states=np.zeros((_STEPS + _HORIZON + 1, 2)),
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
    nmpc = BoundedNMPC(
        problem,
        MPCConfig(_HORIZON, 500, 1.0),
        fallback=lambda state, time_s: np.array([0.0]),
    )
    tvlqr = _tvlqr_controller(predictor)

    for name, controller in (("nmpc", nmpc), ("tvlqr", tvlqr)):
        data = mj.MjData(perturbed)
        data.qpos[0] = 0.4
        initial = np.empty(mj.mj_stateSize(perturbed, mj.mjtState.mjSTATE_INTEGRATION))
        mj.mj_getState(perturbed, data, initial, mj.mjtState.mjSTATE_INTEGRATION)
        direct_q = [float(data.qpos[0])]
        applied = []
        for index in range(_STEPS):
            now = index * _DT
            if name == "nmpc":
                receipt = controller.command_for_step(
                    index,
                    np.array([data.qpos[0], data.qvel[0]]),
                    observation_time_s=now,
                    current_time_s=now,
                )
            else:
                receipt = controller.command_for_step(
                    now, data.qpos.copy(), data.qvel.copy(), _DT
                )
            effort = float(receipt.applied[0])
            assert -2.0 <= effort <= 2.0
            applied.append((effort,))
            data.ctrl[0] = effort
            mj.mj_step(perturbed, data)
            direct_q.append(float(data.qpos[0]))
        applied.append(applied[-1])  # Terminal ZOH sentinel is never stepped.
        bundle = build_native_torque_bundle(
            perturbed_path,
            initial,
            np.arange(_STEPS + 1) * _DT,
            np.array(applied),
            experiment_id=f"f05-{name}-mass-perturbation",
        )
        replay = replay_native_torque_bundle(bundle, perturbed_path)
        np.testing.assert_allclose(replay.qpos[:, 0], direct_q, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(replay.applied_actuator_torques, applied[:-1])
        assert bundle.input_history.input_kind.value == "actuator_torque"
        if name == "nmpc":
            assert any(row.status == "optimized" for row in nmpc.applied_history)
            assert all(row.elapsed_s >= 0.0 for row in nmpc.applied_history)
