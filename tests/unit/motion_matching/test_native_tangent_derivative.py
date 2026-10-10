"""Native multi-DOF tangent derivatives and full-state torque replay (F05c)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.native_tangent_derivative import (
    linearize_native_tangent_step,
)
from src.engines.physics_engines.mujoco.python.native_torque_replay import (
    build_native_torque_bundle,
    replay_native_torque_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _floating_two_hinge_xml() -> str:
    return """<mujoco model="floating-two-hinge">
  <option timestep="0.01" integrator="Euler" gravity="0 0 0"/>
  <worldbody>
    <body name="pelvis">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.07 0.05" mass="1" contype="0" conaffinity="0"/>
      <body name="upper" pos="0.2 0 0">
        <joint name="hip" type="hinge" axis="0 0 1" damping="0.03"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.3"
              contype="0" conaffinity="0"/>
        <body name="lower" pos="0.3 0 0">
          <joint name="knee" type="hinge" axis="0 0 1" damping="0.02"/>
          <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.015" mass="0.2"
                contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_torque" joint="hip" gear="1" ctrllimited="true" ctrlrange="-2 2"/>
    <motor name="knee_torque" joint="knee" gear="1" ctrllimited="true" ctrlrange="-1.5 1.5"/>
  </actuator>
</mujoco>"""


def _initial(model: object) -> np.ndarray:
    mj = pytest.importorskip("mujoco")
    data = mj.MjData(model)
    data.qpos[0:3] = (0.1, -0.2, 0.8)
    data.qpos[3:7] = (np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8))
    mj.mj_normalizeQuat(model, data.qpos)
    data.qpos[7:] = (0.4, -0.3)
    data.qvel[:] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.1, 0.05)
    initial = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(model, data, initial, mj.mjtState.mjSTATE_INTEGRATION)
    return initial


def test_native_tangent_jacobians_match_manifold_perturbed_steps(
    record_property: Callable[[str, object], None],
) -> None:
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string(_floating_two_hinge_xml())
    assert (model.nq, model.nv, model.nu) == (9, 8, 2)
    initial = _initial(model)
    command = np.array([0.6, -0.4])
    result = linearize_native_tangent_step(model, initial, command)
    assert result.A.shape == (16, 16)
    assert result.B.shape == (16, 2)
    assert not result.A.flags.writeable

    def step(
        state_delta: np.ndarray, input_delta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        data = mj.MjData(model)
        mj.mj_setState(model, data, initial, mj.mjtState.mjSTATE_INTEGRATION)
        mj.mj_integratePos(model, data.qpos, state_delta[: model.nv], 1.0)
        data.qvel[:] += state_delta[model.nv :]
        data.ctrl[:] = command + input_delta
        mj.mj_step(model, data)
        return data.qpos.copy(), data.qvel.copy()

    def tangent(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        dq = np.empty(model.nv)
        mj.mj_differentiatePos(model, dq, 1.0, result.next_qpos, qpos)
        return np.r_[dq, qvel - result.next_qvel]

    eps = 1e-6
    for col in (0, 3, 5, 6, 7, 13, 15):
        delta = np.eye(2 * model.nv)[col] * eps
        plus = tangent(*step(delta, np.zeros(model.nu)))
        minus = tangent(*step(-delta, np.zeros(model.nu)))
        np.testing.assert_allclose(
            result.A[:, col], (plus - minus) / (2 * eps), atol=2e-4
        )
    for col in range(model.nu):
        delta = np.eye(model.nu)[col] * eps
        plus = tangent(*step(np.zeros(2 * model.nv), delta))
        minus = tangent(*step(np.zeros(2 * model.nv), -delta))
        np.testing.assert_allclose(
            result.B[:, col], (plus - minus) / (2 * eps), atol=2e-4
        )
    if os.environ.get("F05C_BENCHMARK_RECEIPT") == "1":
        times = []
        for _ in range(10):
            started = time.perf_counter()
            linearize_native_tangent_step(model, initial, command)
            times.append(time.perf_counter() - started)
        record_property("derivative_p50_s", float(np.percentile(times, 50)))
        record_property("derivative_p95_s", float(np.percentile(times, 95)))
        record_property("derivative_worst_s", float(max(times)))
        record_property("derivative_calls", len(times))
        record_property("A_sha256", hashlib.sha256(result.A.tobytes()).hexdigest())
        record_property("B_sha256", hashlib.sha256(result.B.tobytes()).hexdigest())


def test_floating_two_motor_torques_replay_complete_native_state(
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    mj = pytest.importorskip("mujoco")
    path = tmp_path / "floating-two-hinge.xml"
    path.write_text(_floating_two_hinge_xml(), encoding="utf-8")
    model = mj.MjModel.from_xml_path(str(path))
    initial = _initial(model)
    data = mj.MjData(model)
    mj.mj_setState(model, data, initial, mj.mjtState.mjSTATE_INTEGRATION)
    commands = np.array([[0.6, -0.4], [0.3, -0.2], [-0.1, 0.2], [-0.1, 0.2]])
    native_states = [initial.copy()]
    for command in commands[:-1]:
        data.ctrl[:] = command
        mj.mj_step(model, data)
        state = np.empty_like(initial)
        mj.mj_getState(model, data, state, mj.mjtState.mjSTATE_INTEGRATION)
        native_states.append(state)
    started = time.perf_counter()
    bundle = build_native_torque_bundle(
        path,
        initial,
        np.arange(len(commands)) * model.opt.timestep,
        commands,
        experiment_id="f05c-floating-two-hinge",
    )
    bundle_s = time.perf_counter() - started
    started = time.perf_counter()
    replay = replay_native_torque_bundle(bundle, path)
    replay_s = time.perf_counter() - started
    np.testing.assert_allclose(
        replay.integration_states, native_states, atol=1e-12, rtol=0
    )
    np.testing.assert_array_equal(replay.applied_actuator_torques, commands[:-1])
    assert bundle.model.state_schema.components[0].dimension == 9
    if os.environ.get("F05C_BENCHMARK_RECEIPT") == "1":
        record_property("bundle_s", bundle_s)
        record_property("replay_s", replay_s)
        record_property("model_sha256", bundle.model.source_model_sha256)
        record_property("initial_state_sha256", bundle.integrity.initial_state_sha256)
        record_property("applied_input_sha256", bundle.applied_input_sha256)
        record_property("policy_sha256", bundle.policy_sha256)
        record_property("state_schema_sha256", bundle.state_schema_sha256)
        record_property("time_grid_sha256", bundle.time_grid_sha256)


def test_native_tangent_derivative_rejects_rk4() -> None:
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string(
        _floating_two_hinge_xml().replace('integrator="Euler"', 'integrator="RK4"')
    )
    with pytest.raises(ValueError, match="Euler"):
        linearize_native_tangent_step(model, _initial(model), np.array([0.0, 0.0]))


def test_native_tangent_derivative_keeps_motor_order_and_rejects_hidden_drives() -> (
    None
):
    mj = pytest.importorskip("mujoco")
    original = _floating_two_hinge_xml()
    swapped = original.replace(
        '<motor name="hip_torque" joint="hip" gear="1" ctrllimited="true" ctrlrange="-2 2"/>\n'
        '    <motor name="knee_torque" joint="knee" gear="1" ctrllimited="true" ctrlrange="-1.5 1.5"/>',
        '<motor name="knee_torque" joint="knee" gear="1" ctrllimited="true" ctrlrange="-1.5 1.5"/>\n'
        '    <motor name="hip_torque" joint="hip" gear="1" ctrllimited="true" ctrlrange="-2 2"/>',
    )
    assert swapped != original
    nominal = mj.MjModel.from_xml_string(original)
    reordered = mj.MjModel.from_xml_string(swapped)
    normal = linearize_native_tangent_step(
        nominal, _initial(nominal), np.array([0.6, -0.4])
    )
    reversed_order = linearize_native_tangent_step(
        reordered, _initial(reordered), np.array([-0.4, 0.6])
    )
    assert normal.ordered_input_channel_ids == ("hip_torque", "knee_torque")
    assert reversed_order.ordered_input_channel_ids == ("knee_torque", "hip_torque")
    np.testing.assert_allclose(normal.A, reversed_order.A, atol=1e-7)
    np.testing.assert_allclose(normal.B[:, ::-1], reversed_order.B, atol=1e-7)
    np.testing.assert_allclose(normal.next_qpos, reversed_order.next_qpos, atol=1e-12)
    with pytest.raises(ValueError, match="post-limit"):
        linearize_native_tangent_step(nominal, _initial(nominal), np.array([3.0, 0.0]))
    with pytest.raises(ValueError, match="interior"):
        linearize_native_tangent_step(nominal, _initial(nominal), np.array([2.0, 0.0]))
    assisted = mj.MjModel.from_xml_string(
        original.replace('<body name="pelvis">', '<body name="pelvis" gravcomp="0.1">')
    )
    with pytest.raises(ValueError, match="unforced"):
        linearize_native_tangent_step(assisted, _initial(assisted), np.zeros(2))
    with_contact_surface = mj.MjModel.from_xml_string(
        original.replace(
            "<worldbody>",
            '<worldbody><geom name="ground" type="plane" size="1 1 0.1"/>',
        )
    )
    with pytest.raises(ValueError, match="contact-free"):
        linearize_native_tangent_step(
            with_contact_surface, _initial(with_contact_surface), np.zeros(2)
        )
