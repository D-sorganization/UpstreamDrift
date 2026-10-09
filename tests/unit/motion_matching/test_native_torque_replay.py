"""Native MuJoCo replay smoke tests; no shared integration emulation."""

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.native_torque_replay import (
    build_native_torque_bundle,
    replay_native_torque_bundle,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def native_fixture(tmp_path: Path) -> tuple[Path, object, object]:
    mj = pytest.importorskip("mujoco")
    path = tmp_path / "synthetic_native_torque.xml"
    path.write_text(
        """<mujoco><option timestep="0.001" integrator="RK4" gravity="0 0 0"/>
    <worldbody><body name="floating" pos="0 0 1"><freejoint/>
    <geom type="sphere" size="0.1" mass="1"/>
    <body name="link" pos="0.2 0 0"><joint name="hinge" axis="0 1 0"/>
    <geom type="capsule" size="0.02 0.1" mass="0.2"/></body></body></worldbody>
    <actuator><motor name="hinge_torque" joint="hinge" gear="1"
      ctrllimited="true" ctrlrange="-2 2"/></actuator></mujoco>""",
        encoding="utf-8",
    )
    model = mj.MjModel.from_xml_path(str(path))
    data = mj.MjData(model)
    data.qvel[:] = 0.01
    return path, model, data


def test_native_replay_reproduces_held_feedback_inputs_with_quaternion_state(
    native_fixture: tuple[Path, object, object],
) -> None:
    mj = pytest.importorskip("mujoco")
    path, model, data = native_fixture
    initial = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(model, data, initial, mj.mjtState.mjSTATE_INTEGRATION)
    reference = [data.qpos.copy()]
    torques = []
    for _ in range(20):
        # A reference-only controller is sampled once per native step. The
        # independent replay API receives no controller or measured states.
        torque = float(np.clip(-5 * data.qpos[-1] - data.qvel[-1], -2, 2))
        data.ctrl[:] = torque
        torques.append((torque,))
        mj.mj_step(model, data)
        reference.append(data.qpos.copy())
    torques.append(torques[-1])  # Frozen terminal ZOH sentinel, never stepped.
    bundle = build_native_torque_bundle(
        path, initial, np.arange(21) * 0.001, np.array(torques)
    )
    replay = replay_native_torque_bundle(bundle, path)
    assert model.nq != model.nv
    np.testing.assert_allclose(replay.qpos, reference, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(replay.applied_actuator_torques, torques[:-1])
    assert replay.qpos.flags.writeable is False


def test_native_callback_is_rejected_without_evaluating_observations(
    native_fixture: tuple[Path, object, object],
) -> None:
    mj = pytest.importorskip("mujoco")
    path, model, data = native_fixture
    initial = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(model, data, initial, mj.mjtState.mjSTATE_INTEGRATION)
    bundle = build_native_torque_bundle(
        path, initial, np.array([0, 0.001]), np.zeros((2, 1))
    )

    def poisoned(*args: object) -> None:
        pytest.fail(
            "Independent replay accessed a forbidden observation/controller callback"
        )

    previous = mj.get_mjcb_control()
    try:
        mj.set_mjcb_control(poisoned)
        with pytest.raises(ValueError, match="callback"):
            replay_native_torque_bundle(bundle, path)
    finally:
        mj.set_mjcb_control(previous)
