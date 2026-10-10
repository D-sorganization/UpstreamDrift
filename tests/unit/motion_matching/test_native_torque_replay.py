"""Native MuJoCo replay smoke tests; no shared integration emulation."""

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.native_torque_replay import (
    build_native_torque_bundle,
    replay_native_torque_bundle,
)

pytestmark = pytest.mark.unit


def test_contracts_resolve_pinned_tools_without_mutating_the_lab_namespace() -> None:
    import sys

    from sidekick import lab
    from src.engines.native_replay_contracts import native_replay_contract_types

    from src.engines.physics_engines.mujoco.python import native_torque_replay

    before = list(lab.__path__)
    mocap = native_torque_replay._contracts()

    root = Path(__file__).resolve().parents[3]
    vendor_mocap = root / "vendor/ud-tools/src/shared/python/sidekick/lab/mocap"
    assert Path(mocap.__file__).resolve().parent == vendor_mocap.resolve()
    assert hasattr(mocap, "ExperimentReplayBundle")
    assert list(lab.__path__) == before
    assert native_replay_contract_types() is mocap
    assert "sidekick.lab.mocap" not in sys.modules


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
    data.qpos[-1] = 0.5
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
    assert abs(torques[0][0]) == 2  # Saturation happened before saved-input export.
    bundle = build_native_torque_bundle(
        path, initial, np.arange(21) * 0.001, np.array(torques)
    )
    replay = replay_native_torque_bundle(bundle, path)
    assert model.nq != model.nv
    np.testing.assert_allclose(replay.qpos, reference, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(replay.applied_actuator_torques, torques[:-1])
    assert replay.qpos.flags.writeable is False
    np.testing.assert_array_equal(replay.integration_states[0], initial)


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


def _native_state(model: object, data: object) -> np.ndarray:
    mj = pytest.importorskip("mujoco")
    state = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(model, data, state, mj.mjtState.mjSTATE_INTEGRATION)
    return state


@pytest.mark.parametrize("field", ["qfrc_applied", "xfrc_applied"])
def test_external_loads_in_complete_initial_state_are_rejected(
    native_fixture: tuple[Path, object, object],
    field: str,
) -> None:
    path, model, data = native_fixture
    getattr(data, field).flat[0] = 1
    with pytest.raises(ValueError, match="external loads"):
        build_native_torque_bundle(
            path, _native_state(model, data), np.array([0, 0.001]), np.zeros((2, 1))
        )


def test_unormalized_quaternion_is_not_silently_repaired(
    native_fixture: tuple[Path, object, object],
) -> None:
    path, model, data = native_fixture
    data.qpos[3] = 2
    with pytest.raises(ValueError, match="quaternion"):
        build_native_torque_bundle(
            path, _native_state(model, data), np.array([0, 0.001]), np.zeros((2, 1))
        )


def test_missing_warmstart_and_native_input_state_is_not_defaulted(
    native_fixture: tuple[Path, object, object],
) -> None:
    path, model, data = native_fixture
    with pytest.raises(ValueError, match="complete.*state"):
        build_native_torque_bundle(
            path,
            _native_state(model, data)[:-1],
            np.array([0, 0.001]),
            np.zeros((2, 1)),
        )


@pytest.mark.parametrize(
    "replacement",
    [
        'gear="2"',
        'gear="1" forcelimited="true" forcerange="-1 1"',
    ],
)
def test_nonunit_transmission_or_force_clamp_cannot_be_called_saved_torque(
    native_fixture: tuple[Path, object, object],
    replacement: str,
) -> None:
    path, model, data = native_fixture
    path.write_text(path.read_text().replace('gear="1"', replacement))
    with pytest.raises(ValueError, match="unit-gain motors"):
        build_native_torque_bundle(
            path, _native_state(model, data), np.array([0, 0.001]), np.zeros((2, 1))
        )


def test_saved_torques_are_not_silently_clipped(
    native_fixture: tuple[Path, object, object],
) -> None:
    path, model, data = native_fixture
    with pytest.raises(ValueError, match="already satisfy"):
        build_native_torque_bundle(
            path, _native_state(model, data), np.array([0, 0.001]), np.full((2, 1), 2.1)
        )


def test_non_native_time_grid_is_not_interpolated_implicitly(
    native_fixture: tuple[Path, object, object],
) -> None:
    path, model, data = native_fixture
    with pytest.raises(ValueError, match="native timestep"):
        build_native_torque_bundle(
            path, _native_state(model, data), np.array([0, 0.0015]), np.zeros((2, 1))
        )


def test_model_replaced_between_validation_and_execution_is_rejected(
    native_fixture: tuple[Path, object, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.engines.physics_engines.mujoco.python import native_torque_replay as api

    path, model, data = native_fixture
    bundle = build_native_torque_bundle(
        path, _native_state(model, data), np.array([0, 0.001]), np.zeros((2, 1))
    )
    original = api._step_native

    def replace_before_step(bundle: object, path: Path) -> object:
        path.write_text(path.read_text().replace('mass="0.2"', 'mass="0.3"'))
        return original(bundle, path)

    monkeypatch.setattr(api, "_step_native", replace_before_step)
    with pytest.raises(ValueError, match="execution.*identity"):
        replay_native_torque_bundle(bundle, path)


def test_nonzero_numerical_warmstart_is_restored_exactly(
    native_fixture: tuple[Path, object, object],
) -> None:
    path, model, data = native_fixture
    data.qacc_warmstart[:] = np.arange(model.nv) * 0.1
    initial = _native_state(model, data)
    bundle = build_native_torque_bundle(
        path, initial, np.array([0, 0.001]), np.zeros((2, 1))
    )
    result = replay_native_torque_bundle(bundle, path)
    np.testing.assert_array_equal(result.integration_states[0], initial)
