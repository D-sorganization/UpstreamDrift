"""Native floating-root muscle action preserves activation and restart state."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
mj = pytest.importorskip("mujoco")
pytest.importorskip("crocoddyl")
if mj.__version__ != "3.8.0":
    pytest.skip(
        "native activation fixture requires MuJoCo 3.8.0", allow_module_level=True
    )


def test_shared_calculus_bytes_are_bound_to_native_replay_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.engines.physics_engines.mujoco.python import (
        native_activation_manifold as adapter,
        native_manifold_calculus as calculus,
    )

    path, full = _fixture(tmp_path)
    original = adapter.load_native_activation_provider(path)
    changed_source = tmp_path / "changed_calculus.py"
    changed_source.write_bytes(Path(calculus.__file__).read_bytes() + b"\n# changed\n")
    monkeypatch.setattr(calculus, "__file__", str(changed_source))
    changed = adapter.load_native_activation_provider(path)
    assert changed.identity.adapter_sha256 != original.identity.adapter_sha256
    with pytest.raises(ValueError, match="identity changed"):
        adapter.replay_native_activation_commands(
            path, full, np.array([[0.4]]), original.identity
        )


def _model_xml() -> str:
    fixture = (
        Path(__file__).resolve().parents[3]
        / "tests/fixtures/feedback_controls/native_floating_muscle_11958.xml"
    )
    return fixture.read_text(encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, np.ndarray]:
    path = tmp_path / "floating-muscle.xml"
    path.write_text(_model_xml(), encoding="utf-8")
    model = mj.MjModel.from_xml_path(str(path))
    assert (model.nq, model.nv, model.na, model.nu) == (8, 7, 1, 1)
    data = mj.MjData(model)
    data.qpos[:3] = [0.03, -0.02, 0.01]
    data.qpos[3:7] = [0.99, 0.0, 0.0, np.sqrt(1 - 0.99**2)]
    data.qpos[7] = 0.2
    data.qvel[:] = [0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.1]
    data.act[0] = 0.3
    data.qacc_warmstart[:] = np.linspace(0.001, 0.007, model.nv)
    full = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(model, data, full, mj.mjtState.mjSTATE_INTEGRATION)
    return path, full


def _independent_step(
    path: Path, state: np.ndarray, command: float, physical: np.ndarray | None = None
) -> np.ndarray:
    model = mj.MjModel.from_xml_path(str(path))
    model.opt.disableflags |= int(
        mj.mjtDisableBit.mjDSBL_AUTORESET | mj.mjtDisableBit.mjDSBL_WARMSTART
    )
    data = mj.MjData(model)
    mj.mj_setState(model, data, state, mj.mjtState.mjSTATE_INTEGRATION)
    if physical is not None:
        data.qpos[:] = physical[: model.nq]
        data.qvel[:] = physical[model.nq : model.nq + model.nv]
        data.act[:] = physical[model.nq + model.nv :]
    data.ctrl[0] = command
    mj.mj_step(model, data)
    full = np.empty_like(state)
    mj.mj_getState(model, data, full, mj.mjtState.mjSTATE_INTEGRATION)
    return full


def test_floating_activation_action_matches_independent_native_direction(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.mujoco.python.native_activation_manifold import (
        NativeActivationAction,
        load_native_activation_provider,
    )

    path, full = _fixture(tmp_path)
    provider = load_native_activation_provider(path)
    state = provider.state
    physical = provider.project_physical(full)
    assert state.nx == 16 == provider.model.nq + provider.model.nv + provider.model.na
    assert state.ndx == 15 == 2 * provider.model.nv + provider.model.na
    assert provider.identity.input_kind == "actuator_command"
    assert provider.identity.ordered_input_channel_ids == ("drive",)
    assert provider.identity.source_closure == "self_contained_xml"
    assert "warmstart-disabled" in provider.identity.history_policy
    with pytest.raises(ValueError, match="source_model_sha256"):
        replace(provider.identity, source_model_sha256="invalid")
    step = provider.linearize(physical, np.array([0.4]))
    assert step.A.shape == (15, 15)
    assert step.B.shape == (15, 1)
    assert not step.A.flags.writeable
    assert not step.next_physical_state.flags.writeable
    assert abs(step.A[13, 14]) > 1e-5  # activation changes hinge velocity
    assert step.B[14, 0] > 0  # command changes activation
    np.testing.assert_array_equal(
        step.next_physical_state,
        provider.project_physical(_independent_step(path, full, 0.4)),
    )

    dx = np.linspace(-0.2, 0.3, state.ndx)
    du = 0.1
    epsilon = 1e-5
    minus = _independent_step(
        path, full, 0.4 - epsilon * du, state.integrate(physical, -epsilon * dx)
    )
    plus = _independent_step(
        path, full, 0.4 + epsilon * du, state.integrate(physical, epsilon * dx)
    )
    actual = state.diff(
        provider.project_physical(minus), provider.project_physical(plus)
    ) / (2 * epsilon)
    expected = step.A @ dx + step.B[:, 0] * du
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    coarse_epsilon = 2 * epsilon
    coarse_minus = _independent_step(
        path,
        full,
        0.4 - coarse_epsilon * du,
        state.integrate(physical, -coarse_epsilon * dx),
    )
    coarse_plus = _independent_step(
        path,
        full,
        0.4 + coarse_epsilon * du,
        state.integrate(physical, coarse_epsilon * dx),
    )
    coarse = state.diff(
        provider.project_physical(coarse_minus), provider.project_physical(coarse_plus)
    ) / (2 * coarse_epsilon)
    assert np.max(np.abs(actual - expected)) < 2e-7
    assert np.max(np.abs(coarse - expected)) < 2e-6
    assert np.max(np.abs(actual - expected)) <= np.max(np.abs(coarse - expected)) + 1e-7

    action = NativeActivationAction(provider, step.next_physical_state)
    data = action.createData()
    action.calc(data, physical, np.array([0.4]))
    action.calcDiff(data, physical, np.array([0.4]))
    np.testing.assert_array_equal(data.xnext, step.next_physical_state)
    np.testing.assert_allclose(data.Fx, step.A)
    np.testing.assert_allclose(np.asarray(data.Fu).reshape(15, 1), step.B)


@pytest.mark.parametrize("seed", (17, 41))
def test_native_manifold_identities_and_quaternion_sign(
    tmp_path: Path, seed: int
) -> None:
    import crocoddyl

    from src.engines.physics_engines.mujoco.python.native_activation_manifold import (
        load_native_activation_provider,
    )

    path, full = _fixture(tmp_path)
    provider = load_native_activation_provider(path)
    state = provider.state
    base = provider.project_physical(full)
    assert np.all(
        (state.zero()[-provider.model.na :] > 0)
        & (state.zero()[-provider.model.na :] < 1)
    )
    rng = np.random.default_rng(seed)
    tangent = rng.normal(size=state.ndx) * 0.015
    tangent[-1] = 0.02
    displaced = state.integrate(base, tangent)
    np.testing.assert_allclose(state.diff(base, displaced), tangent, atol=2e-10, rtol=0)
    np.testing.assert_allclose(
        state.diff(state.integrate(base, state.diff(base, displaced)), displaced),
        0,
        atol=2e-10,
        rtol=0,
    )
    negated = base.copy()
    negated[3:7] *= -1
    np.testing.assert_allclose(state.diff(base, negated), 0, atol=2e-10, rtol=0)
    first, second = state.Jdiff(base, base)
    np.testing.assert_allclose(first, -np.eye(state.ndx), atol=2e-5, rtol=0)
    np.testing.assert_allclose(second, np.eye(state.ndx), atol=2e-5, rtol=0)
    j_input = state.Jintegrate(base, tangent, crocoddyl.Jcomponent.second)[0]
    j_output = state.Jdiff(base, displaced, crocoddyl.Jcomponent.second)[0]
    np.testing.assert_allclose(j_output @ j_input, np.eye(state.ndx), atol=4e-4, rtol=0)


def test_fresh_replay_binds_compiled_law_and_full_state(tmp_path: Path) -> None:
    from src.engines.physics_engines.mujoco.python.native_activation_manifold import (
        load_native_activation_provider,
        replay_native_activation_commands,
    )

    path, full = _fixture(tmp_path)
    provider = load_native_activation_provider(path)
    commands = np.array([[0.4], [0.5], [0.45]])
    replay = replay_native_activation_commands(path, full, commands, provider.identity)
    assert replay.shape == (4, full.size)
    np.testing.assert_array_equal(replay[0], full)
    independent = full.copy()
    for index, command in enumerate(commands, 1):
        independent = _independent_step(path, independent, float(command[0]))
        np.testing.assert_array_equal(replay[index], independent)
    path.write_text(
        _model_xml().replace('force="100"', 'force="101"'), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identity"):
        replay_native_activation_commands(path, full, commands, provider.identity)


def test_physical_state_reduction_requires_history_independence(tmp_path: Path) -> None:
    from src.engines.physics_engines.mujoco.python.native_activation_manifold import (
        load_native_activation_provider,
    )

    path, full = _fixture(tmp_path)
    provider = load_native_activation_provider(path)
    model = provider.model
    changed = mj.MjData(model)
    mj.mj_setState(model, changed, full, mj.mjtState.mjSTATE_INTEGRATION)
    changed.time = 0.125
    changed.qacc_warmstart[:] = np.linspace(-0.7, 0.4, model.nv)
    second = np.empty_like(full)
    mj.mj_getState(model, changed, second, mj.mjtState.mjSTATE_INTEGRATION)
    assert not np.array_equal(full, second)
    np.testing.assert_array_equal(
        provider.project_physical(full), provider.project_physical(second)
    )
    first_next = _independent_step(path, full, 0.4)
    second_next = _independent_step(path, second, 0.4)
    np.testing.assert_array_equal(
        provider.project_physical(first_next), provider.project_physical(second_next)
    )
    assert model.opt.disableflags & int(mj.mjtDisableBit.mjDSBL_WARMSTART)


def test_source_and_active_constraint_boundaries_fail_closed(tmp_path: Path) -> None:
    from src.engines.physics_engines.mujoco.python.native_activation_manifold import (
        load_native_activation_provider,
    )

    path, full = _fixture(tmp_path)
    path.write_text('<mujoco><include file="other.xml"/></mujoco>', encoding="utf-8")
    with pytest.raises(ValueError, match="source closure"):
        load_native_activation_provider(path)
    path.write_text(
        _model_xml().replace('contype="0"', 'contype="1"'), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="contact"):
        load_native_activation_provider(path)
    path.write_text(_model_xml(), encoding="utf-8")
    provider = load_native_activation_provider(path)
    data = mj.MjData(provider.model)
    mj.mj_setState(provider.model, data, full, mj.mjtState.mjSTATE_INTEGRATION)
    data.qpos[7] = 1.0
    near_limit = np.empty_like(full)
    mj.mj_getState(provider.model, data, near_limit, mj.mjtState.mjSTATE_INTEGRATION)
    with pytest.raises(ValueError, match="limit|constraint"):
        provider.linearize(provider.project_physical(near_limit), np.array([0.4]))


def test_versioned_frozen_command_bundle_replays_full_native_state(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.mujoco.python.native_activation_bundle import (
        build_native_activation_bundle,
        replay_native_activation_bundle,
    )
    from src.engines.physics_engines.mujoco.python.native_torque_replay import (
        _contracts,
    )

    path, full = _fixture(tmp_path)
    commands = np.array([[0.4], [0.5], [0.45]])
    bundle = build_native_activation_bundle(path, full, commands, experiment_id="f05e")
    contracts = _contracts()
    assert (
        bundle.input_history.input_kind == contracts.ActuationInputKind.ACTUATOR_COMMAND
    )
    assert (
        bundle.input_history.interpolation
        == contracts.InputInterpolation.ZERO_ORDER_HOLD
    )
    assert bundle.input_history.channels[0].unit == "1"
    assert bundle.input_history.channels[0].channel_id == "drive"
    assert bundle.input_history.values[-1] == bundle.input_history.values[-2]
    assert any(
        component.role == contracts.StateComponentRole.MUSCLE_ACTIVATION
        for component in bundle.model.state_schema.components
    )
    assert not bundle.policy.observation_access
    assert not bundle.policy.state_feedback_access
    assert not bundle.policy.state_reset_allowed
    replay = replay_native_activation_bundle(bundle, path)
    np.testing.assert_array_equal(replay.integration_states[0], full)
    for index, command in enumerate(commands, 1):
        np.testing.assert_array_equal(
            replay.integration_states[index],
            _independent_step(
                path, replay.integration_states[index - 1], float(command[0])
            ),
        )
    assert replay.applied_input_sha256 == bundle.applied_input_sha256
    assert replay.policy_sha256 == bundle.policy_sha256
    assert not replay.integration_states.flags.writeable
    with pytest.raises(ValueError, match="hash|policy"):
        replace(bundle, policy=replace(bundle.policy, input_player_id="feedback"))
    initial = tuple(
        (
            item.component_id,
            (item.values[0] + 0.01, *item.values[1:])
            if item.component_id == "activation"
            else item.values,
        )
        for item in bundle.initial_state
    )
    conflicting_activation = contracts.build_experiment_replay_bundle(
        bundle.experiment_id,
        bundle.model,
        bundle.capabilities,
        initial,
        bundle.input_history.channels,
        bundle.input_history.input_kind,
        bundle.input_history.interpolation,
        bundle.input_history.time_seconds,
        bundle.input_history.values,
        bundle.policy,
    )
    with pytest.raises(ValueError, match="state|policy"):
        replay_native_activation_bundle(conflicting_activation, path)
    wrong_target = replace(bundle.input_history.channels[0], target_id="other")
    conflicting_channel = contracts.build_experiment_replay_bundle(
        bundle.experiment_id,
        bundle.model,
        bundle.capabilities,
        tuple((item.component_id, item.values) for item in bundle.initial_state),
        (wrong_target,),
        bundle.input_history.input_kind,
        bundle.input_history.interpolation,
        bundle.input_history.time_seconds,
        bundle.input_history.values,
        bundle.policy,
    )
    with pytest.raises(ValueError, match="channel|model"):
        replay_native_activation_bundle(conflicting_channel, path)
