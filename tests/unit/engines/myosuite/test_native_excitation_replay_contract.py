"""MyoSuite F09 admission reuses Tools T01 without qualifying physics."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from src.engines.feedback_comparison import DriveMode, FeedbackComparisonRegistry
from src.engines.feedback_native_execution import (
    NativeAdapterBinding,
    NativeReplayRequest,
    build_native_replay_report,
    execute_native_replay,
    validate_native_replay_output,
)
from src.engines.model_inventory import EngineModelInventory, TARGET_ENGINES
from src.engines.physics_engines.myosuite.python.native_excitation_replay import (
    NativeMyoSuiteExcitationReplay,
    _execute_native_steps,
    _module_bytes,
    validate_myo_suite_bundle_contract,
)
from src.engines.native_replay_contracts import native_replay_contract_types

_mocap = native_replay_contract_types()
ActuationInputKind = _mocap.ActuationInputKind
CapabilityAvailability = _mocap.CapabilityAvailability
CapabilityDeclaration = _mocap.CapabilityDeclaration
CapabilitySupport = _mocap.CapabilitySupport
InitialStateSchema = _mocap.InitialStateSchema
InputChannel = _mocap.InputChannel
InputInterpolation = _mocap.InputInterpolation
ModelIdentity = _mocap.ModelIdentity
ReplayExecutionPolicy = _mocap.ReplayExecutionPolicy
ReplayMode = _mocap.ReplayMode
StateComponentRole = _mocap.StateComponentRole
StateComponentSpec = _mocap.StateComponentSpec
build_experiment_replay_bundle = _mocap.build_experiment_replay_bundle

pytestmark = pytest.mark.unit


def test_provider_source_hashing_only_reads_modules_already_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "provider.py"
    source.write_bytes(b"frozen provider source")
    module = ModuleType("frozen_provider_test")
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    assert _module_bytes(module.__name__) == b"frozen provider source"

    with pytest.raises(ValueError, match="is not loaded"):
        _module_bytes("provider_that_must_not_be_imported")


def _registry() -> FeedbackComparisonRegistry:
    root = Path(__file__).resolve().parents[4]
    return FeedbackComparisonRegistry(EngineModelInventory.load(repo_root=root))


def _case(registry: FeedbackComparisonRegistry):
    row = registry.get("myosuite/driver", "default", DriveMode.MUSCLE_EXCITATION)
    source_sha = hashlib.sha256(b"synthetic independent MyoSuite model").hexdigest()
    native_provider_sha = hashlib.sha256(
        b"synthetic MyoSuite replay provider"
    ).hexdigest()
    components = (
        StateComponentSpec("qpos", StateComponentRole.POSITION, 1, "rad", "test-qpos"),
        StateComponentSpec(
            "qvel", StateComponentRole.VELOCITY, 1, "rad/s", "test-qvel"
        ),
        StateComponentSpec(
            "muscle_activation",
            StateComponentRole.MUSCLE_ACTIVATION,
            2,
            "1",
            "test-act",
        ),
        StateComponentSpec(
            "actuator_internal",
            StateComponentRole.ACTUATOR_INTERNAL_STATE,
            2,
            "1",
            "test-ctrl",
        ),
        StateComponentSpec(
            "integration",
            StateComponentRole.AUXILIARY,
            4,
            "native-SI",
            "test-mjSTATE_INTEGRATION",
        ),
        StateComponentSpec(
            "wrapper_state",
            StateComponentRole.AUXILIARY,
            7,
            "enum-as-float",
            "test-gym-wrapper-state",
        ),
    )
    model = ModelIdentity(
        engine_id="myosuite",
        model_id="test-env-v0",
        variant_id="native-muscle-excitation",
        model_version="3.0.0",
        source_model_sha256=source_sha,
        provider_id="myosuite-native-muscle-excitation-replay",
        provider_version="1.0.0",
        provider_sha256=native_provider_sha,
        state_schema=InitialStateSchema("test-myosuite-state", "1.0.0", components),
        ordered_input_channel_ids=("flexor", "extensor"),
        loaded_native_model_sha256=hashlib.sha256(b"compiled native model").hexdigest(),
    )
    policy = ReplayExecutionPolicy(
        replay_mode=ReplayMode.NATIVE_OWN_CONTACT,
        solver_id="test-solver",
        solver_version="3.6.0",
        integration_method="RK4;frame_skip=1",
        step_policy="fixed",
        step_size_seconds=0.01,
        initialization_policy_id="mjSTATE_INTEGRATION-plus-wrapper-state-restore",
        initialization_policy_version="1.0.0",
        input_player_id="myosuite-native-ctrl-muscle-excitation-zoh-fixed-horizon",
        input_player_version="1.0.0",
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id="myosuite-compiled-model-native-contact",
        contact_policy_version="3.0.0",
        contact_policy_sha256=model.loaded_native_model_sha256,
    )
    state = (
        ("qpos", (0.0,)),
        ("qvel", (0.0,)),
        ("muscle_activation", (0.1, 0.2)),
        ("actuator_internal", (0.1, 0.2)),
        ("integration", (0.0, 0.0, 0.1, 0.2)),
        ("wrapper_state", (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    )
    channels = (
        InputChannel("flexor", "flexor", "1"),
        InputChannel("extensor", "extensor", "1"),
    )
    bundle = build_experiment_replay_bundle(
        "opaque:synthetic-myosuite-excitation",
        model,
        (
            CapabilityDeclaration(
                "native-muscle-excitation-replay",
                True,
                CapabilitySupport.SUPPORTED,
                CapabilityAvailability.AVAILABLE,
            ),
        ),
        state,
        channels,
        ActuationInputKind.MUSCLE_EXCITATION,
        InputInterpolation.ZERO_ORDER_HOLD,
        (0.0, 0.01, 0.02),
        ((0.1, 0.2), (0.3, 0.4), (0.3, 0.4)),
        policy,
    )
    row = replace(
        row,
        source_model_sha256=source_sha,
        provider_id="synthetic-inventory-provider",
        provider_sha256=hashlib.sha256(b"synthetic F01 provider").hexdigest(),
        availability="available",
    )
    registry.rows = (row,)
    binding = NativeAdapterBinding(
        row.package_id,
        row.variant_id,
        row.drive_mode,
        model.model_id,
        model.variant_id,
        model.provider_id,
        model.provider_sha256,
        model.source_model_sha256,
        model.loaded_native_model_sha256,
        bundle.state_schema_sha256,
        bundle.input_channel_schema_sha256,
        model.ordered_input_channel_ids,
    )
    return row, bundle, binding


def test_myo_suite_bundle_contract_accepts_complete_open_loop_excitation() -> None:
    registry = _registry()
    row, bundle, binding = _case(registry)

    validate_myo_suite_bundle_contract(row, binding, bundle)


def test_myo_suite_bundle_contract_rejects_torque_inventory_row() -> None:
    registry = _registry()
    row, bundle, binding = _case(registry)
    row = replace(row, drive_mode=DriveMode.TORQUE)
    with pytest.raises(ValueError, match="torque inventory row"):
        validate_myo_suite_bundle_contract(row, binding, bundle)


def test_wrapper_state_restore_uses_the_explicit_supported_chain() -> None:
    import src.engines.physics_engines.myosuite.python.native_excitation_replay as provider

    def wrapper(module: str, name: str, **attributes: object) -> object:
        wrapper_type = type(name, (), {"__module__": module})
        instance = wrapper_type()
        for attribute, value in attributes.items():
            setattr(instance, attribute, value)
        return instance

    checker = wrapper(
        "gymnasium.wrappers.common",
        "PassiveEnvChecker",
        checked_reset=True,
        checked_step=True,
        checked_render=False,
        close_called=False,
    )
    order = wrapper("gymnasium.wrappers.common", "OrderEnforcing", _has_reset=True)
    limit = wrapper("gymnasium.wrappers.common", "TimeLimit", _elapsed_steps=None)
    myosuite = wrapper(
        "myosuite.envs.wrappers",
        "MjInstabilityTerminationWrapper",
        mj_instability_termination=False,
    )
    checker.env = None
    order.env = checker
    limit.env = order
    myosuite.env = limit

    chain, names = provider._wrapper_chain(myosuite)

    assert chain == [myosuite, limit, order, checker]
    assert names == (
        "myosuite.envs.wrappers.MjInstabilityTerminationWrapper",
        "gymnasium.wrappers.common.TimeLimit",
        "gymnasium.wrappers.common.OrderEnforcing",
        "gymnasium.wrappers.common.PassiveEnvChecker",
    )
    provider._restore_wrapper_state(myosuite, (5.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0))
    assert provider._wrapper_state(myosuite)[1] == (
        5.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
    )


def _native_step_fixture():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """<mujoco model="native-clock-fixture">
          <worldbody><body><joint name="hinge" type="hinge"/>
            <geom type="sphere" size="0.05" mass="1"/>
          </body></worldbody>
          <actuator><motor name="motor" joint="hinge" gear="1"/></actuator>
        </mujoco>"""
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    integration = np.empty(
        mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    )
    mujoco.mj_getState(model, data, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
    times = np.arange(3, dtype=float) * model.opt.timestep
    controls = np.array([[0.1], [0.2], [0.2]])
    states = {
        "integration": integration,
        "wrapper_state": np.zeros(7, dtype=float),
    }
    bundle = SimpleNamespace(
        applied_input_sha256="input-digest", policy_sha256="policy-digest"
    )
    return mujoco, model, data, times, controls, states, bundle


def test_native_myo_output_uses_observed_clock_and_complete_integration_state() -> None:
    mj, model, data, times, controls, states, bundle = _native_step_fixture()
    replay = _execute_native_steps(
        SimpleNamespace(frame_skip=1), model, data, times, controls, states, bundle
    )

    assert np.array_equal(replay.time_seconds, times)
    assert replay.integration_states.shape == (
        len(times),
        len(states["integration"]),
    )
    assert np.array_equal(replay.integration_states[0], states["integration"])
    assert replay.integration_states[-1].shape == states["integration"].shape


def test_native_myo_replay_rejects_global_mujoco_callbacks() -> None:
    mj, model, data, times, controls, states, bundle = _native_step_fixture()
    previous = mj.get_mjcb_control()
    mj.set_mjcb_control(lambda _model, _data: None)
    try:
        with pytest.raises(ValueError, match="global MuJoCo callbacks"):
            _execute_native_steps(
                SimpleNamespace(frame_skip=1),
                model,
                data,
                times,
                controls,
                states,
                bundle,
            )
    finally:
        mj.set_mjcb_control(previous)


def test_native_myo_replay_rejects_clock_drift_and_nonfinite_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mj, model, data, times, controls, states, bundle = _native_step_fixture()
    native_step = mj.mj_step

    def drift_clock(actual_model, actual_data, nstep=1):
        native_step(actual_model, actual_data, nstep=nstep)
        actual_data.time += actual_model.opt.timestep

    monkeypatch.setattr(mj, "mj_step", drift_clock)
    with pytest.raises(ValueError, match="observed native time"):
        _execute_native_steps(
            SimpleNamespace(frame_skip=1),
            model,
            data,
            times,
            controls,
            states,
            bundle,
        )

    monkeypatch.undo()
    mj, model, data, times, controls, states, bundle = _native_step_fixture()
    native_step = mj.mj_step

    def nonfinite_state(actual_model, actual_data, nstep=1):
        native_step(actual_model, actual_data, nstep=nstep)
        actual_data.qpos[0] = np.nan

    monkeypatch.setattr(mj, "mj_step", nonfinite_state)
    with pytest.raises(ValueError, match="nonfinite native state"):
        _execute_native_steps(
            SimpleNamespace(frame_skip=1),
            model,
            data,
            times,
            controls,
            states,
            bundle,
        )


def test_t01_contract_rejects_observation_enabled_replay_policy() -> None:
    registry = _registry()
    _, bundle, _ = _case(registry)

    with pytest.raises(ValueError, match="forbids observation"):
        replace(bundle.policy, observation_access=True)


def test_receipt_binds_exact_applied_excitation_and_keeps_unqualified() -> None:
    registry = _registry()
    row, bundle, binding = _case(registry)
    replay = NativeMyoSuiteExcitationReplay(
        time_seconds=np.array([0.0, 0.01, 0.02]),
        qpos=np.array([[0.0], [0.01], [0.03]]),
        qvel=np.array([[0.0], [1.0], [2.0]]),
        muscle_activations=np.array([[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]]),
        actuator_controls=np.array([[0.1, 0.2], [0.1, 0.2], [0.3, 0.4]]),
        integration_states=np.array(
            [[0.0, 0.0, 0.1, 0.2], [0.0, 0.1, 0.15, 0.25], [0.01, 0.2, 0.2, 0.3]]
        ),
        wrapper_states=np.tile(np.asarray(bundle.initial_state[-1].values), (3, 1)),
        applied_muscle_excitations=np.asarray(bundle.input_history.values[:-1]),
        input_sha256=bundle.applied_input_sha256,
        policy_sha256=bundle.policy_sha256,
    )

    receipt = validate_native_replay_output(row, binding, bundle, replay)

    assert receipt.input_kind == "muscle_excitation"
    assert receipt.initial_state_sha256 == bundle.integrity.initial_state_sha256
    assert receipt.applied_input_sha256 == bundle.applied_input_sha256
    assert receipt.qualification == "unqualified"
    assert receipt.full_state and receipt.full_horizon
    assert receipt.state_reset_count is None
    incomplete_state = replace(
        replay, integration_states=replay.integration_states[:, :-1]
    )
    with pytest.raises(
        ValueError, match="auxiliary state differs from T01 state schema"
    ):
        validate_native_replay_output(row, binding, bundle, incomplete_state)


def test_executor_dispatches_to_explicit_myo_suite_excitation_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.engines.physics_engines.myosuite.python.native_excitation_replay as provider

    registry = _registry()
    row, bundle, binding = _case(registry)
    output = NativeMyoSuiteExcitationReplay(
        time_seconds=np.array([0.0, 0.01, 0.02]),
        qpos=np.array([[0.0], [0.01], [0.03]]),
        qvel=np.array([[0.0], [1.0], [2.0]]),
        muscle_activations=np.array([[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]]),
        actuator_controls=np.array([[0.1, 0.2], [0.1, 0.2], [0.3, 0.4]]),
        integration_states=np.array(
            [[0.0, 0.0, 0.1, 0.2], [0.0, 0.1, 0.15, 0.25], [0.01, 0.2, 0.2, 0.3]]
        ),
        wrapper_states=np.tile(np.asarray(bundle.initial_state[-1].values), (3, 1)),
        applied_muscle_excitations=np.asarray(bundle.input_history.values[:-1]),
        input_sha256=bundle.applied_input_sha256,
        policy_sha256=bundle.policy_sha256,
    )
    observed: list[tuple[object, str]] = []

    def replay(
        actual_bundle: object, environment_id: str
    ) -> NativeMyoSuiteExcitationReplay:
        observed.append((actual_bundle, environment_id))
        return output

    monkeypatch.setattr(provider, "replay_native_myo_suite_excitation_bundle", replay)
    registry.rows = (row,)
    receipt = execute_native_replay(
        NativeReplayRequest(binding, bundle, Path("test-env-v0")), registry
    )

    assert observed == [(bundle, "test-env-v0")]
    assert receipt.engine == "myosuite"
    assert receipt.qualification == "unqualified"
    assert receipt.input_kind == bundle.input_history.input_kind.value
    assert receipt.interpolation == bundle.input_history.interpolation.value
    assert receipt.timebase_id == bundle.input_history.timebase_id
    assert receipt.evidence_mode == bundle.policy.replay_mode.value


def test_required_six_engine_denominator_stays_blocking_without_real_bindings() -> None:
    registry = _registry()
    report = build_native_replay_report(registry, ())

    assert set(report.required_engine_ids) == set(TARGET_ENGINES)
    assert "myosuite" in report.missing_engine_ids
    assert not report.is_complete
