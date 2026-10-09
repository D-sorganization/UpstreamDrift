"""MyoSuite F09 admission reuses Tools T01 without qualifying physics."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

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
    validate_myo_suite_bundle_contract,
)
from src.shared.python._seam_redirect import extend_sidekick_lab_path

extend_sidekick_lab_path()

from sidekick.lab.mocap import (
    ActuationInputKind,
    CapabilityAvailability,
    CapabilityDeclaration,
    CapabilitySupport,
    InitialStateSchema,
    InputChannel,
    InputInterpolation,
    ModelIdentity,
    ReplayExecutionPolicy,
    ReplayMode,
    StateComponentRole,
    StateComponentSpec,
    build_experiment_replay_bundle,
)

pytestmark = pytest.mark.unit


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
