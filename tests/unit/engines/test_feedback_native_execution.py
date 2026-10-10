"""Strict native bundle execution receipts preserve the F01 denominator."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.feedback_comparison import DriveMode, FeedbackComparisonRegistry
from src.engines.feedback_native_execution import (
    NativeAdapterBinding,
    NativeReplayRequest,
    _validate_command_evidence_row,
    build_native_replay_report,
    execute_native_replay,
    validate_native_replay_output,
)
from src.engines.model_inventory import EngineModelInventory, TARGET_ENGINES
from src.shared.python._seam_redirect import load_pinned_tools_package

load_pinned_tools_package("sidekick.lab.mocap")

from _pinned_tools__sidekick__lab__mocap import (  # type: ignore[import-not-found]
    ActuationInputKind,
    CapabilityAvailability,
    CapabilityDeclaration,
    CapabilitySupport,
    InitialStateSchema,
    InputChannel,
    InputInterpolation,
    DriveMode as T02DriveMode,
    ModelIdentity,
    ReplayExecutionPolicy,
    ReplayMode,
    StateComponentRole,
    StateComponentSpec,
    ComparisonEvidenceRow,
    EvidenceArtifactKind,
    EvidenceArtifactReference,
    ImplementationEvidence,
    ImplementationEvidenceKind,
    build_experiment_replay_bundle,
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def registry() -> FeedbackComparisonRegistry:
    root = Path(__file__).resolve().parents[3]
    return FeedbackComparisonRegistry(EngineModelInventory.load(repo_root=root))


def _bundle(registry: FeedbackComparisonRegistry):
    row = registry.get("mujoco/driver", "default", DriveMode.TORQUE)
    source_sha256 = hashlib.sha256(b"independent synthetic source").hexdigest()
    model = ModelIdentity(
        engine_id="mujoco",
        model_id="synthetic-test-model",
        variant_id="test-unit-hinge",
        model_version="1.0.0",
        source_model_sha256=source_sha256,
        provider_id="mujoco-native-torque-replay",
        provider_version="1.0.0",
        provider_sha256="a" * 64,
        state_schema=InitialStateSchema(
            "test-state",
            "1.0.0",
            (
                StateComponentSpec(
                    "qpos", StateComponentRole.POSITION, 1, "rad", "hinge"
                ),
                StateComponentSpec(
                    "qvel", StateComponentRole.VELOCITY, 1, "rad/s", "hinge-rate"
                ),
                StateComponentSpec(
                    "integration",
                    StateComponentRole.AUXILIARY,
                    1,
                    "native-SI",
                    "test-integration-state",
                ),
            ),
        ),
        ordered_input_channel_ids=("motor",),
        loaded_native_model_sha256="b" * 64,
    )
    policy = ReplayExecutionPolicy(
        replay_mode=ReplayMode.NATIVE_OWN_CONTACT,
        solver_id="synthetic-test-solver",
        solver_version="1.0.0",
        integration_method="synthetic-test-step",
        step_policy="fixed",
        step_size_seconds=0.01,
        initialization_policy_id="test-native-state-restore",
        initialization_policy_version="1.0.0",
        input_player_id="test-zoh-player",
        input_player_version="1.0.0",
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id="test-native-contact",
        contact_policy_version="1.0.0",
        contact_policy_sha256="c" * 64,
    )
    bundle = build_experiment_replay_bundle(
        "opaque:test-native-replay",
        model,
        (
            CapabilityDeclaration(
                "native-unit-motor-replay",
                True,
                CapabilitySupport.SUPPORTED,
                CapabilityAvailability.AVAILABLE,
            ),
        ),
        (("qpos", (0.0,)), ("qvel", (0.0,)), ("integration", (0.0,))),
        (InputChannel("motor", "motor", "N*m", "hinge"),),
        ActuationInputKind.ACTUATOR_TORQUE,
        InputInterpolation.ZERO_ORDER_HOLD,
        (0.0, 0.01, 0.02),
        ((0.1,), (0.2,), (0.2,)),
        policy,
    )
    test_row = replace(
        row,
        source_model_sha256=source_sha256,
        provider_id="synthetic-inventory-provider",
        provider_sha256="e" * 64,
        availability="available",
    )
    registry.rows = (test_row,)
    binding = NativeAdapterBinding(
        package_id=test_row.package_id,
        variant_id=test_row.variant_id,
        drive_mode=test_row.drive_mode,
        native_model_id=model.model_id,
        native_variant_id=model.variant_id,
        native_execution_provider_id=model.provider_id,
        native_execution_provider_sha256=model.provider_sha256,
        source_model_sha256=model.source_model_sha256,
        loaded_native_model_sha256=model.loaded_native_model_sha256,
        state_schema_sha256=bundle.state_schema_sha256,
        input_channel_schema_sha256=bundle.input_channel_schema_sha256,
        ordered_input_channel_ids=model.ordered_input_channel_ids,
    )
    return test_row, bundle, binding


def _output(bundle):
    from src.engines.physics_engines.mujoco.python.native_torque_replay import (
        NativeTorqueReplay,
    )

    return NativeTorqueReplay(
        time_seconds=np.asarray(bundle.input_history.time_seconds),
        qpos=np.array([[0.0], [0.001], [0.003]]),
        qvel=np.array([[0.0], [0.1], [0.2]]),
        integration_states=np.array([[0.0], [0.1], [0.2]]),
        applied_actuator_torques=np.asarray(bundle.input_history.values[:-1]),
        generalized_actuator_torques=np.array([[0.1], [0.2]]),
        input_sha256=bundle.applied_input_sha256,
        policy_sha256=bundle.policy_sha256,
    )


def test_native_receipt_binds_bundle_and_actual_output_without_qualification(
    registry: FeedbackComparisonRegistry,
) -> None:
    row, bundle, binding = _bundle(registry)

    receipt = validate_native_replay_output(row, binding, bundle, _output(bundle))

    assert receipt.package_id == row.package_id
    assert receipt.inventory_provider_id == row.provider_id
    assert receipt.native_execution_provider_id == bundle.model.provider_id
    assert receipt.initial_state_sha256 == bundle.integrity.initial_state_sha256
    assert receipt.applied_input_sha256 == bundle.applied_input_sha256
    assert receipt.policy_sha256 == bundle.policy_sha256
    assert receipt.full_horizon
    assert receipt.qualification == "unqualified"
    assert not receipt.state_reset_allowed
    assert receipt.state_reset_count is None


def test_receipt_rejects_changed_native_initial_numerical_state(
    registry: FeedbackComparisonRegistry,
) -> None:
    row, bundle, binding = _bundle(registry)
    output = _output(bundle)
    output.integration_states[0, 0] = 1.0

    with pytest.raises(ValueError, match="initial numerical state"):
        validate_native_replay_output(row, binding, bundle, output)


def test_pinocchio_qv_state_is_validated_without_fabricated_cache_state(
    registry: FeedbackComparisonRegistry,
) -> None:
    from src.engines.physics_engines.pinocchio.python.native_torque_replay import (
        NativePinocchioTorqueReplay,
    )

    inventory_row = registry.get("pinocchio/driver", "default", DriveMode.TORQUE)
    _, source_bundle, _ = _bundle(registry)
    source_model = replace(
        source_bundle.model,
        engine_id="pinocchio",
        provider_id="pinocchio-native-torque-replay",
        state_schema=InitialStateSchema(
            "test-pinocchio-qv-state",
            "1.0.0",
            source_bundle.model.state_schema.components[:2],
        ),
    )
    bundle = build_experiment_replay_bundle(
        source_bundle.experiment_id,
        source_model,
        source_bundle.capabilities,
        tuple(
            (item.component_id, item.values) for item in source_bundle.initial_state[:2]
        ),
        source_bundle.input_history.channels,
        source_bundle.input_history.input_kind,
        source_bundle.input_history.interpolation,
        source_bundle.input_history.time_seconds,
        source_bundle.input_history.values,
        source_bundle.policy,
    )
    source_sha = source_model.source_model_sha256
    row = replace(
        inventory_row,
        source_model_sha256=source_sha,
        provider_id="synthetic-inventory-provider",
        provider_sha256="e" * 64,
        availability="available",
    )
    registry.rows = (row,)
    binding = NativeAdapterBinding(
        package_id=row.package_id,
        variant_id=row.variant_id,
        drive_mode=row.drive_mode,
        native_model_id=source_model.model_id,
        native_variant_id=source_model.variant_id,
        native_execution_provider_id=source_model.provider_id,
        native_execution_provider_sha256=source_model.provider_sha256,
        source_model_sha256=source_model.source_model_sha256,
        loaded_native_model_sha256=source_model.loaded_native_model_sha256,
        state_schema_sha256=bundle.state_schema_sha256,
        input_channel_schema_sha256=bundle.input_channel_schema_sha256,
        ordered_input_channel_ids=source_model.ordered_input_channel_ids,
    )
    output = NativePinocchioTorqueReplay(
        time_seconds=np.asarray(bundle.input_history.time_seconds),
        qpos=np.array([[0.0], [0.001], [0.003]]),
        qvel=np.array([[0.0], [0.1], [0.2]]),
        applied_actuator_torques=np.asarray(bundle.input_history.values[:-1]),
        generalized_actuator_torques=np.array([[0.1], [0.2]]),
        input_sha256=bundle.applied_input_sha256,
        policy_sha256=bundle.policy_sha256,
    )

    receipt = validate_native_replay_output(row, binding, bundle, output)

    assert receipt.engine == "pinocchio"
    assert receipt.nq == receipt.nv == 1
    assert receipt.full_state and receipt.full_horizon
    assert receipt.initial_state_sha256 == bundle.integrity.initial_state_sha256
    assert receipt.state_reset_count is None


@pytest.mark.parametrize(
    ("binding_field", "value", "message"),
    [
        ("source_model_sha256", "d" * 64, "source model"),
        ("native_variant_id", "unit-hinge-motors", "variant"),
        ("native_execution_provider_id", "unregistered-adapter", "provider"),
        ("ordered_input_channel_ids", ("other",), "channel"),
    ],
)
def test_binding_rejects_unmapped_or_stale_native_identity(
    registry: FeedbackComparisonRegistry,
    binding_field: str,
    value: object,
    message: str,
) -> None:
    row, bundle, binding = _bundle(registry)
    stale = replace(binding, **{binding_field: value})

    with pytest.raises(ValueError, match=message):
        validate_native_replay_output(row, stale, bundle, _output(bundle))


def test_receipt_rejects_tampered_output_input_or_short_horizon(
    registry: FeedbackComparisonRegistry,
) -> None:
    row, bundle, binding = _bundle(registry)
    output = _output(bundle)
    output.applied_actuator_torques[0, 0] = 9.0

    with pytest.raises(ValueError, match="actual inputs"):
        validate_native_replay_output(row, binding, bundle, output)


def test_receipt_rejects_unknown_native_output_type(
    registry: FeedbackComparisonRegistry,
) -> None:
    row, bundle, binding = _bundle(registry)

    with pytest.raises(ValueError, match="unknown output type"):
        validate_native_replay_output(row, binding, bundle, object())  # type: ignore[arg-type]


def test_report_keeps_all_six_engines_blocking_without_explicit_bindings(
    registry: FeedbackComparisonRegistry,
) -> None:
    report = build_native_replay_report(registry, ())

    assert set(report.required_engine_ids) == set(TARGET_ENGINES)
    assert report.missing_engine_ids == report.required_engine_ids
    assert report.executed_row_count == 0
    assert not report.is_complete
    assert all(row.status == "missing_binding" for row in report.rows if row.required)


def test_report_rejects_request_for_unregistered_inventory_row(
    registry: FeedbackComparisonRegistry,
) -> None:
    _, bundle, binding = _bundle(registry)
    unregistered = replace(binding, package_id="unregistered/test-model")
    request = NativeReplayRequest(unregistered, bundle, Path("synthetic.xml"))

    with pytest.raises(ValueError, match="unregistered inventory row"):
        build_native_replay_report(registry, (request,))


def test_request_references_bundle_model_path_without_serializing_it(
    registry: FeedbackComparisonRegistry,
    tmp_path: Path,
) -> None:
    row, bundle, binding = _bundle(registry)
    model_path = tmp_path / "private-or-local-model.xml"
    request = NativeReplayRequest(binding, bundle, model_path)

    assert request.model_path == model_path
    assert str(model_path) not in str(request.as_dict())
    assert request.as_dict()["row_key"] == (
        f"{row.package_id}/{row.variant_id}/{row.drive_mode.value}"
    )
    assert request.as_dict()["bundle_schema"] == bundle.schema_version


def test_command_admission_requires_matching_t02_profile_artifact(
    registry: FeedbackComparisonRegistry,
) -> None:
    from _pinned_tools__sidekick__lab__mocap import (
        COMPILED_ACTUATOR_PROFILE_SCHEMA_VERSION,
    )  # type: ignore[import-not-found]

    f01_row = registry.get("myosuite/driver", "default", DriveMode.MUSCLE_EXCITATION)
    _, source_bundle, _ = _bundle(registry)
    bundle = build_experiment_replay_bundle(
        source_bundle.experiment_id,
        source_bundle.model,
        source_bundle.capabilities,
        tuple((item.component_id, item.values) for item in source_bundle.initial_state),
        source_bundle.input_history.channels,
        ActuationInputKind.ACTUATOR_COMMAND,
        InputInterpolation.ZERO_ORDER_HOLD,
        source_bundle.input_history.time_seconds,
        source_bundle.input_history.values,
        source_bundle.policy,
    )
    profile_bytes = json.dumps(
        {"schema_version": COMPILED_ACTUATOR_PROFILE_SCHEMA_VERSION},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    profile_sha = hashlib.sha256(profile_bytes).hexdigest()
    reference = "opaque:test-compiled-profile"
    from _pinned_tools__sidekick__lab__mocap import (  # type: ignore[import-not-found]
        COMPILED_ACTUATOR_PROFILE_ID,
        COMPILED_ACTUATOR_PROFILE_VERSION,
    )

    t02_row = ComparisonEvidenceRow(
        "myosuite/driver/default/muscle_excitation",
        "myosuite/driver",
        "default",
        T02DriveMode.MUSCLE_EXCITATION,
        bundle,
        CapabilitySupport.SUPPORTED,
        CapabilityAvailability.AVAILABLE,
        (
            ImplementationEvidence(
                ImplementationEvidenceKind.ACTUATOR,
                COMPILED_ACTUATOR_PROFILE_ID,
                COMPILED_ACTUATOR_PROFILE_VERSION,
                profile_sha,
                True,
                CapabilitySupport.SUPPORTED,
                CapabilityAvailability.AVAILABLE,
                evidence_reference_id=reference,
            ),
        ),
        (
            EvidenceArtifactReference(
                EvidenceArtifactKind.ACTUATOR, reference, profile_sha
            ),
        ),
    )
    binding = NativeAdapterBinding(
        f01_row.package_id,
        f01_row.variant_id,
        f01_row.drive_mode,
        bundle.model.model_id,
        bundle.model.variant_id,
        bundle.model.provider_id,
        bundle.model.provider_sha256,
        f01_row.source_model_sha256,
        bundle.model.loaded_native_model_sha256 or "",
        bundle.state_schema_sha256,
        bundle.input_channel_schema_sha256,
        bundle.model.ordered_input_channel_ids,
    )
    request = NativeReplayRequest(binding, bundle, Path("synthetic.xml"))

    parsed = _validate_command_evidence_row(request, t02_row, profile_bytes)

    assert parsed["schema_version"] == COMPILED_ACTUATOR_PROFILE_SCHEMA_VERSION
    with pytest.raises(ValueError, match="artifact digest"):
        _validate_command_evidence_row(request, t02_row, profile_bytes + b" ")


def test_execution_calls_native_mujoco_adapter_on_independent_synthetic_model(
    registry: FeedbackComparisonRegistry,
    tmp_path: Path,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    model_path = tmp_path / "one_hinge.xml"
    model_path.write_text(
        """<mujoco model="synthetic">
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody><body name="link"><joint name="hinge" type="hinge"/>
    <geom type="capsule" size="0.02 0.1" mass="1"/>
  </body></worldbody>
  <actuator><motor name="motor" joint="hinge"/></actuator>
</mujoco>""",
        encoding="utf-8",
    )
    native_model = mujoco.MjModel.from_xml_path(str(model_path))
    native_data = mujoco.MjData(native_model)
    native_state = np.empty(
        mujoco.mj_stateSize(native_model, mujoco.mjtState.mjSTATE_INTEGRATION)
    )
    mujoco.mj_getState(
        native_model, native_data, native_state, mujoco.mjtState.mjSTATE_INTEGRATION
    )
    from src.engines.physics_engines.mujoco.python.native_torque_replay import (
        build_native_torque_bundle,
    )

    bundle = build_native_torque_bundle(
        model_path,
        native_state,
        np.asarray([0.0, 0.001, 0.002]),
        np.asarray([[0.1], [0.2], [0.2]]),
        experiment_id="synthetic-native-execution-test",
    )
    base_row = registry.get("mujoco/driver", "default", DriveMode.TORQUE)
    row = replace(
        base_row,
        source_model_sha256=bundle.model.source_model_sha256,
        provider_id="synthetic-inventory-provider",
        provider_sha256="e" * 64,
        availability="available",
    )
    registry.rows = (row,)
    binding = NativeAdapterBinding(
        row.package_id,
        row.variant_id,
        row.drive_mode,
        bundle.model.model_id,
        bundle.model.variant_id,
        bundle.model.provider_id,
        bundle.model.provider_sha256,
        bundle.model.source_model_sha256,
        bundle.model.loaded_native_model_sha256,
        bundle.state_schema_sha256,
        bundle.input_channel_schema_sha256,
        bundle.model.ordered_input_channel_ids,
    )

    receipt = execute_native_replay(
        NativeReplayRequest(binding, bundle, model_path), registry
    )

    assert receipt.engine == "mujoco"
    assert receipt.state_sample_count == 3
    assert receipt.applied_input_sample_count == 2
    assert receipt.qualification == "unqualified"
    assert receipt.inventory_provider_id == row.provider_id
    assert receipt.native_execution_provider_id != row.provider_id


def test_execution_calls_native_drake_adapter_when_provider_is_installed(
    registry: FeedbackComparisonRegistry,
    tmp_path: Path,
) -> None:
    pytest.importorskip("pydrake.multibody.plant")
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import MultibodyPlant

    from src.engines.physics_engines.drake.python.native_torque_replay import (
        build_native_drake_torque_bundle,
    )

    urdf = """<robot name="execution_probe">
  <link name="base"><inertial><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="arm"><inertial><mass value="0.5"/><inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="2" velocity="100"/></joint>
  <transmission name="motor"><type>transmission_interface/SimpleTransmission</type><joint name="hinge"/><actuator name="motor"><mechanicalReduction>1</mechanicalReduction></actuator></transmission>
</robot>"""
    model_path = tmp_path / "one_hinge.urdf"
    model_path.write_text(urdf, encoding="utf-8")
    plant = MultibodyPlant(0.001)
    plant.SetUseSampledOutputPorts(False)
    Parser(plant).AddModelsFromString(urdf, "urdf")
    plant.Finalize()
    context = plant.CreateDefaultContext()
    initial = context.get_discrete_state_vector().CopyToVector()
    bundle = build_native_drake_torque_bundle(
        model_path,
        initial,
        np.asarray([0.0, 0.001, 0.002]),
        np.asarray([[0.1], [0.2], [0.2]]),
    )
    base_row = registry.get("drake/driver", "default", DriveMode.TORQUE)
    row = replace(
        base_row,
        source_model_sha256=bundle.model.source_model_sha256,
        provider_id="synthetic-inventory-provider",
        provider_sha256="e" * 64,
        availability="available",
    )
    registry.rows = (row,)
    binding = NativeAdapterBinding(
        row.package_id,
        row.variant_id,
        row.drive_mode,
        bundle.model.model_id,
        bundle.model.variant_id,
        bundle.model.provider_id,
        bundle.model.provider_sha256,
        bundle.model.source_model_sha256,
        bundle.model.loaded_native_model_sha256,
        bundle.state_schema_sha256,
        bundle.input_channel_schema_sha256,
        bundle.model.ordered_input_channel_ids,
    )

    receipt = execute_native_replay(
        NativeReplayRequest(binding, bundle, model_path), registry
    )

    assert receipt.engine == "drake"
    assert receipt.state_sample_count == 3
    assert receipt.applied_input_sample_count == 2
    assert receipt.qualification == "unqualified"
