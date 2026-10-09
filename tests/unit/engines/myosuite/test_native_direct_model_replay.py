"""Synthetic direct-model command replay tests; no MyoSuite SDK claim."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
    DeclaredModelResource,
    DirectModelRegistration,
    actuator_law_manifest_sha256,
    direct_provider_sha256,
    native_contact_policy_sha256,
    replay_direct_model_actuator_commands,
    resource_closure_sha256,
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
_FACTORY_MODEL_PATHS: list[str] = []
_RETURN_WRAPPER = False
_SET_WARNING = False

_MODEL_XML = """<mujoco model="direct-command-fixture">
  <option timestep="0.001" integrator="RK4"/>
  <worldbody><body name="link"><joint name="hinge" type="hinge"/>
    <geom name="link_geom" type="capsule" fromto="0 0 0 0 0 0.2"
          size="0.03" mass="1"/>
  </body></worldbody>
  <actuator>
    <motor name="motor_command" joint="hinge" gear="2"
           ctrllimited="true" ctrlrange="-1 1"/>
    <general name="filtered_command" joint="hinge" dyntype="filter"
             dynprm="0.02" gaintype="fixed" gainprm="0.5"
             biastype="none" ctrlrange="0 1" ctrllimited="true"/>
  </actuator>
</mujoco>"""


class _DirectModelFixture:
    def __init__(self, model_path: str) -> None:
        import mujoco

        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        if _SET_WARNING:
            self.data.warning[0].number = 1

    def close(self) -> None:
        pass


def _direct_factory(model_path: str) -> _DirectModelFixture:
    _FACTORY_MODEL_PATHS.append(model_path)
    environment = _DirectModelFixture(model_path)
    if _RETURN_WRAPPER:
        return SimpleNamespace(env=environment)
    return environment


def _saved_model_sha(model) -> str:
    import mujoco

    payload = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=payload)
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _state(model, data) -> np.ndarray:
    import mujoco

    payload = np.empty(
        mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION),
        dtype=np.float64,
    )
    mujoco.mj_getState(model, data, payload, mujoco.mjtState.mjSTATE_INTEGRATION)
    return payload


def _require_native_mujoco_38(mujoco: Any) -> None:
    if mujoco.__version__ != "3.8.0":
        pytest.skip("direct native replay fixtures require MuJoCo 3.8.0")


def _case(
    tmp_path: Path,
    engine_id: str = "mujoco",
    timestep_seconds: float = 0.001,
    input_interval_scale: float = 1.0,
    floating_base: bool = False,
):
    import mujoco

    _require_native_mujoco_38(mujoco)

    model_path = tmp_path / "fixture.xml"
    model_xml = _MODEL_XML.replace(
        'timestep="0.001"', f'timestep="{timestep_seconds:.17g}"'
    )
    if floating_base:
        model_xml = model_xml.replace(
            "</worldbody>",
            '<body name="floating"><freejoint name="floating_root"/>'
            '<geom name="floating_geom" type="sphere" size="0.03" mass="1"/>'
            "</body></worldbody>",
        )
    model_path.write_text(model_xml, encoding="utf-8")
    initial_env = _DirectModelFixture(str(model_path))
    model = initial_env.model
    initial_state = _state(model, initial_env.data)
    source_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    loaded_sha = _saved_model_sha(model)
    channels = (
        InputChannel("command:motor", "actuator:motor_command", "1"),
        InputChannel("command:filter", "actuator:filtered_command", "1"),
    )
    law_sha = actuator_law_manifest_sha256(model, channels)
    class_source = Path(inspect.getsourcefile(_DirectModelFixture)).read_bytes()
    factory_source = Path(inspect.getsourcefile(_direct_factory)).read_bytes()
    resources = (DeclaredModelResource("fixture.xml", source_sha),)
    closure_sha = resource_closure_sha256(tmp_path, model_path, resources)
    registration = DirectModelRegistration(
        engine_id=engine_id,
        model_id="synthetic-direct-model",
        variant_id="test-only",
        model_version="3.8.0",
        source_model_sha256=source_sha,
        loaded_native_model_sha256=loaded_sha,
        provider_id="direct-native-mujoco-test-provider",
        provider_version="1.0.0",
        provider_sha256="0" * 64,
        state_schema_sha256="0" * 64,
        input_channel_schema_sha256="0" * 64,
        actuator_law_manifest_sha256=law_sha,
        ordered_channel_ids=tuple(channel.channel_id for channel in channels),
        resource_root=tmp_path,
        model_path=model_path,
        resources=resources,
        environment_class_id=(
            f"{_DirectModelFixture.__module__}.{_DirectModelFixture.__qualname__}"
        ),
        environment_class_sha256=hashlib.sha256(class_source).hexdigest(),
        factory_source_sha256=hashlib.sha256(factory_source).hexdigest(),
        solver_id=mujoco.mjtSolver(model.opt.solver).name,
        integration_method=(
            f"{mujoco.mjtIntegrator(model.opt.integrator).name};native-single-step"
        ),
        contact_policy_id="compiled-native-model-contact",
    )
    provider_sha = direct_provider_sha256(
        registration, _direct_factory, initial_env, closure_sha
    )
    registration = replace(registration, provider_sha256=provider_sha)
    integration_dim = len(initial_state)
    schema = InitialStateSchema(
        "test-native-direct-state",
        "1.0.0",
        (
            StateComponentSpec(
                "qpos", StateComponentRole.POSITION, model.nq, "native", "qpos"
            ),
            StateComponentSpec(
                "qvel", StateComponentRole.VELOCITY, model.nv, "native", "qvel"
            ),
            StateComponentSpec(
                "actuator_activation",
                StateComponentRole.AUXILIARY,
                model.na,
                "native",
                "MuJoCo-data.act",
            ),
            StateComponentSpec(
                "actuator_internal",
                StateComponentRole.ACTUATOR_INTERNAL_STATE,
                model.nu,
                "native",
                "MuJoCo-data.ctrl",
            ),
            StateComponentSpec(
                "integration",
                StateComponentRole.AUXILIARY,
                integration_dim,
                "native",
                "mjSTATE_INTEGRATION",
            ),
        ),
    )
    identity = ModelIdentity(
        engine_id,
        registration.model_id,
        registration.variant_id,
        registration.model_version,
        source_sha,
        registration.provider_id,
        registration.provider_version,
        provider_sha,
        schema,
        registration.ordered_channel_ids,
        loaded_sha,
    )
    policy = ReplayExecutionPolicy(
        replay_mode=ReplayMode.NATIVE_OWN_CONTACT,
        solver_id=registration.solver_id,
        solver_version=mujoco.__version__,
        integration_method=registration.integration_method,
        step_policy="fixed",
        step_size_seconds=float(model.opt.timestep),
        initialization_policy_id="mujoco-integration-state-restore-forward",
        initialization_policy_version="1.0.0",
        input_player_id="native-mj-step-direct-actuator-command",
        input_player_version="1.0.0",
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id=registration.contact_policy_id,
        contact_policy_version="1.0.0",
        contact_policy_sha256=native_contact_policy_sha256(model),
    )
    bundle = build_experiment_replay_bundle(
        "opaque:synthetic-direct-model",
        identity,
        (
            CapabilityDeclaration(
                "native-forward-command-replay",
                True,
                CapabilitySupport.SUPPORTED,
                CapabilityAvailability.AVAILABLE,
            ),
        ),
        (
            ("qpos", tuple(float(x) for x in initial_env.data.qpos)),
            ("qvel", tuple(float(x) for x in initial_env.data.qvel)),
            ("actuator_activation", tuple(float(x) for x in initial_env.data.act)),
            ("actuator_internal", tuple(float(x) for x in initial_env.data.ctrl)),
            ("integration", tuple(float(x) for x in initial_state)),
        ),
        channels,
        ActuationInputKind.ACTUATOR_COMMAND,
        InputInterpolation.ZERO_ORDER_HOLD,
        (
            0.0,
            model.opt.timestep * input_interval_scale,
            2 * model.opt.timestep * input_interval_scale,
        ),
        ((0.1, 0.2), (0.3, 0.6), (0.4, 0.8)),
        policy,
    )
    registration = replace(
        registration,
        state_schema_sha256=bundle.state_schema_sha256,
        input_channel_schema_sha256=bundle.input_channel_schema_sha256,
    )
    return registration, bundle, closure_sha


def _public_receipt_case(resource_root: Path, artifact_root: Path, variant: str):
    import mujoco

    manifest = json.loads((artifact_root / "resource_manifest.json").read_text())
    inventory = json.loads((artifact_root / "receipt.json").read_text())
    assembly = json.loads((artifact_root / "assembly_contact_receipt.json").read_text())
    history = json.loads((artifact_root / f"replay_{variant}_inputs.json").read_text())
    model_entry = next(
        item
        for item in manifest
        if item["path"].replace("\\", "/") == f"golf/body/golfer_myobody_{variant}.xml"
    )
    model_path = resource_root / model_entry["path"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    expected_model = next(
        item for item in inventory["models"] if f"_{variant}.xml" in item["path"]
    )
    assert _saved_model_sha(model) == expected_model["mjb_sha256"]
    prepared = next(item for item in assembly["models"] if item["variant"] == variant)[
        "after"
    ]
    integration = np.asarray(prepared["integration_state"], dtype=np.float64)
    channels = tuple(
        InputChannel(
            item["channel_id"],
            item["target_id"],
            item["unit"],
            item.get("coordinate_id"),
            item.get("frame_id"),
        )
        for item in history["channels"]
    )
    channel_ids = tuple(channel.channel_id for channel in channels)
    resources = tuple(
        DeclaredModelResource(item["path"].replace("\\", "/"), item["sha256"])
        for item in manifest
    )
    closure = resource_closure_sha256(resource_root, model_path, resources)
    source_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    implementation_sha = hashlib.sha256(
        Path(inspect.getsourcefile(_DirectModelFixture)).read_bytes()
    ).hexdigest()
    registration = DirectModelRegistration(
        engine_id="mujoco",
        model_id="mujosim-source-body",
        variant_id=variant,
        model_version=mujoco.__version__,
        source_model_sha256=source_sha,
        loaded_native_model_sha256=_saved_model_sha(model),
        provider_id="test-only-direct-xml-receipt-provider",
        provider_version="0.0.0-experiment",
        provider_sha256="0" * 64,
        state_schema_sha256="0" * 64,
        input_channel_schema_sha256="0" * 64,
        actuator_law_manifest_sha256=actuator_law_manifest_sha256(model, channels),
        ordered_channel_ids=channel_ids,
        resource_root=resource_root,
        model_path=model_path,
        resources=resources,
        environment_class_id=(
            f"{_DirectModelFixture.__module__}.{_DirectModelFixture.__qualname__}"
        ),
        environment_class_sha256=implementation_sha,
        factory_source_sha256=hashlib.sha256(
            Path(inspect.getsourcefile(_direct_factory)).read_bytes()
        ).hexdigest(),
        solver_id=mujoco.mjtSolver(model.opt.solver).name,
        integration_method=(
            f"{mujoco.mjtIntegrator(model.opt.integrator).name};native-single-step"
        ),
        contact_policy_id="compiled-native-contact-policy",
    )
    environment = _DirectModelFixture(str(model_path))
    registration = replace(
        registration,
        provider_sha256=direct_provider_sha256(
            registration, _direct_factory, environment, closure
        ),
    )
    schema = InitialStateSchema(
        "mujoco-integration-state",
        "1.0.0",
        (
            StateComponentSpec(
                "qpos", StateComponentRole.POSITION, model.nq, "native", "qpos"
            ),
            StateComponentSpec(
                "qvel", StateComponentRole.VELOCITY, model.nv, "native", "qvel"
            ),
            StateComponentSpec(
                "actuator_activation",
                StateComponentRole.AUXILIARY,
                model.na,
                "native",
                "act",
            ),
            StateComponentSpec(
                "actuator_internal",
                StateComponentRole.ACTUATOR_INTERNAL_STATE,
                model.nu,
                "native",
                "ctrl",
            ),
            StateComponentSpec(
                "integration",
                StateComponentRole.AUXILIARY,
                len(integration),
                "native",
                "mjSTATE_INTEGRATION",
            ),
        ),
    )
    identity = ModelIdentity(
        "mujoco",
        registration.model_id,
        variant,
        mujoco.__version__,
        source_sha,
        registration.provider_id,
        registration.provider_version,
        registration.provider_sha256,
        schema,
        channel_ids,
        registration.loaded_native_model_sha256,
    )
    policy = ReplayExecutionPolicy(
        replay_mode=ReplayMode.NATIVE_OWN_CONTACT,
        solver_id=registration.solver_id,
        solver_version=mujoco.__version__,
        integration_method=registration.integration_method,
        step_policy="fixed",
        step_size_seconds=float(model.opt.timestep),
        initialization_policy_id="mujoco-integration-state-restore-forward",
        initialization_policy_version="1.0.0",
        input_player_id="native-mj-step-direct-actuator-command",
        input_player_version="1.0.0",
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id=registration.contact_policy_id,
        contact_policy_version="1.0.0",
        contact_policy_sha256=native_contact_policy_sha256(model),
    )
    bundle = build_experiment_replay_bundle(
        f"opaque:public-native-receipt-{variant}",
        identity,
        (
            CapabilityDeclaration(
                "native-forward-command-replay",
                True,
                CapabilitySupport.SUPPORTED,
                CapabilityAvailability.AVAILABLE,
            ),
        ),
        (
            ("qpos", tuple(prepared["qpos"])),
            ("qvel", tuple(prepared["qvel"])),
            ("actuator_activation", tuple(prepared["act"])),
            ("actuator_internal", tuple(prepared["ctrl"])),
            ("integration", tuple(integration)),
        ),
        channels,
        ActuationInputKind.ACTUATOR_COMMAND,
        InputInterpolation.ZERO_ORDER_HOLD,
        history["time_seconds"],
        history["values"],
        policy,
    )
    return replace(
        registration,
        state_schema_sha256=bundle.state_schema_sha256,
        input_channel_schema_sha256=bundle.input_channel_schema_sha256,
    ), bundle


def test_direct_model_replay_preserves_source_path_laws_and_full_state(
    tmp_path: Path,
) -> None:
    registration, bundle, _ = _case(tmp_path)
    _FACTORY_MODEL_PATHS.clear()

    output = replay_direct_model_actuator_commands(
        bundle, registration, _direct_factory
    )

    assert output.time_seconds == pytest.approx(bundle.input_history.time_seconds)
    assert np.array_equal(
        output.applied_actuator_commands,
        np.asarray(bundle.input_history.values[:-1], dtype=np.float64),
    )
    assert bundle.input_history.values[-1] != bundle.input_history.values[-2]
    assert np.array_equal(output.actuator_controls[-1], bundle.input_history.values[-2])
    assert output.integration_states.shape[0] == len(bundle.input_history.time_seconds)
    assert output.actuator_activation.shape[1] == 1
    assert output.input_sha256 == bundle.applied_input_sha256
    assert output.policy_sha256 == bundle.policy_sha256
    assert output.resource_closure_sha256 == resource_closure_sha256(
        registration.resource_root, registration.model_path, registration.resources
    )
    assert [str(registration.model_path.resolve())] == _FACTORY_MODEL_PATHS


def test_direct_model_replay_rejects_missing_or_changed_resource(
    tmp_path: Path,
) -> None:
    registration, bundle, _ = _case(tmp_path)
    missing_resource_registration = replace(registration, resources=())

    with pytest.raises(ValueError, match="resource closure cannot be empty"):
        replay_direct_model_actuator_commands(
            bundle, missing_resource_registration, _direct_factory
        )

    registration.model_path.write_text(_MODEL_XML + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model resource digest differs"):
        replay_direct_model_actuator_commands(bundle, registration, _direct_factory)


def test_direct_registration_validates_digest_and_ordered_channel_contract(
    tmp_path: Path,
) -> None:
    registration, _, _ = _case(tmp_path)

    with pytest.raises(ValueError, match="provider_sha256 must be"):
        replace(registration, provider_sha256="invalid")
    with pytest.raises(ValueError, match="ordered channel IDs must be unique"):
        replace(
            registration,
            ordered_channel_ids=(
                registration.ordered_channel_ids[0],
                registration.ordered_channel_ids[0],
            ),
        )
    with pytest.raises(ValueError, match="model resource sha256 must be"):
        DeclaredModelResource("fixture.xml", "invalid")


@pytest.mark.parametrize("engine_id", ("myosuite", "opensim", "drake", "unknown"))
def test_direct_mujoco_kernel_rejects_other_engine_id_before_factory(
    tmp_path: Path, engine_id: str
) -> None:
    registration, bundle, _ = _case(tmp_path, engine_id=engine_id)
    _FACTORY_MODEL_PATHS.clear()

    with pytest.raises(ValueError, match="only admits engine_id='mujoco'"):
        replay_direct_model_actuator_commands(bundle, registration, _direct_factory)
    assert not _FACTORY_MODEL_PATHS


def test_native_contact_replay_rejects_external_load_payload(tmp_path: Path) -> None:
    registration, bundle, _ = _case(tmp_path)
    policy = replace(bundle.policy, external_loads_sha256="a" * 64)
    loaded_policy_bundle = build_experiment_replay_bundle(
        bundle.experiment_id,
        bundle.model,
        bundle.capabilities,
        tuple((item.component_id, item.values) for item in bundle.initial_state),
        bundle.input_history.channels,
        bundle.input_history.input_kind,
        bundle.input_history.interpolation,
        bundle.input_history.time_seconds,
        bundle.input_history.values,
        policy,
        bundle.input_history.timebase_id,
    )
    _FACTORY_MODEL_PATHS.clear()

    with pytest.raises(ValueError, match="does not consume external-load payloads"):
        replay_direct_model_actuator_commands(
            loaded_policy_bundle, registration, _direct_factory
        )
    assert not _FACTORY_MODEL_PATHS


def test_native_clock_rejects_grid_interval_far_larger_than_native_step(
    tmp_path: Path,
) -> None:
    registration, bundle, _ = _case(
        tmp_path, timestep_seconds=1e-15, input_interval_scale=100.0
    )
    _FACTORY_MODEL_PATHS.clear()

    with pytest.raises(ValueError, match="one native step per sample"):
        replay_direct_model_actuator_commands(bundle, registration, _direct_factory)
    assert not _FACTORY_MODEL_PATHS


def test_direct_model_replay_rejects_nonunit_frozen_free_quaternion(
    tmp_path: Path,
) -> None:
    import mujoco

    registration, bundle, _ = _case(tmp_path, floating_base=True)
    model = mujoco.MjModel.from_xml_path(str(registration.model_path))
    data = mujoco.MjData(model)
    integration = np.asarray(
        next(
            item.values
            for item in bundle.initial_state
            if item.component_id == "integration"
        ),
        dtype=np.float64,
    )
    mujoco.mj_setState(model, data, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_root")
    qpos_address = int(model.jnt_qposadr[joint_id])
    data.qpos[qpos_address + 3 : qpos_address + 7] = (2.0, 0.0, 0.0, 0.0)
    malformed_integration = _state(model, data)
    state_values = tuple(
        (
            item.component_id,
            tuple(
                float(value)
                for value in (
                    data.qpos
                    if item.component_id == "qpos"
                    else malformed_integration
                    if item.component_id == "integration"
                    else item.values
                )
            ),
        )
        for item in bundle.initial_state
    )
    malformed_bundle = build_experiment_replay_bundle(
        bundle.experiment_id,
        bundle.model,
        bundle.capabilities,
        state_values,
        bundle.input_history.channels,
        bundle.input_history.input_kind,
        bundle.input_history.interpolation,
        bundle.input_history.time_seconds,
        bundle.input_history.values,
        bundle.policy,
        bundle.input_history.timebase_id,
    )
    _FACTORY_MODEL_PATHS.clear()

    with pytest.raises(
        ValueError, match="native quaternion configuration must be normalized"
    ):
        replay_direct_model_actuator_commands(
            malformed_bundle, registration, _direct_factory
        )


def test_direct_model_replay_rejects_actuator_law_manifest_mismatch(
    tmp_path: Path,
) -> None:
    registration, bundle, _ = _case(tmp_path)
    bad_registration = replace(registration, actuator_law_manifest_sha256="f" * 64)

    with pytest.raises(ValueError, match="actuator law/channel manifest differs"):
        replay_direct_model_actuator_commands(bundle, bad_registration, _direct_factory)


def test_direct_model_replay_rejects_wrapper_factory_result(tmp_path: Path) -> None:
    registration, bundle, _ = _case(tmp_path)
    global _RETURN_WRAPPER
    _RETURN_WRAPPER = True
    try:
        with pytest.raises(ValueError, match="unwrapped environment"):
            replay_direct_model_actuator_commands(bundle, registration, _direct_factory)
    finally:
        _RETURN_WRAPPER = False


def test_direct_model_replay_rejects_callbacks_before_resource_or_factory_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mujoco

    import src.engines.physics_engines.myosuite.python.native_direct_model_replay as direct_replay

    registration, bundle, _ = _case(tmp_path)

    def forbidden_preflight(*_args, **_kwargs):
        raise AssertionError("native source preflight must not run with callbacks")

    monkeypatch.setattr(
        direct_replay, "_discover_native_resource_files", forbidden_preflight
    )
    mujoco.set_mjcb_control(lambda _model, _data: None)
    _FACTORY_MODEL_PATHS.clear()
    try:
        with pytest.raises(
            ValueError, match="process-global MuJoCo callbacks are forbidden"
        ):
            replay_direct_model_actuator_commands(bundle, registration, _direct_factory)
        assert not _FACTORY_MODEL_PATHS
    finally:
        mujoco.set_mjcb_control(None)


def test_direct_model_replay_rejects_native_warning(tmp_path: Path) -> None:
    registration, bundle, _ = _case(tmp_path)
    global _SET_WARNING
    _SET_WARNING = True
    try:
        with pytest.raises(ValueError, match="produced a MuJoCo warning"):
            replay_direct_model_actuator_commands(bundle, registration, _direct_factory)
    finally:
        _SET_WARNING = False


def test_contact_policy_digest_changes_with_compiled_contact_rules(
    tmp_path: Path,
) -> None:
    import mujoco

    model_path = tmp_path / "fixture.xml"
    model_path.write_text(_MODEL_XML, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    original = native_contact_policy_sha256(model)
    model.geom_contype[0] = int(model.geom_contype[0]) + 1

    assert native_contact_policy_sha256(model) != original


def test_actuator_law_digest_changes_with_compiled_native_law(
    tmp_path: Path,
) -> None:
    import mujoco

    _require_native_mujoco_38(mujoco)
    model_path = tmp_path / "fixture.xml"
    model_path.write_text(_MODEL_XML, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    channels = (
        InputChannel("command:motor", "actuator:motor_command", "1"),
        InputChannel("command:filter", "actuator:filtered_command", "1"),
    )
    original = actuator_law_manifest_sha256(model, channels)
    model.actuator_gainprm[0, 0] += 0.25

    assert actuator_law_manifest_sha256(model, channels) != original


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("UD_MYO_NATIVE_RESOURCE_ROOT")
    or not os.environ.get("UD_MYO_NATIVE_ARTIFACT_ROOT"),
    reason="public MyoSim native replay artifacts are not configured",
)
def test_public_driver_and_iron_native_receipts_replay_exactly() -> None:
    resource_root = Path(os.environ["UD_MYO_NATIVE_RESOURCE_ROOT"])
    artifact_root = Path(os.environ["UD_MYO_NATIVE_ARTIFACT_ROOT"])
    for variant in ("driver", "iron"):
        registration, bundle = _public_receipt_case(
            resource_root, artifact_root, variant
        )
        output = replay_direct_model_actuator_commands(
            bundle, registration, _direct_factory
        )
        with np.load(
            artifact_root / f"replay_{variant}_producer.npz", allow_pickle=False
        ) as producer:
            expected = producer["state"]
        assert np.array_equal(output.integration_states, expected)
        assert output.applied_actuator_commands.shape == (30, 100)


def _pinned_myoarm_diagnostic_case(resource_root: Path, pulse: bool):
    import mujoco

    model_path = resource_root / "arm" / "myoarm.xml"
    environment = _DirectModelFixture(str(model_path))
    model = environment.model
    data = environment.data
    if model.nu != 63 or model.na != 63:
        raise AssertionError(
            "pinned MyoArm model must expose 63 actuator controls/activations"
        )
    channels = tuple(
        InputChannel(
            f"command:{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)}",
            f"actuator:{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)}",
            "1",
        )
        for index in range(model.nu)
    )
    source_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    resources = tuple(
        DeclaredModelResource(
            path.relative_to(resource_root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(resource_root.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(resource_root).parts
    )
    resource_digest = resource_closure_sha256(resource_root, model_path, resources)
    test_source = Path(inspect.getsourcefile(_DirectModelFixture)).read_bytes()
    factory_source = Path(inspect.getsourcefile(_direct_factory)).read_bytes()
    state = _state(model, data)
    schema = InitialStateSchema(
        "pinned-myoarm-native-integration-state",
        "1.0.0",
        (
            StateComponentSpec(
                "qpos", StateComponentRole.POSITION, model.nq, "native", "data.qpos"
            ),
            StateComponentSpec(
                "qvel", StateComponentRole.VELOCITY, model.nv, "native", "data.qvel"
            ),
            StateComponentSpec(
                "actuator_activation",
                StateComponentRole.AUXILIARY,
                model.na,
                "native",
                "data.act",
            ),
            StateComponentSpec(
                "actuator_internal",
                StateComponentRole.ACTUATOR_INTERNAL_STATE,
                model.nu,
                "native",
                "data.ctrl",
            ),
            StateComponentSpec(
                "integration",
                StateComponentRole.AUXILIARY,
                len(state),
                "native",
                "mjSTATE_INTEGRATION",
            ),
        ),
    )
    loaded_sha = _saved_model_sha(model)
    provider_id = "f09g-direct-mujoco-diagnostic"
    provider_version = "0.1.0"
    model_identity = ModelIdentity(
        "mujoco",
        "myosim-myoarm",
        "arm-v0.01",
        (resource_root / "VERSION").read_text(encoding="utf-8").strip(),
        source_sha,
        provider_id,
        provider_version,
        "0" * 64,
        schema,
        tuple(item.channel_id for item in channels),
        loaded_sha,
    )
    implementation_sha = hashlib.sha256(test_source).hexdigest()
    registration = DirectModelRegistration(
        engine_id="mujoco",
        model_id="myosim-myoarm",
        variant_id="arm-v0.01",
        model_version=model_identity.model_version,
        source_model_sha256=source_sha,
        loaded_native_model_sha256=loaded_sha,
        provider_id=provider_id,
        provider_version=provider_version,
        provider_sha256="0" * 64,
        state_schema_sha256="0" * 64,
        input_channel_schema_sha256="0" * 64,
        actuator_law_manifest_sha256=actuator_law_manifest_sha256(model, channels),
        ordered_channel_ids=tuple(item.channel_id for item in channels),
        resource_root=resource_root,
        model_path=model_path,
        resources=resources,
        environment_class_id=f"{_DirectModelFixture.__module__}.{_DirectModelFixture.__qualname__}",
        environment_class_sha256=implementation_sha,
        factory_source_sha256=hashlib.sha256(factory_source).hexdigest(),
        solver_id=mujoco.mjtSolver(model.opt.solver).name,
        integration_method=f"{mujoco.mjtIntegrator(model.opt.integrator).name};native-single-step",
        contact_policy_id="pinned-myoarm-native-contact",
    )
    registration = replace(
        registration,
        provider_sha256=direct_provider_sha256(
            registration, _direct_factory, environment, resource_digest
        ),
    )
    model_identity = replace(
        model_identity, provider_sha256=registration.provider_sha256
    )
    capabilities = (
        CapabilityDeclaration(
            "native-direct-actuator-command-replay",
            True,
            CapabilitySupport.SUPPORTED,
            CapabilityAvailability.AVAILABLE,
        ),
    )
    policy = ReplayExecutionPolicy(
        replay_mode=ReplayMode.NATIVE_OWN_CONTACT,
        solver_id=registration.solver_id,
        solver_version=mujoco.__version__,
        integration_method=registration.integration_method,
        step_policy="fixed",
        step_size_seconds=float(model.opt.timestep),
        initialization_policy_id="mujoco-integration-state-restore-forward",
        initialization_policy_version="1.0.0",
        input_player_id="native-mj-step-direct-actuator-command",
        input_player_version="1.0.0",
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id=registration.contact_policy_id,
        contact_policy_version="1.0.0",
        contact_policy_sha256=native_contact_policy_sha256(model),
    )
    count = 31
    values = np.zeros((count, model.nu), dtype=np.float64)
    if pulse:
        values[5:30, 0] = 0.05
    bundle = build_experiment_replay_bundle(
        "opaque:pinned-myoarm-diagnostic",
        model_identity,
        capabilities,
        (
            ("qpos", tuple(float(value) for value in data.qpos)),
            ("qvel", tuple(float(value) for value in data.qvel)),
            ("actuator_activation", tuple(float(value) for value in data.act)),
            ("actuator_internal", tuple(float(value) for value in data.ctrl)),
            ("integration", tuple(float(value) for value in state)),
        ),
        channels,
        ActuationInputKind.ACTUATOR_COMMAND,
        InputInterpolation.ZERO_ORDER_HOLD,
        tuple(float(index * model.opt.timestep) for index in range(count)),
        tuple(tuple(float(value) for value in row) for row in values),
        policy,
    )
    registration = replace(
        registration,
        state_schema_sha256=bundle.state_schema_sha256,
        input_channel_schema_sha256=bundle.input_channel_schema_sha256,
    )
    return registration, bundle, environment, resources, resource_digest


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("UD_MYO_ARM_RESOURCE_ROOT"),
    reason="pinned MyoSim arm model root is not configured",
)
def test_pinned_myoarm_native_command_replay_is_exactly_repeatable() -> None:
    import mujoco

    resource_root = Path(os.environ["UD_MYO_ARM_RESOURCE_ROOT"])
    registration, pulse_bundle, initial, resources, resource_digest = (
        _pinned_myoarm_diagnostic_case(resource_root, pulse=True)
    )
    baseline_values = tuple(
        tuple(0.0 for _ in row) for row in pulse_bundle.input_history.values
    )
    baseline_bundle = build_experiment_replay_bundle(
        pulse_bundle.experiment_id,
        pulse_bundle.model,
        pulse_bundle.capabilities,
        tuple((item.component_id, item.values) for item in pulse_bundle.initial_state),
        pulse_bundle.input_history.channels,
        pulse_bundle.input_history.input_kind,
        pulse_bundle.input_history.interpolation,
        pulse_bundle.input_history.time_seconds,
        baseline_values,
        pulse_bundle.policy,
        pulse_bundle.input_history.timebase_id,
    )
    equality_mask = initial.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    equality_residual = initial.data.efc_pos[equality_mask]
    contacts = []
    for index in range(initial.data.ncon):
        contact = initial.data.contact[index]
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(initial.model, initial.data, index, force)
        contacts.append(
            {
                "geom1": mujoco.mj_id2name(
                    initial.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                ),
                "geom2": mujoco.mj_id2name(
                    initial.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                ),
                "distance_m": float(contact.dist),
                "force_in_contact_frame": force.tolist(),
            }
        )
    law_counts: dict[str, int] = {}
    for index in range(initial.model.nu):
        law = "/".join(
            (
                mujoco.mjtDyn(int(initial.model.actuator_dyntype[index])).name,
                mujoco.mjtGain(int(initial.model.actuator_gaintype[index])).name,
                mujoco.mjtBias(int(initial.model.actuator_biastype[index])).name,
                mujoco.mjtTrn(int(initial.model.actuator_trntype[index])).name,
            )
        )
        law_counts[law] = law_counts.get(law, 0) + 1
    _FACTORY_MODEL_PATHS.clear()

    pulse = replay_direct_model_actuator_commands(
        pulse_bundle, registration, _direct_factory
    )
    repeated = replay_direct_model_actuator_commands(
        pulse_bundle, registration, _direct_factory
    )
    baseline = replay_direct_model_actuator_commands(
        baseline_bundle, registration, _direct_factory
    )
    assert np.array_equal(pulse.integration_states, repeated.integration_states)
    assert np.array_equal(pulse.integration_states[:6], baseline.integration_states[:6])
    assert not np.array_equal(
        pulse.integration_states[6:], baseline.integration_states[6:]
    )
    assert np.isfinite(pulse.integration_states).all()
    print(
        json.dumps(
            {
                "evidence": "unqualified-underlying-mujoco-native-diagnostic",
                "pin": "33f3ded946f55adbdcf963c99999587aadaf975f",
                "model_source_sha256": registration.source_model_sha256,
                "loaded_native_model_sha256": registration.loaded_native_model_sha256,
                "resource_file_count": len(resources),
                "declared_resource_set_sha256": resource_digest,
                "provider_sha256": registration.provider_sha256,
                "initial_state_sha256": pulse_bundle.integrity.initial_state_sha256,
                "applied_input_sha256": pulse_bundle.applied_input_sha256,
                "policy_sha256": pulse_bundle.policy_sha256,
                "license_sha256": hashlib.sha256(
                    (resource_root / "LICENSE").read_bytes()
                ).hexdigest(),
                "mujoco_version": mujoco.__version__,
                "dimensions": {
                    "nq": initial.model.nq,
                    "nv": initial.model.nv,
                    "nu": initial.model.nu,
                    "na": initial.model.na,
                    "ngeom": initial.model.ngeom,
                    "neq": initial.model.neq,
                },
                "actuator_laws": law_counts,
                "native_default": {
                    "contact_count": initial.data.ncon,
                    "equality_count": int(equality_residual.size),
                    "equality_max_abs": float(np.max(np.abs(equality_residual))),
                    "active_equality_count": int(np.sum(initial.data.eq_active)),
                    "contacts": contacts,
                    "qacc_max_abs": float(np.max(np.abs(initial.data.qacc))),
                    "qfrc_passive_max_abs": float(
                        np.max(np.abs(initial.data.qfrc_passive))
                    ),
                    "qfrc_actuator_max_abs": float(
                        np.max(np.abs(initial.data.qfrc_actuator))
                    ),
                    "qfrc_constraint_max_abs": float(
                        np.max(np.abs(initial.data.qfrc_constraint))
                    ),
                    "initial_activation_max": float(np.max(np.abs(initial.data.act))),
                    "initial_control_max": float(np.max(np.abs(initial.data.ctrl))),
                    "warnings": sum(int(item.number) for item in initial.data.warning),
                },
                "replay": {
                    "samples": int(pulse.time_seconds.size),
                    "applied_channels": int(pulse.applied_actuator_commands.shape[1]),
                    "repeat_full_state_exact": True,
                    "unchanged_prefix_samples": 6,
                    "pulse_channel": "DELT1",
                    "pulse_value": 0.05,
                    "max_qpos_delta_from_zero_input": float(
                        np.max(np.abs(pulse.qpos[6:] - baseline.qpos[6:]))
                    ),
                    "max_activation_delta_from_zero_input": float(
                        np.max(
                            np.abs(
                                pulse.actuator_activation[6:]
                                - baseline.actuator_activation[6:]
                            )
                        )
                    ),
                },
            },
            sort_keys=True,
        )
    )
