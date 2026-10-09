"""Export SDK-produced commands through existing T01 and compiled profiles.

The returned object groups existing interchange artifacts; it is not a new wire
format. A repeated final command is an unused ZOH boundary sample. Native replay
uses the independent model provider and the existing direct-command executor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.myosuite_project_task_producer import ProjectTaskCommandHistory
from src.engines.native_direct_model_provider import (
    NativeDirectModel,
    create_native_direct_model,
)
from src.engines.native_replay_contracts import native_replay_contract_types


@dataclass(frozen=True)
class FrozenProjectTaskReplay:
    """Existing replay contracts plus separately retained producer lineage."""

    registration: Any
    bundle: Any
    compiled_profile_bytes: bytes
    producer_history: ProjectTaskCommandHistory


def freeze_project_task_history(
    task: Any,
    history: ProjectTaskCommandHistory,
    *,
    resource_root: Path,
    resources: tuple[Any, ...],
    model_id: str,
    variant_id: str,
    experiment_id: str,
) -> FrozenProjectTaskReplay:
    """Freeze a history and independently verify it using the existing executor.

    Preconditions: exact native/source identity, finite complete history and
    immutable post-mapping command samples. Resource admission recompiles the
    original closure and must match the producer's executed compiled model.
    """
    import mujoco as mj
    from src.engines.myosuite_project_task_producer import _admit_task
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        compiled_actuator_profile_bytes,
        resource_closure_sha256,
    )

    _admit_task(task, mj)
    _validate_history(task, history, mj)
    contracts = native_replay_contract_types()
    path = Path(task.model_path).resolve(strict=True)
    closure = resource_closure_sha256(
        resource_root,
        path,
        resources,
        expected_loaded_native_model_sha256=history.loaded_native_model_sha256,
    )
    if closure != history.resource_closure_sha256:
        raise ValueError(
            "export source resource closure differs from the executed producer"
        )
    environment = create_native_direct_model(str(path))
    try:
        model, data = environment.model, environment.data
        mj.mj_setState(
            model, data, history.integration_states[0], mj.mjtState.mjSTATE_INTEGRATION
        )
        channels = tuple(
            contracts.InputChannel(f"command:{name}", f"actuator:{name}", "1")
            for index in range(model.nu)
            for name in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, index),)
        )
        registration = _registration(
            environment,
            history,
            resource_root,
            resources,
            channels,
            closure,
            model_id,
            variant_id,
            mj,
        )
        schema, values = _initial_state(data, history.integration_states[0], contracts)
        identity = contracts.ModelIdentity(
            registration.engine_id,
            model_id,
            variant_id,
            registration.model_version,
            registration.source_model_sha256,
            registration.provider_id,
            registration.provider_version,
            registration.provider_sha256,
            schema,
            registration.ordered_channel_ids,
            registration.loaded_native_model_sha256,
        )
        commands = np.vstack(
            (history.applied_actuator_commands, history.applied_actuator_commands[-1])
        )
        times = history.time_seconds - history.time_seconds[0]
        bundle = contracts.build_experiment_replay_bundle(
            experiment_id,
            identity,
            (
                contracts.CapabilityDeclaration(
                    "native-forward-command-replay",
                    True,
                    contracts.CapabilitySupport.SUPPORTED,
                    contracts.CapabilityAvailability.AVAILABLE,
                ),
            ),
            values,
            channels,
            contracts.ActuationInputKind.ACTUATOR_COMMAND,
            contracts.InputInterpolation.ZERO_ORDER_HOLD,
            tuple(times),
            tuple(tuple(row) for row in commands),
            _policy(model, registration, contracts, mj),
        )
        registration = replace(
            registration,
            state_schema_sha256=bundle.state_schema_sha256,
            input_channel_schema_sha256=bundle.input_channel_schema_sha256,
        )
        profile = compiled_actuator_profile_bytes(bundle, registration, model, closure)
        artifact = FrozenProjectTaskReplay(registration, bundle, profile, history)
        _verify_history(artifact)
        return artifact
    finally:
        environment.close()


def _validate_history(task: Any, history: ProjectTaskCommandHistory, mj: Any) -> None:
    from src.engines.myosuite_project_task_producer import (
        _model_sha256,
        _require_native_step_clock,
    )

    if not isinstance(history, ProjectTaskCommandHistory):
        raise TypeError("an explicit project task command history is required")
    model = task.model
    if (
        history.sdk_binding != task.project_sdk_binding
        or history.project_task_source_sha256 != task.project_task_source_sha256
        or history.loaded_native_model_sha256 != _model_sha256(mj, task.model)
        or history.source_model_sha256 != task.project_model_source.source_model_sha256
        or history.resource_closure_sha256
        != task.project_model_source.resource_closure_sha256
    ):
        raise ValueError("producer history differs from the admitted task identity")
    if history.applied_actuator_commands.ndim != 2:
        raise ValueError("producer commands require ordered two-dimensional samples")
    count = history.applied_actuator_commands.shape[0]
    dimension = mj.mj_stateSize(task.model, mj.mjtState.mjSTATE_INTEGRATION)
    if (
        count < 1
        or history.time_seconds.shape != (count + 1,)
        or history.integration_states.shape != (count + 1, dimension)
        or history.applied_actuator_commands.shape != (count, task.model.nu)
    ):
        raise ValueError("producer history has incomplete native sample dimensions")
    for array in (
        history.time_seconds,
        history.integration_states,
        history.applied_actuator_commands,
    ):
        if not np.isfinite(array).all():
            raise ValueError("producer history must contain finite native samples")
    from src.engines.myosuite_project_task_producer import native_action_bounds

    low, high = native_action_bounds(task.model)
    if np.any(history.applied_actuator_commands < low) or np.any(
        history.applied_actuator_commands > high
    ):
        raise ValueError(
            "producer post-mapping commands violate native actuator limits"
        )
    initial = mj.MjData(task.model)
    mj.mj_setState(
        task.model,
        initial,
        history.integration_states[0],
        mj.mjtState.mjSTATE_INTEGRATION,
    )
    if float(initial.time) != float(history.time_seconds[0]):
        raise ValueError(
            "producer initial integration time differs from its recorded clock"
        )
    for before, after in zip(
        history.time_seconds[:-1], history.time_seconds[1:], strict=True
    ):
        _require_native_step_clock(
            float(before), float(after), float(model.opt.timestep)
        )


def _verify_history(artifact: FrozenProjectTaskReplay) -> None:
    """Verify claimed SDK history with the canonical observation-free executor."""
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        replay_direct_model_actuator_commands,
    )

    replay = replay_direct_model_actuator_commands(
        artifact.bundle,
        artifact.registration,
        create_native_direct_model,
        compiled_profile_bytes=artifact.compiled_profile_bytes,
    )
    history = artifact.producer_history
    if (
        not np.array_equal(replay.integration_states, history.integration_states)
        or not np.array_equal(
            replay.applied_actuator_commands, history.applied_actuator_commands
        )
        or not np.array_equal(
            replay.time_seconds, history.time_seconds - history.time_seconds[0]
        )
    ):
        raise ValueError(
            "producer state history differs from independently executed native transitions"
        )


def _registration(
    environment: NativeDirectModel,
    history: ProjectTaskCommandHistory,
    resource_root: Path,
    resources: tuple[Any, ...],
    channels: tuple[Any, ...],
    closure: str,
    model_id: str,
    variant_id: str,
    mj: Any,
) -> Any:
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        DirectModelRegistration,
        actuator_law_manifest_sha256,
        direct_provider_sha256,
    )

    provider_source = hashlib.sha256(
        Path(__file__).with_name("native_direct_model_provider.py").read_bytes()
    ).hexdigest()
    model = environment.model
    registration = DirectModelRegistration(
        engine_id="mujoco",
        model_id=model_id,
        variant_id=variant_id,
        model_version=mj.__version__,
        source_model_sha256=hashlib.sha256(
            Path(environment.model_path).read_bytes()
        ).hexdigest(),
        loaded_native_model_sha256=history.loaded_native_model_sha256,
        provider_id="native-direct-mujoco-model",
        provider_version="1.0.0",
        provider_sha256="0" * 64,
        state_schema_sha256="0" * 64,
        input_channel_schema_sha256="0" * 64,
        actuator_law_manifest_sha256=actuator_law_manifest_sha256(model, channels),
        ordered_channel_ids=tuple(channel.channel_id for channel in channels),
        resource_root=resource_root,
        model_path=Path(environment.model_path),
        resources=resources,
        environment_class_id=f"{NativeDirectModel.__module__}.{NativeDirectModel.__qualname__}",
        environment_class_sha256=provider_source,
        factory_source_sha256=provider_source,
        solver_id=mj.mjtSolver(model.opt.solver).name,
        integration_method=f"{mj.mjtIntegrator(model.opt.integrator).name};native-single-step",
        contact_policy_id="compiled-native-model-contact",
    )
    return replace(
        registration,
        provider_sha256=direct_provider_sha256(
            registration,
            create_native_direct_model,
            environment,
            closure,
        ),
    )


def _initial_state(
    data: Any, integration: Any, contracts: Any
) -> tuple[Any, tuple[Any, ...]]:
    rows = [
        ("qpos", contracts.StateComponentRole.POSITION, data.qpos, "qpos"),
        ("qvel", contracts.StateComponentRole.VELOCITY, data.qvel, "qvel"),
    ]
    if data.act.size:
        rows.append(
            (
                "actuator_activation",
                contracts.StateComponentRole.AUXILIARY,
                data.act,
                "MuJoCo-data.act",
            )
        )
    rows.extend(
        (
            (
                "actuator_internal",
                contracts.StateComponentRole.ACTUATOR_INTERNAL_STATE,
                data.ctrl,
                "MuJoCo-data.ctrl",
            ),
            (
                "integration",
                contracts.StateComponentRole.AUXILIARY,
                integration,
                "mjSTATE_INTEGRATION",
            ),
        )
    )
    schema = contracts.InitialStateSchema(
        "native-direct-mujoco-integration",
        "1.0.0",
        tuple(
            contracts.StateComponentSpec(name, role, len(values), "native", meaning)
            for name, role, values, meaning in rows
        ),
    )
    return schema, tuple(
        (name, tuple(float(value) for value in values)) for name, _, values, _ in rows
    )


def _policy(model: Any, registration: Any, contracts: Any, mj: Any) -> Any:
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        native_contact_policy_sha256,
    )

    return contracts.ReplayExecutionPolicy(
        replay_mode=contracts.ReplayMode.NATIVE_OWN_CONTACT,
        solver_id=registration.solver_id,
        solver_version=mj.__version__,
        integration_method=registration.integration_method,
        step_policy="fixed",
        step_size_seconds=float(model.opt.timestep),
        initialization_policy_id="mujoco-integration-state-restore-forward",
        initialization_policy_version="1.1.0",
        input_player_id="native-mj-step-direct-actuator-command",
        input_player_version="1.0.0",
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id=registration.contact_policy_id,
        contact_policy_version="1.0.0",
        contact_policy_sha256=native_contact_policy_sha256(model),
    )
