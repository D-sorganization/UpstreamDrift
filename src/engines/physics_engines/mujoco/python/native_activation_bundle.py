"""Versioned frozen-command transport for a restricted native muscle action."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco as mj
import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.native_activation_manifold import (
    NativeActivationProvider,
    load_native_activation_provider,
    replay_native_activation_commands,
)
from src.engines.physics_engines.mujoco.python.native_torque_replay import _contracts

if TYPE_CHECKING:
    from sidekick.lab.mocap import ExperimentReplayBundle

Array = NDArray[np.float64]
_VERSION = "1.0.0"


@dataclass(frozen=True)
class NativeActivationReplay:
    """Actual uninterrupted full-state replay; no scientific acceptance verdict."""

    time_seconds: Array
    physical_states: Array
    integration_states: Array
    applied_commands: Array
    applied_input_sha256: str
    policy_sha256: str
    compiled_law_sha256: str


def _model_identity(provider: NativeActivationProvider, contracts: Any) -> Any:
    model, native = provider.model, provider.identity
    components = (
        contracts.StateComponentSpec(
            "qpos",
            contracts.StateComponentRole.POSITION,
            model.nq,
            "native-SI",
            "mujoco-configuration",
        ),
        contracts.StateComponentSpec(
            "qvel",
            contracts.StateComponentRole.VELOCITY,
            model.nv,
            "native-SI",
            "mujoco-tangent-velocity",
        ),
        contracts.StateComponentSpec(
            "activation",
            contracts.StateComponentRole.MUSCLE_ACTIVATION,
            model.na,
            "1",
            "mujoco-native-muscle-activation",
        ),
        contracts.StateComponentSpec(
            "integration",
            contracts.StateComponentRole.AUXILIARY,
            mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION),
            "native-SI",
            "mjSTATE_INTEGRATION-warmstart-disabled",
        ),
    )
    provider_digest = hashlib.sha256(
        bytes.fromhex(native.provider_sha256)
        + bytes.fromhex(native.adapter_sha256)
        + bytes.fromhex(native.compiled_law_sha256)
    ).hexdigest()
    return contracts.ModelIdentity(
        "mujoco",
        "native-muscle-fixture",
        "smooth-floating-muscle",
        _VERSION,
        native.source_model_sha256,
        "mujoco-native-activation-action",
        native.provider_version,
        provider_digest,
        contracts.InitialStateSchema(
            "mujoco-native-muscle-integration", _VERSION, components
        ),
        native.ordered_input_channel_ids,
        native.loaded_native_model_sha256,
    )


def _execution_policy(
    provider: NativeActivationProvider, identity: Any, contracts: Any
) -> Any:
    model = provider.model
    return contracts.ReplayExecutionPolicy(
        replay_mode=contracts.ReplayMode.NATIVE_OWN_CONTACT,
        solver_id=mj.mjtSolver(model.opt.solver).name,
        solver_version=mj.__version__,
        integration_method=mj.mjtIntegrator(model.opt.integrator).name,
        step_policy="fixed",
        step_size_seconds=float(model.opt.timestep),
        initialization_policy_id="mjSTATE_INTEGRATION-warmstart-disabled-no-resets",
        initialization_policy_version=_VERSION,
        input_player_id="native-muscle-command-zoh-terminal-sentinel",
        input_player_version=_VERSION,
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id="mujoco-collision-disabled-smooth-fixture",
        contact_policy_version=_VERSION,
        contact_policy_sha256=identity.loaded_native_model_sha256,
    )


def _native_times(steps: int, step_size_s: float) -> tuple[float, ...]:
    times = [0.0]
    for _ in range(steps):
        times.append(times[-1] + step_size_s)
    return tuple(times)


def build_native_activation_bundle(
    model_path: str | Path,
    initial_integration_state: Array,
    applied_commands: Array,
    *,
    experiment_id: str,
) -> ExperimentReplayBundle:
    """Freeze exact post-limit native commands and complete initial state."""
    contracts = _contracts()
    provider = load_native_activation_provider(model_path)
    model = provider.model
    commands = np.asarray(applied_commands, dtype=np.float64)
    if (
        commands.ndim != 2
        or len(commands) < 1
        or commands.shape[1] != model.nu
        or not np.isfinite(commands).all()
    ):
        raise ValueError("native actuator commands require finite ordered rows")
    full = np.asarray(initial_integration_state, dtype=np.float64)
    provider.project_physical(full)
    data = mj.MjData(model)
    mj.mj_setState(model, data, full, mj.mjtState.mjSTATE_INTEGRATION)
    if data.time != 0:
        raise ValueError("bundle execution clock must start at zero")
    for command in commands:
        full = provider.step_full(full, command)
    identity = _model_identity(provider, contracts)
    initial = np.asarray(initial_integration_state, dtype=np.float64)
    mj.mj_setState(model, data, initial, mj.mjtState.mjSTATE_INTEGRATION)
    channels = tuple(
        contracts.InputChannel(
            name,
            name,
            "1",
            mj.mj_id2name(
                model, mj.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[index, 0])
            ),
        )
        for index, name in enumerate(identity.ordered_input_channel_ids)
    )
    values = np.vstack((commands, commands[-1]))
    return contracts.build_experiment_replay_bundle(
        experiment_id,
        identity,
        (
            contracts.CapabilityDeclaration(
                "native-builtin-muscle-command-replay",
                True,
                contracts.CapabilitySupport.SUPPORTED,
                contracts.CapabilityAvailability.AVAILABLE,
            ),
        ),
        (
            ("qpos", tuple(float(v) for v in data.qpos)),
            ("qvel", tuple(float(v) for v in data.qvel)),
            ("activation", tuple(float(v) for v in data.act)),
            ("integration", tuple(float(v) for v in initial)),
        ),
        channels,
        contracts.ActuationInputKind.ACTUATOR_COMMAND,
        contracts.InputInterpolation.ZERO_ORDER_HOLD,
        _native_times(len(commands), float(model.opt.timestep)),
        tuple(tuple(float(value) for value in row) for row in values),
        _execution_policy(provider, identity, contracts),
    )


def replay_native_activation_bundle(
    bundle: ExperimentReplayBundle, model_path: str | Path
) -> NativeActivationReplay:
    """Revalidate T01 integrity, then replay commands on a freshly loaded model."""
    contracts = _contracts()
    bundle = contracts.load_experiment_replay_bundle(
        contracts.dumps_experiment_replay_bundle(bundle)
    )
    if bundle.blocking_capabilities:
        raise ValueError("required native muscle replay capability unavailable")
    state = {item.component_id: item.values for item in bundle.initial_state}
    if "integration" not in state:
        raise ValueError("complete native integration state missing")
    full = np.asarray(state["integration"], dtype=np.float64)
    commands = np.asarray(bundle.input_history.values[:-1], dtype=np.float64)
    expected = build_native_activation_bundle(
        model_path, full, commands, experiment_id=bundle.experiment_id
    )
    if bundle != expected:
        raise ValueError("native model, law, state, channels, input or policy differs")
    provider = load_native_activation_provider(model_path)
    history = replay_native_activation_commands(
        model_path, full, commands, provider.identity
    )
    physical = np.stack([provider.project_physical(row) for row in history])
    times = np.asarray(bundle.input_history.time_seconds, dtype=np.float64)
    data = mj.MjData(provider.model)
    for index, row in enumerate(history):
        mj.mj_setState(provider.model, data, row, mj.mjtState.mjSTATE_INTEGRATION)
        if data.time != times[index]:
            raise ValueError("native replay clock differs from frozen input grid")
    for array in (history, physical, times, commands):
        array.setflags(write=False)
    return NativeActivationReplay(
        times,
        physical,
        history,
        commands,
        bundle.applied_input_sha256,
        bundle.policy_sha256,
        provider.identity.compiled_law_sha256,
    )
