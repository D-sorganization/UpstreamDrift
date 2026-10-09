"""Open-loop muscle-excitation replay through a pinned MyoSuite plant.

The bundle input is the normalized excitation written directly to native
MuJoCo actuator controls. Gym actions, observations, rewards, and tracking
wrappers are outside this input boundary. Receipts prove execution only.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from src.engines.native_replay_contracts import (
    native_replay_admission_bytes,
    native_replay_contract_types,
    validate_native_replay_bundle,
)

_VERSION = "1.0.0"
_WRAPPER_STATE = (
    ("gymnasium.wrappers.common.TimeLimit", "_elapsed_steps", "counter"),
    ("gymnasium.wrappers.common.OrderEnforcing", "_has_reset", "boolean"),
    ("gymnasium.wrappers.common.PassiveEnvChecker", "checked_reset", "boolean"),
    ("gymnasium.wrappers.common.PassiveEnvChecker", "checked_step", "boolean"),
    ("gymnasium.wrappers.common.PassiveEnvChecker", "checked_render", "boolean"),
    ("gymnasium.wrappers.common.PassiveEnvChecker", "close_called", "boolean"),
    (
        "myosuite.envs.wrappers.MjInstabilityTerminationWrapper",
        "mj_instability_termination",
        "boolean",
    ),
)


@dataclass(frozen=True)
class NativeMyoSuiteExcitationReplay:
    """Actual native states and applied excitation samples for one run."""

    time_seconds: NDArray[np.float64]
    qpos: NDArray[np.float64]
    qvel: NDArray[np.float64]
    muscle_activations: NDArray[np.float64]
    actuator_controls: NDArray[np.float64]
    integration_states: NDArray[np.float64]
    wrapper_states: NDArray[np.float64]
    applied_muscle_excitations: NDArray[np.float64]
    input_sha256: str
    policy_sha256: str


def _contracts() -> Any:
    return native_replay_contract_types()


def _wrapper_state(env: Any) -> tuple[tuple[str, ...], tuple[float, ...]]:
    chain: list[Any] = []
    current = env
    while hasattr(current, "env"):
        chain.append(current)
        current = current.env
    names = tuple(f"{type(item).__module__}.{type(item).__name__}" for item in chain)
    expected = (
        "myosuite.envs.wrappers.MjInstabilityTerminationWrapper",
        "gymnasium.wrappers.common.TimeLimit",
        "gymnasium.wrappers.common.OrderEnforcing",
        "gymnasium.wrappers.common.PassiveEnvChecker",
    )
    if names != expected:
        raise ValueError("MyoSuite wrapper chain is not explicitly supported")
    lookup = dict(zip(names, chain, strict=True))
    values: list[float] = []
    for class_name, attribute, kind in _WRAPPER_STATE:
        wrapper = lookup.get(class_name)
        if wrapper is None or not hasattr(wrapper, attribute):
            raise ValueError("MyoSuite wrapper state is incomplete")
        value = getattr(wrapper, attribute)
        if kind == "counter":
            values.append(-1.0 if value is None else float(value))
        else:
            values.append(float(bool(value)))
    return names, tuple(values)


def _load_environment(environment_id: str) -> tuple[Any, Any, Any, Any, Path]:
    import gymnasium as gym
    import mujoco as mj
    import myosuite  # noqa: F401

    if not environment_id or gym.spec(environment_id).id != environment_id:
        raise ValueError("a registered MyoSuite environment id is required")
    spec = gym.spec(environment_id)
    spec_kwargs = spec.kwargs
    source_path_text = spec_kwargs.get("model_path")
    source_path = Path(source_path_text) if source_path_text else None
    if source_path is None or not source_path.is_file():
        raise ValueError("registered MyoSuite model source is not an owned file")
    source_before = hashlib.sha256(source_path.read_bytes()).digest()
    env = gym.make(environment_id)
    plant = cast(Any, env.unwrapped)
    model = getattr(plant, "model", None)
    data = getattr(plant, "data", None)
    if model is None or data is None:
        env.close()
        raise ValueError("MyoSuite runtime lacks its native model or data")
    frame_skip = getattr(plant, "frame_skip", None)
    if not isinstance(frame_skip, int) or frame_skip <= 0:
        env.close()
        raise ValueError("MyoSuite frame_skip must be a positive integer")
    environment_spec = env.spec
    if environment_spec is None:
        env.close()
        raise ValueError("MyoSuite environment has no registered Gym specification")
    environment_kwargs = environment_spec.kwargs
    model_path = Path(environment_kwargs.get("model_path", ""))
    if (
        model_path != source_path
        or hashlib.sha256(model_path.read_bytes()).digest() != source_before
    ):
        env.close()
        raise ValueError("MyoSuite model source changed during native loading")
    if model.na <= 0 or model.na != model.nu:
        env.close()
        raise ValueError("every muscle actuator needs one native activation state")
    if not np.array_equal(
        np.asarray(model.actuator_actadr), np.arange(model.nu)
    ) or not np.array_equal(np.asarray(model.actuator_actnum), np.ones(model.nu)):
        env.close()
        raise ValueError("muscle activation state order is not actuator order")
    if model.nplugin or not np.all(np.asarray(plant._muscle_act_ind, dtype=bool)):
        env.close()
        raise ValueError(
            "plugins and non-muscle actuator channels need another adapter"
        )
    if getattr(plant, "mujoco_render_frames", False):
        env.close()
        raise ValueError("render callbacks are forbidden during frozen replay")
    if getattr(plant, "normalize_act", False) is not True:
        env.close()
        raise ValueError("MyoSuite action normalization policy is not declared")
    if getattr(plant, "ctrl_stages", ()):
        env.close()
        raise ValueError("stateful MyoSuite control stages need an explicit adapter")
    action_shape = env.action_space.shape
    if action_shape is None or len(action_shape) != 1 or action_shape[0] != model.nu:
        env.close()
        raise ValueError(
            "MyoSuite action-to-muscle mapping differs from native channels"
        )
    model.opt.disableflags |= int(mj.mjtDisableBit.mjDSBL_AUTORESET)
    return env, plant, model, data, model_path


def _module_bytes(module_name: str) -> bytes:
    module = importlib.import_module(module_name)
    path = getattr(module, "__file__", None)
    if not path:
        raise ValueError(f"native provider module {module_name!r} has no file")
    return Path(path).read_bytes()


def _provider_digest(environment_id: str, env: Any) -> str:
    module_names = (
        "myosuite.envs.myo.tasks.basic.muscle_mixin",
        "myosuite.envs.gymnasium_env",
        "myosuite.envs.wrappers",
        "mujoco._functions",
        "mujoco._structs",
    )
    digest = hashlib.sha256()
    for name in module_names:
        digest.update(name.encode("utf-8") + b"\0" + _module_bytes(name))
    digest.update(Path(__file__).read_bytes())
    digest.update(native_replay_admission_bytes())
    spec = env.spec
    plant = env.unwrapped
    spec_kwargs = {
        key: value for key, value in spec.kwargs.items() if key != "model_path"
    }
    digest.update(
        json.dumps(
            {
                "environment_id": environment_id,
                "entry_point": spec.entry_point,
                "kwargs": spec_kwargs,
                "frame_skip": int(plant.frame_skip),
                "action_space_low": np.asarray(plant.action_space.low).tolist(),
                "action_space_high": np.asarray(plant.action_space.high).tolist(),
                "control_boundary": "data.ctrl-direct-normalized-excitation",
                "termination_policy": "fixed-horizon-no-gym-step-no-auto-reset",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _initial_state_schema(
    model: Any,
    wrapper_values: tuple[float, ...],
    wrapper_names: tuple[str, ...],
    contracts: Any,
) -> Any:
    import mujoco as mj

    wrapper_representation = (
        ";".join(wrapper_names)
        + "|"
        + ";".join(
            f"{class_name}:{field}:{kind}" for class_name, field, kind in _WRAPPER_STATE
        )
    )
    components = (
        contracts.StateComponentSpec(
            "qpos",
            contracts.StateComponentRole.POSITION,
            model.nq,
            "native-SI",
            "MuJoCo-configuration",
        ),
        contracts.StateComponentSpec(
            "qvel",
            contracts.StateComponentRole.VELOCITY,
            model.nv,
            "native-SI",
            "MuJoCo-tangent-velocity",
        ),
        contracts.StateComponentSpec(
            "muscle_activation",
            contracts.StateComponentRole.MUSCLE_ACTIVATION,
            model.na,
            "1",
            "MuJoCo-data.act-in-actuator-order",
        ),
        contracts.StateComponentSpec(
            "actuator_internal",
            contracts.StateComponentRole.ACTUATOR_INTERNAL_STATE,
            model.nu,
            "1",
            "MuJoCo-data.ctrl-in-actuator-order",
        ),
        contracts.StateComponentSpec(
            "integration",
            contracts.StateComponentRole.AUXILIARY,
            mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION),
            "native-SI",
            "mjSTATE_INTEGRATION-including-warmstart",
        ),
        contracts.StateComponentSpec(
            "wrapper_state",
            contracts.StateComponentRole.AUXILIARY,
            len(wrapper_values),
            "enum-as-float",
            wrapper_representation,
        ),
    )
    return contracts.InitialStateSchema(
        "myosuite-native-excitation-state", _VERSION, components
    )


def _native_identity(
    environment_id: str, env: Any, model: Any, model_path: Path, contracts: Any
) -> tuple[Any, tuple[str, ...], tuple[float, ...]]:
    import mujoco as mj

    wrapper_names, wrapper_values = _wrapper_state(env)
    names = tuple(
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    )
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("native muscle actuators require unique explicit names")
    native = np.zeros(mj.mj_sizeModel(model), dtype=np.uint8)
    mj.mj_saveModel(model, buffer=native)
    loaded_hash = hashlib.sha256(native.tobytes()).hexdigest()
    provider_hash = _provider_digest(environment_id, env)
    schema = _initial_state_schema(model, wrapper_values, wrapper_names, contracts)
    source = model_path.read_bytes()
    identity = contracts.ModelIdentity(
        "myosuite",
        environment_id,
        "native-muscle-excitation",
        importlib.metadata.version("MyoSuite"),
        hashlib.sha256(source).hexdigest(),
        "myosuite-native-muscle-excitation-replay",
        _VERSION,
        provider_hash,
        schema,
        names,
        loaded_hash,
    )
    return identity, wrapper_names, wrapper_values


def _execution_policy(model: Any, plant: Any, identity: Any, contracts: Any) -> Any:
    import mujoco as mj

    frame_skip = int(plant.frame_skip)
    return contracts.ReplayExecutionPolicy(
        replay_mode=contracts.ReplayMode.NATIVE_OWN_CONTACT,
        solver_id=mj.mjtSolver(model.opt.solver).name,
        solver_version=mj.__version__,
        integration_method=(
            f"{mj.mjtIntegrator(model.opt.integrator).name};frame_skip={frame_skip}"
        ),
        step_policy="fixed",
        step_size_seconds=float(model.opt.timestep),
        initialization_policy_id="mjSTATE_INTEGRATION-plus-wrapper-state-restore",
        initialization_policy_version=_VERSION,
        input_player_id="myosuite-native-ctrl-muscle-excitation-zoh-fixed-horizon",
        input_player_version=_VERSION,
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id="myosuite-compiled-model-native-contact",
        contact_policy_version=identity.model_version,
        contact_policy_sha256=identity.loaded_native_model_sha256,
    )


def _state_values(
    env: Any, model: Any, data: Any
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    import mujoco as mj

    integration = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(model, data, integration, mj.mjtState.mjSTATE_INTEGRATION)
    _, wrapper_values = _wrapper_state(env)
    return (
        ("qpos", tuple(np.asarray(data.qpos, dtype=float))),
        ("qvel", tuple(np.asarray(data.qvel, dtype=float))),
        ("muscle_activation", tuple(np.asarray(data.act, dtype=float))),
        ("actuator_internal", tuple(np.asarray(data.ctrl, dtype=float))),
        ("integration", tuple(integration)),
        ("wrapper_state", wrapper_values),
    )


def build_native_myo_suite_excitation_bundle(
    environment_id: str,
    time_seconds: NDArray[np.float64],
    muscle_excitations: NDArray[np.float64],
    *,
    experiment_id: str = "myosuite-native-excitation-replay",
) -> Any:
    """Bind complete current native state and post-mapping excitation inputs."""
    import mujoco as mj

    contracts = _contracts()
    env, plant, model, data, model_path = _load_environment(environment_id)
    try:
        identity, _, _ = _native_identity(
            environment_id, env, model, model_path, contracts
        )
        policy = _execution_policy(model, plant, identity, contracts)
        times = np.asarray(time_seconds, dtype=np.float64)
        values = np.asarray(muscle_excitations, dtype=np.float64)
        dt = float(model.opt.timestep) * int(plant.frame_skip)
        if (
            times.ndim != 1
            or len(times) < 2
            or times[0] != 0
            or not np.isfinite(times).all()
            or not np.allclose(np.diff(times), dt, atol=1e-12, rtol=0)
        ):
            raise ValueError("excitation time grid must match native frame_skip")
        if values.shape != (len(times), model.nu) or not np.isfinite(values).all():
            raise ValueError(
                "excitation rows must match ordered native muscle channels"
            )
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("muscle excitation must be normalized to [0, 1]")
        limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
        control_ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
        if np.any(values[:, limited] < control_ranges[limited, 0]) or np.any(
            values[:, limited] > control_ranges[limited, 1]
        ):
            raise ValueError(
                "saved excitations must already satisfy native ctrl limits"
            )
        if not np.array_equal(values[-1], values[-2]):
            raise ValueError(
                "terminal ZOH sentinel must repeat the last applied excitation"
            )
        channels = tuple(
            contracts.InputChannel(name, name, "1")
            for name in identity.ordered_input_channel_ids
        )
        return contracts.build_experiment_replay_bundle(
            experiment_id,
            identity,
            (
                contracts.CapabilityDeclaration(
                    "native-muscle-excitation-replay",
                    True,
                    contracts.CapabilitySupport.SUPPORTED,
                    contracts.CapabilityAvailability.AVAILABLE,
                ),
            ),
            _state_values(env, model, data),
            channels,
            contracts.ActuationInputKind.MUSCLE_EXCITATION,
            contracts.InputInterpolation.ZERO_ORDER_HOLD,
            tuple(times),
            tuple(tuple(row) for row in values),
            policy,
        )
    finally:
        env.close()


def validate_myo_suite_bundle_contract(row: Any, binding: Any, bundle: Any) -> None:
    """Check MyoSuite-specific input and state semantics before native loading."""
    contracts = _contracts()
    validate_native_replay_bundle(bundle, contracts)
    if row.drive_mode.value != "muscle_excitation":
        raise ValueError("MyoSuite excitation cannot satisfy a torque inventory row")
    if bundle.schema_version != "experiment-replay/1.0.0":
        raise ValueError("unsupported frozen MyoSuite bundle schema")
    if bundle.model.engine_id != "myosuite":
        raise ValueError("bundle engine identity is not MyoSuite")
    if (
        bundle.input_history.input_kind
        is not contracts.ActuationInputKind.MUSCLE_EXCITATION
    ):
        raise ValueError("MyoSuite replay requires muscle_excitation input")
    if (
        bundle.input_history.interpolation
        is not contracts.InputInterpolation.ZERO_ORDER_HOLD
    ):
        raise ValueError("MyoSuite replay requires zero-order-held excitation")
    if bundle.input_history.timebase_id != "simulation_relative":
        raise ValueError("MyoSuite replay requires simulation-relative time")
    if bundle.policy.replay_mode is not contracts.ReplayMode.NATIVE_OWN_CONTACT:
        raise ValueError("MyoSuite replay requires native model-owned contact")
    if (
        bundle.policy.observation_access
        or bundle.policy.state_feedback_access
        or bundle.policy.state_reset_allowed
    ):
        raise ValueError("MyoSuite replay forbids observations, feedback and resets")
    if bundle.policy.step_policy != "fixed":
        raise ValueError("MyoSuite replay requires fixed native integration")
    for actual, expected, label in (
        (
            bundle.model.source_model_sha256,
            row.source_model_sha256,
            "inventory source model",
        ),
        (
            bundle.model.source_model_sha256,
            binding.source_model_sha256,
            "bound source model",
        ),
        (bundle.model.model_id, binding.native_model_id, "native model"),
        (bundle.model.variant_id, binding.native_variant_id, "native variant"),
        (
            bundle.model.provider_id,
            binding.native_execution_provider_id,
            "execution provider",
        ),
        (
            bundle.model.provider_sha256,
            binding.native_execution_provider_sha256,
            "execution provider hash",
        ),
        (
            bundle.model.loaded_native_model_sha256 or "",
            binding.loaded_native_model_sha256,
            "loaded native model",
        ),
        (bundle.state_schema_sha256, binding.state_schema_sha256, "state schema"),
        (
            bundle.input_channel_schema_sha256,
            binding.input_channel_schema_sha256,
            "input channel schema",
        ),
    ):
        if actual != expected:
            raise ValueError(f"MyoSuite {label} identity differs from explicit binding")
    component_ids = tuple(item.component_id for item in bundle.initial_state)
    expected_components = (
        "qpos",
        "qvel",
        "muscle_activation",
        "actuator_internal",
        "integration",
        "wrapper_state",
    )
    if component_ids != expected_components:
        raise ValueError("MyoSuite complete physical and wrapper state is required")
    model_state_schema = bundle.model.state_schema
    if (
        tuple(item.component_id for item in model_state_schema.components)
        != expected_components
    ):
        raise ValueError("MyoSuite initial-state schema is incomplete or reordered")
    _validate_myo_input_history(bundle, binding)


def _validate_myo_input_history(bundle: Any, binding: Any) -> None:
    """Validate ordered channels and the finite normalized ZOH history."""
    channels = tuple(item.channel_id for item in bundle.input_history.channels)
    if (
        channels != binding.ordered_input_channel_ids
        or channels != bundle.model.ordered_input_channel_ids
    ):
        raise ValueError("MyoSuite ordered actuator mapping differs")
    if any(channel.unit != "1" for channel in bundle.input_history.channels):
        raise ValueError("MyoSuite muscle excitation channels require normalized units")
    values = np.asarray(bundle.input_history.values, dtype=np.float64)
    times = np.asarray(bundle.input_history.time_seconds, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("MyoSuite excitation history must be finite and within [0, 1]")
    if not np.array_equal(values[-1], values[-2]):
        raise ValueError("MyoSuite terminal ZOH sentinel differs from last input")
    if not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("MyoSuite simulation time grid is invalid")


def _restore_wrapper_state(env: Any, values: tuple[float, ...]) -> None:
    chain: list[Any] = []
    current = env
    while hasattr(current, "env"):
        chain.append(current)
        current = current.env
    names = tuple(f"{type(item).__module__}.{type(item).__name__}" for item in chain)
    expected, _ = _wrapper_state(env)
    if names != expected or len(values) != len(_WRAPPER_STATE):
        raise ValueError("native wrapper state schema differs")
    lookup = dict(zip(names, chain, strict=True))
    for value, (class_name, attribute, kind) in zip(
        values, _WRAPPER_STATE, strict=True
    ):
        wrapper = lookup[class_name]
        if kind == "counter":
            if value != int(value) or value < -1:
                raise ValueError("invalid frozen Gymnasium elapsed-step state")
            setattr(wrapper, attribute, None if value == -1 else int(value))
        else:
            if value not in (0.0, 1.0):
                raise ValueError("invalid frozen MyoSuite wrapper flag")
            setattr(wrapper, attribute, bool(value))


def _validate_replay_request(bundle: Any, environment_id: str, contracts: Any) -> Any:
    bundle = validate_native_replay_bundle(bundle, contracts)
    if bundle.schema_version != "experiment-replay/1.0.0":
        raise ValueError("unsupported frozen MyoSuite replay schema")
    if (
        bundle.input_history.input_kind
        is not contracts.ActuationInputKind.MUSCLE_EXCITATION
    ):
        raise ValueError("MyoSuite replay requires muscle_excitation input")
    if (
        bundle.input_history.interpolation
        is not contracts.InputInterpolation.ZERO_ORDER_HOLD
    ):
        raise ValueError(
            "MyoSuite replay currently requires zero-order-held excitation"
        )
    if bundle.input_history.timebase_id != "simulation_relative":
        raise ValueError("MyoSuite replay requires simulation-relative time")
    if (
        bundle.blocking_capabilities
        or bundle.policy.observation_access
        or bundle.policy.state_feedback_access
        or bundle.policy.state_reset_allowed
    ):
        raise ValueError(
            "MyoSuite replay requires available, open-loop, no-reset policy"
        )
    if bundle.model.engine_id != "myosuite" or bundle.model.model_id != environment_id:
        raise ValueError("frozen native MyoSuite environment identity differs")
    return bundle


def _restore_replay_state(
    env: Any, model: Any, data: Any, bundle: Any
) -> dict[str, NDArray[np.float64]]:
    import mujoco as mj

    states = {
        item.component_id: np.asarray(item.values, dtype=np.float64)
        for item in bundle.initial_state
    }
    expected = (
        "qpos",
        "qvel",
        "muscle_activation",
        "actuator_internal",
        "integration",
        "wrapper_state",
    )
    if tuple(states) != expected:
        raise ValueError(
            "complete ordered MyoSuite physical and wrapper state is required"
        )
    mj.mj_setState(model, data, states["integration"], mj.mjtState.mjSTATE_INTEGRATION)
    _restore_wrapper_state(env, tuple(states["wrapper_state"]))
    if not np.array_equal(data.qpos, states["qpos"]) or not np.array_equal(
        data.qvel, states["qvel"]
    ):
        raise ValueError("restored MyoSuite configuration or velocity differs")
    if not np.array_equal(data.act, states["muscle_activation"]) or not np.array_equal(
        data.ctrl, states["actuator_internal"]
    ):
        raise ValueError("restored MyoSuite muscle or actuator state differs")
    if data.time != 0.0 or np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("initial native time and external-load state must be zero")
    if not np.array_equal(
        np.asarray(_state_values(env, model, data)[-1][1]), states["wrapper_state"]
    ):
        raise ValueError("restored Gymnasium wrapper state differs")
    return states


def _execute_native_steps(
    plant: Any,
    model: Any,
    data: Any,
    times: NDArray[np.float64],
    expected_input: NDArray[np.float64],
    states: dict[str, NDArray[np.float64]],
    bundle: Any,
) -> NativeMyoSuiteExcitationReplay:
    import mujoco as mj

    qpos = [data.qpos.copy()]
    qvel = [data.qvel.copy()]
    activations = [data.act.copy()]
    controls = [data.ctrl.copy()]
    integration = [states["integration"].copy()]
    wrapper_states = [states["wrapper_state"].copy()]
    applied = []
    for excitation in expected_input[:-1]:
        data.ctrl[:] = excitation
        actual = np.asarray(data.ctrl, dtype=np.float64).copy()
        if not np.array_equal(actual, excitation):
            raise ValueError("native applied excitation readback differs")
        limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
        limits = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
        if np.any(actual[limited] < limits[limited, 0]) or np.any(
            actual[limited] > limits[limited, 1]
        ):
            raise ValueError("native applied excitation violates actuator limits")
        applied.append(actual)
        mj.mj_step(model, data, nstep=int(plant.frame_skip))
        mj.mj_forward(model, data)
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        activations.append(data.act.copy())
        controls.append(data.ctrl.copy())
        state = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
        mj.mj_getState(model, data, state, mj.mjtState.mjSTATE_INTEGRATION)
        integration.append(state)
        wrapper_states.append(states["wrapper_state"].copy())
    return NativeMyoSuiteExcitationReplay(
        times.copy(),
        np.asarray(qpos),
        np.asarray(qvel),
        np.asarray(activations),
        np.asarray(controls),
        np.asarray(integration),
        np.asarray(wrapper_states),
        np.asarray(applied),
        bundle.applied_input_sha256,
        bundle.policy_sha256,
    )


def replay_native_myo_suite_excitation_bundle(
    bundle: Any, environment_id: str
) -> NativeMyoSuiteExcitationReplay:
    """Replay a validated T01 excitation bundle through native MyoSuite/MuJoCo."""
    contracts = _contracts()
    bundle = _validate_replay_request(bundle, environment_id, contracts)
    env, plant, model, data, model_path = _load_environment(environment_id)
    try:
        identity, _, _ = _native_identity(
            environment_id, env, model, model_path, contracts
        )
        policy = _execution_policy(model, plant, identity, contracts)
        if identity != bundle.model or policy != bundle.policy:
            raise ValueError("MyoSuite model/provider or executed policy differs")
        if (
            tuple(channel.channel_id for channel in bundle.input_history.channels)
            != identity.ordered_input_channel_ids
        ):
            raise ValueError("ordered native muscle-channel mapping differs")
        expected_times = np.asarray(bundle.input_history.time_seconds, dtype=np.float64)
        expected_input = np.asarray(bundle.input_history.values, dtype=np.float64)
        if expected_input.shape != (len(expected_times), model.nu):
            raise ValueError(
                "frozen excitation dimensions differ from native actuators"
            )
        if not np.allclose(
            np.diff(expected_times),
            model.opt.timestep * plant.frame_skip,
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("frozen input clock differs from native frame_skip")
        states = _restore_replay_state(env, model, data, bundle)
        return _execute_native_steps(
            plant, model, data, expected_times, expected_input, states, bundle
        )
    finally:
        env.close()


__all__ = [
    "NativeMyoSuiteExcitationReplay",
    "build_native_myo_suite_excitation_bundle",
    "replay_native_myo_suite_excitation_bundle",
]
