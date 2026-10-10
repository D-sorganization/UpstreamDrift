"""Frozen Tools bundles replayed by actual native MuJoCo stepping (F06).

This first adapter admits direct unit-gain hinge motors only. It preserves
native configuration/velocity dimensions and complete mjSTATE_INTEGRATION,
including numerical warm-start state. Unsupported actuator dynamics are
rejected rather than represented as generalized torque.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.engines.native_replay_contracts import (
    _MUJOCO_GLOBAL_CALLBACKS,
    native_replay_admission_bytes,
    native_replay_contract_types,
    require_no_global_mujoco_callbacks,
    validate_native_replay_bundle,
)

if TYPE_CHECKING:
    from sidekick.lab.mocap import ExperimentReplayBundle

_VERSION = "1.0.0"
# Kept for sibling native replay modules that guard the same global callbacks.
_CALLBACKS = _MUJOCO_GLOBAL_CALLBACKS


@dataclass(frozen=True)
class NativeTorqueReplay:
    """Uninterrupted native evidence; no capture or parity acceptance verdict."""

    time_seconds: NDArray[np.float64]
    qpos: NDArray[np.float64]
    qvel: NDArray[np.float64]
    integration_states: NDArray[np.float64]
    applied_actuator_torques: NDArray[np.float64]
    generalized_actuator_torques: NDArray[np.float64]
    input_sha256: str
    policy_sha256: str


def _contracts() -> Any:
    return native_replay_contract_types()


def _load_native(path: Path) -> tuple[Any, Any]:
    import mujoco as mj

    require_no_global_mujoco_callbacks(mj)
    source_hash = hashlib.sha256(path.read_bytes()).digest()
    model = mj.MjModel.from_xml_path(str(path))
    if hashlib.sha256(path.read_bytes()).digest() != source_hash:
        raise ValueError("native model source changed during loading")
    if model.nplugin or model.nmocap or not model.nu:
        raise ValueError(
            "plugins, mocap drive or absent motors need another native adapter"
        )
    if model.opt.disableactuator or np.any(model.jnt_actfrclimited):
        raise ValueError(
            "disabled actuator groups or joint force clamps are unsupported"
        )
    if np.any(model.body_gravcomp):
        raise ValueError(
            "active gravity compensation needs an explicit assistance policy"
        )
    _validate_unit_motors(model, mj)
    # A numerical failure must fail, never silently reset physical state.
    model.opt.disableflags |= int(mj.mjtDisableBit.mjDSBL_AUTORESET)
    return model, mj.MjData(model)


def native_initial_state_from_joint_state(
    model_path: Path, joint_state: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Export the complete native fixture state without importing MuJoCo in shared code."""
    import mujoco as mj

    values = np.asarray(joint_state, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("one-hinge fixture requires finite position and velocity")
    model, data = _load_native(model_path)
    if model.nq != 1 or model.nv != 1 or model.nu != 1:
        raise ValueError("native fixture must have one hinge and one unit motor")
    data.qpos[0], data.qvel[0] = values
    state = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(model, data, state, mj.mjtState.mjSTATE_INTEGRATION)
    return state


def _validate_unit_motors(model: Any, mj: Any) -> None:
    """Prove that saved actuator torque equals the admitted motor's input."""
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("native motors must have unique explicit names")
    gear = np.zeros((model.nu, 6))
    gear[:, 0] = 1
    if (
        np.any(model.actuator_trntype != mj.mjtTrn.mjTRN_JOINT)
        or np.any(model.actuator_dyntype != mj.mjtDyn.mjDYN_NONE)
        or np.any(model.actuator_gaintype != mj.mjtGain.mjGAIN_FIXED)
        or np.any(model.actuator_biastype != mj.mjtBias.mjBIAS_NONE)
        or np.any(model.actuator_gainprm[:, 0] != 1)
        or not np.array_equal(model.actuator_gear, gear)
        or np.any(model.actuator_forcelimited)
    ):
        raise ValueError(
            "native adapter supports direct unit-gain motors without internal dynamics or force clamps"
        )
    joints = model.actuator_trnid[:, 0]
    if np.any(model.jnt_type[joints] != mj.mjtJoint.mjJNT_HINGE):
        raise ValueError("native torque inputs require hinge-joint motor targets")


def _restore_native(model: Any, data: Any, values: NDArray[np.float64]) -> None:
    import mujoco as mj

    size = mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION)
    state = np.array(values, dtype=np.float64, copy=True)
    if state.shape != (size,) or not np.isfinite(state).all():
        raise ValueError("complete finite native integration state is required")
    mj.mj_setState(model, data, state, mj.mjtState.mjSTATE_INTEGRATION)
    normalized = data.qpos.copy()
    mj.mj_normalizeQuat(model, normalized)
    if not np.allclose(data.qpos, normalized, atol=1e-12, rtol=0):
        raise ValueError("initial native quaternion configuration must be normalized")
    if data.time != 0 or np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("initial replay time must be zero and external loads absent")


def _native_identity(model: Any, path: Path, contracts: Any) -> Any:
    import mujoco as mj

    binary = np.zeros(mj.mj_sizeModel(model), dtype=np.uint8)
    mj.mj_saveModel(model, buffer=binary)
    loaded_hash = hashlib.sha256(binary.tobytes()).hexdigest()
    native_module = importlib.import_module("mujoco._functions")
    if native_module.__file__ is None:
        raise ValueError("native provider has no hashable module artifact")
    native_bytes = Path(native_module.__file__).read_bytes()
    provider_hash = hashlib.sha256(
        native_bytes + Path(__file__).read_bytes() + native_replay_admission_bytes()
    ).hexdigest()
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
            "integration",
            contracts.StateComponentRole.AUXILIARY,
            mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION),
            "native-SI",
            "mjSTATE_INTEGRATION-including-warmstart",
        ),
    )
    names = tuple(
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    )
    return contracts.ModelIdentity(
        "mujoco",
        "native-model",
        "unit-hinge-motors",
        _VERSION,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        "mujoco-native-torque-replay",
        mj.__version__,
        provider_hash,
        contracts.InitialStateSchema("mujoco-native-integration", _VERSION, components),
        names,
        loaded_hash,
    )


def _execution_policy(model: Any, identity: Any, contracts: Any) -> Any:
    import mujoco as mj

    return contracts.ReplayExecutionPolicy(
        replay_mode=contracts.ReplayMode.NATIVE_OWN_CONTACT,
        solver_id=mj.mjtSolver(model.opt.solver).name,
        solver_version=mj.__version__,
        integration_method=mj.mjtIntegrator(model.opt.integrator).name,
        step_policy="fixed",
        step_size_seconds=float(model.opt.timestep),
        initialization_policy_id="mjSTATE_INTEGRATION-restore-no-resets",
        initialization_policy_version=_VERSION,
        input_player_id="native-unit-motor-zoh-terminal-sentinel",
        input_player_version=_VERSION,
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id="mujoco-compiled-model-owned-contact",
        contact_policy_version=_VERSION,
        contact_policy_sha256=identity.loaded_native_model_sha256,
    )


def _validate_history(
    model: Any, times: NDArray[np.float64], values: NDArray[np.float64]
) -> None:
    if (
        times.ndim != 1
        or len(times) < 2
        or times[0] != 0
        or not np.isfinite(times).all()
        or not np.allclose(np.diff(times), model.opt.timestep, atol=1e-12, rtol=0)
    ):
        raise ValueError("time grid must begin at zero and match every native timestep")
    if values.shape != (len(times), model.nu) or not np.isfinite(values).all():
        raise ValueError(
            "finite torque rows must match times and native actuator order"
        )
    if not np.array_equal(values[-1], values[-2]):
        raise ValueError("terminal ZOH sentinel must equal the last executed input")
    limited = model.actuator_ctrllimited.astype(bool)
    ranges = model.actuator_ctrlrange[limited]
    if np.any(values[:, limited] < ranges[:, 0]) or np.any(
        values[:, limited] > ranges[:, 1]
    ):
        raise ValueError(
            "saved native motor torques must already satisfy command limits"
        )


def build_native_torque_bundle(
    model_path: str | Path,
    initial_integration_state: NDArray[np.float64],
    time_seconds: NDArray[np.float64],
    applied_motor_torques: NDArray[np.float64],
    *,
    experiment_id: str = "native-torque-replay",
) -> ExperimentReplayBundle:
    """Bind complete native initialization and exact post-limit held torque.

    The last input row is an explicit unused terminal sentinel. One preceding
    row is held throughout each actual native timestep. This builder accepts
    recorded inputs, never a controller or observation callback.
    """
    import mujoco as mj

    contracts = _contracts()
    path = Path(model_path)
    model, data = _load_native(path)
    times = np.array(time_seconds, dtype=np.float64, copy=True)
    values = np.array(applied_motor_torques, dtype=np.float64, copy=True)
    _validate_history(model, times, values)
    _restore_native(model, data, initial_integration_state)
    identity = _native_identity(model, path, contracts)
    channels = tuple(
        contracts.InputChannel(
            name,
            name,
            "N*m",
            mj.mj_id2name(
                model, mj.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0])
            ),
        )
        for i, name in enumerate(identity.ordered_input_channel_ids)
    )
    return contracts.build_experiment_replay_bundle(
        experiment_id,
        identity,
        (
            contracts.CapabilityDeclaration(
                "native-unit-motor-replay",
                True,
                contracts.CapabilitySupport.SUPPORTED,
                contracts.CapabilityAvailability.AVAILABLE,
            ),
        ),
        (
            ("qpos", tuple(data.qpos)),
            ("qvel", tuple(data.qvel)),
            ("integration", tuple(initial_integration_state)),
        ),
        channels,
        contracts.ActuationInputKind.ACTUATOR_TORQUE,
        contracts.InputInterpolation.ZERO_ORDER_HOLD,
        tuple(times),
        tuple(tuple(row) for row in values),
        _execution_policy(model, identity, contracts),
    )


def replay_native_torque_bundle(
    bundle: ExperimentReplayBundle,
    model_path: str | Path,
) -> NativeTorqueReplay:
    """Replay only frozen inputs through a fresh native plant, without resets.

    Revalidates Tools integrity and actual loaded model/provider/policy/channel
    identity. Native transmission effort is checked against saved motor torque.
    No reference states, feedback law or observation provider can be supplied.
    """
    contracts = _contracts()
    bundle = validate_native_replay_bundle(bundle, contracts)
    initial = {
        component.component_id: component.values for component in bundle.initial_state
    }
    if "integration" not in initial:
        raise ValueError("complete native integration state is missing")
    expected = build_native_torque_bundle(
        model_path,
        np.array(initial["integration"]),
        np.array(bundle.input_history.time_seconds),
        np.array(bundle.input_history.values),
        experiment_id=bundle.experiment_id,
    )
    if (
        bundle.model,
        bundle.initial_state,
        bundle.input_history.channels,
        bundle.policy,
    ) != (
        expected.model,
        expected.initial_state,
        expected.input_history.channels,
        expected.policy,
    ):
        raise ValueError(
            "native model, complete state, ordered channels or executed policy differs"
        )
    if (
        bundle.input_history.input_kind != contracts.ActuationInputKind.ACTUATOR_TORQUE
        or bundle.input_history.interpolation
        != contracts.InputInterpolation.ZERO_ORDER_HOLD
    ):
        raise ValueError("native direct-motor replay requires held actuator torque")
    return _step_native(bundle, Path(model_path))


def native_marker_positions_from_replay(
    bundle: ExperimentReplayBundle,
    model_path: str | Path,
    replay: NativeTorqueReplay,
    attachments: tuple[tuple[str, str, tuple[float, float, float]], ...],
) -> Any:
    """Map replay configurations to named body-local points using MuJoCo FK."""
    import mujoco as mj

    from src.shared.python.motion_matching.replay_metrics import (
        NativeMarkerPositionOutput,
    )

    contracts = _contracts()
    bundle = validate_native_replay_bundle(bundle, contracts)
    path = Path(model_path)
    model, data = _load_native(path)
    if _native_identity(model, path, contracts) != bundle.model:
        raise ValueError("marker FK model identity differs from frozen replay bundle")
    times = np.asarray(replay.time_seconds, dtype=np.float64)
    qpos = np.asarray(replay.qpos, dtype=np.float64)
    if (
        qpos.shape != (len(times), model.nq)
        or not np.isfinite(times).all()
        or not np.isfinite(qpos).all()
        or len(times) < 2
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError("marker FK requires finite full-horizon native qpos samples")
    labels = tuple(item[0] for item in attachments)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("marker labels must be non-empty, unique and ordered")
    resolved: list[tuple[int, np.ndarray]] = []
    for label, body_name, offset in attachments:
        if not label or not body_name:
            raise ValueError("marker labels and native body names must be explicit")
        local = np.asarray(offset, dtype=np.float64)
        if local.shape != (3,) or not np.isfinite(local).all():
            raise ValueError("marker body-local offset must be a finite 3-vector")
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"unknown native marker body: {body_name}")
        resolved.append((body_id, local))
    positions: NDArray[np.float64] = np.empty(
        (len(times), len(attachments), 3), dtype=np.float64
    )
    for sample, configuration in enumerate(qpos):
        data.qpos[:] = configuration
        mj.mj_forward(model, data)
        for marker, (body_id, local) in enumerate(resolved):
            positions[sample, marker] = (
                data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ local
            )
    return NativeMarkerPositionOutput(
        times,
        positions,
        labels,
        "world",
        bundle.input_history.timebase_id,
    )


def _step_native(bundle: Any, path: Path) -> NativeTorqueReplay:
    import mujoco as mj

    model, data = _load_native(path)
    if _native_identity(model, path, _contracts()) != bundle.model:
        raise ValueError("execution native model/provider identity differs")
    state = next(
        item.values
        for item in bundle.initial_state
        if item.component_id == "integration"
    )
    _restore_native(model, data, np.array(state))
    times = np.array(bundle.input_history.time_seconds)
    inputs = np.array(bundle.input_history.values[:-1])
    full = np.empty((len(times), len(state)))
    qs, vs = np.empty((len(times), model.nq)), np.empty((len(times), model.nv))
    generalized = np.empty((len(inputs), model.nv))
    for row in range(len(times)):
        mj.mj_getState(model, data, full[row], mj.mjtState.mjSTATE_INTEGRATION)
        qs[row], vs[row] = data.qpos, data.qvel
        if row == len(inputs):
            break
        data.ctrl[:] = inputs[row]
        mj.mj_step(model, data)
        expected = np.zeros(model.nv)
        np.add.at(expected, model.jnt_dofadr[model.actuator_trnid[:, 0]], inputs[row])
        generalized[row] = data.qfrc_actuator
        if not np.allclose(generalized[row], expected, atol=1e-12, rtol=0):
            raise RuntimeError(
                "native generalized actuator effort differs from saved torque"
            )
        if not np.isfinite(data.qpos).all() or not np.isclose(
            data.time, times[row + 1], atol=1e-12, rtol=0
        ):
            raise RuntimeError("native integration failed or reset its state/time")
    for array in (times, qs, vs, full, inputs, generalized):
        if not np.isfinite(array).all():
            raise RuntimeError("native replay produced nonfinite evidence")
        array.setflags(write=False)
    return NativeTorqueReplay(
        times,
        qs,
        vs,
        full,
        inputs,
        generalized,
        bundle.applied_input_sha256,
        bundle.policy_sha256,
    )
