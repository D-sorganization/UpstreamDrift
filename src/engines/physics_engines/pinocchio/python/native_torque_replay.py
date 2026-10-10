"""Independent native Pinocchio ABA replay through the existing public engine.

This collision-free unit-motor policy admits floating configurations and
revolute torque inputs. Native Data caches are recomputed from q/v; they are
not extra physical state. Contact, actuator dynamics and external forces
require another policy and are rejected. No acceptance verdict is produced.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from defusedxml import ElementTree as SafeET
from numpy.typing import NDArray

from src.engines.native_replay_contracts import (
    native_replay_admission_bytes,
    native_replay_contract_types,
    require_native_replay_equivalence,
    validate_frozen_torque_history,
    validate_native_replay_bundle,
)

if TYPE_CHECKING:
    from sidekick.lab.mocap import ExperimentReplayBundle

    from src.shared.python.motion_matching.replay_metrics import (
        NativeMarkerPositionOutput,
    )

_VERSION = "1.0.0"
_SOURCE_TAGS = frozenset(
    "robot link inertial origin mass inertia visual geometry mesh material color "
    "texture box sphere cylinder joint parent child axis limit transmission type "
    "actuator hardwareInterface mechanicalReduction".split()
)


@dataclass(frozen=True)
class NativePinocchioTorqueReplay:
    """Read-only native trajectory and effective input audit, without qualification."""

    time_seconds: NDArray[np.float64]
    qpos: NDArray[np.float64]
    qvel: NDArray[np.float64]
    applied_actuator_torques: NDArray[np.float64]
    generalized_actuator_torques: NDArray[np.float64]
    input_sha256: str
    policy_sha256: str


def _motors(raw: bytes, model: Any) -> tuple[tuple[str, str, int, float], ...]:
    root = SafeET.fromstring(raw, forbid_dtd=True, forbid_entities=True)
    if root.tag != "robot":
        raise ValueError("native replay requires a URDF robot")
    for element in root.iter():
        if element.tag not in _SOURCE_TAGS or "filename" in element.attrib:
            raise ValueError("source semantics need another native replay policy")
        if element.tag == "mechanicalReduction" and float(element.text or "nan") != 1:
            raise ValueError("native torque requires unit transmission reduction")
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    if any(
        joint.get("type") not in {"fixed", "floating", "revolute"}
        for joint in joints.values()
    ):
        raise ValueError(
            "only fixed, floating and bounded revolute joints are admitted"
        )
    motors: list[tuple[str, str, int, float]] = []
    for transmission in root.findall("transmission"):
        kind = transmission.get("type") or transmission.findtext("type")
        if kind not in {
            "SimpleTransmission",
            "transmission_interface/SimpleTransmission",
        }:
            raise ValueError("one-to-one SimpleTransmission is required")
        declared_joints = transmission.findall("joint")
        actuators = transmission.findall("actuator")
        if len(declared_joints) != 1 or len(actuators) != 1:
            raise ValueError("one motor must map to one revolute joint")
        name, motor = declared_joints[0].get("name"), actuators[0].get("name")
        if not name or not motor or name not in joints:
            raise ValueError("motor and joint identities must resolve uniquely")
        joint_id = model.getJointId(name)
        if joint_id >= model.njoints or joints[name].get("type") != "revolute":
            raise ValueError("motor must resolve to a native revolute joint")
        joint = model.joints[joint_id]
        if joint.nq != 1 or joint.nv != 1:
            raise ValueError("motor joint must have one configuration and velocity")
        interfaces = transmission.findall(".//hardwareInterface")
        if any(
            item.text
            not in {"EffortJointInterface", "hardware_interface/EffortJointInterface"}
            for item in interfaces
        ):
            raise ValueError("motor interface must declare effort")
        motors.append(
            (motor, name, int(joint.idx_v), float(model.effortLimit[joint.idx_v]))
        )
    if (
        not motors
        or len({m[0] for m in motors}) != len(motors)
        or len({m[2] for m in motors}) != len(motors)
    ):
        raise ValueError("native motors and generalized channels must be unique")
    return tuple(sorted(motors, key=lambda item: item[2]))


def _load(path: Path) -> tuple[Any, tuple[tuple[str, str, int, float], ...]]:
    from .pinocchio_physics_engine import PinocchioPhysicsEngine

    raw = path.read_bytes()
    # Parse securely before the native loader sees any external declaration.
    source = SafeET.fromstring(raw, forbid_dtd=True, forbid_entities=True)
    if any(
        item.tag not in _SOURCE_TAGS or "filename" in item.attrib
        for item in source.iter()
    ):
        raise ValueError("source semantics need another native replay policy")
    engine = PinocchioPhysicsEngine()
    engine.load_from_string(raw.decode("utf-8"), "urdf")
    if raw != path.read_bytes():
        raise ValueError("native model source changed while loading")
    return engine, _motors(raw, engine.model)


def _restore(engine: Any, q: Any, v: Any) -> None:
    import pinocchio as pin

    position, velocity = np.asarray(q, dtype=float), np.asarray(v, dtype=float)
    if position.shape != (engine.model.nq,) or velocity.shape != (engine.model.nv,):
        raise ValueError("complete native configuration and tangent velocity required")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("native initial state must be finite")
    if not pin.isNormalized(engine.model, position, 1e-12):
        raise ValueError("initial native quaternion must already be normalized")
    engine.set_state(position, velocity)
    if not np.array_equal(engine.q, position) or not np.array_equal(engine.v, velocity):
        raise ValueError("native restoration changed physical state")


def _identity(engine: Any, path: Path, motors: tuple[Any, ...], contracts: Any) -> Any:
    import pinocchio as pin

    from . import pinocchio_physics_engine

    native = importlib.import_module("pinocchio.pinocchio_pywrap_default")
    provider = hashlib.sha256(native_replay_admission_bytes())
    for artifact_name in (
        __file__,
        pinocchio_physics_engine.__file__,
        native.__file__,
    ):
        if not isinstance(artifact_name, str):
            raise ValueError("native provider source artifact is unavailable")
        provider.update(Path(artifact_name).read_bytes())
    specs = tuple(
        contracts.StateComponentSpec(name, role, size, "native-SI", frame)
        for name, role, size, frame in (
            (
                "qpos",
                contracts.StateComponentRole.POSITION,
                engine.model.nq,
                "pinocchio-configuration",
            ),
            (
                "qvel",
                contracts.StateComponentRole.VELOCITY,
                engine.model.nv,
                "pinocchio-tangent-velocity",
            ),
        )
    )
    return contracts.ModelIdentity(
        "pinocchio",
        "native-model",
        "unit-revolute-motors",
        _VERSION,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        "pinocchio-native-torque-replay",
        pin.__version__,
        provider.hexdigest(),
        contracts.InitialStateSchema("pinocchio-native-q-v", _VERSION, specs),
        tuple(motor[0] for motor in motors),
        hashlib.sha256(engine.model.saveToString().encode()).hexdigest(),
    )


def _policy(identity: Any, dt: float, contracts: Any) -> Any:
    return contracts.ReplayExecutionPolicy(
        replay_mode=contracts.ReplayMode.NATIVE_OWN_CONTACT,
        solver_id="pinocchio-native-aba",
        solver_version=identity.provider_version,
        integration_method="PinocchioPhysicsEngine-rk4-manifold",
        step_policy="fixed",
        step_size_seconds=dt,
        initialization_policy_id="fresh-engine-q-v-time-zero-derived-caches",
        initialization_policy_version=_VERSION,
        input_player_id="unit-generalized-scatter-zoh-terminal-sentinel",
        input_player_version=_VERSION,
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id="pinocchio-no-contact-no-external-force",
        contact_policy_version=_VERSION,
        contact_policy_sha256=identity.loaded_native_model_sha256,
    )


def build_native_pinocchio_torque_bundle(
    model_path: str | Path,
    initial_q: NDArray[np.float64],
    initial_v: NDArray[np.float64],
    time_seconds: NDArray[np.float64],
    applied_motor_torques: NDArray[np.float64],
    *,
    time_step: float = 0.001,
    experiment_id: str = "native-pinocchio-replay",
) -> ExperimentReplayBundle:
    """Bind complete physical q/v, actual native model/provider and frozen inputs."""
    contracts = native_replay_contract_types()
    path = Path(model_path)
    engine, motors = _load(path)
    times = np.array(time_seconds, dtype=float, copy=True)
    values = np.array(applied_motor_torques, dtype=float, copy=True)
    validate_frozen_torque_history(
        times, values, time_step, np.array([m[3] for m in motors])
    )
    _restore(engine, initial_q, initial_v)
    identity = _identity(engine, path, motors, contracts)
    channels = tuple(
        contracts.InputChannel(motor, motor, "N*m", joint)
        for motor, joint, _, _ in motors
    )
    capability = contracts.CapabilityDeclaration(
        "native-unit-motor-replay",
        True,
        contracts.CapabilitySupport.SUPPORTED,
        contracts.CapabilityAvailability.AVAILABLE,
    )
    return contracts.build_experiment_replay_bundle(
        experiment_id,
        identity,
        (capability,),
        (("qpos", tuple(engine.q)), ("qvel", tuple(engine.v))),
        channels,
        contracts.ActuationInputKind.ACTUATOR_TORQUE,
        contracts.InputInterpolation.ZERO_ORDER_HOLD,
        tuple(times),
        tuple(tuple(row) for row in values),
        _policy(identity, time_step, contracts),
    )


def replay_native_pinocchio_torque_bundle(
    bundle: ExperimentReplayBundle,
    model_path: str | Path,
) -> NativePinocchioTorqueReplay:
    """Run an uninterrupted fresh-engine replay without observations or feedback."""
    contracts = native_replay_contract_types()
    bundle = validate_native_replay_bundle(bundle, contracts)
    state = {item.component_id: np.array(item.values) for item in bundle.initial_state}
    if set(state) != {"qpos", "qvel"} or bundle.policy.step_size_seconds is None:
        raise ValueError("complete native q/v and explicit fixed stepping required")
    expected = build_native_pinocchio_torque_bundle(
        model_path,
        state["qpos"],
        state["qvel"],
        np.array(bundle.input_history.time_seconds),
        np.array(bundle.input_history.values),
        time_step=bundle.policy.step_size_seconds,
        experiment_id=bundle.experiment_id,
    )
    require_native_replay_equivalence(bundle, expected, contracts)
    path = Path(model_path)
    engine, motors = _load(path)
    if _identity(engine, path, motors, contracts) != bundle.model:
        raise ValueError("execution native model/provider identity differs")
    _restore(engine, state["qpos"], state["qvel"])
    return _step(engine, motors, bundle)


def native_marker_positions_from_replay(
    bundle: ExperimentReplayBundle,
    model_path: str | Path,
    replay: NativePinocchioTorqueReplay,
    attachments: tuple[tuple[str, str, tuple[float, float, float]], ...],
) -> NativeMarkerPositionOutput:
    """Evaluate explicit frame-local points at every actual replay q state."""
    from src.shared.python.motion_matching.replay_metrics import (
        NativeMarkerPositionOutput,
    )

    contracts = native_replay_contract_types()
    bundle = validate_native_replay_bundle(bundle, contracts)
    times = np.asarray(replay.time_seconds, dtype=np.float64)
    expected_times = np.asarray(bundle.input_history.time_seconds, dtype=np.float64)
    q_history = np.asarray(replay.qpos, dtype=np.float64)
    v_history = np.asarray(replay.qvel, dtype=np.float64)
    state = {
        item.component_id: np.asarray(item.values) for item in bundle.initial_state
    }
    if tuple(state) != ("qpos", "qvel"):
        raise ValueError("Pinocchio FK requires complete native qpos and qvel state")
    if (
        not np.array_equal(times, expected_times)
        or q_history.shape != (len(times), len(state["qpos"]))
        or v_history.shape != (len(times), len(state["qvel"]))
        or not np.isfinite(times).all()
        or not np.isfinite(q_history).all()
        or not np.isfinite(v_history).all()
    ):
        raise ValueError("Pinocchio FK requires the exact complete native output")
    if (
        replay.input_sha256 != bundle.applied_input_sha256
        or replay.policy_sha256 != bundle.policy_sha256
    ):
        raise ValueError("Pinocchio FK replay input or policy identity differs")
    if not np.array_equal(q_history[0], state["qpos"]) or not np.array_equal(
        v_history[0], state["qvel"]
    ):
        raise ValueError("Pinocchio FK initial state differs from frozen bundle")
    if not attachments:
        raise ValueError("Pinocchio FK requires explicit marker attachments")
    labels = tuple(item[0] for item in attachments)
    if any(not label.strip() for label in labels) or len(labels) != len(set(labels)):
        raise ValueError("Pinocchio marker labels must be non-empty and unique")
    frame_ids = tuple(item[1] for item in attachments)
    if any(not frame_id.strip() for frame_id in frame_ids):
        raise ValueError("Pinocchio marker frame names must be explicit")
    offsets = np.asarray([item[2] for item in attachments], dtype=np.float64)
    if offsets.shape != (len(attachments), 3) or not np.isfinite(offsets).all():
        raise ValueError("Pinocchio marker offsets must be finite local 3-vectors")

    engine, motors = _load(Path(model_path))
    if _identity(engine, Path(model_path), motors, contracts) != bundle.model:
        raise ValueError("Pinocchio marker FK model/provider identity differs")
    if engine.model is None:
        raise RuntimeError("Pinocchio native model failed to initialize")
    for frame_id in frame_ids:
        matches = [
            index
            for index, frame in enumerate(engine.model.frames)
            if frame.name == frame_id
        ]
        if len(matches) != 1:
            raise ValueError("unknown or ambiguous native marker frame")

    positions = np.empty((len(times), len(attachments), 3), dtype=np.float64)
    for sample, (q, v) in enumerate(zip(q_history, v_history, strict=True)):
        _restore(engine, q, v)
        transforms = engine.get_link_transforms()
        for marker_index, (frame_id, local_offset) in enumerate(
            zip(frame_ids, offsets, strict=True)
        ):
            transform = transforms[frame_id]
            positions[sample, marker_index] = (
                transform[:3, 3] + transform[:3, :3] @ local_offset
            )
    if not np.isfinite(positions).all():
        raise RuntimeError("Pinocchio native marker FK produced nonfinite positions")
    for array in (times, positions):
        array.setflags(write=False)
    return NativeMarkerPositionOutput(
        times,
        positions,
        labels,
        "world",
        bundle.input_history.timebase_id,
    )


def _step(
    engine: Any, motors: tuple[Any, ...], bundle: Any
) -> NativePinocchioTorqueReplay:
    times = np.array(bundle.input_history.time_seconds)
    inputs = np.array(bundle.input_history.values[:-1])
    positions, velocities = [engine.q.copy()], [engine.v.copy()]
    efforts = np.zeros((len(inputs), engine.model.nv))
    indices = [item[2] for item in motors]
    for row, saved in enumerate(inputs):
        efforts[row, indices] = saved
        engine.set_control(efforts[row])
        if not np.array_equal(engine.tau, efforts[row]):
            raise RuntimeError(
                "actual native generalized input differs from frozen input"
            )
        engine.step(bundle.policy.step_size_seconds, integrator="rk4")
        if not np.isclose(engine.get_time(), times[row + 1], atol=1e-12, rtol=0):
            raise RuntimeError("native stepping failed to reach required time")
        positions.append(engine.q.copy())
        velocities.append(engine.v.copy())
    q, v = np.asarray(positions), np.asarray(velocities)
    for array in (times, inputs, q, v, efforts):
        if not np.isfinite(array).all():
            raise RuntimeError("native replay produced nonfinite evidence")
        array.setflags(write=False)
    return NativePinocchioTorqueReplay(
        times, q, v, inputs, efforts, bundle.applied_input_sha256, bundle.policy_sha256
    )
