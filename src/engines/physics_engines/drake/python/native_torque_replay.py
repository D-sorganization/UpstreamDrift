"""Independent native Drake discrete-plant replay of frozen unit motor torque.

The owned plant admits collision-free URDFs and an explicit unsampled-output
policy. Complete discrete state is required; abstract/continuous state and
unsupported transmission declarations fail admission. This is not full-body
or contact qualification.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from defusedxml import ElementTree as SafeET
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sidekick.lab.mocap import ExperimentReplayBundle

_VERSION = "1.0.0"


@dataclass(frozen=True)
class NativeDrakeTorqueReplay:
    """Uninterrupted native evidence, without a scientific acceptance verdict."""

    time_seconds: NDArray[np.float64]
    qpos: NDArray[np.float64]
    qvel: NDArray[np.float64]
    discrete_states: NDArray[np.float64]
    applied_actuator_torques: NDArray[np.float64]
    generalized_actuator_torques: NDArray[np.float64]
    input_sha256: str
    policy_sha256: str


def _contracts() -> Any:
    from src.shared.python._seam_redirect import extend_sidekick_lab_path

    extend_sidekick_lab_path()
    from sidekick.lab import mocap

    if not hasattr(mocap, "ExperimentReplayBundle"):
        raise RuntimeError("native replay requires the merged Tools T01 contract")
    return mocap


def _admit_source(raw: bytes) -> None:
    """Reject source declarations Drake could ignore or resolve externally."""
    root = SafeET.fromstring(raw, forbid_dtd=True, forbid_entities=True)
    if root.tag != "robot":
        raise ValueError("native adapter requires a URDF robot")
    for element in root.iter():
        if "}" in element.tag or element.tag in {"collision", "gazebo", "mimic"}:
            raise ValueError(
                "extensions, collision and mimic need another native policy"
            )
        if "filename" in element.attrib:
            raise ValueError("external model resources need complete resource identity")
        if element.tag == "mechanicalReduction":
            reduction = float(element.text or "nan")
            if not np.isfinite(reduction) or reduction != 1:
                raise ValueError("nonunit transmission reduction is unsupported")
    for transmission in root.findall("transmission"):
        declared_type = transmission.get("type") or transmission.findtext("type")
        if declared_type not in {
            "SimpleTransmission",
            "transmission_interface/SimpleTransmission",
        }:
            raise ValueError("native torque requires a SimpleTransmission")
        if (
            len(transmission.findall("joint")) != 1
            or len(transmission.findall("actuator")) != 1
        ):
            raise ValueError("transmission must map one motor to one joint")


def _load_native(path: Path, timestep: float) -> tuple[Any, Any]:
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import MultibodyPlant

    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("native timestep must be finite and positive")
    raw = path.read_bytes()
    _admit_source(raw)
    plant = MultibodyPlant(timestep)
    plant.SetUseSampledOutputPorts(False)
    Parser(plant).AddModelsFromString(raw.decode("utf-8"), "urdf")
    if path.read_bytes() != raw:
        raise ValueError("native model source changed during loading")
    plant.Finalize()
    context = plant.CreateDefaultContext()
    if (
        context.num_continuous_states()
        or context.num_abstract_states()
        or context.num_discrete_state_groups() != 1
    ):
        raise ValueError("native policy requires complete numeric discrete state only")
    _abstract_parameter_identity(plant, context)
    actuators = _actuators(plant)
    if not actuators:
        raise ValueError("native model has no torque inputs")
    for actuator in actuators:
        if actuator.joint().type_name() != "revolute" or actuator.num_inputs() != 1:
            raise ValueError("native torque requires one-input revolute-joint motors")
        if not np.isfinite(actuator.effort_limit()) or actuator.effort_limit() <= 0:
            raise ValueError("native motor effort limits must be finite and positive")
    return plant, context


def _abstract_parameter_identity(plant: Any, context: Any) -> dict[str, Any]:
    """Admit only reviewed factory defaults, never caller-supplied parameters.

    Drake v1.57.0 plant DeclareParameters creates two internal constraint maps.
    Their Python values are unbound. GetConstraintIds must prove both empty;
    the owned default context and boolean prefix are recorded explicitly.
    Another native version/layout requires another reviewed policy.
    """
    if importlib.metadata.version("drake") != "1.57.0":
        raise ValueError("native abstract parameter policy requires Drake 1.57.0")
    if plant.GetConstraintIds() or plant.num_constraints():
        raise ValueError("constraint parameters require another native policy")
    size = context.num_abstract_parameters()
    if size != plant.num_joints() + 3:
        raise ValueError("unreviewed native abstract parameter layout")
    flags = [context.get_abstract_parameter(i).get_value() for i in range(size - 2)]
    if any(type(value) is not bool for value in flags):
        raise ValueError("unreviewed native abstract parameter type")
    return {
        "factory_default_boolean_parameters": flags,
        "constraint_active_status": [],
        "distance_constraint_parameters": [],
        "layout_source": "drake-v1.57.0-MultibodyPlant-DeclareParameters",
    }


def _actuators(plant: Any) -> tuple[Any, ...]:
    return tuple(
        sorted(
            (
                plant.get_joint_actuator(index)
                for index in plant.GetJointActuatorIndices()
            ),
            key=lambda item: item.input_start(),
        )
    )


def _restore_native(plant: Any, context: Any, values: NDArray[np.float64]) -> None:
    state = np.array(values, dtype=np.float64, copy=True)
    expected_size = context.get_discrete_state_vector().size()
    if state.shape != (expected_size,) or not np.isfinite(state).all():
        raise ValueError("complete finite native discrete state is required")
    context.get_mutable_discrete_state_vector().SetFromVector(state)
    q = plant.GetPositions(context)
    for index in plant.GetFloatingBaseBodies():
        body = plant.get_body(index)
        if body.has_quaternion_dofs():
            start = body.floating_positions_start()
            if not np.isclose(
                np.linalg.norm(q[start : start + 4]), 1, atol=1e-12, rtol=0
            ):
                raise ValueError("initial native quaternion must be normalized")
    if not np.array_equal(plant.GetPositionsAndVelocities(context), state):
        raise ValueError("native state layout needs another explicit policy")


def _identity(plant: Any, context: Any, path: Path, contracts: Any) -> Any:
    from pydrake.multibody import parsing, tree
    from pydrake.multibody import plant as plant_module
    from pydrake.systems import analysis

    provider = hashlib.sha256(Path(__file__).read_bytes())
    for module in (parsing, plant_module, tree, analysis):
        if module.__file__ is None:
            raise ValueError("native provider artifact is unavailable")
        provider.update(Path(module.__file__).read_bytes())
    parameters = [
        {
            "size": context.get_numeric_parameter(i).size(),
            "ieee754_sha256": hashlib.sha256(
                np.asarray(
                    context.get_numeric_parameter(i).get_value(), dtype="<f8"
                ).tobytes()
            ).hexdigest(),
        }
        for i in range(context.num_numeric_parameter_groups())
    ]
    loaded = {
        "topology": plant.GetTopologyGraphvizString(),
        "parameters": parameters,
        "abstract_parameter_policy": _abstract_parameter_identity(plant, context),
        "nq": plant.num_positions(),
        "nv": plant.num_velocities(),
        "time_step": plant.time_step(),
        "discrete_contact_solver": str(plant.get_discrete_contact_solver()),
        "discrete_contact_approximation": str(
            plant.get_discrete_contact_approximation()
        ),
        "actuators": [
            (a.name(), a.joint().name(), a.input_start(), a.effort_limit())
            for a in _actuators(plant)
        ],
    }
    loaded_hash = hashlib.sha256(
        json.dumps(loaded, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    specs = (
        contracts.StateComponentSpec(
            "qpos",
            contracts.StateComponentRole.POSITION,
            plant.num_positions(),
            "native-SI",
            "drake-configuration",
        ),
        contracts.StateComponentSpec(
            "qvel",
            contracts.StateComponentRole.VELOCITY,
            plant.num_velocities(),
            "native-SI",
            "drake-tangent-velocity",
        ),
        contracts.StateComponentSpec(
            "discrete",
            contracts.StateComponentRole.AUXILIARY,
            context.get_discrete_state_vector().size(),
            "native-SI",
            "drake-unsampled-complete-discrete",
        ),
    )
    return contracts.ModelIdentity(
        "drake",
        "native-model",
        "unit-hinge-motors-unsampled",
        _VERSION,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        "drake-native-torque-replay",
        importlib.metadata.version("drake"),
        provider.hexdigest(),
        contracts.InitialStateSchema("drake-native-discrete", _VERSION, specs),
        tuple(a.name() for a in _actuators(plant)),
        loaded_hash,
    )


def _policy(plant: Any, identity: Any, contracts: Any) -> Any:
    return contracts.ReplayExecutionPolicy(
        replay_mode=contracts.ReplayMode.NATIVE_OWN_CONTACT,
        solver_id=(
            "drake-discrete-"
            + str(plant.get_discrete_contact_solver())
            + "-"
            + str(plant.get_discrete_contact_approximation())
        ),
        solver_version=identity.provider_version,
        integration_method="native-discrete-AdvanceTo-preupdate-boundary",
        step_policy="fixed",
        step_size_seconds=float(plant.time_step()),
        initialization_policy_id="complete-discrete-restore-Initialize-once-unsampled",
        initialization_policy_version=_VERSION,
        input_player_id="native-fixed-port-zoh-terminal-sentinel",
        input_player_version=_VERSION,
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id="drake-model-no-collision-geometry",
        contact_policy_version=_VERSION,
        contact_policy_sha256=identity.loaded_native_model_sha256,
    )


def _history(
    plant: Any, times: NDArray[np.float64], values: NDArray[np.float64]
) -> None:
    if (
        times.ndim != 1
        or len(times) < 2
        or times[0] != 0
        or not np.isfinite(times).all()
        or not np.allclose(np.diff(times), plant.time_step(), atol=1e-12, rtol=0)
    ):
        raise ValueError("time grid must begin at zero and match native timestep")
    if (
        values.shape != (len(times), plant.num_actuated_dofs())
        or not np.isfinite(values).all()
    ):
        raise ValueError("finite torque rows must match ordered native motors")
    if not np.array_equal(values[-1], values[-2]):
        raise ValueError("terminal ZOH sentinel must equal the last executed input")
    limits = np.array([a.effort_limit() for a in _actuators(plant)])
    if np.any(np.abs(values) > limits):
        raise ValueError("saved torque must already satisfy native effort limits")


def build_native_drake_torque_bundle(
    model_path: str | Path,
    initial_discrete_state: NDArray[np.float64],
    time_seconds: NDArray[np.float64],
    applied_motor_torques: NDArray[np.float64],
    *,
    time_step: float = 0.001,
    experiment_id: str = "native-drake-replay",
) -> ExperimentReplayBundle:
    """Bind actual native state, source/provider, fixed port and stepping policy."""
    contracts = _contracts()
    path = Path(model_path)
    plant, context = _load_native(path, time_step)
    times = np.array(time_seconds, dtype=np.float64, copy=True)
    values = np.array(applied_motor_torques, dtype=np.float64, copy=True)
    _history(plant, times, values)
    _restore_native(plant, context, initial_discrete_state)
    identity = _identity(plant, context, path, contracts)
    channels = tuple(
        contracts.InputChannel(a.name(), a.name(), "N*m", a.joint().name())
        for a in _actuators(plant)
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
            ("qpos", tuple(plant.GetPositions(context))),
            ("qvel", tuple(plant.GetVelocities(context))),
            ("discrete", tuple(initial_discrete_state)),
        ),
        channels,
        contracts.ActuationInputKind.ACTUATOR_TORQUE,
        contracts.InputInterpolation.ZERO_ORDER_HOLD,
        tuple(times),
        tuple(tuple(row) for row in values),
        _policy(plant, identity, contracts),
    )


def replay_native_drake_torque_bundle(
    bundle: ExperimentReplayBundle, model_path: str | Path
) -> NativeDrakeTorqueReplay:
    """Execute frozen inputs in a fresh owned plant; no observations or resets."""
    contracts = _contracts()
    bundle = contracts.load_experiment_replay_bundle(
        contracts.dumps_experiment_replay_bundle(bundle)
    )
    if bundle.blocking_capabilities:
        raise ValueError("required native replay capabilities are unavailable")
    initial = {item.component_id: item.values for item in bundle.initial_state}
    if "discrete" not in initial or bundle.policy.step_size_seconds is None:
        raise ValueError("complete native discrete state and stepping policy required")
    expected = build_native_drake_torque_bundle(
        model_path,
        np.array(initial["discrete"]),
        np.array(bundle.input_history.time_seconds),
        np.array(bundle.input_history.values),
        time_step=bundle.policy.step_size_seconds,
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
            "native model, state, channel or executed policy identity differs"
        )
    if (
        bundle.input_history.input_kind != contracts.ActuationInputKind.ACTUATOR_TORQUE
        or bundle.input_history.interpolation
        != contracts.InputInterpolation.ZERO_ORDER_HOLD
    ):
        raise ValueError("native replay requires held actuator torque")
    return _step_native(bundle, Path(model_path))


def _step_native(bundle: Any, path: Path) -> NativeDrakeTorqueReplay:
    from pydrake.systems.analysis import Simulator

    plant, context = _load_native(path, bundle.policy.step_size_seconds)
    if _identity(plant, context, path, _contracts()) != bundle.model:
        raise ValueError("execution native model/provider identity differs")
    state = next(
        item.values for item in bundle.initial_state if item.component_id == "discrete"
    )
    _restore_native(plant, context, np.array(state))
    times = np.array(bundle.input_history.time_seconds)
    inputs = np.array(bundle.input_history.values[:-1])
    full = np.empty((len(times), len(state)))
    qs = np.empty((len(times), plant.num_positions()))
    vs = np.empty((len(times), plant.num_velocities()))
    generalized = np.empty((len(inputs), plant.num_velocities()))
    plant.get_actuation_input_port().FixValue(
        context, np.zeros(plant.num_actuated_dofs())
    )
    simulator = Simulator(plant, context)
    simulator.Initialize()
    if not np.array_equal(context.get_discrete_state_vector().CopyToVector(), state):
        raise RuntimeError("native initialization changed exported state")
    for row in range(len(times)):
        full[row] = context.get_discrete_state_vector().CopyToVector()
        qs[row], vs[row] = plant.GetPositions(context), plant.GetVelocities(context)
        if row == len(inputs):
            break
        plant.get_actuation_input_port().FixValue(context, inputs[row])
        actual = np.asarray(plant.get_net_actuation_output_port().Eval(context))
        if not np.allclose(actual, inputs[row], atol=1e-12, rtol=0):
            raise RuntimeError("actual native motor effort differs from saved torque")
        generalized[row] = plant.MakeActuationMatrix() @ actual
        simulator.AdvanceTo(float(times[row + 1]))
        if not np.isclose(context.get_time(), times[row + 1], atol=1e-12, rtol=0):
            raise RuntimeError("native simulation failed to reach the boundary")
    for array in (times, qs, vs, full, inputs, generalized):
        if not np.isfinite(array).all():
            raise RuntimeError("native replay produced nonfinite evidence")
        array.setflags(write=False)
    return NativeDrakeTorqueReplay(
        times,
        qs,
        vs,
        full,
        inputs,
        generalized,
        bundle.applied_input_sha256,
        bundle.policy_sha256,
    )
