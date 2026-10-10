"""Versioned constrained-muscle cold start and exact T01 excitation replay.

This narrow policy admits native couplers/locks and simple physical muscle
paths. It is not a contact, wrapped-path, anatomy or physiology certificate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.engines.native_replay_contracts import (
    native_replay_contract_types,
    validate_native_replay_bundle,
)

from . import muscle_replay, native_muscle_bundle
from .native_constraint_state import _restore_named_state, owned_native_source_state
from .native_prepared_state import (
    DeclaredColdStart,
    NativeConstraintStateAudit,
    _admission_identity,
    _apply_declared_locks,
    observe_declared_native_sample,
)

if TYPE_CHECKING:
    from sidekick.lab.mocap import ExperimentReplayBundle

_VERSION = "2.0.0"
_ACCURACY = 1e-8
_ALLOWED_COMPONENTS = native_muscle_bundle._COMPONENTS | {
    "CoordinateCouplerConstraint",
    "CustomJoint",
    "LinearFunction",
    "MovingPathPoint",
}


@dataclass(frozen=True)
class ConstrainedMuscleReplay:
    """Sampled native result, with every declared constraint observed."""

    state_names: tuple[str, ...]
    muscle_names: tuple[str, ...]
    times: NDArray[np.float64]
    states: NDArray[np.float64]
    applied_excitations: NDArray[np.float64]
    muscle_forces_n: NDArray[np.float64]
    constraint_audits: tuple[NativeConstraintStateAudit, ...]
    input_sha256: str


def _audit_moving_path_points(
    components: tuple[Any, ...], declaration: DeclaredColdStart, osim: Any
) -> None:
    for component in components:
        if component.getConcreteClassName() != "MovingPathPoint":
            continue
        point = osim.MovingPathPoint.safeDownCast(component)
        if point is None:
            raise ValueError("moving path point needs its native concrete class")
        for coordinate, function in (
            (point.getXCoordinate(), point.get_x_location()),
            (point.getYCoordinate(), point.get_y_location()),
            (point.getZCoordinate(), point.get_z_location()),
        ):
            if (
                coordinate is None
                or coordinate.getAbsolutePathString() not in declaration.chart_bounds
            ):
                raise ValueError("moving path coordinate needs a declared chart")
            if (linear := osim.LinearFunction.safeDownCast(function)) is not None:
                values = (
                    linear.getCoefficients().get(0),
                    linear.getCoefficients().get(1),
                )
                if not np.isfinite(values).all():
                    raise ValueError("moving path law must be finite")
            elif (
                constant := osim.Constant.safeDownCast(function)
            ) is None or not np.isfinite(constant.getValue()):
                raise ValueError(
                    "moving path law is outside the reviewed linear profile"
                )


def _audit_source_components(
    model: Any, muscles: Any, declaration: DeclaredColdStart
) -> None:
    import opensim as osim

    components = tuple(model.getComponentsList())
    for component in components:
        kind = component.getConcreteClassName()
        if kind not in _ALLOWED_COMPONENTS:
            raise ValueError(f"unreviewed constrained muscle component: {kind}")
        if osim.Controller.safeDownCast(component) is not None:
            raise ValueError("source controller is forbidden in excitation replay")
        if osim.PositionMotion.safeDownCast(component) is not None:
            raise ValueError("prescribed motion is forbidden in excitation replay")
        if (
            osim.Force.safeDownCast(component) is not None
            and osim.Muscle.safeDownCast(component) is None
        ):
            raise ValueError("non-muscle force is forbidden in this policy")
    registered = {
        muscles.get(index).getAbsolutePathString() for index in range(muscles.getSize())
    }
    recursive = {
        component.getAbsolutePathString()
        for component in components
        if osim.Muscle.safeDownCast(component) is not None
    }
    if registered != recursive or not registered:
        raise ValueError("native muscle registry must cover every recursive muscle")
    _audit_moving_path_points(components, declaration, osim)
    constraints = model.getConstraintSet()
    for index in range(constraints.getSize()):
        constraint = constraints.get(index)
        if constraint.getConcreteClassName() != "CoordinateCouplerConstraint":
            raise ValueError("only coordinate couplers have this constraint policy")
    joints = model.getJointSet()
    for index in range(joints.getSize()):
        joint = joints.get(index)
        if joint.getConcreteClassName() != "CustomJoint":
            continue
        custom = osim.CustomJoint.safeDownCast(joint)
        if custom is None or custom.numCoordinates() != 1:
            raise ValueError("CustomJoint needs the reviewed one-coordinate chart")
        transform = custom.getSpatialTransform()
        coordinate = custom.getCoordinate(0).getName()
        for axis_index in range(6):
            axis = transform.getTransformAxis(axis_index)
            names = axis.getCoordinateNames()
            function = axis.getFunction()
            if axis_index == 0:
                linear = osim.LinearFunction.safeDownCast(function)
                if (
                    names.size() != 1
                    or names.getValue(0) != coordinate
                    or linear is None
                    or linear.getCoefficients().get(0) != 1.0
                    or linear.getCoefficients().get(1) != 0.0
                ):
                    raise ValueError("CustomJoint rotation chart is not unit linear")
            elif (
                names.size() != 0
                or (constant := osim.Constant.safeDownCast(function)) is None
                or constant.getValue() != 0.0
            ):
                raise ValueError("CustomJoint other transform axes must be fixed zero")


def _owned_prepared(
    declaration: DeclaredColdStart,
    grid: NDArray[np.float64],
    controls: Mapping[str, NDArray[np.float64]],
) -> Any:
    """Yield a fresh model with one time-only player and explicit lock recipe."""
    from contextlib import contextmanager

    @contextmanager
    def prepared() -> Any:
        import opensim as osim

        path = declaration.model_path
        raw = path.read_bytes()
        native_muscle_bundle._validate_self_contained_source(raw)
        if hashlib.sha256(raw).hexdigest() != declaration.source_sha256:
            raise ValueError("native constrained muscle source identity differs")

        def before_initialize(model: Any) -> None:
            model.finalizeConnections()
            _audit_source_components(model, model.getMuscles(), declaration)
            muscles = model.getMuscles()
            names = tuple(muscles.get(i).getName() for i in range(muscles.getSize()))
            if len(set(names)) != len(names) or set(names) != set(controls):
                raise ValueError("excitation channels must exactly cover named muscles")
            if any(
                model.getCoordinateSet().get(i).getDefaultIsPrescribed()
                for i in range(model.getCoordinateSet().getSize())
            ):
                raise ValueError("prescribed coordinate is forbidden")
            muscle_replay._configure_input_player(model, muscles, names, grid, controls)
            model.finalizeConnections()

        with owned_native_source_state(path, before_initialize=before_initialize) as (
            model,
            state,
            source_sha,
        ):
            if source_sha != declaration.source_sha256:
                raise ValueError("source changed during native load")
            _apply_declared_locks(model, state, declaration.lock_targets)
            _restore_named_state(model, state, declaration.named_state)
            state.setTime(declaration.time_seconds)
            audit = observe_declared_native_sample(model, state, declaration)
            muscle_replay._validate_muscle_state(
                declaration.named_state,
                muscle_replay._muscle_state_domains(model, osim),
            )
            yield model, state, audit

    return prepared()


def _bundle_for(
    declaration: DeclaredColdStart,
    grid: NDArray[np.float64],
    controls: Mapping[str, NDArray[np.float64]],
    experiment_id: str,
) -> ExperimentReplayBundle:
    contracts = native_replay_contract_types()
    with _owned_prepared(declaration, grid, controls) as (model, state, _audit):
        muscles = model.getMuscles()
        names = tuple(muscles.get(i).getName() for i in range(muscles.getSize()))
        state_names = tuple(declaration.named_state)
        options = native_muscle_bundle._registered_options(model, state, muscles)
        identity = native_muscle_bundle._identity(
            declaration.model_path, model, state_names, names, options, contracts
        )
        adapter_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        provider_sha = hashlib.sha256(
            bytes.fromhex(identity.provider_sha256) + bytes.fromhex(adapter_sha)
        ).hexdigest()
        identity = replace(
            identity,
            variant_id="declared-constrained-muscles",
            model_version=_VERSION,
            provider_sha256=provider_sha,
        )
        policy = native_muscle_bundle._policy(identity, contracts)
        policy = replace(
            policy,
            initialization_policy_id="declared-coupler-lock-muscle-cold-start",
            initialization_policy_version=_VERSION,
            contact_policy_id="native-coupler-lock-no-contact",
            contact_policy_version=_VERSION,
            contact_policy_sha256=_admission_identity(declaration, adapter_sha),
        )
        values = tuple(
            (name, (declaration.named_state[name],)) for name in state_names
        ) + tuple(options.items())
        capability = contracts.CapabilityDeclaration(
            "declared-native-constrained-muscle-cold-start",
            True,
            contracts.CapabilitySupport.SUPPORTED,
            contracts.CapabilityAvailability.AVAILABLE,
        )
        return contracts.build_experiment_replay_bundle(
            experiment_id,
            identity,
            (capability,),
            values,
            tuple(contracts.InputChannel(name, name, "1") for name in names),
            contracts.ActuationInputKind.MUSCLE_EXCITATION,
            contracts.InputInterpolation.LINEAR,
            tuple(grid),
            tuple(tuple(controls[name][i] for name in names) for i in range(len(grid))),
            policy,
        )


def build_constrained_muscle_bundle(
    declaration: DeclaredColdStart,
    times: NDArray[np.float64],
    excitations: Mapping[str, NDArray[np.float64]],
    *,
    experiment_id: str = "native-constrained-muscle-replay",
) -> ExperimentReplayBundle:
    """Bind native source, complete initial muscle state and declared constraints."""
    if (
        declaration.constant_commands
        or declaration.allow_source_controllers_for_observation
    ):
        raise ValueError("constrained muscle policy cannot mix mechanical commands")
    if declaration.source_controller_replacement or declaration.scheduled_command_step:
        raise ValueError(
            "constrained muscle policy forbids source-controller replacement"
        )
    grid, controls, _ = muscle_replay._validated_inputs(
        times, excitations, declaration.named_state, _ACCURACY
    )
    if grid[0] != declaration.time_seconds or grid[0] != 0.0:
        raise ValueError("T01 constrained muscle clock must start at zero")
    return _bundle_for(declaration, grid, controls, experiment_id)


def replay_constrained_muscle_bundle(
    bundle: ExperimentReplayBundle, declaration: DeclaredColdStart
) -> ConstrainedMuscleReplay:
    """Independently reconstruct and replay the same frozen excitation policy."""
    import opensim as osim

    contracts = native_replay_contract_types()
    bundle = validate_native_replay_bundle(bundle, contracts)
    history = bundle.input_history
    grid = np.asarray(history.time_seconds, dtype=np.float64)
    controls = {
        channel.channel_id: np.asarray(
            [row[i] for row in history.values], dtype=np.float64
        )
        for i, channel in enumerate(history.channels)
    }
    raw = declaration.model_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != bundle.model.source_model_sha256:
        raise ValueError("native constrained muscle source identity differs")
    with TemporaryDirectory(prefix="opensim-constrained-muscle-") as directory:
        snapshot = Path(directory) / "frozen.osim"
        snapshot.write_bytes(raw)
        frozen = replace(declaration, model_path=snapshot)
        expected = build_constrained_muscle_bundle(
            frozen, grid, controls, experiment_id=bundle.experiment_id
        )
        for field in (
            "model",
            "initial_state",
            "policy",
            "input_history",
            "capabilities",
        ):
            if getattr(bundle, field) != getattr(expected, field):
                raise ValueError(
                    "native constrained muscle policy or input identity differs"
                )
        with _owned_prepared(frozen, grid, controls) as (model, state, initial_audit):
            muscles = model.getMuscles()
            names = tuple(muscles.get(i).getName() for i in range(muscles.getSize()))
            state_names = tuple(declaration.named_state)
            domains = muscle_replay._muscle_state_domains(model, osim)
            manager = osim.Manager(model)
            manager.setIntegratorMethod(osim.Manager.IntegratorMethod_RungeKuttaMerson)
            manager.setIntegratorAccuracy(_ACCURACY)
            manager.setWriteToStorage(False)
            manager.initialize(state)
            states = np.empty((len(grid), len(state_names)))
            forces = np.empty((len(grid), len(names)))
            applied = np.empty_like(forces)
            audits = []
            for index, time in enumerate(grid):
                state = state if index == 0 else manager.integrate(float(time))
                if state.getTime() != time:
                    raise RuntimeError("native integration missed exact replay clock")
                model.realizeDynamics(state)
                audits.append(observe_declared_native_sample(model, state, frozen))
                model.realizeDynamics(state)
                states[index] = [
                    model.getStateVariableValue(state, name) for name in state_names
                ]
                muscle_replay._validate_muscle_state(
                    dict(zip(state_names, states[index], strict=True)), domains
                )
                forces[index] = [
                    muscles.get(i).getActuation(state) for i in range(len(names))
                ]
                applied[index] = [
                    muscles.get(i).getExcitation(state) for i in range(len(names))
                ]
            if audits[0] != initial_audit:
                raise RuntimeError(
                    "native prepared state changed before first replay sample"
                )
            expected_input = np.column_stack([controls[name] for name in names])
            if not np.allclose(applied, expected_input, rtol=0, atol=1e-12):
                raise RuntimeError("native applied excitation differs from saved input")
    for array in (grid, states, forces, applied):
        array.setflags(write=False)
    return ConstrainedMuscleReplay(
        state_names,
        names,
        grid,
        states,
        applied,
        forces,
        tuple(audits),
        bundle.applied_input_sha256,
    )


def observe_constrained_markers(
    declaration: DeclaredColdStart,
    replay: ConstrainedMuscleReplay,
    bindings: Mapping[str, tuple[str, tuple[float, float, float]]],
    indices: NDArray[np.intp],
) -> tuple[NDArray[np.float64], str]:
    """Sample declared source frames on replayed states without native assembly."""
    import opensim as osim
    from .native_prepared_state import reconstruct_declared_cold_start

    if set(replay.state_names) != set(declaration.named_state) or not bindings:
        raise ValueError(
            "complete replay state and explicit marker placements required"
        )
    if np.any(indices < 0) or np.any(indices >= len(replay.times)):
        raise ValueError("marker sample index is outside frozen replay clock")
    with reconstruct_declared_cold_start(declaration) as prepared:
        model, state = prepared.model, prepared.state
        frames = []
        for label, (path, offset) in bindings.items():
            if (
                not label
                or not path.startswith("/")
                or len(offset) != 3
                or not np.isfinite(offset).all()
                or not model.hasComponent(path)
            ):
                raise ValueError("marker placement needs a native frame and 3D offset")
            frame = osim.PhysicalFrame.safeDownCast(model.getComponent(path))
            if frame is None:
                raise ValueError("marker placement is not attached to a physical frame")
            frames.append((frame, offset))
        points = np.empty((len(indices), len(frames), 3), dtype=float)
        for row, index in enumerate(indices):
            _restore_named_state(
                model,
                state,
                dict(zip(replay.state_names, replay.states[index], strict=True)),
            )
            state.setTime(float(replay.times[index]))
            observe_declared_native_sample(model, state, declaration)
            model.realizePosition(state)
            for column, (frame, offset) in enumerate(frames):
                point = frame.findStationLocationInGround(state, osim.Vec3(*offset))
                points[row, column] = [point.get(axis) for axis in range(3)]
    if not np.isfinite(points).all():
        raise RuntimeError("native constrained marker geometry is nonfinite")
    identity = {
        "source_sha256": declaration.source_sha256,
        "replay_input_sha256": replay.input_sha256,
        "provider_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bindings": tuple(
            (name, path, offset) for name, (path, offset) in bindings.items()
        ),
        "times": tuple(float(replay.times[index]) for index in indices),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    points.setflags(write=False)
    return points, digest
