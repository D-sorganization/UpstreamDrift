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
from typing import TYPE_CHECKING, Any, Protocol

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
from .native_wrapped_path import (
    WrappedPathDeclaration,
    validate_wrapped_path_declaration,
)
from .native_mechanics_profile import (
    NativeMechanicsExecutionProfile,
    validate_native_mechanics_execution_profile,
)
from .native_contact_policy import (
    NativeContactExecutionDeclaration,
    native_contact_provider_sha256,
    validate_native_contact_execution,
)

if TYPE_CHECKING:
    from sidekick.lab.mocap import ExperimentReplayBundle

_VERSION = "2.1.0"
_EXACT_557_POLICY_ID = "exact-557-coupler-muscle-cold-start"
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
    contact_wrenches: tuple[tuple[Any, ...], ...] = ()


class NativeNamedStateReplay(Protocol):
    """Minimal named-trajectory interface required for marker FK."""

    @property
    def state_names(self) -> tuple[str, ...]: ...

    @property
    def times(self) -> NDArray[np.float64]: ...

    @property
    def states(self) -> NDArray[np.float64]: ...

    @property
    def input_sha256(self) -> str: ...


def _reject_exact_557_mixing(exact_557: bool, *declarations: object) -> None:
    """Keep the exact-source diagnostic free of any other admission surface."""
    if exact_557 and any(item is not None for item in declarations):
        raise ValueError("exact 557 diagnostic cannot combine other declarations")


def _audit_moving_path_points(
    components: tuple[Any, ...],
    declaration: DeclaredColdStart,
    osim: Any,
    native_mechanics_profile: NativeMechanicsExecutionProfile | None = None,
) -> None:
    delegated = (
        set()
        if native_mechanics_profile is None
        else {item.absolute_path for item in native_mechanics_profile.components}
    )
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
            if component.getAbsolutePathString() in delegated:
                if osim.Function.safeDownCast(function) is None:
                    raise ValueError(
                        "delegated moving path law is not an OpenSim function"
                    )
                continue
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
    model: Any,
    muscles: Any,
    declaration: DeclaredColdStart,
    mechanical_paths: frozenset[str] = frozenset(),
    wrapped_path_declaration: WrappedPathDeclaration | None = None,
    native_mechanics_profile: NativeMechanicsExecutionProfile | None = None,
    contact_declaration: NativeContactExecutionDeclaration | None = None,
) -> None:
    import opensim as osim

    if wrapped_path_declaration is not None:
        validate_wrapped_path_declaration(
            model, wrapped_path_declaration, declaration.source_sha256
        )
    if native_mechanics_profile is not None:
        validate_native_mechanics_execution_profile(
            model, native_mechanics_profile, declaration.source_sha256
        )
    if contact_declaration is not None:
        validate_native_contact_execution(
            model, contact_declaration, declaration.source_sha256
        )
    components = tuple(model.getComponentsList())
    _audit_component_policy(
        components,
        mechanical_paths,
        wrapped_path_declaration is not None,
        contact_declaration,
        osim,
    )
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
    _audit_moving_path_points(components, declaration, osim, native_mechanics_profile)
    _audit_source_constraints(model)
    delegated_paths = (
        set()
        if native_mechanics_profile is None
        else {item.absolute_path for item in native_mechanics_profile.components}
    )
    _audit_custom_joint_charts(model, delegated_paths, osim)


def _audit_component_policy(
    components: tuple[Any, ...],
    mechanical_paths: frozenset[str],
    wrapped_declared: bool,
    contact_declaration: NativeContactExecutionDeclaration | None,
    osim: Any,
) -> None:
    """Reject source components outside the declared replay ownership policy."""
    declared_contact_paths = (
        frozenset()
        if contact_declaration is None
        else frozenset(contact_declaration.force_paths)
    )
    wrapped_components = {
        "WrapCylinder",
        "WrapEllipsoid",
        "WrapSphere",
        "PathWrap",
        "PathWrapPoint",
        "ConditionalPathPoint",
    }
    for component in components:
        kind = component.getConcreteClassName()
        structural_weld = kind == "WeldJoint"
        if structural_weld:
            _audit_structural_weld_joint(component, osim)
        declared_mechanical = (
            kind == "CoordinateActuator"
            and component.getAbsolutePathString() in mechanical_paths
        )
        declared_wrap_component = wrapped_declared and kind in wrapped_components
        declared_contact_force = (
            component.getAbsolutePathString() in declared_contact_paths
        )
        declared_contact_geometry = contact_declaration is not None and kind in {
            "ContactSphere",
            "ContactHalfSpace",
        }
        if (
            kind not in _ALLOWED_COMPONENTS
            and not structural_weld
            and not declared_mechanical
            and not declared_wrap_component
            and not declared_contact_force
            and not declared_contact_geometry
        ):
            raise ValueError(f"unreviewed constrained muscle component: {kind}")
        if osim.Controller.safeDownCast(component) is not None:
            raise ValueError("source controller is forbidden in excitation replay")
        if osim.PositionMotion.safeDownCast(component) is not None:
            raise ValueError("prescribed motion is forbidden in excitation replay")
        if (
            osim.Force.safeDownCast(component) is not None
            and osim.Muscle.safeDownCast(component) is None
            and not declared_mechanical
            and not declared_contact_force
        ):
            raise ValueError("non-muscle force is forbidden in this policy")


def _audit_structural_weld_joint(component: Any, osim: Any) -> None:
    """Admit only native, zero-DOF welds joining two named physical frames."""
    joint = osim.WeldJoint.safeDownCast(component)
    if joint is None:
        raise ValueError("structural WeldJoint needs its native concrete class")
    if joint.numCoordinates() != 0:
        raise ValueError("structural WeldJoint must have zero coordinates")

    parent = osim.PhysicalFrame.safeDownCast(joint.getParentFrame())
    child = osim.PhysicalFrame.safeDownCast(joint.getChildFrame())
    if parent is None:
        raise ValueError("structural WeldJoint needs a valid parent frame")
    if child is None:
        raise ValueError("structural WeldJoint needs a valid child frame")
    allowed_frame_types = {"Body", "Ground", "PhysicalOffsetFrame"}
    if parent.getConcreteClassName() not in allowed_frame_types:
        raise ValueError("structural WeldJoint parent frame type is not reviewed")
    if child.getConcreteClassName() not in allowed_frame_types:
        raise ValueError("structural WeldJoint child frame type is not reviewed")
    parent_path = parent.getAbsolutePathString()
    child_path = child.getAbsolutePathString()
    if (
        not isinstance(parent_path, str)
        or not parent_path.startswith("/")
        or "/../" in parent_path
    ):
        raise ValueError("structural WeldJoint needs a valid parent frame path")
    if (
        not isinstance(child_path, str)
        or not child_path.startswith("/")
        or "/../" in child_path
    ):
        raise ValueError("structural WeldJoint needs a valid child frame path")
    if parent_path == child_path:
        raise ValueError("structural WeldJoint needs distinct frames")


def _audit_source_constraints(model: Any) -> None:
    """Limit this cold-start constraint policy to native coordinate couplers."""
    constraints = model.getConstraintSet()
    for index in range(constraints.getSize()):
        constraint = constraints.get(index)
        if constraint.getConcreteClassName() != "CoordinateCouplerConstraint":
            raise ValueError("only coordinate couplers have this constraint policy")


def _audit_custom_joint_charts(
    model: Any, delegated_paths: set[str], osim: Any
) -> None:
    """Validate every non-delegated CustomJoint as the reviewed scalar chart."""
    joints = model.getJointSet()
    for index in range(joints.getSize()):
        joint = joints.get(index)
        if joint.getConcreteClassName() != "CustomJoint":
            continue
        custom = osim.CustomJoint.safeDownCast(joint)
        if joint.getAbsolutePathString() in delegated_paths:
            continue
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
    wrapped_path_declaration: WrappedPathDeclaration | None = None,
    native_mechanics_profile: NativeMechanicsExecutionProfile | None = None,
    contact_declaration: NativeContactExecutionDeclaration | None = None,
    *,
    exact_557: bool = False,
) -> Any:
    """Yield a fresh model with one time-only player and explicit lock recipe."""
    _reject_exact_557_mixing(
        exact_557,
        wrapped_path_declaration,
        native_mechanics_profile,
        contact_declaration,
    )
    from contextlib import contextmanager

    @contextmanager
    def prepared() -> Any:
        import opensim as osim

        path = declaration.model_path
        raw = path.read_bytes()
        if exact_557:
            from .native_exact_557_profile import validate_exact_source_bytes

            validate_exact_source_bytes(raw)
        else:
            native_muscle_bundle._validate_self_contained_source(raw)
        if hashlib.sha256(raw).hexdigest() != declaration.source_sha256:
            raise ValueError("native constrained muscle source identity differs")

        def before_initialize(model: Any) -> None:
            model.finalizeConnections()
            if exact_557:
                from .native_exact_557_profile import validate_exact_loaded_model

                validate_exact_loaded_model(model, declaration)
            else:
                _audit_source_components(
                    model,
                    model.getMuscles(),
                    declaration,
                    wrapped_path_declaration=wrapped_path_declaration,
                    native_mechanics_profile=native_mechanics_profile,
                    contact_declaration=contact_declaration,
                )
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
    wrapped_path_declaration: WrappedPathDeclaration | None = None,
    native_mechanics_profile: NativeMechanicsExecutionProfile | None = None,
    contact_declaration: NativeContactExecutionDeclaration | None = None,
    *,
    exact_557: bool = False,
) -> ExperimentReplayBundle:
    contracts = native_replay_contract_types()
    with _owned_prepared(
        declaration,
        grid,
        controls,
        wrapped_path_declaration,
        native_mechanics_profile,
        contact_declaration,
        exact_557=exact_557,
    ) as (
        model,
        state,
        _audit,
    ):
        muscles = model.getMuscles()
        names = tuple(muscles.get(i).getName() for i in range(muscles.getSize()))
        state_names = tuple(declaration.named_state)
        options = native_muscle_bundle._registered_options(model, state, muscles)
        identity, adapter_sha = _constrained_bundle_identity(
            native_muscle_bundle.BundleSubject(
                declaration.model_path, model, state_names, names, options
            ),
            contracts,
            native_mechanics_profile,
            exact_557=exact_557,
        )
        policy = _constrained_bundle_policy(
            declaration,
            identity,
            adapter_sha,
            wrapped_path_declaration,
            native_mechanics_profile,
            contact_declaration,
            contracts,
            exact_557=exact_557,
        )
        values, capability, channels = _constrained_bundle_channels(
            declaration, state_names, options, names, contracts
        )
        return contracts.build_experiment_replay_bundle(
            experiment_id,
            identity,
            (capability,),
            values,
            channels,
            contracts.ActuationInputKind.MUSCLE_EXCITATION,
            contracts.InputInterpolation.LINEAR,
            tuple(grid),
            tuple(tuple(controls[name][i] for name in names) for i in range(len(grid))),
            policy,
        )


def _constrained_bundle_identity(
    subject: native_muscle_bundle.BundleSubject,
    contracts: Any,
    mechanics_profile: NativeMechanicsExecutionProfile | None,
    *,
    exact_557: bool = False,
) -> tuple[Any, str]:
    coupled_rotation_paths: frozenset[str] = frozenset()
    variant, version, profile_material = "declared-constrained-muscles", _VERSION, b""
    if exact_557:
        from . import native_exact_557_profile as exact

        coupled_rotation_paths = exact.COUPLED_ROTATION_PATHS
        variant, version = exact.VARIANT_ID, exact.PROFILE_VERSION
        profile_material = bytes.fromhex(exact.profile_source_sha256())
    identity = native_muscle_bundle._identity(
        subject,
        contracts,
        mechanics_profile,
        coupled_rotation_paths=coupled_rotation_paths,
    )
    adapter_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provider_sources = (
        Path(__file__).with_name("native_prepared_state.py"),
        Path(__file__).with_name("native_constraint_state.py"),
        Path(__file__).with_name("native_mechanics_profile.py"),
        Path(__file__).with_name("native_muscle_bundle.py"),
        Path(__file__).with_name("native_contact_policy.py"),
        Path(__file__).parents[1] / "opensim_force_torque.py",
    )
    provider_sha = hashlib.sha256(
        bytes.fromhex(identity.provider_sha256)
        + bytes.fromhex(adapter_sha)
        + profile_material
        + b"".join(path.read_bytes() for path in provider_sources)
    ).hexdigest()
    return (
        replace(
            identity,
            variant_id=variant,
            model_version=version,
            provider_sha256=provider_sha,
        ),
        adapter_sha,
    )


def _constrained_bundle_policy(
    declaration: DeclaredColdStart,
    identity: Any,
    adapter_sha: str,
    wrapped_path_declaration: WrappedPathDeclaration | None,
    mechanics_profile: NativeMechanicsExecutionProfile | None,
    contact_declaration: NativeContactExecutionDeclaration | None,
    contracts: Any,
    *,
    exact_557: bool = False,
) -> Any:
    policy = native_muscle_bundle._policy(identity, contracts)
    version = identity.model_version
    if exact_557:
        from .native_exact_557_profile import profile_source_sha256

        adapter_sha = hashlib.sha256(
            bytes.fromhex(adapter_sha) + bytes.fromhex(profile_source_sha256())
        ).hexdigest()
    return replace(
        policy,
        initialization_policy_id=(
            _EXACT_557_POLICY_ID
            if exact_557
            else "declared-coupler-lock-muscle-cold-start"
        ),
        initialization_policy_version=version,
        contact_policy_id=(
            "native-declared-sphere-halfspace-contact"
            if contact_declaration is not None
            else "native-coupler-lock-no-contact"
        ),
        contact_policy_version=version,
        contact_policy_sha256=_admission_identity(
            declaration,
            adapter_sha,
            wrapped_path_declaration.declaration_sha256
            if wrapped_path_declaration is not None
            else None,
            mechanics_profile.profile_sha256 if mechanics_profile is not None else None,
            contact_declaration.declaration_sha256
            if contact_declaration is not None
            else None,
            native_contact_provider_sha256()
            if contact_declaration is not None
            else None,
        ),
    )


def _constrained_bundle_channels(
    declaration: DeclaredColdStart,
    state_names: tuple[str, ...],
    options: Mapping[str, Any],
    muscle_names: tuple[str, ...],
    contracts: Any,
) -> tuple[tuple[tuple[str, tuple[float, ...]], ...], Any, tuple[Any, ...]]:
    values = tuple(
        (name, (declaration.named_state[name],)) for name in state_names
    ) + tuple(options.items())
    capability = contracts.CapabilityDeclaration(
        "declared-native-constrained-muscle-cold-start",
        True,
        contracts.CapabilitySupport.SUPPORTED,
        contracts.CapabilityAvailability.AVAILABLE,
    )
    channels = tuple(contracts.InputChannel(name, name, "1") for name in muscle_names)
    return values, capability, channels


def build_constrained_muscle_bundle(
    declaration: DeclaredColdStart,
    times: NDArray[np.float64],
    excitations: Mapping[str, NDArray[np.float64]],
    *,
    experiment_id: str = "native-constrained-muscle-replay",
    wrapped_path_declaration: WrappedPathDeclaration | None = None,
    native_mechanics_profile: NativeMechanicsExecutionProfile | None = None,
    contact_declaration: NativeContactExecutionDeclaration | None = None,
    exact_557: bool = False,
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
    return _bundle_for(
        declaration,
        grid,
        controls,
        experiment_id,
        wrapped_path_declaration,
        native_mechanics_profile,
        contact_declaration,
        exact_557=exact_557,
    )


def replay_constrained_muscle_bundle(
    bundle: ExperimentReplayBundle,
    declaration: DeclaredColdStart,
    *,
    wrapped_path_declaration: WrappedPathDeclaration | None = None,
    native_mechanics_profile: NativeMechanicsExecutionProfile | None = None,
    contact_declaration: NativeContactExecutionDeclaration | None = None,
    exact_557: bool = False,
) -> ConstrainedMuscleReplay:
    """Independently reconstruct and replay the same frozen excitation policy."""
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
            frozen,
            grid,
            controls,
            experiment_id=bundle.experiment_id,
            wrapped_path_declaration=wrapped_path_declaration,
            native_mechanics_profile=native_mechanics_profile,
            contact_declaration=contact_declaration,
            exact_557=exact_557,
        )
        _require_matching_constrained_bundle(bundle, expected)
        return _replay_constrained_snapshot(
            bundle,
            frozen,
            grid,
            controls,
            wrapped_path_declaration,
            native_mechanics_profile,
            contact_declaration,
            exact_557=exact_557,
        )


def _require_matching_constrained_bundle(
    bundle: ExperimentReplayBundle, expected: ExperimentReplayBundle
) -> None:
    for field in ("model", "initial_state", "policy", "input_history", "capabilities"):
        if getattr(bundle, field) != getattr(expected, field):
            raise ValueError(
                "native constrained muscle policy or input identity differs"
            )


def _replay_constrained_snapshot(
    bundle: ExperimentReplayBundle,
    declaration: DeclaredColdStart,
    grid: NDArray[np.float64],
    controls: Mapping[str, NDArray[np.float64]],
    wrapped_path_declaration: WrappedPathDeclaration | None,
    native_mechanics_profile: NativeMechanicsExecutionProfile | None,
    contact_declaration: NativeContactExecutionDeclaration | None,
    *,
    exact_557: bool = False,
) -> ConstrainedMuscleReplay:
    """Integrate and detach numeric replay output before native teardown."""
    import opensim as osim
    from .native_scalar_replay import (
        NativeScalarReplayPolicy,
        integrate_native_scalar_replay,
    )

    with _owned_prepared(
        declaration,
        grid,
        controls,
        wrapped_path_declaration,
        native_mechanics_profile,
        contact_declaration,
        exact_557=exact_557,
    ) as (model, state, initial_audit):
        muscles = model.getMuscles()
        names = tuple(muscles.get(i).getName() for i in range(muscles.getSize()))
        state_names = tuple(declaration.named_state)
        paths = tuple(
            muscles.get(i).getAbsolutePathString() for i in range(muscles.getSize())
        )
        samples = integrate_native_scalar_replay(
            model,
            state,
            state_names,
            paths,
            grid,
            muscle_replay._muscle_state_domains(model, osim),
            NativeScalarReplayPolicy(
                _ACCURACY,
                contact_force_paths=(
                    ()
                    if contact_declaration is None
                    else contact_declaration.force_paths
                ),
                constrained_cold_start=declaration,
            ),
        )
        states, forces, applied = (
            samples.states,
            samples.actuations,
            samples.applied_controls,
        )
        audits = samples.state_observations
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
        tuple(declaration.named_state),
        names,
        grid,
        states,
        applied,
        forces,
        tuple(audits),
        bundle.applied_input_sha256,
        samples.contact_wrenches,
    )


def observe_constrained_markers(
    declaration: DeclaredColdStart,
    replay: NativeNamedStateReplay,
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
        points: NDArray[np.float64] = np.empty(
            (len(indices), len(frames), 3), dtype=float
        )
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
