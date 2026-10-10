"""Observe constrained native state without certifying an arbitrary restart.

OpenSim's registered state maps omit Simbody constraint enable flags and the
model-owned mutable coordinate lock target. This diagnostic preserves that gap;
it never equates an observation fingerprint with a complete restart artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator


@dataclass(frozen=True)
class CoordinateConstraintObservation:
    """Actual coordinate state; declared target is not a native target readback."""

    path: str
    motion_type: int
    value: float
    speed: float
    locked: bool
    prescribed: bool
    clamped: bool
    dependent: bool
    source_default_value: float
    source_default_locked: bool
    declared_lock_target: float | None


@dataclass(frozen=True)
class ConstraintObservation:
    """Native enforcement may differ from the serialized default property."""

    path: str
    concrete_class: str
    enforced: bool
    serialized_sha256: str


@dataclass(frozen=True)
class ActuatorObservation:
    """Native actuator values; actuation units belong to its concrete force law."""

    path: str
    concrete_class: str
    is_muscle: bool
    enabled: bool
    control: float
    actuation: float
    power_w: float
    overridden: bool


@dataclass(frozen=True)
class NativeConstraintStateAudit:
    """Immutable observations, with unsupported full-state semantics explicit."""

    loaded_model_sha256: str
    runtime: str
    runtime_extension_sha256: str
    simbody_extension_sha256: str
    observer_sha256: str
    time_seconds: float
    coordinates: tuple[CoordinateConstraintObservation, ...]
    constraints: tuple[ConstraintObservation, ...]
    actuators: tuple[ActuatorObservation, ...]
    named_state: tuple[tuple[str, float], ...]
    registered_discrete: tuple[tuple[str, float], ...]
    modeling_options: tuple[tuple[str, float], ...]
    derivatives: tuple[tuple[str, float], ...]
    muscle_forces: tuple[tuple[str, float], ...]
    position_errors: tuple[float, ...]
    velocity_errors: tuple[float, ...]
    acceleration_errors: tuple[float, ...]
    constraint_jacobian_shape: tuple[int, int]
    blockers: tuple[str, ...]
    source_sha256: str | None = None
    qualification: str = field(default="not-qualified-for-native-restart", init=False)

    @property
    def observation_sha256(self) -> str:
        """Bind observations only; never authenticate or qualify a native state."""
        text = json.dumps(asdict(self), sort_keys=True, allow_nan=False)
        return _digest(text.encode())


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("native state/target values must be real numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("native state/target values must be finite")
    return result


def _vector(value: Any) -> tuple[float, ...]:
    return tuple(_finite(value.get(index)) for index in range(value.size()))


def _named_values(
    names: Any, getter: Callable[[str], float]
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (names.get(index), _finite(getter(names.get(index))))
        for index in range(names.getSize())
    )


def _coordinate_observations(
    model: Any, state: Any, targets: Mapping[str, float]
) -> tuple[CoordinateConstraintObservation, ...]:
    result = []
    coordinates = model.getCoordinateSet()
    paths = {coordinate.getAbsolutePathString() for coordinate in coordinates}
    if set(targets) - paths:
        raise ValueError("declared lock targets name unknown coordinate paths")
    for coordinate in coordinates:
        path = coordinate.getAbsolutePathString()
        if path in targets and not coordinate.getLocked(state):
            raise ValueError(
                "declared lock target requires an actually locked coordinate"
            )
        result.append(
            CoordinateConstraintObservation(
                path=path,
                motion_type=int(coordinate.getMotionType()),
                value=_finite(coordinate.getValue(state)),
                speed=_finite(coordinate.getSpeedValue(state)),
                locked=bool(coordinate.getLocked(state)),
                prescribed=bool(coordinate.isPrescribed(state)),
                clamped=bool(coordinate.getClamped(state)),
                dependent=bool(coordinate.isDependent(state)),
                source_default_value=_finite(coordinate.getDefaultValue()),
                source_default_locked=bool(coordinate.getDefaultLocked()),
                declared_lock_target=targets.get(path),
            )
        )
    return tuple(result)


def _constraint_observations(
    model: Any, state: Any, osim: Any
) -> tuple[ConstraintObservation, ...]:
    result = []
    constraint_type = osim.Constraint
    for component in tuple(model.getComponentsList()):
        constraint = constraint_type.safeDownCast(component)
        if constraint is not None:
            result.append(
                ConstraintObservation(
                    path=constraint.getAbsolutePathString(),
                    concrete_class=constraint.getConcreteClassName(),
                    enforced=bool(constraint.isEnforced(state)),
                    serialized_sha256=_digest(constraint.dump().encode()),
                )
            )
    return tuple(result)


def native_muscles(model: Any, osim: Any) -> tuple[Any, ...]:
    """Return the complete native muscle registry or reject hidden components."""
    muscles = tuple(model.getMuscles())
    registered = {muscle.getAbsolutePathString() for muscle in muscles}
    recursive = {
        component.getAbsolutePathString()
        for component in model.getComponentsList()
        if osim.Muscle.safeDownCast(component) is not None
    }
    if recursive != registered:
        raise ValueError("native muscle registry must cover all recursive muscles")
    return muscles


def _blockers(
    coordinates: tuple[CoordinateConstraintObservation, ...],
    constraints: tuple[ConstraintObservation, ...],
    actuators: tuple[ActuatorObservation, ...],
) -> tuple[str, ...]:
    result = [
        "complete-native-state-unverified",
        "constraint-residual-acceptance-unqualified",
    ]
    if any(item.locked for item in coordinates):
        result.append("native-lock-target-unverified")
    if any(item.prescribed for item in coordinates):
        result.append("native-prescribed-motion-unqualified")
    if constraints:
        result.append("native-constraint-state-unqualified")
    if any(not item.is_muscle for item in actuators):
        result.append("nonmuscle-assistance-unqualified")
    if any(
        item.declared_lock_target is not None
        and item.value != item.declared_lock_target
        for item in coordinates
    ):
        result.append("declared-lock-target-mismatch")
    return tuple(result)


def _actuator_observations(
    model: Any, state: Any, osim: Any
) -> tuple[ActuatorObservation, ...]:
    result = []
    actuator_type = osim.ScalarActuator
    muscle_type = osim.Muscle
    for component in tuple(model.getComponentsList()):
        actuator = actuator_type.safeDownCast(component)
        if actuator is None:
            continue
        result.append(
            ActuatorObservation(
                path=actuator.getAbsolutePathString(),
                concrete_class=actuator.getConcreteClassName(),
                is_muscle=muscle_type.safeDownCast(component) is not None,
                enabled=bool(actuator.appliesForce(state)),
                control=_finite(actuator.getControl(state)),
                actuation=_finite(actuator.getActuation(state)),
                power_w=_finite(actuator.getPower(state)),
                overridden=bool(actuator.isActuationOverridden(state)),
            )
        )
    return tuple(result)


def audit_native_constraint_state(
    model: Any, state: Any, declared_lock_targets: Mapping[str, float] | None = None
) -> NativeConstraintStateAudit:
    """Observe a native model/state without changing its continuous or discrete state.

    The caller must own the model and state and prevent concurrent mutation. A
    copied State isolates realization caches, but model-owned constraint targets
    remain shared. No missing target is inferred from q or source defaults. This
    function does not integrate, assemble, equilibrate, or admit replay.
    """
    import opensim as osim

    if not isinstance(model, osim.Model) or not isinstance(state, osim.State):
        raise TypeError("audit requires actual native OpenSim Model and State")
    if declared_lock_targets is not None and not isinstance(
        declared_lock_targets, Mapping
    ):
        raise TypeError("declared lock targets must be a mapping")
    targets = {
        key: _finite(value) for key, value in (declared_lock_targets or {}).items()
    }
    working = osim.State(state)
    model.realizeAcceleration(working)
    coordinates = _coordinate_observations(model, working, targets)
    constraints = _constraint_observations(model, working, osim)
    actuators = _actuator_observations(model, working, osim)
    matter = model.getMatterSubsystem()
    jacobian = osim.Matrix()
    matter.calcG(working, jacobian)
    return NativeConstraintStateAudit(
        loaded_model_sha256=_digest(model.dump().encode()),
        runtime=osim.GetVersionAndDate(),
        runtime_extension_sha256=_digest(Path(osim._simulation.__file__).read_bytes()),
        simbody_extension_sha256=_digest(Path(osim._simbody.__file__).read_bytes()),
        observer_sha256=_digest(Path(__file__).read_bytes()),
        time_seconds=_finite(working.getTime()),
        coordinates=coordinates,
        constraints=constraints,
        actuators=actuators,
        named_state=_named_values(
            model.getStateVariableNames(),
            lambda name: model.getStateVariableValue(working, name),
        ),
        registered_discrete=_named_values(
            model.getDiscreteVariableNames(),
            lambda name: model.getDiscreteVariableValue(working, name),
        ),
        modeling_options=_named_values(
            model.getModelingOptionNames(),
            lambda name: model.getModelingOption(working, name),
        ),
        derivatives=_named_values(
            model.getStateVariableNames(),
            lambda name: model.getStateVariableDerivativeValue(working, name),
        ),
        muscle_forces=tuple(
            (muscle.getAbsolutePathString(), _finite(muscle.getActuation(working)))
            for muscle in native_muscles(model, osim)
        ),
        position_errors=_vector(working.getQErr()),
        velocity_errors=_vector(working.getUErr()),
        acceleration_errors=_vector(working.getUDotErr()),
        constraint_jacobian_shape=(jacobian.nrow(), jacobian.ncol()),
        blockers=_blockers(coordinates, constraints, actuators),
    )


def _restore_named_state(model: Any, state: Any, initial: Mapping[str, float]) -> None:
    names = model.getStateVariableNames()
    expected = {names.get(index) for index in range(names.getSize())}
    if set(initial) != expected:
        raise ValueError("initial state must have complete native named-state coverage")
    frozen = {name: _finite(value) for name, value in initial.items()}
    for name, value in frozen.items():
        model.setStateVariableValue(state, name, value)
    if any(
        model.getStateVariableValue(state, name) != value
        for name, value in frozen.items()
    ):
        raise ValueError("native achieved state differs from requested restore")


@contextmanager
def owned_native_source_state(
    model_path: Path, initial_state: Mapping[str, float] | None = None
) -> Iterator[tuple[Any, Any, str]]:
    """Load an owned XML copy, optionally restoring every named state value.

    This is a read-only observation boundary, not a native restart certificate.
    Resource closure is not verified; callers must account for external assets.
    Native initSystem includes OpenSim's initialization assembly; restoring named
    continuous values afterward does not invoke a second assembly here.
    """
    import opensim as osim

    source = Path(model_path)
    payload = source.read_bytes()
    with NamedTemporaryFile(
        prefix="opensim-constraint-audit-", suffix=".osim", delete=False
    ) as stream:
        stream.write(payload)
        owned = Path(stream.name)
    try:
        model = osim.Model(str(owned))
        model.finalizeConnections()
        state = model.initSystem()
        if initial_state is not None:
            _restore_named_state(model, state, initial_state)
        yield model, state, _digest(payload)
        if source.read_bytes() != payload:
            raise ValueError("source XML changed during native observation")
    finally:
        owned.unlink()


def audit_source_constraint_state(
    model_path: Path,
    initial_state: Mapping[str, float],
    declared_lock_targets: Mapping[str, float] | None = None,
) -> NativeConstraintStateAudit:
    """Evaluate frozen XML in an isolated fresh native model with its source locks.

    Restore every named continuous value, rejecting silently ignored assignments.
    Registered options and unexposed state retain native initialization defaults;
    this is explicitly not an arbitrary-state restart. External resources and
    transitive runtime dependencies are not certified by this diagnostic.
    """
    with owned_native_source_state(model_path, initial_state) as (
        model,
        state,
        source_sha,
    ):
        result = audit_native_constraint_state(model, state, declared_lock_targets)
    return replace(result, source_sha256=source_sha)


__all__ = [
    "ActuatorObservation",
    "CoordinateConstraintObservation",
    "ConstraintObservation",
    "NativeConstraintStateAudit",
    "audit_native_constraint_state",
    "audit_source_constraint_state",
    "owned_native_source_state",
    "native_muscles",
]
