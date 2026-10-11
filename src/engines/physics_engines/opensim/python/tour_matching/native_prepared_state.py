"""Reconstruct a declared OpenSim constrained cold start on an owned model.

This is an exact-source preparation boundary, not arbitrary SimTK State
serialization or physiological admission. Model-owned lock targets are set from
an explicit declaration; they are never inferred from named q or XML defaults.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .native_constraint_state import (
    NativeConstraintStateAudit,
    _restore_named_state,
    audit_native_constraint_state,
    owned_native_source_state,
)


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _finite_mapping(values: Mapping[str, float], name: str) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.startswith("/"):
            raise ValueError(f"{name} keys must be absolute native paths")
        result[key] = _finite(value, name)
    return MappingProxyType(result)


def _validated_command_step(
    value: tuple[float, Mapping[str, float]] | None,
    initial_time: float,
    commands: Mapping[str, float],
) -> tuple[float, Mapping[str, float]] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("scheduled command step must be a time and channel mapping")
    time = _finite(value[0], "scheduled command time")
    future = _finite_mapping(value[1], "scheduled actuator command")
    if time <= initial_time or set(future) != set(commands):
        raise ValueError("scheduled command step needs a future exact channel set")
    return time, future


@dataclass(frozen=True)
class DeclaredColdStart:
    """Caller-owned cold-start values and explicit native constraint recipe."""

    model_path: Path
    source_sha256: str
    named_state: Mapping[str, float]
    time_seconds: float
    lock_targets: Mapping[str, float]
    chart_bounds: Mapping[str, tuple[float, float]]
    constraint_enforcement: Mapping[str, bool]
    residual_tolerance: float
    constant_commands: Mapping[str, float] = field(default_factory=dict)
    allow_source_controllers_for_observation: bool = False
    source_controller_replacement: tuple[str, str, float] | None = None
    scheduled_command_step: tuple[float, Mapping[str, float]] | None = None
    linear_chart_bounds: Mapping[str, tuple[Mapping[str, float], float, float]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        path = Path(self.model_path)
        if not path.is_file():
            raise ValueError("declared native source file is unavailable")
        if len(self.source_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.source_sha256
        ):
            raise ValueError("source SHA-256 must be a lowercase hex digest")
        object.__setattr__(self, "model_path", path.resolve())
        object.__setattr__(
            self, "named_state", _finite_mapping(self.named_state, "named state")
        )
        object.__setattr__(
            self, "lock_targets", _finite_mapping(self.lock_targets, "lock target")
        )
        object.__setattr__(
            self,
            "constant_commands",
            _finite_mapping(self.constant_commands, "actuator command"),
        )
        if not isinstance(self.chart_bounds, Mapping):
            raise TypeError("chart bounds must be a mapping")
        bounds: dict[str, tuple[float, float]] = {}
        for coordinate_path, interval in self.chart_bounds.items():
            if not isinstance(coordinate_path, str) or not coordinate_path.startswith(
                "/"
            ):
                raise ValueError("chart bound keys must be absolute native paths")
            if not isinstance(interval, tuple) or len(interval) != 2:
                raise ValueError("chart bounds must be finite lower/upper pairs")
            lower, upper = (_finite(value, "chart bound") for value in interval)
            if lower >= upper:
                raise ValueError("chart bounds must have lower < upper")
            bounds[coordinate_path] = (lower, upper)
        object.__setattr__(self, "chart_bounds", MappingProxyType(bounds))
        if not isinstance(self.linear_chart_bounds, Mapping):
            raise TypeError("linear chart bounds must be a mapping")
        linear = {}
        for name, (terms, lower, upper) in self.linear_chart_bounds.items():
            if not isinstance(name, str) or not name or not terms:
                raise ValueError("linear chart rules need names and terms")
            frozen_terms = _finite_mapping(terms, "linear chart term")
            interval = (_finite(lower, "linear lower"), _finite(upper, "linear upper"))
            if interval[0] >= interval[1]:
                raise ValueError("linear chart bounds must have lower < upper")
            linear[name] = (frozen_terms, *interval)
        object.__setattr__(self, "linear_chart_bounds", MappingProxyType(linear))
        if not isinstance(self.constraint_enforcement, Mapping):
            raise TypeError("constraint enforcement must be a mapping")
        constraints: dict[str, bool] = {}
        for constraint_path, enforced in self.constraint_enforcement.items():
            if not isinstance(constraint_path, str) or not constraint_path.startswith(
                "/"
            ):
                raise ValueError("constraint paths must be absolute")
            if not isinstance(enforced, bool):
                raise TypeError("constraint enforcement must be boolean")
            constraints[constraint_path] = enforced
        object.__setattr__(
            self, "constraint_enforcement", MappingProxyType(constraints)
        )
        _finite(self.time_seconds, "time")
        tolerance = _finite(self.residual_tolerance, "residual tolerance")
        if tolerance <= 0:
            raise ValueError("residual tolerance must be positive")
        if not isinstance(self.allow_source_controllers_for_observation, bool):
            raise TypeError("source-controller observation flag must be boolean")
        if self.allow_source_controllers_for_observation and self.constant_commands:
            raise ValueError("owned commands cannot mix with source controllers")
        object.__setattr__(
            self,
            "scheduled_command_step",
            _validated_command_step(
                self.scheduled_command_step,
                self.time_seconds,
                self.constant_commands,
            ),
        )
        replacement = self.source_controller_replacement
        if replacement is not None:
            if not isinstance(replacement, tuple):
                raise TypeError("source-controller replacement must be a tuple")
            if (
                self.allow_source_controllers_for_observation
                or not self.constant_commands
            ):
                raise ValueError(
                    "source-controller replacement requires owned commands"
                )
            if len(replacement) != 3 or not replacement[0].startswith("/"):
                raise ValueError("source-controller replacement identity is malformed")
            if len(replacement[1]) != 64 or any(
                char not in "0123456789abcdef" for char in replacement[1]
            ):
                raise ValueError("source-controller replacement digest is malformed")
            _finite(replacement[2], "original source-controller command")


@dataclass(frozen=True)
class ReconstructedColdStart:
    """Actual owned native objects and the independently observed state."""

    model: Any
    state: Any
    audit: NativeConstraintStateAudit
    source_controller_observed: bool


@dataclass(frozen=True)
class DeclaredColdStartReplay:
    """Uninterrupted mechanical diagnostic; no muscular qualification."""

    time_seconds: tuple[float, ...]
    actuator_paths: tuple[str, ...]
    applied_commands: tuple[tuple[float, ...], ...]
    audits: tuple[NativeConstraintStateAudit, ...]
    source_sha256: str
    loaded_model_sha256: str
    input_sha256: str
    admission_sha256: str
    adapter_source_sha256: str


def _apply_declared_locks(model: Any, state: Any, targets: Mapping[str, float]) -> None:
    coordinates = tuple(model.getCoordinateSet())
    locked = {
        coordinate.getAbsolutePathString()
        for coordinate in coordinates
        if coordinate.getLocked(state)
    }
    if set(targets) != locked:
        raise ValueError("every native lock target needs an explicit declaration")
    for coordinate in coordinates:
        path = coordinate.getAbsolutePathString()
        if path not in targets:
            continue
        coordinate.setLocked(state, False)
        coordinate.setValue(state, targets[path], False)
        coordinate.setLocked(state, True)


def _remove_matching_source_controller(
    controllers: Any,
    replacement: tuple[str, str, float],
    commands: Mapping[str, float],
) -> None:
    import opensim as osim

    if controllers.getSize() != 1:
        raise ValueError("source controller identity requires exactly one controller")
    original = controllers.get(0)
    original_path, original_sha, original_command = replacement
    if (
        original.getAbsolutePathString() != original_path
        or hashlib.sha256(original.dump().encode()).hexdigest() != original_sha
    ):
        raise ValueError("source controller identity differs from declaration")
    prescribed = osim.PrescribedController.safeDownCast(original)
    if (
        original.getConcreteClassName() != "PrescribedController"
        or prescribed is None
        or prescribed.get_ControlFunctions().getSize() != 1
    ):
        raise ValueError("source controller law is not single-channel prescribed")
    source_function = prescribed.get_ControlFunctions().get(0)
    constant = osim.Constant.safeDownCast(source_function)
    socket = prescribed.getSocket("actuators")
    if (
        source_function.getConcreteClassName() != "Constant"
        or constant is None
        or constant.getValue() != original_command
        or socket.getNumConnectees() != 1
        or set(commands) != {str(socket.getConnecteePath(0))}
    ):
        raise ValueError("source controller law or actuator differs")
    if not controllers.remove(0):
        raise RuntimeError("native source controller removal failed")


def _build_prescribed_player(
    native: Mapping[str, Any],
    commands: Mapping[str, float],
    initial_time: float,
    command_step: tuple[float, Mapping[str, float]] | None,
) -> Any:
    import opensim as osim

    player = osim.PrescribedController()
    player.setName("owned_time_only_cold_start_player")
    for path, actuator in native.items():
        command = commands[path]
        scalar = osim.ScalarActuator.safeDownCast(actuator)
        if scalar is None:
            raise ValueError("only scalar actuator commands have this native policy")
        if osim.Muscle.safeDownCast(actuator) is not None:
            raise ValueError("muscle excitation needs its dedicated native player")
        if not scalar.getMinControl() <= command <= scalar.getMaxControl():
            raise ValueError("command exceeds native actuator bounds")
        future_command = command if command_step is None else command_step[1][path]
        if not scalar.getMinControl() <= future_command <= scalar.getMaxControl():
            raise ValueError("scheduled command exceeds native actuator bounds")
        player.addActuator(scalar)
        if command_step is None:
            function = osim.Constant(command)
        else:
            function = osim.PiecewiseConstantFunction()
            function.addPoint(initial_time, command)
            function.addPoint(command_step[0], future_command)
        player.prescribeControlForActuator(actuator.getName(), function)
    return player


def _install_owned_input_player(
    model: Any,
    commands: Mapping[str, float],
    allow_source_controllers: bool,
    replacement: tuple[str, str, float] | None,
    initial_time: float,
    command_step: tuple[float, Mapping[str, float]] | None,
) -> bool:
    import opensim as osim

    controllers = model.updControllerSet()
    components = tuple(model.getComponentsList())
    registered_controllers = {
        controllers.get(index).getAbsolutePathString()
        for index in range(controllers.getSize())
    }
    discovered_controllers = {
        component.getAbsolutePathString()
        for component in components
        if osim.Controller.safeDownCast(component) is not None
    }
    if discovered_controllers != registered_controllers:
        raise ValueError("unregistered native controller needs a reviewed policy")
    source_controller = controllers.getSize() > 0
    if replacement is not None:
        _remove_matching_source_controller(controllers, replacement, commands)
        source_controller = False
    if source_controller and not allow_source_controllers:
        raise ValueError("source controller is forbidden in independent replay")
    if not commands:
        return source_controller
    forces = model.getForceSet()
    registered_forces = {
        forces.get(index).getAbsolutePathString() for index in range(forces.getSize())
    }
    discovered_forces = {
        component.getAbsolutePathString()
        for component in components
        if osim.Force.safeDownCast(component) is not None
    }
    if discovered_forces != registered_forces:
        raise ValueError("unregistered native force needs a reviewed policy")
    if any(
        osim.Actuator.safeDownCast(forces.get(index)) is None
        for index in range(forces.getSize())
    ):
        raise ValueError("non-actuator force needs a separately reviewed native policy")
    actuators = model.getActuators()
    native = {
        actuators.get(index).getAbsolutePathString(): actuators.get(index)
        for index in range(actuators.getSize())
    }
    if set(commands) != set(native):
        raise ValueError("constant commands need exact native actuator coverage")
    player = _build_prescribed_player(native, commands, initial_time, command_step)
    model.addController(player)
    return False


def _admit_observation(
    audit: NativeConstraintStateAudit, declaration: DeclaredColdStart
) -> None:
    actual_constraints = {item.path: item.enforced for item in audit.constraints}
    if actual_constraints != declaration.constraint_enforcement:
        raise ValueError("native constraint enforcement differs from declaration")
    coordinates = {item.path: item for item in audit.coordinates}
    if set(declaration.chart_bounds) - coordinates.keys():
        raise ValueError("chart bound names unknown native coordinate")
    for path, (lower, upper) in declaration.chart_bounds.items():
        if not lower <= coordinates[path].value <= upper:
            raise ValueError(f"native state is outside declared chart: {path}")
    for name, (terms, lower, upper) in declaration.linear_chart_bounds.items():
        if set(terms) - coordinates.keys():
            raise ValueError(f"linear chart rule names unknown coordinate: {name}")
        value = sum(weight * coordinates[path].value for path, weight in terms.items())
        if not lower <= value <= upper:
            raise ValueError(f"native state violates linear source chart: {name}")
    residuals = (*audit.position_errors, *audit.velocity_errors)
    if any(abs(value) > declaration.residual_tolerance for value in residuals):
        raise ValueError("native constraint residual exceeds declared tolerance")
    if any(item.prescribed for item in audit.coordinates):
        raise ValueError("prescribed coordinates need a separate input policy")


def observe_declared_native_sample(
    model: Any, state: Any, declaration: DeclaredColdStart
) -> NativeConstraintStateAudit:
    """Read one native sample and enforce the same chart/constraint admission."""
    audit = audit_native_constraint_state(model, state, declaration.lock_targets)
    _admit_observation(audit, declaration)
    return audit


@contextmanager
def reconstruct_declared_cold_start(
    declaration: DeclaredColdStart,
    *,
    expected_audit: NativeConstraintStateAudit | None = None,
) -> Iterator[ReconstructedColdStart]:
    """Rebuild a declared cold start, preserving original XML and native policy.

    A fresh model is owned for the context lifetime. Initial named state, lock
    targets, constraint flags and chart are checked before yielding. If an
    expected producer audit is given, every observed native field must match.
    The result remains a diagnostic, not a full native-restart certificate.
    """
    source = declaration.model_path
    if hashlib.sha256(source.read_bytes()).hexdigest() != declaration.source_sha256:
        raise ValueError("native source SHA-256 changed before cold start")
    source_controller = False

    def configure(model: Any) -> None:
        nonlocal source_controller
        source_controller = _install_owned_input_player(
            model,
            declaration.constant_commands,
            declaration.allow_source_controllers_for_observation,
            declaration.source_controller_replacement,
            declaration.time_seconds,
            declaration.scheduled_command_step,
        )

    with owned_native_source_state(
        source,
        before_initialize=configure,
    ) as (model, state, source_sha):
        if source_sha != declaration.source_sha256:
            raise ValueError("native source changed during owned load")
        _apply_declared_locks(model, state, declaration.lock_targets)
        _restore_named_state(model, state, declaration.named_state)
        state.setTime(declaration.time_seconds)
        audit = observe_declared_native_sample(model, state, declaration)
        if expected_audit is not None and audit != expected_audit:
            raise ValueError("fresh native observation differs from declared producer")
        yield ReconstructedColdStart(model, state, audit, source_controller)


def _replay_grid(values: tuple[float, ...], initial_time: float) -> tuple[float, ...]:
    if not isinstance(values, tuple) or len(values) < 2:
        raise ValueError("native replay needs at least two exact time knots")
    grid = tuple(_finite(value, "replay time") for value in values)
    if grid[0] != initial_time or any(
        later <= earlier for earlier, later in zip(grid, grid[1:], strict=False)
    ):
        raise ValueError("native replay clock must start at declared time and advance")
    return grid


def _admission_identity(declaration: DeclaredColdStart, adapter_sha256: str) -> str:
    payload = {
        "source_sha256": declaration.source_sha256,
        "adapter_source_sha256": adapter_sha256,
        "lock_targets": dict(declaration.lock_targets),
        "chart_bounds": dict(declaration.chart_bounds),
        "linear_chart_bounds": {
            name: (dict(terms), lower, upper)
            for name, (terms, lower, upper) in declaration.linear_chart_bounds.items()
        },
        "constraint_enforcement": dict(declaration.constraint_enforcement),
        "residual_tolerance": declaration.residual_tolerance,
        "source_controller_replacement": declaration.source_controller_replacement,
        "source_controller_observation": declaration.allow_source_controllers_for_observation,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def replay_declared_time_only_input(
    declaration: DeclaredColdStart,
    time_seconds: tuple[float, ...],
    *,
    accuracy: float = 1e-9,
    expected_initial_audit: NativeConstraintStateAudit | None = None,
) -> DeclaredColdStartReplay:
    """Freshly replay a declared ZOH mechanical signal without corrections.

    This diagnostic exercises exact declared constraints and chart at every
    sample. It cannot qualify muscle excitation, contact or full-swing replay.
    """
    grid = _replay_grid(time_seconds, declaration.time_seconds)
    step = declaration.scheduled_command_step
    if step is not None and step[0] not in grid[1:-1]:
        raise ValueError("scheduled command time must be an interior replay knot")
    tolerance = _finite(accuracy, "integrator accuracy")
    if not 0 < tolerance < 1:
        raise ValueError("integrator accuracy must be between zero and one")
    if not declaration.constant_commands:
        raise ValueError("mechanical replay needs declared actuator commands")
    if declaration.allow_source_controllers_for_observation:
        raise ValueError("source-controller observation cannot enter replay")
    adapter_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # Local import avoids a cycle with the executor's declared sample policy.
    from . import native_scalar_replay

    executor_path = Path(native_scalar_replay.__file__)
    executor_sha256 = hashlib.sha256(executor_path.read_bytes()).hexdigest()
    with reconstruct_declared_cold_start(
        declaration, expected_audit=expected_initial_audit
    ) as prepared:
        manager, initial = native_scalar_replay._initialize_manager(
            prepared.model, prepared.state, tuple(declaration.named_state), tolerance
        )
        paths = tuple(declaration.constant_commands)
        initial_commands = tuple(declaration.constant_commands[path] for path in paths)
        audits = []
        applied = []
        for index, time in enumerate(grid):
            state = initial if index == 0 else manager.integrate(time)
            if state.getTime() != time:
                raise RuntimeError("native integration did not reach exact replay knot")
            audit = observe_declared_native_sample(prepared.model, state, declaration)
            actual = {item.path: item.control for item in audit.actuators}
            observed = tuple(actual[path] for path in paths)
            expected = (
                initial_commands
                if step is None or time < step[0]
                else tuple(step[1][path] for path in paths)
            )
            if any(
                abs(left - right) > 1e-12
                for left, right in zip(observed, expected, strict=True)
            ):
                raise RuntimeError(
                    "native applied command differs from time-only input"
                )
            audits.append(audit)
            applied.append(observed)
        identity = {
            "source_sha256": declaration.source_sha256,
            "loaded_model_sha256": audits[0].loaded_model_sha256,
            "initial_observation_sha256": audits[0].observation_sha256,
            "time_seconds": grid,
            "actuator_paths": paths,
            "commands": initial_commands,
            "scheduled_command_step": None
            if step is None
            else (step[0], tuple(step[1][path] for path in paths)),
            "source_controller_replacement": declaration.source_controller_replacement,
            "interpolation": "zero_order_hold",
            "integrator": "RungeKuttaMerson",
            "accuracy": tolerance,
            "integrator_provider_sha256": executor_sha256,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != adapter_sha256:
        raise RuntimeError("native preparation adapter changed during replay")
    if hashlib.sha256(executor_path.read_bytes()).hexdigest() != executor_sha256:
        raise RuntimeError("native integrator provider changed during replay")
    return DeclaredColdStartReplay(
        grid,
        paths,
        tuple(applied),
        tuple(audits),
        declaration.source_sha256,
        audits[0].loaded_model_sha256,
        digest,
        _admission_identity(declaration, adapter_sha256),
        adapter_sha256,
    )


def replay_declared_constant_input(
    declaration: DeclaredColdStart,
    time_seconds: tuple[float, ...],
    *,
    accuracy: float = 1e-9,
    expected_initial_audit: NativeConstraintStateAudit | None = None,
) -> DeclaredColdStartReplay:
    """Compatibility entry point restricted to a constant mechanical signal."""
    if declaration.scheduled_command_step is not None:
        raise ValueError("constant replay cannot contain a scheduled command step")
    return replay_declared_time_only_input(
        declaration,
        time_seconds,
        accuracy=accuracy,
        expected_initial_audit=expected_initial_audit,
    )


__all__ = [
    "DeclaredColdStart",
    "DeclaredColdStartReplay",
    "ReconstructedColdStart",
    "observe_declared_native_sample",
    "reconstruct_declared_cold_start",
    "replay_declared_constant_input",
    "replay_declared_time_only_input",
]
