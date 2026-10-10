"""Explicit immutable native state and scalar-control constraints for Moco."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Any


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Binding values must be real numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Binding values must be finite")
    return result


def _freeze_bounds(
    bounds: Mapping[str, tuple[float, float]],
) -> Mapping[str, tuple[float, float]]:
    result = {}
    for name, interval in bounds.items():
        if not isinstance(name, str) or not name.startswith("/") or len(name) < 2:
            raise ValueError("Binding names must be absolute native component paths")
        if len(interval) != 2:
            raise ValueError("Bounds require exactly two endpoints")
        lower, upper = (_finite_number(value) for value in interval)
        if lower > upper:
            raise ValueError("Lower bound must not exceed upper bound")
        result[name] = (lower, upper)
    return MappingProxyType(result)


@dataclass(frozen=True)
class MocoInitialBindings:
    """Caller-supplied bounds and fixed initial continuous native state.

    Bounds must come from the model and experiment protocol. This object does
    not invent physiological limits or serialize discrete/native model state.
    """

    state_bounds: Mapping[str, tuple[float, float]]
    initial_state: Mapping[str, float]
    control_bounds: Mapping[str, tuple[float, float]]

    def __post_init__(self) -> None:
        states = _freeze_bounds(self.state_bounds)
        controls = _freeze_bounds(self.control_bounds)
        if states.keys() != self.initial_state.keys():
            raise ValueError("Initial state and state bounds must have identical names")
        initial = {}
        for name, value in self.initial_state.items():
            scalar = _finite_number(value)
            lower, upper = states[name]
            if not lower <= scalar <= upper:
                raise ValueError(f"Initial state outside bounds: {name}")
            initial[name] = scalar
        object.__setattr__(self, "state_bounds", states)
        object.__setattr__(self, "control_bounds", controls)
        object.__setattr__(self, "initial_state", MappingProxyType(initial))


def apply_moco_initial_bindings(
    problem: Any, model_path: str, bindings: MocoInitialBindings, osim: Any
) -> None:
    """Require complete native coverage and apply constraints before guesses.

    Loading and initializing the unchanged model discovers its continuous state
    names and actual scalar actuator paths. No equilibration or model transforms
    are applied. Multi-control actuators require a separately defined mapping.
    """
    model = osim.Model(model_path)
    components = tuple(model.getComponentsList())
    if any(osim.Controller.safeDownCast(item) is not None for item in components):
        raise ValueError("Existing controller requires a separate Moco binding policy")
    if any(osim.PositionMotion.safeDownCast(item) is not None for item in components):
        raise ValueError("Prescribed motion requires a separate Moco binding policy")
    model.initSystem()
    names = model.getStateVariableNames()
    native_states = {str(names.get(index)) for index in range(names.getSize())}
    actuators = model.getActuators()
    native_controls = set()
    for index in range(actuators.getSize()):
        actuator = actuators.get(index)
        if actuator.numControls() != 1:
            raise ValueError(
                "Complete native control mapping requires scalar actuators"
            )
        native_controls.add(str(actuator.getAbsolutePathString()))
    if set(bindings.state_bounds) != native_states:
        raise ValueError("Bindings must cover the complete native continuous state")
    if set(bindings.control_bounds) != native_controls:
        raise ValueError("Bindings must cover the complete native scalar controls")
    for name, (lower, upper) in bindings.state_bounds.items():
        problem.setStateInfo(
            name,
            osim.MocoBounds(lower, upper),
            osim.MocoInitialBounds(bindings.initial_state[name]),
        )
    for name, (lower, upper) in bindings.control_bounds.items():
        problem.setControlInfo(name, osim.MocoBounds(lower, upper))
