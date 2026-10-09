"""Uninterrupted native excitation replay without observations or feedback.

This development boundary restores every named continuous OpenSim state and
cold-starts a native integrator once. It does not certify model anatomy, contact,
discrete/plugin state support, or physiological validity. Such qualification
belongs to the model-specific F07/F08 gates, not to successful integration.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NativeMuscleReplayResult:
    """Native continuous-state replay evidence; never an acceptance certificate."""

    state_names: tuple[str, ...]
    muscle_names: tuple[str, ...]
    times: NDArray[np.float64]
    states: NDArray[np.float64]
    muscle_forces_n: NDArray[np.float64]
    applied_excitations: NDArray[np.float64]
    model_sha256: str
    input_sha256: str
    contact_geometry_count: int
    policy: Mapping[str, str | float | bool]
    mode: str = "native-muscle-excitation-replay"


def _validated_inputs(
    times: NDArray[np.float64],
    excitations: Mapping[str, NDArray[np.float64]],
    initial_state: Mapping[str, float],
    accuracy: float,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]], dict[str, float]]:
    """Copy caller inputs before passing ownership to a native simulator."""
    grid = np.array(times, dtype=np.float64, copy=True)
    if (
        grid.ndim != 1
        or len(grid) < 2
        or not np.isfinite(grid).all()
        or not np.all(np.diff(grid) > 0)
    ):
        raise ValueError("times must be finite and strictly increasing (at least two)")
    if not np.isfinite(accuracy) or not 0 < accuracy < 1:
        raise ValueError("integrator accuracy must be finite and between zero and one")
    controls = {}
    for name, values in excitations.items():
        values = np.array(values, dtype=np.float64, copy=True)
        if (
            values.shape != grid.shape
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values > 1)
        ):
            raise ValueError(f"excitation {name!r} must match times and lie in [0, 1]")
        controls[name] = values
    initial = {name: float(value) for name, value in initial_state.items()}
    if not all(np.isfinite(value) for value in initial.values()):
        raise ValueError("complete initial state must contain only finite values")
    return grid, controls, initial


def _audit_drive_components(model: Any, osim: Any) -> set[str]:
    """Audit recursive native components, including those outside legacy sets.

    SWIG exposes heterogeneous Component proxies without Python type stubs;
    native safeDownCast is the runtime type authority at this boundary.
    """
    muscle_paths = set()
    # Exhaust SWIG's iterator before raising; early generator close in 4.6 can
    # otherwise emit an unraisable GeneratorExit warning.
    for component in tuple(model.getComponentsList()):
        if osim.Controller.safeDownCast(component) is not None:
            raise ValueError("existing controller is forbidden in independent replay")
        if osim.PositionMotion.safeDownCast(component) is not None:
            raise ValueError("prescribed motion is forbidden in independent replay")
        if osim.Constraint.safeDownCast(component) is not None:
            raise ValueError("constraint needs a qualified initialization policy")
        if osim.Muscle.safeDownCast(component) is not None:
            muscle_paths.add(component.getAbsolutePathString())
            continue
        if osim.Actuator.safeDownCast(component) is not None:
            raise ValueError("non-muscle actuator is forbidden in muscle-only replay")
        if osim.Force.safeDownCast(component) is not None:
            raise ValueError(
                "non-muscle force requires a separate qualified passive/contact policy"
            )
    return muscle_paths


def _muscle_state_domains(model: Any, osim: Any) -> dict[str, tuple[float, float]]:
    """Return model-owned state bounds for the explicitly supported muscle law."""
    domains = {}
    for component in tuple(model.getComponentsList()):
        muscle = osim.Muscle.safeDownCast(component)
        if muscle is None:
            continue
        millard = osim.Millard2012EquilibriumMuscle.safeDownCast(component)
        if millard is None:
            raise ValueError(
                "muscle type needs an explicit qualified state-domain policy"
            )
        prefix = muscle.getAbsolutePathString()
        domains[prefix + "/activation"] = (millard.getMinimumActivation(), 1.0)
        domains[prefix + "/fiber_length"] = (millard.getMinimumFiberLength(), np.inf)
    return domains


def _validate_muscle_state(
    values: Mapping[str, float], domains: Mapping[str, tuple[float, float]]
) -> None:
    """Reject model-domain violations without silently clamping native state."""
    for name, (lower, upper) in domains.items():
        if name in values and not lower <= values[name] <= upper:
            raise ValueError(
                f"muscle state domain violated: {name} outside [{lower}, {upper}]"
            )


def _configure_input_player(
    model: Any,
    muscles: Any,
    muscle_names: tuple[str, ...],
    grid: NDArray[np.float64],
    controls: Mapping[str, NDArray[np.float64]],
) -> None:
    """Install only the internally owned time-driven excitation player."""
    import opensim as osim

    player = osim.PrescribedController()
    player.setName("independent_time_only_excitation_player")
    for name in muscle_names:
        function = osim.PiecewiseLinearFunction()
        for time, value in zip(grid, controls[name], strict=True):
            function.addPoint(float(time), float(value))
        player.addActuator(muscles.get(name))
        player.prescribeControlForActuator(name, function)
    model.addController(player)


def _restore_continuous_state(
    model: Any,
    initial: Mapping[str, float],
    initial_time: float,
) -> tuple[Any, tuple[str, ...], dict[str, tuple[float, float]]]:
    """Initialize once and restore every supplied physical continuous state."""
    import opensim as osim

    state = model.initSystem()
    names = model.getStateVariableNames()
    state_names = tuple(names.get(i) for i in range(names.getSize()))
    if set(initial) != set(state_names):
        raise ValueError("complete initial state must match every native state name")
    domains = _muscle_state_domains(model, osim)
    _validate_muscle_state(initial, domains)
    for name in state_names:
        value = initial[name]
        if name.endswith("/activation") and not 0 <= value <= 1:
            raise ValueError("initial muscle activation must lie in [0, 1]")
        if name.endswith("/fiber_length") and value <= 0:
            raise ValueError("initial muscle fiber length must be positive")
        model.setStateVariableValue(state, name, value)
    state.setTime(float(initial_time))
    model.realizeDynamics(state)
    return state, state_names, domains


def _integrate_native_replay(
    model: Any,
    muscles: Any,
    state: Any,
    state_names: tuple[str, ...],
    grid: NDArray[np.float64],
    domains: Mapping[str, tuple[float, float]],
    accuracy: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Advance one native manager without reinitialization or corrections."""
    import opensim as osim

    manager = osim.Manager(model)
    manager.setIntegratorMethod(osim.Manager.IntegratorMethod_RungeKuttaMerson)
    manager.setIntegratorAccuracy(float(accuracy))
    manager.setWriteToStorage(False)
    manager.initialize(state)
    states = np.empty((len(grid), len(state_names)), dtype=np.float64)
    forces = np.empty((len(grid), muscles.getSize()), dtype=np.float64)
    applied = np.empty_like(forces)
    for row, time in enumerate(grid):
        if row:
            state = manager.integrate(float(time))
        model.realizeDynamics(state)
        states[row] = [model.getStateVariableValue(state, n) for n in state_names]
        _validate_muscle_state(
            dict(zip(state_names, states[row], strict=True)), domains
        )
        forces[row] = [
            muscles.get(i).getActuation(state) for i in range(muscles.getSize())
        ]
        applied[row] = [
            muscles.get(i).getExcitation(state) for i in range(muscles.getSize())
        ]
    return states, forces, applied


def _admit_native_muscles(
    model: Any,
    controls: Mapping[str, NDArray[np.float64]],
) -> tuple[Any, tuple[str, ...]]:
    """Reject hidden drive mechanisms before adding the owned input player."""
    import opensim as osim

    recursive_muscle_paths = _audit_drive_components(model, osim)
    muscles = model.getMuscles()
    muscle_names = tuple(muscles.get(i).getName() for i in range(muscles.getSize()))
    registered_paths = {
        muscles.get(i).getAbsolutePathString() for i in range(muscles.getSize())
    }
    if recursive_muscle_paths != registered_paths:
        raise ValueError("native muscle registry omits recursively owned muscles")
    if not muscle_names or len(set(muscle_names)) != len(muscle_names):
        raise ValueError("model must contain uniquely named muscles")
    coordinates = model.getCoordinateSet()
    if any(
        coordinates.get(i).getDefaultIsPrescribed()
        for i in range(coordinates.getSize())
    ):
        raise ValueError("prescribed coordinate is forbidden in independent replay")
    if any(coordinates.get(i).getDefaultLocked() for i in range(coordinates.getSize())):
        raise ValueError("locked coordinate needs a qualified initialization policy")
    if set(controls) != set(muscle_names):
        raise ValueError("excitation names must exactly match native muscle names")

    return muscles, muscle_names


def replay_muscle_excitations(
    model_path: str | Path,
    initial_state: Mapping[str, float],
    times: NDArray[np.float64],
    excitations: Mapping[str, NDArray[np.float64]],
    *,
    accuracy: float = 1e-8,
) -> NativeMuscleReplayResult:
    """Replay bounded muscle excitations through a fresh native OpenSim model.

    Linear interpolation is bounded between samples. A time-only native input
    player is constructed internally; callers cannot supply controllers, state
    corrections, or observation callbacks. Existing controllers, prescribed
    coordinates, non-muscle actuators and other forces are rejected to avoid
    hidden assistance. Passive/contact force support needs a separate qualified
    policy. Full named initial state
    is mandatory and is never replaced by muscle equilibration/defaults.

    The caller must independently establish that named continuous state is a
    complete model initialization contract. This API does not serialize native
    numerical history; it declares a cold-start Runge-Kutta-Merson policy.
    """
    import opensim as osim

    grid, controls, initial = _validated_inputs(
        times, excitations, initial_state, accuracy
    )
    path = Path(model_path)
    model_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    model = osim.Model(str(path))
    muscles, muscle_names = _admit_native_muscles(model, controls)
    _configure_input_player(model, muscles, muscle_names, grid, controls)
    model.finalizeConnections()
    state, state_names, domains = _restore_continuous_state(
        model, initial, float(grid[0])
    )
    if any(
        muscles.get(i).isActuationOverridden(state) for i in range(len(muscle_names))
    ):
        raise ValueError("overridden muscle force is forbidden in excitation replay")

    policy: dict[str, str | float | bool] = {
        "adapter": "native-muscle-replay/1.0.0",
        "input_boundary": "muscle_excitation",
        "interpolation": "linear",
        "state_resets": False,
        "initialization": "complete-named-continuous-state-cold-start",
        "integrator": "RungeKuttaMerson",
        "accuracy": float(accuracy),
        "provider": osim.GetVersionAndDate(),
        "force_policy": "muscle-and-gravity-only",
        "constraint_policy": "unconstrained-unlocked-only",
        "muscle_domain_policy": "Millard2012EquilibriumMuscle-native-minima",
    }
    payload = {
        "model_sha256": model_digest,
        "state_names": state_names,
        "initial_state": [initial[name] for name in state_names],
        "times": grid.tolist(),
        "muscle_names": muscle_names,
        "excitations": [controls[name].tolist() for name in muscle_names],
        "policy": policy,
    }
    input_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    states, forces, applied = _integrate_native_replay(
        model, muscles, state, state_names, grid, domains, accuracy
    )
    if not np.isfinite(states).all() or not np.isfinite(forces).all():
        raise RuntimeError("native muscle replay produced nonfinite state or force")
    expected = np.column_stack([controls[name] for name in muscle_names])
    if not np.allclose(applied, expected, rtol=0, atol=1e-12):
        raise RuntimeError("native excitation differs from the saved input samples")
    for array in (grid, states, forces, applied):
        array.setflags(write=False)
    return NativeMuscleReplayResult(
        state_names=state_names,
        muscle_names=muscle_names,
        times=grid,
        states=states,
        muscle_forces_n=forces,
        applied_excitations=applied,
        model_sha256=model_digest,
        input_sha256=input_digest,
        contact_geometry_count=model.getContactGeometrySet().getSize(),
        policy=MappingProxyType(policy),
    )
