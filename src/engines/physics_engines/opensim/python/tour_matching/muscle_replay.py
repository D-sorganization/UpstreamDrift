"""Uninterrupted native excitation replay without observations or feedback.

This development boundary restores every named continuous OpenSim state and
cold-starts a native integrator once. It does not certify model anatomy, contact,
discrete/plugin state support, or physiological validity. Such qualification
belongs to the model-specific F07/F08 gates, not to successful integration.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from src.shared.python.force_overlay import OverlayWrench


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
    contact_wrenches: tuple[tuple[OverlayWrench, ...], ...] = ()
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


def _audit_drive_components(
    model: Any, osim: Any, contact_paths: tuple[str, ...]
) -> set[str]:
    """Audit recursive native components, including those outside legacy sets.

    SWIG exposes heterogeneous Component proxies without Python type stubs;
    Native casts establish inheritance; exact concrete-law identity separately
    restricts muscles to the explicitly tested state and force policies.
    """
    muscle_paths = set()
    found_contacts = set()
    # Exhaust SWIG's iterator before raising; early generator close in 4.6 can
    # otherwise emit an unraisable GeneratorExit warning.
    for component in tuple(model.getComponentsList()):
        if osim.Controller.safeDownCast(component) is not None:
            raise ValueError("existing controller is forbidden in independent replay")
        if osim.PositionMotion.safeDownCast(component) is not None:
            raise ValueError("prescribed motion is forbidden in independent replay")
        if osim.Constraint.safeDownCast(component) is not None:
            raise ValueError("constraint needs a qualified initialization policy")
        muscle = osim.Muscle.safeDownCast(component)
        if muscle is not None:
            if muscle.getConcreteClassName() not in (
                "Millard2012EquilibriumMuscle",
                "Thelen2003Muscle",
            ):
                raise ValueError(
                    "concrete muscle law needs an explicit qualified state/force policy"
                )
            if (
                muscle.get_ignore_activation_dynamics()
                or muscle.get_ignore_tendon_compliance()
            ):
                raise ValueError("ignored muscle dynamics need a separate state policy")
            muscle_paths.add(component.getAbsolutePathString())
            continue
        if osim.Actuator.safeDownCast(component) is not None:
            raise ValueError("non-muscle actuator is forbidden in muscle-only replay")
        if osim.Force.safeDownCast(component) is not None:
            path = component.getAbsolutePathString()
            if path in contact_paths:
                if component.getConcreteClassName() not in {
                    "HuntCrossleyForce",
                    "SmoothSphereHalfSpaceForce",
                }:
                    raise ValueError(
                        "listed contact paths must name supported native contact forces"
                    )
                if not re.fullmatch(r"[A-Za-z0-9_.:-]+", component.getName()):
                    raise ValueError(
                        "contact force names must have unambiguous overlay labels"
                    )
                found_contacts.add(path)
                continue
            raise ValueError(
                "non-muscle force requires a separate qualified passive/contact policy"
            )
    if found_contacts != set(contact_paths):
        raise ValueError(
            "contact force paths must exactly match native contact components"
        )
    forces = model.getForceSet()
    registered = {
        forces.get(i).getAbsolutePathString() for i in range(forces.getSize())
    }
    if not found_contacts <= registered:
        raise ValueError("contact paths must belong to the native force registry")
    return muscle_paths


def _contact_sampler(model: Any, paths: tuple[str, ...]) -> Any:
    """Reuse the existing native wrench provider, requiring complete contact output."""
    if not paths:
        return None
    from src.engines.physics_engines.opensim.python.opensim_force_torque import (
        OpenSimForceTorqueSource,
    )

    sampler = OpenSimForceTorqueSource(model, include_muscles=False)
    sampler.validate_ground_contact_paths(paths)
    return sampler


def _sample_contacts(
    sampler: Any, state: Any, paths: tuple[str, ...]
) -> tuple[OverlayWrench, ...]:
    """Fail on unsupported geometry or omitted evidence rather than filling zeros."""
    if sampler is None:
        return ()
    from src.shared.python.force_overlay import WrenchKind

    contacts = tuple(sampler.sample(state).by_kind(WrenchKind.CONTACT))
    expected = {"contact:" + path.rsplit("/", 1)[-1] for path in paths}
    if len(contacts) != len(paths) or {w.label for w in contacts} != expected:
        raise ValueError("native contact wrench evidence is incomplete or unsupported")
    for wrench in contacts:
        if wrench.force_n is None or wrench.torque_nm is None:
            raise ValueError(
                "native contact wrench evidence needs both force and torque"
            )
        if not np.isfinite((*wrench.force_n, *wrench.torque_nm, *wrench.point_m)).all():
            raise RuntimeError("native contact wrench evidence must be finite")
    return contacts


def _muscle_state_domains(model: Any, osim: Any) -> dict[str, tuple[float, float]]:
    """Return model-owned state bounds for the explicitly supported muscle law."""
    domains = {}
    for component in tuple(model.getComponentsList()):
        muscle = osim.Muscle.safeDownCast(component)
        if muscle is None:
            continue
        law = osim.Millard2012EquilibriumMuscle.safeDownCast(component)
        if law is None:
            law = osim.Thelen2003Muscle.safeDownCast(component)
        if law is None:
            raise ValueError(
                "muscle type needs an explicit qualified state-domain policy"
            )
        prefix = muscle.getAbsolutePathString()
        domains[prefix + "/activation"] = (law.getMinimumActivation(), 1.0)
        domains[prefix + "/fiber_length"] = (law.getMinimumFiberLength(), np.inf)
    return domains


def _validate_muscle_state(
    values: Mapping[str, float], domains: Mapping[str, tuple[float, float]]
) -> None:
    """Reject model-domain violations without silently clamping native state."""
    for name, (lower, upper) in domains.items():
        if name in values and (
            not lower <= values[name] <= upper
            or (name.endswith("/fiber_length") and values[name] <= 0)
        ):
            raise ValueError(
                f"muscle state domain violated: {name}; bounds [{lower}, {upper}], positive fiber length required"
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
    contact_force_paths: tuple[str, ...],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    tuple[tuple[OverlayWrench, ...], ...],
]:
    """Advance one native manager without reinitialization or corrections."""
    from .native_scalar_replay import (
        NativeScalarReplayPolicy,
        integrate_native_scalar_replay,
    )

    samples = integrate_native_scalar_replay(
        model,
        state,
        state_names,
        tuple(muscles.get(i).getAbsolutePathString() for i in range(muscles.getSize())),
        grid,
        domains,
        NativeScalarReplayPolicy(accuracy, contact_force_paths),
    )
    return (
        samples.states,
        samples.actuations,
        samples.applied_controls,
        samples.contact_wrenches,
    )


def _admit_native_muscles(
    model: Any,
    controls: Mapping[str, NDArray[np.float64]],
    contact_force_paths: tuple[str, ...],
) -> tuple[Any, tuple[str, ...]]:
    """Reject hidden drive mechanisms before adding the owned input player."""
    import opensim as osim

    recursive_muscle_paths = _audit_drive_components(model, osim, contact_force_paths)
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


def _native_replay_policy(
    model: Any, accuracy: float, contact_force_paths: tuple[str, ...]
) -> dict[str, str | float | bool]:
    """Identify executed drive, native force and numerical policies."""
    import opensim as osim

    muscles = model.getMuscles()
    return {
        "adapter": "native-muscle-replay/1.3.1",
        "input_boundary": "muscle_excitation",
        "interpolation": "linear",
        "state_resets": False,
        "initialization": "complete-named-continuous-state-cold-start",
        "integrator": "RungeKuttaMerson",
        "accuracy": float(accuracy),
        "provider": osim.GetVersionAndDate(),
        "force_policy": (
            "muscle-gravity-and-listed-native-contact"
            if contact_force_paths
            else "muscle-and-gravity-only"
        ),
        "contact_force_paths": json.dumps(contact_force_paths),
        "contact_force_laws": json.dumps(
            {
                path: model.getComponent(path).getConcreteClassName()
                for path in contact_force_paths
            },
            sort_keys=True,
        ),
        "contact_frame": "world-z-up",
        "constraint_policy": "unconstrained-unlocked-only",
        "muscle_domain_policy": "explicit-equilibrium-muscle-native-minima/1.1.0",
        "muscle_class_policy": "exact-supported-concrete-law/1.0.0",
        "muscle_laws": json.dumps(
            [muscles.get(i).getConcreteClassName() for i in range(muscles.getSize())],
            separators=(",", ":"),
        ),
    }


def replay_muscle_excitations(
    model_path: str | Path,
    initial_state: Mapping[str, float],
    times: NDArray[np.float64],
    excitations: Mapping[str, NDArray[np.float64]],
    *,
    accuracy: float = 1e-8,
    contact_force_paths: tuple[str, ...] = (),
) -> NativeMuscleReplayResult:
    """Replay bounded muscle excitations through a fresh native OpenSim model.

    Linear interpolation is bounded between samples. A time-only native input
    player is constructed internally; callers cannot supply controllers, state
    corrections, or observation callbacks. Existing controllers, prescribed
    coordinates and non-muscle actuators are rejected to avoid hidden assistance.
    Contact requires an exact path allowlist of native HuntCrossley or smooth
    sphere/half-space forces and complete existing-provider wrench evidence.
    Other non-muscle forces remain forbidden. Full named initial state
    is mandatory and is never replaced by muscle equilibration/defaults.

    The caller must independently establish that named continuous state is a
    complete model initialization contract. This API does not serialize native
    numerical history; it declares a cold-start Runge-Kutta-Merson policy.
    """
    import opensim as osim

    grid, controls, initial = _validated_inputs(
        times, excitations, initial_state, accuracy
    )
    if (
        not isinstance(contact_force_paths, tuple)
        or any(
            not isinstance(p, str) or not p.startswith("/") for p in contact_force_paths
        )
        or len(set(contact_force_paths)) != len(contact_force_paths)
    ):
        raise ValueError(
            "contact force paths must be a tuple of unique absolute native paths"
        )
    contact_force_paths = tuple(sorted(contact_force_paths))
    path = Path(model_path)
    model_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    model = osim.Model(str(path))
    muscles, muscle_names = _admit_native_muscles(model, controls, contact_force_paths)
    _configure_input_player(model, muscles, muscle_names, grid, controls)
    model.finalizeConnections()
    state, state_names, domains = _restore_continuous_state(
        model, initial, float(grid[0])
    )
    if any(
        muscles.get(i).isActuationOverridden(state) for i in range(len(muscle_names))
    ):
        raise ValueError("overridden muscle force is forbidden in excitation replay")

    policy = _native_replay_policy(model, accuracy, contact_force_paths)
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
    states, forces, applied, contact_wrenches = _integrate_native_replay(
        model, muscles, state, state_names, grid, domains, accuracy, contact_force_paths
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
        contact_wrenches=tuple(contact_wrenches),
    )
