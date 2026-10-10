"""One owned native Manager and engine-owned scalar-actuator sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from .native_prepared_state import DeclaredColdStart, observe_declared_native_sample
from .moco_initial_bindings import _finite_number


@dataclass(frozen=True)
class NativeScalarReplayPolicy:
    """Integrator accuracy and read-only native sample observations."""

    accuracy: float
    contact_force_paths: tuple[str, ...] = ()
    constrained_cold_start: DeclaredColdStart | None = None

    def __post_init__(self) -> None:
        if _finite_number(self.accuracy) <= 0:
            raise ValueError("native integration accuracy must be positive")
        if not isinstance(self.contact_force_paths, tuple):
            raise TypeError("native contact paths must be immutable")
        if any(
            not isinstance(path, str) or not path.startswith("/")
            for path in self.contact_force_paths
        ) or len(set(self.contact_force_paths)) != len(self.contact_force_paths):
            raise ValueError("native contact paths must be absolute and unique")
        if self.constrained_cold_start is not None and not isinstance(
            self.constrained_cold_start, DeclaredColdStart
        ):
            raise TypeError("native observations require a declared constraint policy")


@dataclass(frozen=True)
class NativeScalarSamples:
    states: NDArray[np.float64]
    actuations: NDArray[np.float64]
    applied_controls: NDArray[np.float64]
    powers_w: NDArray[np.float64]
    contact_wrenches: tuple[Any, ...]
    state_observations: tuple[Any, ...] = ()


def _initialize_manager(
    model: Any, state: Any, names: tuple[str, ...], accuracy: float
) -> tuple[Any, Any]:
    """Reject internal projection instead of recording an unexecuted seed."""
    import opensim as osim

    requested = tuple(model.getStateVariableValue(state, name) for name in names)
    manager = osim.Manager(model)
    manager.setIntegratorMethod(osim.Manager.IntegratorMethod_RungeKuttaMerson)
    manager.setIntegratorAccuracy(float(accuracy))
    manager.setWriteToStorage(False)
    manager.initialize(state)
    executed = manager.getState()
    observed = tuple(model.getStateVariableValue(executed, name) for name in names)
    if observed != requested or executed.getTime() != state.getTime():
        raise RuntimeError("native Manager changed the requested initial state seed")
    return manager, executed


def integrate_native_scalar_replay(
    model: Any,
    state: Any,
    state_names: tuple[str, ...],
    channel_paths: tuple[str, ...],
    grid: NDArray[np.float64],
    domains: Mapping[str, tuple[float, float]],
    policy: NativeScalarReplayPolicy,
) -> NativeScalarSamples:
    """Sample physical native outputs without feedback inputs or state resets.

    Admission and complete initial-state reconstruction belong to the calling
    engine profile. This internal executor owns the only integration loop.
    A fixed declared policy can read native constraint residuals at each sample;
    observations never supply or alter the saved actuator commands.
    """
    import opensim as osim

    from .muscle_replay import (
        _contact_sampler,
        _sample_contacts,
        _validate_muscle_state,
    )

    actuators = tuple(
        osim.ScalarActuator.safeDownCast(model.getComponent(path))
        for path in channel_paths
    )
    if not actuators or any(item is None for item in actuators):
        raise ValueError("native sampling requires admitted scalar actuators")
    muscles = tuple(osim.Muscle.safeDownCast(item) for item in actuators)
    sampler = _contact_sampler(model, policy.contact_force_paths)
    manager, state = _initialize_manager(model, state, state_names, policy.accuracy)
    states: NDArray[np.float64] = np.empty(
        (len(grid), len(state_names)), dtype=np.float64
    )
    forces: NDArray[np.float64] = np.empty(
        (len(grid), len(actuators)), dtype=np.float64
    )
    controls = np.empty_like(forces)
    powers = np.empty_like(forces)
    contacts = []
    observations = []
    for row, time in enumerate(grid):
        if row:
            state = manager.integrate(float(time))
        if float(state.getTime()) != float(time):
            raise RuntimeError(
                "native replay time differs from requested physical time"
            )
        model.realizeDynamics(state)
        if policy.constrained_cold_start is not None:
            observations.append(
                observe_declared_native_sample(
                    model, state, policy.constrained_cold_start
                )
            )
            model.realizeDynamics(state)
        states[row] = [model.getStateVariableValue(state, name) for name in state_names]
        _validate_muscle_state(
            dict(zip(state_names, states[row], strict=True)), domains
        )
        for column, (actuator, muscle) in enumerate(
            zip(actuators, muscles, strict=True)
        ):
            forces[row, column] = actuator.getActuation(state)
            controls[row, column] = (
                muscle.getExcitation(state)
                if muscle is not None
                else actuator.getControl(state)
            )
            powers[row, column] = actuator.getPower(state)
        contacts.append(_sample_contacts(sampler, state, policy.contact_force_paths))
    if not all(
        np.isfinite(array).all() for array in (states, forces, controls, powers)
    ):
        raise RuntimeError("native scalar replay produced nonfinite physical output")
    return NativeScalarSamples(
        states, forces, controls, powers, tuple(contacts), tuple(observations)
    )
