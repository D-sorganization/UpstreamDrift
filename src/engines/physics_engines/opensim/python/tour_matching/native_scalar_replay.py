"""One owned native Manager and engine-owned scalar-actuator sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NativeScalarSamples:
    states: NDArray[np.float64]
    actuations: NDArray[np.float64]
    applied_controls: NDArray[np.float64]
    powers_w: NDArray[np.float64]
    contact_wrenches: tuple[Any, ...]


def integrate_native_scalar_replay(
    model: Any,
    state: Any,
    state_names: tuple[str, ...],
    channel_paths: tuple[str, ...],
    grid: NDArray[np.float64],
    domains: Mapping[str, tuple[float, float]],
    accuracy: float,
    contact_force_paths: tuple[str, ...] = (),
) -> NativeScalarSamples:
    """Sample physical native outputs without callbacks, resets or observations.

    Admission and complete initial-state reconstruction belong to the calling
    engine profile. This internal executor owns the only integration loop.
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
    sampler = _contact_sampler(model, contact_force_paths)
    manager = osim.Manager(model)
    manager.setIntegratorMethod(osim.Manager.IntegratorMethod_RungeKuttaMerson)
    manager.setIntegratorAccuracy(float(accuracy))
    manager.setWriteToStorage(False)
    manager.initialize(state)
    states: NDArray[np.float64] = np.empty(
        (len(grid), len(state_names)), dtype=np.float64
    )
    forces: NDArray[np.float64] = np.empty(
        (len(grid), len(actuators)), dtype=np.float64
    )
    controls = np.empty_like(forces)
    powers = np.empty_like(forces)
    contacts = []
    for row, time in enumerate(grid):
        if row:
            state = manager.integrate(float(time))
        if float(state.getTime()) != float(time):
            raise RuntimeError(
                "native replay time differs from requested physical time"
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
        contacts.append(_sample_contacts(sampler, state, contact_force_paths))
    if not all(
        np.isfinite(array).all() for array in (states, forces, controls, powers)
    ):
        raise RuntimeError("native scalar replay produced nonfinite physical output")
    return NativeScalarSamples(states, forces, controls, powers, tuple(contacts))
