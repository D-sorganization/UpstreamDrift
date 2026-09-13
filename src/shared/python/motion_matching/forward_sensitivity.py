"""Integrate Euclidean forward state and parameter sensitivities without resets."""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.continuous_forward import (
    ContinuousForwardResult,
    integrate_forward,
)

Array = NDArray[np.float64]
Linearization = Callable[[float, Array], tuple[Array, Array, Array]]


class ForwardSensitivityResult(NamedTuple):
    """Physical state samples and time-by-state-by-parameter derivatives."""

    integration: ContinuousForwardResult
    state_parameter_jacobian: Array


def integrate_sensitivities(
    initial_state: Array,
    time: Array,
    linearize: Linearization,
    parameter_count: int,
    *,
    initial_sensitivity: Array | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = 0.001,
    max_evaluations: int | None = None,
) -> ForwardSensitivityResult:
    """Integrate x'=f and S'=df/dx S + df/dp using shared forward integration.

    The callback returns f, df/dx and df/dp at the same absolute time and state.
    Engine adapters must supply constraint-consistent derivatives. Adaptive error
    control includes the augmented sensitivity variables; qualify primal replay
    accuracy and runtime before using the result as an optimizer Jacobian.
    max_evaluations caps actual linearization callbacks through the shared
    integrator, including rejected steps; exhaustion never returns partial data.
    """
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count <= 0
    ):
        raise ValueError("parameter_count must be a positive integer")
    initial = np.asarray(initial_state, dtype=float)
    if initial.ndim != 1 or not initial.size or not np.isfinite(initial).all():
        raise ValueError("initial_state must be a nonempty finite vector")
    n = initial.size
    shape = (n, parameter_count)
    sensitivity = (
        np.zeros(shape)
        if initial_sensitivity is None
        else np.asarray(initial_sensitivity, dtype=float)
    )
    if sensitivity.shape != shape or not np.isfinite(sensitivity).all():
        raise ValueError("Invalid initial_sensitivity")

    def derivative(t: float, augmented: Array) -> Array:
        f, a, b = (
            np.asarray(value, dtype=float) for value in linearize(t, augmented[:n])
        )
        if (
            f.shape != (n,)
            or a.shape != (n, n)
            or b.shape != shape
            or not all(np.isfinite(value).all() for value in (f, a, b))
        ):
            raise ValueError("Invalid finite forward linearization")
        rate = a @ augmented[n:].reshape(shape) + b
        return np.concatenate((f, rate.ravel()))

    augmented = integrate_forward(
        np.concatenate((initial, sensitivity.ravel())),
        time,
        derivative,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        max_evaluations=max_evaluations,
    )
    states = augmented.state[:, :n].copy()
    derivatives = augmented.state[:, n:].reshape((-1, n, parameter_count)).copy()
    states.setflags(write=False)
    derivatives.setflags(write=False)
    result = ContinuousForwardResult(
        augmented.time, states, augmented.evaluations, augmented.elapsed_s
    )
    return ForwardSensitivityResult(result, derivatives)
