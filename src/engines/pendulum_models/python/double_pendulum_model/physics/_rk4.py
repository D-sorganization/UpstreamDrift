"""Shared classical Runge-Kutta (RK4) integrator for the pendulum models.

Both :mod:`double_pendulum` and :mod:`triple_pendulum` previously
reimplemented the same fixed-step RK4 update (the four stage evaluations
plus the ``y + dt/6 * (k1 + 2 k2 + 2 k3 + k4)`` combination) inline in
their ``step`` methods. This module extracts a single generic
:func:`rk4_step` that works on a flat tuple of floats so both models share
one implementation.

The arithmetic is intentionally written element-wise and in the same order
as the previous inline code so the integrator is *bit-identical* to the
hand-written versions it replaces.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

# A state vector is a flat tuple of floats; the derivatives function maps a
# time and a state vector to the corresponding derivative vector.
StateVector = tuple[float, ...]
DerivativesFn = Callable[[float, "StateVector"], Sequence[float]]


def _increment(
    state: Sequence[float], scale: float, derivs: Sequence[float]
) -> StateVector:
    """Return ``state + scale * derivs`` element-wise."""
    return tuple(s + scale * d for s, d in zip(state, derivs, strict=True))


def rk4_step(
    derivatives_fn: DerivativesFn,
    state_vector: Sequence[float],
    t: float,
    dt: float,
) -> StateVector:
    """Advance ``state_vector`` by one classical RK4 step.

    Args:
        derivatives_fn: ``f(t, y)`` returning the derivative vector for the
            state vector ``y`` at time ``t``. The returned sequence must have
            the same length as ``state_vector``.
        state_vector: Current state as a flat sequence of floats.
        t: Current time (passed through to ``derivatives_fn``).
        dt: Integration step.

    Returns:
        The new state vector as a tuple of floats.

    Postconditions:
        The result has the same length as ``state_vector`` and is computed
        with the same per-component arithmetic as a hand-written RK4 update.
    """
    y = tuple(float(v) for v in state_vector)

    k1 = derivatives_fn(t, y)
    k2 = derivatives_fn(t + dt / 2.0, _increment(y, dt / 2.0, k1))
    k3 = derivatives_fn(t + dt / 2.0, _increment(y, dt / 2.0, k2))
    k4 = derivatives_fn(t + dt, _increment(y, dt, k3))

    return tuple(
        y[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(len(y))
    )
