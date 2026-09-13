"""Fourth-order forward integration in rebased local configuration charts.

Configuration dimension nq and tangent dimension nv are independent. Each step
integrates chart displacement u and tangent velocity v with classical RK4,
then changes the chart anchor without changing the physical state. There are no
constraint projections, target states, feedback, or intermediate state resets.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Acceleration = Callable[[float, Array, Array], Array]
Retraction = Callable[[Array, Array], Array]
DifferenceRate = Callable[[Array, Array, Array], Array]


class ManifoldForwardResult(NamedTuple):
    """Owned read-only samples, with explicit configuration/tangent dimensions."""

    time: Array
    configuration: Array
    velocity: Array
    evaluations: int
    steps: int
    elapsed_s: float


def _vector(value: Array, size: int, name: str) -> Array:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must return a finite vector of length {size}")
    result.setflags(write=False)
    return result


def _initial_data(
    initial_configuration: Array, initial_velocity: Array, time: Array, max_step: float
) -> tuple[Array, Array, Array]:
    initial_q = np.array(initial_configuration, dtype=float, copy=True)
    initial_v = np.array(initial_velocity, dtype=float, copy=True)
    for name, value in (
        ("initial_configuration", initial_q),
        ("initial_velocity", initial_v),
    ):
        if value.ndim != 1 or not value.size or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a nonempty finite vector")
    clock = np.array(time, dtype=float, copy=True)
    if (
        clock.ndim != 1
        or clock.size < 2
        or not np.isfinite(clock).all()
        or clock[0] != 0
        or np.any(np.diff(clock) <= 0)
    ):
        raise ValueError("time must be finite and strictly increasing from zero")
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be finite and positive")
    return initial_q, initial_v, clock


@dataclass(frozen=True)
class _ChartStepper:
    """One RK4 kernel shared by fixed stepping and adaptive step doubling."""

    nq: int
    nv: int
    acceleration: Acceleration
    integrate: Retraction
    difference_rate: DifferenceRate

    def step(self, q: Array, v: Array, start: float, end: float) -> tuple[Array, Array]:
        h = end - start
        midpoint = start + h / 2
        if not start < midpoint < end:
            raise RuntimeError(f"Manifold integration step underflow at t={start:.17g}")

        def derivative(t: float, u: Array, tangent: Array) -> tuple[Array, Array]:
            displacement = _vector(u, self.nv, "local displacement")
            rate = _vector(tangent, self.nv, "stage velocity")
            state = _vector(self.integrate(q, displacement), self.nq, "integrate")
            udot = _vector(
                self.difference_rate(q, state, rate), self.nv, "difference_rate"
            )
            vdot = _vector(self.acceleration(t, state, rate), self.nv, "acceleration")
            return udot, vdot

        k1u, k1v = derivative(start, np.zeros(self.nv), v)
        k2u, k2v = derivative(midpoint, h / 2 * k1u, v + h / 2 * k1v)
        k3u, k3v = derivative(midpoint, h / 2 * k2u, v + h / 2 * k2v)
        k4u, k4v = derivative(end, h * k3u, v + h * k3v)
        u = h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        next_q = _vector(self.integrate(q, u), self.nq, "integrate")
        next_v = _vector(
            v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v), self.nv, "velocity"
        )
        return next_q, next_v


def _result(
    clock: Array, q: Array, v: Array, evaluations: int, steps: int, start: float
) -> ManifoldForwardResult:
    for value in (clock, q, v):
        value.setflags(write=False)
    return ManifoldForwardResult(
        clock, q, v, evaluations, steps, perf_counter() - start
    )


def integrate_manifold_forward(
    initial_configuration: Array,
    initial_velocity: Array,
    time: Array,
    acceleration: Acceleration,
    *,
    integrate: Retraction,
    difference_rate: DifferenceRate,
    max_step: float = 0.001,
) -> ManifoldForwardResult:
    """Integrate one initial state, landing exactly at all requested sample times.

    `integrate(anchor,u)` retracts an nv displacement into nq configuration.
    `difference_rate(anchor,q,v)` computes the time derivative of
    difference(anchor,q) when q's instantaneous tangent velocity is v. It is
    generally NOT equal to v. Adapters own chart validity and branch rejection.
    Acceleration receives absolute physical time, current q and tangent v;
    its output is the derivative of that velocity convention. Callback inputs
    are read-only, outputs must be finite with the exact stated dimensions.

    This fixed-step solver has no adaptive error estimate. Acceptance requires
    step refinement on the actual problem, including constraints and effort
    conversion. max_step bounds every numerical step; sample boundaries may
    make individual steps shorter. Failure raises instead of returning a partial
    trajectory as successful. It does not normalize/project engine state.
    """
    initial_q, initial_v, clock = _initial_data(
        initial_configuration, initial_velocity, time, max_step
    )
    nq, nv = initial_q.size, initial_v.size
    q_samples, v_samples = np.empty((clock.size, nq)), np.empty((clock.size, nv))
    q_samples[0], v_samples[0] = initial_q, initial_v
    q, v = _vector(initial_q, nq, "initial_configuration"), initial_v
    steps = 0
    stepper = _ChartStepper(nq, nv, acceleration, integrate, difference_rate)
    start = perf_counter()
    for sample in range(1, clock.size):
        t0, t1 = float(clock[sample - 1]), float(clock[sample])
        quotient = (t1 - t0) / max_step
        if not np.isfinite(quotient):
            raise ValueError("max_step is too small for the sample interval")
        count = max(1, ceil(quotient))
        h = (t1 - t0) / count
        for step in range(count):
            physical_time = t0 + step * h
            endpoint = t1 if step == count - 1 else t0 + (step + 1) * h
            q, v = stepper.step(q, v, physical_time, endpoint)
            steps += 1
        q_samples[sample], v_samples[sample] = q, v
    return _result(clock, q_samples, v_samples, 4 * steps, steps, start)


def integrate_manifold_adaptive(
    initial_configuration: Array,
    initial_velocity: Array,
    time: Array,
    acceleration: Acceleration,
    *,
    integrate: Retraction,
    difference_rate: DifferenceRate,
    difference: Retraction,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = 0.001,
    max_evaluations: int = 1_000_000,
) -> ManifoldForwardResult:
    """Adaptive local-chart RK4 with step doubling and no physical projections.

    The accepted state is the two-half-step result, without extrapolation.
    Estimated configuration error is difference(q_full,q_fine)/15 in tangent
    units; velocity error is (v_fine-v_full)/15 in the declared component
    convention. Configuration scales are atol + rtol*max(abs(h*v_start),
    abs(h*v_fine)); velocity scales are atol + rtol*max(abs(v_start),abs(v_fine)).
    Thus atol applies to tangent displacement and velocity components (e.g.
    radians and rad/s or meters and m/s). Relative error uses LOCAL displacement,
    not a global configuration norm: these tolerances are not numerically
    interchangeable with Euclidean solve_ivp tolerances. No tangent transport
    or Richardson extrapolation is applied to the accepted physical state.

    The maximum componentwise scaled error controls step size with exponent
    1/5 and safety factor 0.9. evaluations includes rejected attempts; steps counts
    accepted macrosteps (each contains two half steps). Every attempt uses 12
    RHS evaluations. Budget exhaustion or representational time underflow
    raises explicitly; partial trajectories are never returned as successes.
    Callback failures propagate. Constraint/trajectory acceptance still requires
    independent replay convergence beyond the local error estimate.
    """
    q, v, clock = _initial_data(initial_configuration, initial_velocity, time, max_step)
    if any(not np.isfinite(value) or value <= 0 for value in (rtol, atol)):
        raise ValueError("rtol and atol must be finite and positive")
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, int)
        or max_evaluations <= 0
    ):
        raise ValueError("max_evaluations must be a positive integer")
    nq, nv = q.size, v.size
    stepper = _ChartStepper(nq, nv, acceleration, integrate, difference_rate)
    q_samples, v_samples = np.empty((clock.size, nq)), np.empty((clock.size, nv))
    q_samples[0], v_samples[0] = q, v
    q, v = _vector(q, nq, "initial_configuration"), _vector(v, nv, "initial_velocity")
    steps = evaluations = 0
    t, proposed_h = 0.0, max_step
    start = perf_counter()
    for sample in range(1, clock.size):
        boundary = float(clock[sample])
        while t < boundary:
            end = min(t + proposed_h, boundary)
            h = end - t
            midpoint = t + h / 2
            if not t < midpoint < end:
                raise RuntimeError(f"Manifold integration step underflow at t={t:.17g}")
            if evaluations + 12 > max_evaluations:
                raise RuntimeError(
                    f"Manifold evaluation budget exhausted at t={t:.17g}: {evaluations}/{max_evaluations}"
                )
            full_q, full_v = stepper.step(q, v, t, end)
            half_q, half_v = stepper.step(q, v, t, midpoint)
            fine_q, fine_v = stepper.step(half_q, half_v, midpoint, end)
            evaluations += 12
            q_error = _vector(difference(full_q, fine_q), nv, "difference") / 15
            v_error = (fine_v - full_v) / 15
            q_scale = atol + rtol * np.maximum(abs(h * v), abs(h * fine_v))
            v_scale = atol + rtol * np.maximum(abs(v), abs(fine_v))
            error = max(
                float(np.max(abs(q_error) / q_scale)),
                float(np.max(abs(v_error) / v_scale)),
            )
            if not np.isfinite(error):
                raise RuntimeError("Nonfinite adaptive manifold error estimate")
            if error <= 1:
                q, v, t = fine_q, fine_v, end
                steps += 1
            factor = 5.0 if error == 0 else min(5.0, max(0.1, 0.9 * error ** (-0.2)))
            proposed_h = min(max_step, h * factor)
        q_samples[sample], v_samples[sample] = q, v
    return _result(clock, q_samples, v_samples, evaluations, steps, start)
