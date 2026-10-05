"""Local feedback policy and replay acceptance for a fitted trajectory (MOSAIC).

After the joint fit, the recovered inputs ``u*`` and states ``x*`` define a
nominal open-loop motion.  Linearising the (any-engine) discrete step along
that motion gives time-varying ``(A_t, B_t)``; a backward Riccati pass yields
gains ``K_t`` so that ``u = u*_t - K_t (x - x*_t)`` is the locally optimal
linear policy around the fit.  This pair ``(u*, K)`` is the "locally learned
input": the same object a whole-body controller consumes on a humanoid.

Acceptance is honest: the *open-loop* replay from one initial state is the
test of a physically consistent fit; the closed-loop replay quantifies how
much feedback effort is needed to hold the motion, which exposes sensitivity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import ensure, require

FloatArray: TypeAlias = npt.NDArray[np.float64]
StepFunction = Callable[[FloatArray, FloatArray], FloatArray]


@dataclass(frozen=True)
class LinearizedDynamics:
    """Discrete linearisation ``x_{t+1} ~ A_t dx + B_t du`` along a trajectory."""

    a: FloatArray
    b: FloatArray

    def __post_init__(self) -> None:
        require(
            self.a.ndim == 3 and self.a.shape[1] == self.a.shape[2],
            "A must be (T, nx, nx)",
        )
        require(self.b.shape[:2] == self.a.shape[:2], "B must be (T, nx, nu)")

    @property
    def n_steps(self) -> int:
        return int(self.a.shape[0])


@dataclass(frozen=True)
class ReplayReport:
    """Measured deviation of open- and closed-loop replay from the reference."""

    open_loop_rms: float
    closed_loop_rms: float
    feedback_effort_rms: float
    open_loop_accepted: bool
    closed_loop_accepted: bool
    divergence_time_s: float | None


def linearize_along_trajectory(
    step: StepFunction, states: FloatArray, inputs: FloatArray, eps: float
) -> LinearizedDynamics:
    """Vectorized central-difference linearisation at every node simultaneously."""
    require(states.ndim == 2 and inputs.ndim == 2, "states/inputs must be 2-D")
    require(states.shape[0] == inputs.shape[0], "one input per state node")
    require(eps > 0.0, "eps must be positive", eps)

    def column(index: int, perturb_state: bool) -> FloatArray:
        if perturb_state:
            bump = np.zeros_like(states)
            bump[:, index] = eps
            return (step(states + bump, inputs) - step(states - bump, inputs)) / (
                2.0 * eps
            )
        bump = np.zeros_like(inputs)
        bump[:, index] = eps
        return (step(states, inputs + bump) - step(states, inputs - bump)) / (2.0 * eps)

    a = np.stack([column(i, True) for i in range(states.shape[1])], axis=-1)
    b = np.stack([column(i, False) for i in range(inputs.shape[1])], axis=-1)
    return LinearizedDynamics(a, b)


def tvlqr_gains(
    linearization: LinearizedDynamics,
    state_weight: FloatArray,
    input_weight: FloatArray,
    terminal_weight: FloatArray | None = None,
) -> FloatArray:
    """Backward Riccati recursion; returns ``K`` with shape ``(T, nu, nx)``.

    Postcondition: all gains are finite.
    """
    n_steps, n_x, n_u = linearization.b.shape
    require(state_weight.shape == (n_x, n_x), "state_weight must be (nx, nx)")
    require(input_weight.shape == (n_u, n_u), "input_weight must be (nu, nu)")
    value = state_weight if terminal_weight is None else terminal_weight
    gains = np.empty((n_steps, n_u, n_x), dtype=np.float64)
    for t in range(n_steps - 1, -1, -1):
        a_t, b_t = linearization.a[t], linearization.b[t]
        gain = np.linalg.solve(input_weight + b_t.T @ value @ b_t, b_t.T @ value @ a_t)
        closed = a_t - b_t @ gain
        value = state_weight + gain.T @ input_weight @ gain + closed.T @ value @ closed
        value = 0.5 * (value + value.T)
        gains[t] = gain
    ensure(bool(np.all(np.isfinite(gains))), "Riccati gains finite")
    return gains


def replay(
    step: StepFunction,
    initial_state: FloatArray,
    inputs: FloatArray,
    gains: FloatArray | None = None,
    reference: FloatArray | None = None,
) -> FloatArray:
    """Roll the fitted inputs forward; with ``gains`` apply ``u - K (x - x_ref)``."""
    require(initial_state.ndim == 1, "initial_state must be 1-D")
    if gains is not None:
        require(
            reference is not None, "closed-loop replay needs a reference trajectory"
        )
        require(gains.shape[0] == inputs.shape[0], "one gain per input node")
    states = [initial_state]
    for t in range(inputs.shape[0]):
        command = inputs[t]
        if gains is not None and reference is not None:
            command = command - gains[t] @ (states[-1] - reference[t])
        states.append(step(states[-1][None], command[None])[0])
    return np.stack(states)


def replay_acceptance(
    step: StepFunction,
    reference: FloatArray,
    inputs: FloatArray,
    gains: FloatArray | None,
    initial_perturbation: FloatArray,
    position_tolerance: float,
    dt: float,
) -> ReplayReport:
    """Compare open- and closed-loop replay from a perturbed start against ``reference``.

    Deviation is measured on the configuration half of the state.  The
    divergence time is the first instant the open-loop configuration error
    exceeds ``position_tolerance``.
    """
    require(position_tolerance > 0.0 and dt > 0.0, "tolerance and dt must be positive")
    n_q = reference.shape[1] // 2
    start = reference[0] + initial_perturbation
    open_loop = replay(step, start, inputs)
    error_open = np.linalg.norm(open_loop[:, :n_q] - reference[:, :n_q], axis=1)
    open_rms = float(np.sqrt(np.mean(error_open**2)))
    exceed = np.flatnonzero(error_open > position_tolerance)
    divergence = None if exceed.size == 0 else float(exceed[0] * dt)
    if gains is None:
        return ReplayReport(
            open_rms, open_rms, 0.0, exceed.size == 0, exceed.size == 0, divergence
        )
    closed_loop = replay(step, start, inputs, gains, reference)
    error_closed = np.linalg.norm(closed_loop[:, :n_q] - reference[:, :n_q], axis=1)
    feedback = np.einsum("tij,tj->ti", gains, closed_loop[:-1] - reference[:-1])
    return ReplayReport(
        open_rms,
        float(np.sqrt(np.mean(error_closed**2))),
        float(np.sqrt(np.mean(feedback**2))),
        exceed.size == 0,
        bool(np.all(error_closed <= position_tolerance)),
        divergence,
    )
