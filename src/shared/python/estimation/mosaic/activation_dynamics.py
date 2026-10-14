"""Declared joint-level activation dynamics (MOSAIC actuation layer).

The applied joint torque follows the commanded torque through a *declared*
first-order lag with separate activation/deactivation time constants, hard
bounds and an optional slew-rate limit.  The model is declared (with
provenance), not identified from muscle data.

Continuous model per joint (``u`` clipped to ``[lower, upper]``)::

    da/dt = (u - a) / tau,   tau = tau_act if u > a else tau_deact

Each interval uses a zero-order-hold command, so the update is the exact
solution ``a+ = u + (a - u) exp(-dt / tau)``; a step response therefore equals
``1 - exp(-t / tau)`` to machine precision.  The optional rate limit clamps
``|a+ - a| <= rate_limit * dt`` afterwards.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import ensure, require
from src.shared.python.estimation.mosaic.local_policy import StepFunction, replay

if TYPE_CHECKING:
    from src.shared.python.estimation.mosaic.outer_solve import TrialData

FloatArray: TypeAlias = npt.NDArray[np.float64]
ArrayLike: TypeAlias = FloatArray | float | Sequence[float]
StepFactory: TypeAlias = Callable[[FloatArray, FloatArray, float], StepFunction]

_EPS = 1e-9


def _per_joint(value: object) -> FloatArray:
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
    require(arr.ndim == 1 and arr.size >= 1, "parameter must be scalar or (nj,)")
    require(bool(np.all(np.isfinite(arr))), "parameter must be finite", arr)
    return arr


@dataclass(frozen=True)
class ActivationModel:
    """Declared per-joint activation model (scalars broadcast to all joints)."""

    tau_act: FloatArray
    tau_deact: FloatArray
    lower: FloatArray
    upper: FloatArray
    rate_limit: FloatArray | None = None
    provenance: str = "declared"

    def __init__(
        self,
        tau_act: ArrayLike,
        tau_deact: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rate_limit: ArrayLike | None = None,
        provenance: str = "declared",
    ) -> None:
        fields = {
            "tau_act": _per_joint(tau_act),
            "tau_deact": _per_joint(tau_deact),
            "lower": _per_joint(lower),
            "upper": _per_joint(upper),
        }
        n_joints = max(v.size for v in fields.values())
        for name, val in fields.items():
            require(val.size in (1, n_joints), f"{name} size mismatch", val.size)
            object.__setattr__(self, name, np.broadcast_to(val, (n_joints,)).copy())
        if rate_limit is not None:
            rate = _per_joint(rate_limit)
            require(rate.size in (1, n_joints), "rate_limit size mismatch", rate.size)
            object.__setattr__(
                self, "rate_limit", np.broadcast_to(rate, (n_joints,)).copy()
            )
            require(bool(np.all(rate > 0.0)), "rate_limit must be > 0", rate)
        else:
            object.__setattr__(self, "rate_limit", None)
        object.__setattr__(self, "provenance", provenance)
        require(bool(np.all(self.tau_act > 0.0)), "tau_act must be > 0")
        require(bool(np.all(self.tau_deact > 0.0)), "tau_deact must be > 0")
        require(bool(np.all(self.lower < self.upper)), "lower must be < upper")

    @property
    def n_joints(self) -> int:
        """Number of joints the model is declared for."""
        return int(self.tau_act.size)


_REQUIRED = ("tau_act", "tau_deact", "lower", "upper")


@dataclass(frozen=True)
class TimeConstantFit:
    """Recovered time constants; ``None`` means unobservable from the data."""

    tau_act: float | None
    tau_deact: float | None
    n_act_steps: int
    n_deact_steps: int


@dataclass(frozen=True)
class ActivationReport:
    """Declared activation model and side-by-side open-loop replay results."""

    model: ActivationModel
    provenance: str
    time_constants_fit: tuple[TimeConstantFit, ...]
    open_loop_rms_without_actuation: tuple[float, ...]
    open_loop_rms_with_actuation: tuple[float, ...]


def _check_times(times: FloatArray) -> None:
    require(times.ndim == 1 and times.size >= 2, "times must be 1-D, >= 2 samples")
    require(bool(np.all(np.isfinite(times))), "times must be finite")
    require(bool(np.all(np.diff(times) > 0.0)), "times must be strictly increasing")


def simulate_activation(
    model: ActivationModel,
    times: FloatArray,
    commands: FloatArray,
    a0: FloatArray | None = None,
) -> FloatArray:
    """Integrate the applied activation over ``times`` for ``(N, nj)`` commands.

    Preconditions: finite strictly increasing ``times``; finite ``commands``
    with one row per time and ``model.n_joints`` columns; ``a0`` (default: the
    clipped first command) finite and within bounds.
    Postcondition: the result is ``(N, nj)``, finite and within
    ``[lower, upper]``.
    """
    _check_times(times)
    require(
        commands.shape == (times.size, model.n_joints),
        "commands must be (N, nj)",
        commands.shape,
    )
    require(bool(np.all(np.isfinite(commands))), "commands must be finite")
    u = np.clip(commands, model.lower, model.upper)
    state = u[0].copy() if a0 is None else np.asarray(a0, dtype=np.float64).copy()
    require(state.shape == (model.n_joints,), "a0 must be (nj,)", state.shape)
    in_bounds = bool(np.all((state >= model.lower) & (state <= model.upper)))
    require(in_bounds, "a0 must lie within bounds", state)
    out = np.empty_like(u)
    out[0] = state
    for k in range(times.size - 1):
        dt = float(times[k + 1] - times[k])
        tau = np.where(u[k] > state, model.tau_act, model.tau_deact)
        nxt = u[k] + (state - u[k]) * np.exp(-dt / tau)
        if model.rate_limit is not None:
            step = model.rate_limit * dt
            nxt = state + np.clip(nxt - state, -step, step)
        state = np.clip(nxt, model.lower, model.upper)
        out[k + 1] = state
    ensure(bool(np.all(np.isfinite(out))), "activation must be finite")
    ensure(
        bool(np.all((out >= model.lower - _EPS) & (out <= model.upper + _EPS))),
        "activation must respect bounds",
    )
    return out


def _decay_tau(
    d0: FloatArray, d1: FloatArray, dt: float, min_steps: int
) -> float | None:
    """Least-squares ``tau`` from ``d1 = exp(-dt/tau) d0``; ``None`` if ill-posed."""
    if d0.size < min_steps:
        return None
    denom = float(np.dot(d0, d0))
    if denom <= _EPS:
        return None
    phi = float(np.dot(d0, d1)) / denom
    if not 0.0 < phi < 1.0:
        return None
    return -dt / float(np.log(phi))


def fit_time_constants(
    times: FloatArray,
    command: FloatArray,
    activation: FloatArray,
    *,
    lower: float,
    upper: float,
    min_steps: int = 3,
) -> TimeConstantFit:
    """Recover ``(tau_act, tau_deact)`` of one joint from a measured response.

    Uses the exact per-step relation ``a+ - u = exp(-dt/tau) (a - u)`` (clipped
    ``u``), separately over rising (``u > a``) and falling steps.  Assumes no
    rate limit is active.  A constant is reported as ``None`` (unobservable)
    when fewer than ``min_steps`` informative steps exist or the estimated
    decay factor is not in ``(0, 1)``.  Preconditions: uniform increasing
    ``times``; equal-length finite 1-D series; ``lower < upper``.
    """
    _check_times(times)
    require(lower < upper, "lower must be < upper")
    require(min_steps >= 1, "min_steps must be >= 1")
    require(command.shape == times.shape == activation.shape, "series length mismatch")
    require(bool(np.all(np.isfinite(command) & np.isfinite(activation))), "finite data")
    steps = np.diff(times)
    dt = float(steps.mean())
    require(bool(np.allclose(steps, dt, rtol=1e-6)), "times must be uniform", dt)
    u = np.clip(command, lower, upper)[:-1]
    d0 = activation[:-1] - u
    d1 = activation[1:] - u
    rising = d0 < -_EPS
    falling = d0 > _EPS
    return TimeConstantFit(
        tau_act=_decay_tau(d0[rising], d1[rising], dt, min_steps),
        tau_deact=_decay_tau(d0[falling], d1[falling], dt, min_steps),
        n_act_steps=int(rising.sum()),
        n_deact_steps=int(falling.sum()),
    )


def evaluate_actuation_replays(
    model: ActivationModel,
    step_factory: StepFactory,
    geometry: FloatArray,
    parameters: FloatArray,
    trials: Sequence[TrialData],
    coefficients: Sequence[FloatArray],
    inputs: FloatArray,
    initial_perturbation: FloatArray | None = None,
) -> ActivationReport:
    """Compare open-loop rollouts with and without the declared actuation layer.

    Evaluates side-by-side open-loop RMS error against kinematic reference,
    and estimates time constants from the simulated response.
    """
    require(len(trials) >= 1, "at least one trial required")
    require(len(trials) == len(coefficients), "mismatched trials and coefficients")

    rms_without: list[float] = []
    rms_with: list[float] = []
    time_fits: list[TimeConstantFit] = []

    offset = 0
    for trial, coeff in zip(trials, coefficients, strict=True):
        trial_inputs = inputs[offset : offset + trial.n_times]
        offset += trial.n_times

        steps = np.diff(trial.times)
        dt = float(steps[0])

        step = step_factory(geometry, parameters, dt)
        q, v, _ = trial.basis.evaluate(coeff)
        reference = np.concatenate([q, v], axis=1)
        n_q = reference.shape[1] // 2

        start = (
            reference[0]
            if initial_perturbation is None
            else reference[0] + initial_perturbation
        )

        nominal_without = trial_inputs[:-1]
        traj_without = replay(step, start, nominal_without)
        err_without = np.linalg.norm(traj_without[:, :n_q] - reference[:, :n_q], axis=1)
        rms_without.append(float(np.sqrt(np.mean(err_without**2))))

        act_inputs = simulate_activation(model, trial.times, trial_inputs)
        nominal_with = act_inputs[:-1]
        traj_with = replay(step, start, nominal_with)
        err_with = np.linalg.norm(traj_with[:, :n_q] - reference[:, :n_q], axis=1)
        rms_with.append(float(np.sqrt(np.mean(err_with**2))))

        if not time_fits:
            for j in range(model.n_joints):
                fit_j = fit_time_constants(
                    trial.times,
                    trial_inputs[:, j],
                    act_inputs[:, j],
                    lower=float(model.lower[j]),
                    upper=float(model.upper[j]),
                )
                time_fits.append(fit_j)

    report = ActivationReport(
        model=model,
        provenance=model.provenance,
        time_constants_fit=tuple(time_fits),
        open_loop_rms_without_actuation=tuple(rms_without),
        open_loop_rms_with_actuation=tuple(rms_with),
    )
    ensure(
        len(report.open_loop_rms_without_actuation) == len(trials),
        "without-actuation RMS count mismatch",
    )
    ensure(
        len(report.open_loop_rms_with_actuation) == len(trials),
        "with-actuation RMS count mismatch",
    )
    return report
