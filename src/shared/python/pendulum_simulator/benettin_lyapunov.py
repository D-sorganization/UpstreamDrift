r"""Benettin largest Lyapunov exponent check and receipt for dynamical systems.

Implements the classical renormalization algorithm of Benettin et al. (1980)
for estimating the largest Lyapunov exponent (LLE, \lambda_1) of continuous
dynamical systems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.shared.python.pendulum_simulator.triple_pendulum_spec import (
    TRIPLE_PENDULUM_SPEC,
)


@dataclass(frozen=True)
class LyapunovRecord:
    """Snapshot of estimated LLE at a given integration horizon."""

    horizon_s: float
    exponent: float


@dataclass(frozen=True)
class BenettinReceipt:
    """Design-by-Contract verification receipt for Lyapunov exponent estimation."""

    horizon_s: float
    renormalization_interval_s: float
    initial_perturbation_norm: float
    largest_lyapunov_exponent: float
    convergence_history: list[LyapunovRecord] = field(default_factory=list)
    system_name: str = "generic"


def _rk4_step(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    h: float,
) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _validate_benettin_inputs(
    horizon_s: float,
    renormalization_interval_s: float,
    dt: float,
    d0: float,
    transient_horizon_s: float,
) -> None:
    if not math.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError(f"horizon_s must be finite and positive, got {horizon_s}")
    if (
        not math.isfinite(renormalization_interval_s)
        or renormalization_interval_s <= 0.0
    ):
        raise ValueError(
            f"renormalization_interval_s must be finite and positive, got {renormalization_interval_s}"
        )
    if not math.isfinite(dt) or dt <= 0.0 or dt > renormalization_interval_s:
        raise ValueError(
            f"dt must be positive and <= renormalization_interval_s, got dt={dt}"
        )
    if not math.isfinite(d0) or d0 <= 0.0:
        raise ValueError(f"d0 must be finite and positive, got {d0}")
    if not math.isfinite(transient_horizon_s) or transient_horizon_s < 0.0:
        raise ValueError(
            f"transient_horizon_s must be non-negative, got {transient_horizon_s}"
        )


def _integrate_benettin(
    step_fn: Callable[[float, np.ndarray, float], np.ndarray],
    x_init: np.ndarray,
    y_init: np.ndarray,
    horizon_s: float,
    tau: float,
    dt: float,
    d0: float,
) -> tuple[float, list[LyapunovRecord], float]:
    n_intervals = int(math.floor(horizon_s / tau))
    if n_intervals == 0:
        raise ValueError("horizon_s is too short for renormalization_interval_s")

    sub_steps = max(1, int(round(tau / dt)))
    actual_dt = tau / sub_steps

    accumulated_log_ratio = 0.0
    convergence_history: list[LyapunovRecord] = []
    current_time = 0.0
    x = x_init
    y = y_init

    for k in range(1, n_intervals + 1):
        for _ in range(sub_steps):
            x = step_fn(current_time, x, actual_dt)
            y = step_fn(current_time, y, actual_dt)
            current_time += actual_dt

        diff = y - x
        d_k = float(np.linalg.norm(diff))
        if not math.isfinite(d_k) or d_k <= 0.0:
            d_k = 1e-15

        accumulated_log_ratio += math.log(d_k / d0)
        current_horizon = k * tau
        lle_k = accumulated_log_ratio / current_horizon
        convergence_history.append(
            LyapunovRecord(horizon_s=current_horizon, exponent=lle_k)
        )
        y = x + (d0 / d_k) * diff

    return current_time, convergence_history, accumulated_log_ratio


def compute_benettin_lle(
    initial_state: np.ndarray | tuple[float, ...],
    horizon_s: float,
    *,
    renormalization_interval_s: float = 0.05,
    dt: float = 0.001,
    d0: float = 1e-8,
    transient_horizon_s: float = 0.0,
    step_fn: Callable[[float, np.ndarray, float], np.ndarray] | None = None,
    deriv_fn: Callable[[float, np.ndarray], np.ndarray] | None = None,
) -> BenettinReceipt:
    """Compute the Largest Lyapunov Exponent using Benettin renormalization."""
    _validate_benettin_inputs(
        horizon_s, renormalization_interval_s, dt, d0, transient_horizon_s
    )
    if step_fn is None and deriv_fn is None:
        raise ValueError("Either step_fn or deriv_fn must be provided")

    x = np.array(initial_state, dtype=float)
    dim = len(x)
    if dim == 0:
        raise ValueError("initial_state must not be empty")

    if step_fn is None:
        assert deriv_fn is not None
        df = deriv_fn

        def _default_step(t: float, y_val: np.ndarray, h: float) -> np.ndarray:
            return _rk4_step(df, t, y_val, h)

        step_fn = _default_step

    current_time = 0.0
    if transient_horizon_s > 0.0:
        n_transient_steps = int(round(transient_horizon_s / dt))
        transient_dt = transient_horizon_s / max(1, n_transient_steps)
        for _ in range(n_transient_steps):
            x = step_fn(current_time, x, transient_dt)
            current_time += transient_dt

    p_dir = np.ones(dim, dtype=float) / math.sqrt(dim)
    y = x + d0 * p_dir

    elapsed, history, _ = _integrate_benettin(
        step_fn, x, y, horizon_s, renormalization_interval_s, dt, d0
    )
    final_lle = history[-1].exponent if history else 0.0

    return BenettinReceipt(
        horizon_s=elapsed,
        renormalization_interval_s=renormalization_interval_s,
        initial_perturbation_norm=d0,
        largest_lyapunov_exponent=final_lle,
        convergence_history=history,
    )


def _build_triple_pendulum_stepper(
    dynamics: object,
) -> Callable[[float, np.ndarray, float], np.ndarray]:
    from src.engines.pendulum_models.python.double_pendulum_model.physics.triple_pendulum import (
        TriplePendulumDynamics,
        TriplePendulumState,
    )

    dyn = dynamics if isinstance(dynamics, TriplePendulumDynamics) else None
    assert dyn is not None
    zero_control = (0.0, 0.0, 0.0)

    def _step(t: float, state_vec: np.ndarray, step_dt: float) -> np.ndarray:
        st = TriplePendulumState(
            theta1=float(state_vec[0]),
            theta2=float(state_vec[1]),
            theta3=float(state_vec[2]),
            omega1=float(state_vec[3]),
            omega2=float(state_vec[4]),
            omega3=float(state_vec[5]),
        )
        next_st = dyn.step(t, st, step_dt, zero_control)
        return np.array(
            [
                next_st.theta1,
                next_st.theta2,
                next_st.theta3,
                next_st.omega1,
                next_st.omega2,
                next_st.omega3,
            ],
            dtype=float,
        )

    return _step


def compute_canonical_triple_pendulum_lle(
    initial_angles: tuple[float, float, float] = (2.0, 2.0, 2.0),
    horizon_s: float = 30.0,
    *,
    renormalization_interval_s: float = 0.1,
    dt: float = 0.002,
    d0: float = 1e-8,
) -> BenettinReceipt:
    """Compute Benettin LLE for canonical unforced, undamped 3DOF triple pendulum."""
    from src.engines.pendulum_models.python.double_pendulum_model.physics.triple_pendulum import (
        TriplePendulumDynamics,
        TriplePendulumParameters,
        TripleSegmentProperties,
    )

    segments = tuple(
        TripleSegmentProperties(
            length_m=seg.length_m,
            mass_kg=seg.mass_kg,
            center_of_mass_ratio=seg.center_of_mass_ratio,
            inertia_about_com=seg.inertia_about_com,
        )
        for seg in TRIPLE_PENDULUM_SPEC.segments
    )
    params = TriplePendulumParameters(
        segments=(segments[0], segments[1], segments[2]),
        damping=(0.0, 0.0, 0.0),
        gravity_enabled=True,
        gravity_m_s2=TRIPLE_PENDULUM_SPEC.gravity_m_s2,
    )
    dynamics = TriplePendulumDynamics(params)
    initial_vec = np.array(
        [
            initial_angles[0],
            initial_angles[1],
            initial_angles[2],
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )
    stepper = _build_triple_pendulum_stepper(dynamics)
    return compute_benettin_lle(
        initial_state=initial_vec,
        horizon_s=horizon_s,
        renormalization_interval_s=renormalization_interval_s,
        dt=dt,
        d0=d0,
        step_fn=stepper,
    )
