"""Benettin largest Lyapunov exponent (LLE) calculation for dynamic systems.

Implements the classical algorithm of Benettin et al. (1980) for estimating
the largest Lyapunov exponent (LLE) of continuous dynamical systems via
periodic renormalization of a shadow perturbation trajectory.

Interpretation Note: Finite-Time vs. Asymptotic Exponents
---------------------------------------------------------
In short-duration observations (such as a 0.35--0.60 s downswing), finite-time
Lyapunov exponents (FTLE) or singular values of the state transition matrix
measure transient kinematic and centrifugal amplification. Because a golf swing
is an unconstrained or driven accelerating whip, local perturbation amplification
rates (e.g. sigma_max ~ 124, FTLE ~ 8.3 /s) arise primarily from rapid proximal
deceleration transferring momentum into distal segments, rather than asymptotic
chaos. True asymptotic chaos requires recurrence over long horizons (e.g. 30 s),
where the Benettin renormalization measures whether nearby trajectories diverge
exponentially on average without saturation. For the free, unforced, undamped
triple pendulum, high-energy states (e.g. initial angles at 2.0 rad) yield a
persistent positive LLE (~2.6 /s), whereas low-energy states (e.g. 0.2 rad) exhibit
decaying LLE (~0.15 /s at 30 s), characteristic of regular quasi-periodic motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.engines.pendulum_models.python.double_pendulum_model.physics.triple_pendulum import (
    TriplePendulumDynamics,
    TriplePendulumParameters,
    TriplePendulumState,
    TripleSegmentProperties,
)
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
    """Convergence receipt for Benettin largest Lyapunov exponent estimation."""

    system_name: str
    horizon_s: float
    renormalization_interval_s: float
    initial_perturbation_norm: float
    largest_lyapunov_exponent: float
    convergence_history: list[LyapunovRecord] = field(default_factory=list)


def _rk4_step(
    deriv_fn: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    h: float,
) -> np.ndarray:
    """Advance ODE state vector by one fourth-order Runge-Kutta step."""
    k1 = deriv_fn(t, y)
    k2 = deriv_fn(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = deriv_fn(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = deriv_fn(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


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
    perturbation_direction: np.ndarray | None = None,
    system_name: str = "generic",
) -> BenettinReceipt:
    """Compute the Largest Lyapunov Exponent using the Benettin renormalization method.

    Parameters
    ----------
    initial_state : array-like
        Initial state vector x_0.
    horizon_s : float
        Total duration of integration in seconds (must be > 0).
    renormalization_interval_s : float
        Interval tau between renormalizations (must be > 0 and <= horizon_s).
    dt : float
        Integration sub-step size in seconds (must be > 0 and <= renormalization_interval_s).
    d0 : float
        Magnitude of the perturbation vector (must be > 0).
    step_fn : callable, optional
        Custom stepper with signature step_fn(t, state, dt) -> next_state.
    deriv_fn : callable, optional
        Derivative function with signature deriv_fn(t, state) -> dstate/dt.
    perturbation_direction : array-like, optional
        Unit direction vector for initial displacement. Defaults to normalized [1, 1, ...].
    system_name : str
        Human-readable descriptor for receipt.

    Returns
    -------
    BenettinReceipt
        Receipt containing estimated LLE and step-by-step convergence history.
    """
    # Design by Contract validations
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

    if step_fn is None and deriv_fn is None:
        raise ValueError("Either step_fn or deriv_fn must be provided")

    x = np.array(initial_state, dtype=float)
    dim = len(x)
    if dim == 0:
        raise ValueError("initial_state must not be empty")

    if step_fn is None:
        assert deriv_fn is not None
        df = deriv_fn

        def _default_step(t: float, y: np.ndarray, h: float) -> np.ndarray:
            return _rk4_step(df, t, y, h)

        step_fn = _default_step

    # Discard initial transient if requested to let state settle on attractor
    current_time = 0.0
    if transient_horizon_s > 0.0:
        n_transient_steps = int(round(transient_horizon_s / dt))
        transient_dt = transient_horizon_s / max(1, n_transient_steps)
        for _ in range(n_transient_steps):
            x = step_fn(current_time, x, transient_dt)
            current_time += transient_dt

    # Initial perturbation vector
    if perturbation_direction is not None:
        p_dir = np.array(perturbation_direction, dtype=float)
        p_norm = float(np.linalg.norm(p_dir))
        if p_norm == 0.0 or not math.isfinite(p_norm):
            raise ValueError("perturbation_direction must have non-zero finite norm")
        p_dir = p_dir / p_norm
    else:
        p_dir = np.ones(dim, dtype=float) / math.sqrt(dim)

    y = x + d0 * p_dir

    # Renormalization loop
    n_intervals = int(math.floor(horizon_s / renormalization_interval_s))
    if n_intervals == 0:
        raise ValueError("horizon_s is too short for renormalization_interval_s")

    sub_steps = max(1, int(round(renormalization_interval_s / dt)))
    actual_dt = renormalization_interval_s / sub_steps

    accumulated_log_ratio = 0.0
    convergence_history: list[LyapunovRecord] = []
    current_time = 0.0

    for k in range(1, n_intervals + 1):
        # Integrate both fiducial and perturbed trajectories over tau
        for _ in range(sub_steps):
            x = step_fn(current_time, x, actual_dt)
            y = step_fn(current_time, y, actual_dt)
            current_time += actual_dt

        # Measure displacement
        diff = y - x
        d_k = float(np.linalg.norm(diff))

        if not math.isfinite(d_k) or d_k <= 0.0:
            d_k = 1e-15  # Numerical floor preventing log domain error

        # Accumulate expansion rate
        accumulated_log_ratio += math.log(d_k / d0)
        current_horizon = k * renormalization_interval_s
        lle_k = accumulated_log_ratio / current_horizon

        convergence_history.append(
            LyapunovRecord(horizon_s=current_horizon, exponent=lle_k)
        )

        # Renormalize shadow trajectory back to distance d0 along the same direction
        y = x + (d0 / d_k) * diff

    final_lle = convergence_history[-1].exponent if convergence_history else 0.0

    return BenettinReceipt(
        system_name=system_name,
        horizon_s=current_time,
        renormalization_interval_s=renormalization_interval_s,
        initial_perturbation_norm=d0,
        largest_lyapunov_exponent=final_lle,
        convergence_history=convergence_history,
    )


def compute_canonical_triple_pendulum_lle(
    initial_angles: tuple[float, float, float] = (2.0, 2.0, 2.0),
    horizon_s: float = 30.0,
    *,
    renormalization_interval_s: float = 0.1,
    dt: float = 0.002,
    d0: float = 1e-8,
) -> BenettinReceipt:
    """Compute Benettin largest Lyapunov exponent for the canonical unforced, undamped 3DOF triple pendulum.

    Parameters
    ----------
    initial_angles : tuple of 3 floats
        Initial joint angles (theta1, theta2, theta3) in radians. Initial velocities default to 0.
    horizon_s : float
        Total duration in seconds.
    renormalization_interval_s : float
        Interval between shadow renormalizations.
    dt : float
        Integration sub-step size in seconds.
    d0 : float
        Perturbation displacement magnitude.

    Returns
    -------
    BenettinReceipt
        Receipt with estimated LLE and convergence history.
    """
    # Build unforced, undamped triple pendulum from canonical spec
    segments = tuple(
        TripleSegmentProperties(
            length_m=seg.length_m,
            mass_kg=seg.mass_kg,
            center_of_mass_ratio=seg.center_of_mass_ratio,
            inertia_about_com=seg.inertia_about_com,
        )
        for seg in TRIPLE_PENDULUM_SPEC.segments
    )
    undamped_params = TriplePendulumParameters(
        segments=(segments[0], segments[1], segments[2]),
        damping=(0.0, 0.0, 0.0),
        gravity_enabled=True,
        gravity_m_s2=TRIPLE_PENDULUM_SPEC.gravity_m_s2,
    )
    dynamics = TriplePendulumDynamics(undamped_params)

    # State vector: [theta1, theta2, theta3, omega1, omega2, omega3]
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

    zero_control = (0.0, 0.0, 0.0)

    def _triple_pendulum_step(
        t: float, state_vec: np.ndarray, step_dt: float
    ) -> np.ndarray:
        st = TriplePendulumState(
            theta1=float(state_vec[0]),
            theta2=float(state_vec[1]),
            theta3=float(state_vec[2]),
            omega1=float(state_vec[3]),
            omega2=float(state_vec[4]),
            omega3=float(state_vec[5]),
        )
        next_st = dynamics.step(t, st, step_dt, zero_control)
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

    return compute_benettin_lle(
        initial_state=initial_vec,
        horizon_s=horizon_s,
        renormalization_interval_s=renormalization_interval_s,
        dt=dt,
        d0=d0,
        step_fn=_triple_pendulum_step,
        system_name=f"canonical_triple_pendulum_angles_{initial_angles}",
    )
