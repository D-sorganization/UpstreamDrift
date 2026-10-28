"""Shared state transition matrix (STM) sensitivity harness and parity receipts (SAP-4, #13793).

Design by Contract & DRY
-------------------------
- Pre: Positive finite difference step sizes and positive state scales (fail closed).
- Pre: Valid tolerance and epsilon with tolerance >= epsilon for agreement horizon.
- Post: SensitivityReceipt records sigma_max, FTLE, agreement horizon,
  dominant sensitivity direction, FD step, and state scale conventions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StateScales:
    """Positive scales for state coordinates."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(v) for v in self.values)
        if len(values) == 0 or any(not math.isfinite(v) or v <= 0.0 for v in values):
            raise ValueError("state scales must contain finite positive scale values")
        object.__setattr__(self, "values", values)

    @property
    def array(self) -> FloatArray:
        return np.asarray(self.values, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class FiniteTimeSpectra:
    singular_values: FloatArray
    exponents_per_s: FloatArray


def normalized_transition(
    transition: npt.ArrayLike, state_scales: StateScales
) -> FloatArray:
    matrix = np.asarray(transition, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("transition must be a nonempty finite square matrix")
    scales = state_scales.array
    if matrix.shape != (scales.size, scales.size):
        raise ValueError("transition dimension must match state scales")
    return matrix * scales[np.newaxis, :] / scales[:, np.newaxis]


def finite_time_spectra(
    transitions: npt.ArrayLike, elapsed_time_s: npt.ArrayLike
) -> FiniteTimeSpectra:
    maps = np.asarray(transitions, dtype=np.float64)
    times = np.asarray(elapsed_time_s, dtype=np.float64)
    if (
        maps.ndim != 3
        or maps.shape[0] != times.size
        or maps.shape[1] == 0
        or maps.shape[1] != maps.shape[2]
        or not np.all(np.isfinite(maps))
    ):
        raise ValueError("transitions must be finite square maps on the time grid")
    if times.size < 2 or times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("elapsed time must start at zero and increase strictly")
    singular = np.asarray(
        [np.linalg.svd(m, compute_uv=False) for m in maps], dtype=np.float64
    )
    exponents = np.full_like(singular, np.nan)
    exponents[1:] = np.log(singular[1:]) / times[1:, np.newaxis]
    return FiniteTimeSpectra(singular_values=singular, exponents_per_s=exponents)


def compute_agreement_horizon(tolerance: float, epsilon: float, ftle: float) -> float:
    if (
        not math.isfinite(tolerance)
        or not math.isfinite(epsilon)
        or tolerance <= 0.0
        or epsilon <= 0.0
    ):
        raise ValueError("tolerance and epsilon must be finite and positive")
    if tolerance < epsilon:
        raise ValueError(
            f"tolerance ({tolerance}) must be greater than or equal to epsilon ({epsilon})"
        )
    return float("inf") if ftle <= 0.0 else float(math.log(tolerance / epsilon) / ftle)


@dataclass(frozen=True)
class SensitivityReceipt:
    sigma_max: float
    ftle_per_s: float
    agreement_horizon_s: float
    dominant_direction: tuple[float, ...]
    fd_step: float
    state_scales: tuple[float, ...]
    sigma_max_phases: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.fd_step) or self.fd_step <= 0.0:
            raise ValueError(f"fd_step must be finite and positive, got {self.fd_step}")
        if not math.isfinite(self.sigma_max) or self.sigma_max < 0.0:
            raise ValueError(
                f"sigma_max must be finite and non-negative, got {self.sigma_max}"
            )
        if not math.isfinite(self.ftle_per_s):
            raise ValueError(f"ftle_per_s must be finite, got {self.ftle_per_s}")
        if len(self.state_scales) == 0 or any(
            not math.isfinite(s) or s <= 0.0 for s in self.state_scales
        ):
            raise ValueError("state_scales must contain finite positive values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sigma_max": self.sigma_max,
            "ftle_per_s": self.ftle_per_s,
            "agreement_horizon_s": self.agreement_horizon_s,
            "dominant_direction": list(self.dominant_direction),
            "fd_step": self.fd_step,
            "state_scales": list(self.state_scales),
            "sigma_max_phases": dict(self.sigma_max_phases),
            "metadata": dict(self.metadata),
        }


def compute_step_jacobian(
    step_fn: Callable[[FloatArray, float, float, Any], FloatArray],
    state: FloatArray,
    t: float,
    dt: float,
    control: Any,
    fd_step: float,
) -> FloatArray:
    n = state.size
    jacobian = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        upper, lower = state.copy(), state.copy()
        upper[j] += fd_step
        lower[j] -= fd_step
        jacobian[:, j] = (
            step_fn(upper, t, dt, control) - step_fn(lower, t, dt, control)
        ) / (2.0 * fd_step)
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("step Jacobian contains non-finite values")
    return jacobian


def propagate_state_transition_matrix(
    step_fn: Callable[[FloatArray, float, float, Any], FloatArray],
    initial_state: FloatArray,
    controls: Sequence[Any],
    time_grid: Sequence[float],
    dt: float,
    fd_step: float = 1e-6,
) -> tuple[FloatArray, FloatArray]:
    n, num_steps = initial_state.size, len(controls)
    states, transitions = (
        np.empty((num_steps + 1, n), dtype=np.float64),
        np.empty((num_steps + 1, n, n), dtype=np.float64),
    )
    states[0], transitions[0] = initial_state, np.eye(n, dtype=np.float64)
    curr_state = initial_state.copy()
    for k, ctrl in enumerate(controls):
        t_curr = time_grid[k]
        transitions[k + 1] = (
            compute_step_jacobian(step_fn, curr_state, t_curr, dt, ctrl, fd_step)
            @ transitions[k]
        )
        curr_state = step_fn(curr_state, t_curr, dt, ctrl)
        states[k + 1] = curr_state
    return states, transitions


def compute_linear_system_stm(
    A: npt.ArrayLike, t_end: float = 0.5, dt: float = 0.001, fd_step: float = 1e-5
) -> FloatArray:
    a_mat = np.asarray(A, dtype=np.float64)

    def _rk4_step(x: FloatArray, t: float, step_dt: float, ctrl: Any) -> FloatArray:
        k1 = a_mat @ x
        k2 = a_mat @ (x + 0.5 * step_dt * k1)
        k3 = a_mat @ (x + 0.5 * step_dt * k2)
        k4 = a_mat @ (x + step_dt * k3)
        return x + (step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    num_steps = int(round(t_end / dt))
    _, transitions = propagate_state_transition_matrix(
        _rk4_step,
        np.ones(a_mat.shape[0], dtype=np.float64),
        [None] * num_steps,
        [k * dt for k in range(num_steps)],
        dt,
        fd_step=fd_step,
    )
    return transitions[-1]


def run_pendulum_sensitivity_study(
    engine: Any,
    torque_profile: Any = None,
    state_scales: StateScales | None = None,
    t_end: float = 0.2,
    dt: float = 0.001,
    fd_step: float = 1e-6,
    tolerance: float = 0.20,
    epsilon: float = 1e-6,
    initial_angles: tuple[float, float, float] = (-1.5, 0.5, 0.2),
    initial_velocities: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SensitivityReceipt:
    if not math.isfinite(fd_step) or fd_step <= 0.0:
        raise ValueError(f"fd_step must be finite and positive, got {fd_step}")
    if not math.isfinite(t_end) or t_end <= 0.0 or t_end <= dt:
        raise ValueError("t_end must be positive and greater than dt")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")

    scales = state_scales or StateScales((1.0, 1.0, 1.0, 10.0, 10.0, 10.0))
    from src.shared.python.pendulum_simulator.step_convergence import (
        BreakpointTorqueProfile,
    )

    profile = torque_profile or BreakpointTorqueProfile(
        initial_torque=(80.0, 30.0, 10.0)
    )

    def _step_fn(
        x: FloatArray, t: float, step_dt: float, ctrl: tuple[float, float, float]
    ) -> FloatArray:
        engine.reset(tuple(x[:3]), tuple(x[3:]))
        engine.step(t, step_dt, ctrl)
        if hasattr(engine, "state"):
            s = engine.state
            return np.array(
                [s.theta1, s.theta2, s.theta3, s.omega1, s.omega2, s.omega3],
                dtype=np.float64,
            )
        if hasattr(engine, "data"):
            return np.concatenate([engine.data.qpos, engine.data.qvel]).astype(
                np.float64
            )
        raise TypeError(f"unsupported engine adapter type: {type(engine)}")

    num_steps = int(round(t_end / dt))
    _, transitions = propagate_state_transition_matrix(
        _step_fn,
        np.array(list(initial_angles) + list(initial_velocities), dtype=np.float64),
        [profile.evaluate(k * dt) for k in range(num_steps)],
        [k * dt for k in range(num_steps)],
        dt,
        fd_step=fd_step,
    )
    m_final_norm = normalized_transition(transitions[-1], scales)
    _, s_vals, vt = np.linalg.svd(m_final_norm)
    sigma_max = float(s_vals[0])
    ftle = float(math.log(max(sigma_max, 1e-15)) / t_end)

    phases = {"release": 1.0}
    if num_steps >= 2:
        m_mid_norm = normalized_transition(transitions[num_steps // 2], scales)
        phases["mid_swing"] = float(np.linalg.svd(m_mid_norm, compute_uv=False)[0])
    phases["impact"] = sigma_max

    return SensitivityReceipt(
        sigma_max=sigma_max,
        ftle_per_s=ftle,
        agreement_horizon_s=compute_agreement_horizon(
            tolerance=tolerance, epsilon=epsilon, ftle=ftle
        ),
        dominant_direction=tuple(float(v) for v in vt[0]),
        fd_step=fd_step,
        state_scales=tuple(scales.values),
        sigma_max_phases=phases,
        metadata={
            "engine": getattr(engine, "engine_name", type(engine).__name__),
            "t_end_s": t_end,
            "dt_s": dt,
        },
    )
