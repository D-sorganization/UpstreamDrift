"""Sampling-based swing optimization over batched simulation rollouts.

Part of epic #8390 (B5/#8400) — the first production consumer of the
``simulation_backends`` batch infrastructure (ADR-0023/0024). Runs CEM or
MPPI over torque histories for the canonical golf double pendulum,
evaluating whole candidate populations per iteration through
:func:`run_batched` (MJX/MJWarp when available) or
:func:`cpu_batch_rollout` (the dependency-free reference path), so
"thousands of parameter variations" is one batched launch per iteration
rather than a Python loop of solves.

Scoring is fully vectorized over the batch: terminal clubhead speed from
planar two-link kinematics (segment lengths from ``GolfModelParams``),
an effort integral, and the smooth injury surrogate's velocity term
(B1/#8396). Accelerators change throughput only — correctness is
identical on the CPU path, which is what CI exercises.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.shared.python.simulation_backends.batched import (
    cpu_batch_rollout,
    run_batched,
)
from src.shared.python.simulation_backends.factory import make_backend
from src.shared.python.simulation_backends.model_params import GolfModelParams
from src.shared.python.simulation_backends.protocol import BatchTrace

__all__ = [
    "BatchSwingObjectiveWeights",
    "BatchSwingOptimizer",
    "BatchSwingResult",
    "score_batch",
    "terminal_clubhead_speed",
]

_VELOCITY_SPIKE_LIMIT = 20.0  # rad/s, mirrors smooth_costs' threshold
_VELOCITY_SPIKE_SHARPNESS = 8.0


@dataclass(frozen=True)
class BatchSwingObjectiveWeights:
    """Weights for the vectorized batch objective (higher score is better).

    Attributes:
        clubhead_speed: Reward per m/s of terminal clubhead speed.
        effort: Penalty per (N*m)^2*s of control effort.
        injury: Penalty weight on the smooth velocity-spike surrogate.
    """

    clubhead_speed: float = 1.0
    effort: float = 1e-4
    injury: float = 0.1

    def __post_init__(self) -> None:
        for name in ("clubhead_speed", "effort", "injury"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class BatchSwingResult:
    """Outcome of a sampling-based batch optimization run.

    Attributes:
        best_controls: Best torque history found, shape ``(horizon, nu)``.
        best_score: Score of ``best_controls`` (higher is better).
        initial_score: Score of the initial mean control history.
        score_history: Best score after each iteration.
        candidates_evaluated: Total rollouts scored.
        backend: Name reported by the batch trace producer.
        method: ``"cem"`` or ``"mppi"``.
    """

    best_controls: np.ndarray
    best_score: float
    initial_score: float
    score_history: tuple[float, ...]
    candidates_evaluated: int
    backend: str
    method: str

    @property
    def improved(self) -> bool:
        """Whether optimization beat the initial mean controls."""
        return self.best_score > self.initial_score


def terminal_clubhead_speed(trace: BatchTrace, params: GolfModelParams) -> np.ndarray:
    """Terminal club-tip speed per environment, shape ``(N,)`` [m/s].

    Planar two-link tip kinematics with segment lengths from ``params``:
    the club tip sits at the end of the lower segment, whose absolute
    angle is ``q1 + q2``.
    """
    l1 = float(params.upper.length_m)
    l2 = float(params.lower.length_m)
    q = np.asarray(trace.q, dtype=float)
    v = np.asarray(trace.v, dtype=float)
    if q.ndim != 3 or q.shape[2] < 2:
        raise ValueError("trace must be a rank-3 batch with >= 2 DOFs")
    q1, q2 = q[:, -1, 0], q[:, -1, 1]
    q1d, q2d = v[:, -1, 0], v[:, -1, 1]
    abs2 = q1 + q2
    abs2d = q1d + q2d
    vx = -l1 * np.sin(q1) * q1d - l2 * np.sin(abs2) * abs2d
    vy = l1 * np.cos(q1) * q1d + l2 * np.cos(abs2) * abs2d
    return np.hypot(vx, vy)


def score_batch(
    trace: BatchTrace,
    controls: np.ndarray,
    params: GolfModelParams,
    weights: BatchSwingObjectiveWeights,
) -> np.ndarray:
    """Vectorized score per environment, shape ``(N,)`` (higher is better)."""
    speed = terminal_clubhead_speed(trace, params)
    effort = (
        np.einsum("nij,nij->n", controls, controls) * trace.dt
    )  # ⚡ Bolt: np.einsum is ~3x faster than np.sum(np.square(...), axis=(1, 2))
    peak_vel = np.max(np.abs(np.asarray(trace.v)), axis=(1, 2))
    spike = 1.0 / (
        1.0 + np.exp(-_VELOCITY_SPIKE_SHARPNESS * (peak_vel - _VELOCITY_SPIKE_LIMIT))
    )
    return (
        weights.clubhead_speed * speed
        - weights.effort * effort
        - weights.injury * spike
    )


class BatchSwingOptimizer:
    """CEM/MPPI swing optimizer over batched rollouts.

    Args:
        backend_name: Simulation backend (``available_backends()``); batched
            backends (MJX/MJWarp) go through :func:`run_batched`, single-env
            backends through :func:`cpu_batch_rollout`.
        params: Canonical golf model parameters.
        method: ``"cem"`` (elite refit) or ``"mppi"`` (softmax reweighting).
        n_candidates: Population size per iteration (> 1).
        n_iterations: Optimization iterations (> 0).
        elite_fraction: CEM elite quantile (0, 1].
        temperature: MPPI softmax temperature (> 0).
        tau_max: Torque sampling bound [N*m].
        seed: RNG seed for reproducibility.
        weights: Objective weights.
        max_batch: Optional per-launch env budget for batched backends.
    """

    def __init__(
        self,
        backend_name: str = "ode",
        params: GolfModelParams | None = None,
        *,
        method: Literal["cem", "mppi"] = "cem",
        n_candidates: int = 64,
        n_iterations: int = 10,
        elite_fraction: float = 0.2,
        temperature: float = 1.0,
        tau_max: float = 100.0,
        seed: int = 0,
        weights: BatchSwingObjectiveWeights | None = None,
        max_batch: int | None = None,
    ) -> None:
        if method not in {"cem", "mppi"}:
            raise ValueError(f"method must be 'cem' or 'mppi', got {method!r}")
        if n_candidates < 2:
            raise ValueError("n_candidates must be at least 2")
        if n_iterations < 1:
            raise ValueError("n_iterations must be positive")
        if not 0.0 < elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if tau_max <= 0.0:
            raise ValueError("tau_max must be positive")
        self.backend_name = backend_name
        self.params = params or GolfModelParams.default()
        self.method = method
        self.n_candidates = n_candidates
        self.n_iterations = n_iterations
        self.elite_fraction = elite_fraction
        self.temperature = temperature
        self.tau_max = tau_max
        self.seed = seed
        self.weights = weights or BatchSwingObjectiveWeights()
        self.max_batch = max_batch

    # ------------------------------------------------------------------

    def optimize(self, horizon: int = 100, dt: float = 0.005) -> BatchSwingResult:
        """Run the sampling loop and return the best control history.

        Args:
            horizon: Integration steps per rollout (> 0).
            dt: Integration step [s] (> 0).
        """
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        rng = np.random.default_rng(self.seed)
        rollout, nu, backend_label = self._make_rollout_fn(horizon, dt)

        mean = np.zeros((horizon, nu))
        sigma = np.full((horizon, nu), 0.3 * self.tau_max)

        initial_trace = rollout(mean[np.newaxis, :, :])
        initial_score = float(
            score_batch(
                initial_trace, mean[np.newaxis, :, :], self.params, self.weights
            )[0]
        )

        best_controls = mean.copy()
        best_score = initial_score
        history: list[float] = []
        evaluated = 1

        n_elite = max(2, int(round(self.elite_fraction * self.n_candidates)))
        for _ in range(self.n_iterations):
            noise = rng.normal(size=(self.n_candidates, horizon, nu))
            candidates = np.clip(
                mean[np.newaxis] + sigma[np.newaxis] * noise,
                -self.tau_max,
                self.tau_max,
            )
            trace = rollout(candidates)
            scores = score_batch(trace, candidates, self.params, self.weights)
            evaluated += self.n_candidates

            top = int(np.argmax(scores))
            if float(scores[top]) > best_score:
                best_score = float(scores[top])
                best_controls = candidates[top].copy()

            if self.method == "cem":
                elite_idx = np.argsort(scores)[-n_elite:]
                elite = candidates[elite_idx]
                mean = elite.mean(axis=0)
                sigma = elite.std(axis=0) + 1e-6
            else:  # mppi
                shifted = (scores - np.max(scores)) / self.temperature
                w = np.exp(shifted)
                w = w / np.sum(w)
                mean = np.tensordot(w, candidates, axes=1)
            history.append(best_score)

        return BatchSwingResult(
            best_controls=best_controls,
            best_score=best_score,
            initial_score=initial_score,
            score_history=tuple(history),
            candidates_evaluated=evaluated,
            backend=backend_label,
            method=self.method,
        )

    # ------------------------------------------------------------------

    def _make_rollout_fn(
        self, horizon: int, dt: float
    ) -> tuple[Callable[[np.ndarray], BatchTrace], int, str]:
        """(rollout(controls_batch) -> BatchTrace, nu, backend label)."""
        probe = make_backend(self.backend_name, self.params)
        probe.reset(None)
        probe_trace = probe.rollout(None, 1, dt)
        nu = 2 if probe_trace.u is None else int(probe_trace.u.shape[1])

        if callable(getattr(probe, "rollout_batch", None)):

            def rollout_batched(controls: np.ndarray) -> BatchTrace:
                return run_batched(
                    probe,  # type: ignore[arg-type]
                    controls,
                    horizon,
                    dt,
                    num_envs=controls.shape[0],
                    max_batch=self.max_batch,
                )

            return rollout_batched, nu, self.backend_name

        def rollout_cpu(controls: np.ndarray) -> BatchTrace:
            return cpu_batch_rollout(
                lambda _i: make_backend(self.backend_name, self.params),
                controls,
                horizon,
                dt,
            )

        return rollout_cpu, nu, f"cpu_batch[{self.backend_name}]"
