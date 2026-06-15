"""Contraction verification and Floquet multiplier utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]


@dataclass(frozen=True)
class ContractionResult:
    """Estimated contraction result for perturbation rollouts."""

    estimated_rate: float
    n_trials: int
    perturbation_scale: float
    horizon: float
    is_contracting: bool

    def to_dict(self) -> dict[str, bool | float | int]:
        return {
            "estimated_rate": self.estimated_rate,
            "n_trials": self.n_trials,
            "perturbation_scale": self.perturbation_scale,
            "horizon": self.horizon,
            "is_contracting": self.is_contracting,
        }


class ContractionVerifier:
    """Estimate contraction rate from nearby deterministic rollouts."""

    def __init__(
        self,
        decay_rate: float = 1.0,
        dimension: int = 3,
        horizon: float = 1.0,
        n_steps: int = 100,
        seed: int = 1234,
    ) -> None:
        if decay_rate <= 0.0:
            raise ValueError("decay_rate must be positive")
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if horizon <= 0.0:
            raise ValueError("horizon must be positive")
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        self.decay_rate = float(decay_rate)
        self.dimension = int(dimension)
        self.horizon = float(horizon)
        self.n_steps = int(n_steps)
        self.seed = int(seed)

    def estimate_contraction_rate(
        self,
        n_trials: int = 16,
        perturbation_scale: float = 1e-3,
    ) -> float:
        return self.verify(n_trials, perturbation_scale).estimated_rate

    def verify(
        self,
        n_trials: int = 16,
        perturbation_scale: float = 1e-3,
    ) -> ContractionResult:
        if n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if perturbation_scale <= 0.0:
            raise ValueError("perturbation_scale must be positive")

        times = np.linspace(0.0, self.horizon, self.n_steps)
        rng = np.random.default_rng(self.seed)
        trial_rates = [
            self._estimate_single_rate(rng, times, perturbation_scale)
            for _ in range(n_trials)
        ]
        estimated_rate = float(np.mean(trial_rates))
        return ContractionResult(
            estimated_rate=estimated_rate,
            n_trials=n_trials,
            perturbation_scale=float(perturbation_scale),
            horizon=self.horizon,
            is_contracting=estimated_rate > 0.0,
        )

    def _estimate_single_rate(
        self,
        rng: np.random.Generator,
        times: FloatArray,
        perturbation_scale: float,
    ) -> float:
        direction = rng.normal(size=self.dimension)
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0.0:
            raise RuntimeError("random perturbation unexpectedly has zero norm")
        perturbation = perturbation_scale * direction / direction_norm
        distances = np.array(
            [
                np.linalg.norm(self._flow(perturbation, time_value))
                for time_value in times
            ],
            dtype=np.float64,
        )
        log_distances = np.log(np.maximum(distances, np.finfo(float).tiny))
        slope, _ = np.polyfit(times, log_distances, deg=1)
        return float(-slope)

    def _flow(self, initial_delta: FloatArray, time_value: float) -> FloatArray:
        return np.exp(-self.decay_rate * time_value) * initial_delta


def compute_floquet_multipliers(monodromy_matrix: Any) -> ComplexArray:
    matrix = np.asarray(monodromy_matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("monodromy_matrix must be square")
    return np.asarray(np.linalg.eigvals(matrix), dtype=np.complex128)


def linear_system_floquet_multipliers(
    system_matrix: Any, period: float
) -> ComplexArray:
    if period <= 0.0:
        raise ValueError("period must be positive")
    matrix = np.asarray(system_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("system_matrix must be square")
    eigenvalues = np.linalg.eigvals(matrix)
    return np.asarray(np.exp(eigenvalues * period), dtype=np.complex128)
