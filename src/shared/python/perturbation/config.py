"""Configuration classes and protocols for perturbation parity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class TrialFailure:
    """Structured metadata for a trial that failed during perturbation analysis."""

    trial_index: int
    seed: int
    stage: str
    error_type: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert failure metadata to a JSON-serializable dictionary."""
        return {
            "trial_index": self.trial_index,
            "seed": self.seed,
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass
class PerturbationConfig:
    """Unified configuration for Monte Carlo perturbation analysis across engines.

    Attributes
    ----------
    n_trials : int
        Number of Monte Carlo simulations to run.
    noise_type : str
        Noise distribution: 'white', 'pink', or 'brown'.
    noise_amplitude : float
        Scalar standard deviation of the noise.
    perturb_mode : str
        'additive', 'multiplicative', or 'both'. Additive logic uses absolute
        values, multiplicative replaces scalar with `val * (1 + noise)`.
    seed : int, optional
        Base seed for reproducibility.
    """

    n_trials: int = 100
    noise_type: str = "white"
    noise_amplitude: float = 0.1
    perturb_mode: str = "additive"
    seed: int | None = None
    min_success_rate: float = 0.95
    raise_on_partial_results: bool = False

    def __post_init__(self) -> None:
        if not (self.n_trials > 0):
            raise ValueError(f"n_trials must be positive, got {self.n_trials}")
        if not (self.noise_amplitude >= 0):
            raise ValueError(
                f"noise_amplitude must be non-negative, got {self.noise_amplitude}"
            )
        if self.noise_type not in {"white", "pink", "brown"}:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")
        if self.perturb_mode not in {"additive", "multiplicative", "both"}:
            raise ValueError(f"Unknown perturb_mode: {self.perturb_mode}")


@dataclass
class PerturbationSummary:
    """Container for the full statistical summary of a batch perturbation run."""

    engine_name: str
    config: PerturbationConfig
    robustness_score: float
    metrics: dict[str, Any]  # Dictionary mapping metric name to MetricStatistics
    success_rate: float
    execution_time_sec: float
    failures: list[TrialFailure] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to JSON-serializable dictionary."""
        return {
            "engine_name": self.engine_name,
            "config": {
                "n_trials": self.config.n_trials,
                "noise_type": self.config.noise_type,
                "noise_amplitude": self.config.noise_amplitude,
                "perturb_mode": self.config.perturb_mode,
                "seed": self.config.seed,
            },
            "robustness_score": self.robustness_score,
            "metrics": {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.metrics.items()
            },
            "success_rate": self.success_rate,
            "execution_time_sec": self.execution_time_sec,
            "failures": [failure.to_dict() for failure in self.failures],
        }


class PerturbationAnalyzer(Protocol):
    """Protocol defining the parity API for engine perturbation analyzers."""

    def set_base_torque_profile(self, profile: object) -> None:
        """Set the nominal torque profile for the analysis."""
        ...

    def perturb_torque(self, config: PerturbationConfig, seed: int) -> object:
        """Apply perturbation to the base torque profile yielding a new profile."""
        ...

    def run_batch(self, config: PerturbationConfig) -> PerturbationSummary:
        """Run full Monte Carlo batch and compute statistics."""
        ...

    def extract_metrics(self, sim_result: object) -> dict[str, float | np.ndarray]:
        """Extract the mandatory metrics from a given simulation result."""
        ...
