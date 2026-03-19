"""
Monte Carlo perturbation analysis for swing consistency evaluation.

Formalizes the pendulum perturbation analysis into the unified PerturbationAnalyzer
protocol defined in the perturbation analysis guidelines.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

import numpy as np

# Re-exporting shared configuration classes so old imports don't break
from ..perturbation.config import (
    PerturbationAnalyzer,
    PerturbationConfig,
    PerturbationSummary,
)
from ..perturbation.noise import generate_noise
from ..perturbation.robustness_score import compute_robustness_score
from ..perturbation.statistics import MetricStatistics, compute_metric_statistics

logger = logging.getLogger(__name__)


class SimulateFn(Protocol):
    """Protocol for simulation callable."""
    def __call__(self, coeffs: list[list[float]]) -> object: ...


class ExtractFn(Protocol):
    """Protocol for metric extraction callable."""
    def __call__(self, result: object) -> dict[str, float | np.ndarray]: ...


# ---------------------------------------------------------------------------
# Backward Compatibility & Torque Utils
# ---------------------------------------------------------------------------

def perturb_torque_coeffs(
    coeffs: list[list[float]],
    noise_amplitude: float,
    noise_type: str = "white",
    seed: int | None = None,
    perturb_mode: str = "additive",
) -> list[list[float]]:
    """Perturb polynomial torque coefficients with noise."""
    assert noise_amplitude >= 0
    assert noise_type in {"white", "pink", "brown"}
    assert perturb_mode in {"additive", "multiplicative", "both"}

    if noise_amplitude == 0.0:
        return [list(c) for c in coeffs]

    total = sum(len(c) for c in coeffs)
    noise = generate_noise(noise_type, total, noise_amplitude, seed)

    idx = 0
    result = []
    for joint_coeffs in coeffs:
        n = len(joint_coeffs)
        joint_noise = noise[idx : idx + n]
        
        perturbed: list[float] = []
        for i, c in enumerate(joint_coeffs):
            p = float(c)
            # Apply additive
            if perturb_mode in {"additive", "both"}:
                p += float(joint_noise[i])
            # Apply multiplicative
            if perturb_mode in {"multiplicative", "both"}:
                p *= (1.0 + float(joint_noise[i]))
            perturbed.append(p)
            
        result.append(perturbed)
        idx += n

    return result


def variability_summary(
    results: list[dict],
) -> dict[str, float | np.ndarray]:
    """Backward-compatible summary matching the old format."""
    assert len(results) > 0, "results must be non-empty"

    speeds = np.array([r["tip_speed_final"] for r in results])
    positions = np.array([r["tip_position_final"] for r in results])

    speed_mean = float(np.mean(speeds))
    speed_std = float(np.std(speeds))
    speed_cv = speed_std / speed_mean if speed_mean != 0 else 0.0

    summary: dict[str, float | np.ndarray] = {
        "tip_speed_mean": speed_mean,
        "tip_speed_std": speed_std,
        "tip_speed_cv": speed_cv,
        "tip_speed_min": float(np.min(speeds)),
        "tip_speed_max": float(np.max(speeds)),
        "tip_position_mean": np.mean(positions, axis=0),
        "tip_position_std": np.std(positions, axis=0),
        "n_trials": len(results),
    }

    return summary


# ---------------------------------------------------------------------------
# Core Analyzer Protocol Implementation
# ---------------------------------------------------------------------------

class PendulumPerturbationAnalyzer(PerturbationAnalyzer):
    """Implements the Parity Guidelines PerturbationAnalyzer for pendulum."""

    def __init__(self, simulate_fn: SimulateFn, extract_fn: ExtractFn) -> None:
        self.simulate_fn = simulate_fn
        self.extract_fn = extract_fn
        self._base_coeffs: list[list[float]] = []

    def set_base_torque_profile(self, profile: Any) -> None:
        """Set the nominal torque profile. Expects list of lists."""
        assert isinstance(profile, list)
        self._base_coeffs = profile

    def perturb_torque(self, config: PerturbationConfig, seed: int) -> list[list[float]]:
        """Apply perturbation to base torque."""
        assert self._base_coeffs, "Base torque profile must be set first"
        return perturb_torque_coeffs(
            self._base_coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            seed=seed,
            perturb_mode=config.perturb_mode,
        )

    def extract_metrics(self, sim_result: object) -> dict[str, float | np.ndarray]:
        """Extract the mandatory metrics from a given simulation result."""
        # Use underlying extraction and standardize keys
        extracted = self.extract_fn(sim_result)
        
        # Mandatory mapping as per guidelines 
        mapped: dict[str, float | np.ndarray] = {}
        if "tip_position_final" in extracted:
            mapped["end_effector_position_final"] = extracted["tip_position_final"]
        if "tip_speed_final" in extracted:
            mapped["end_effector_speed_final"] = float(extracted["tip_speed_final"])
        
        # Propagate the rest
        for k, v in extracted.items():
            if k not in mapped:
                mapped[k] = v
                
        return mapped

    def run_batch(self, config: PerturbationConfig) -> PerturbationSummary:
        """Run full Monte Carlo batch and compute statistics."""
        assert self._base_coeffs, "Base torque profile must be set first"
        
        start_time = time.perf_counter()
        raw_metrics_list = []
        base_seed = config.seed if config.seed is not None else 0

        for i in range(config.n_trials):
            trial_seed = base_seed + i
            perturbed = self.perturb_torque(config, trial_seed)
            try:
                sim_result = self.simulate_fn(perturbed)
                metrics = self.extract_metrics(sim_result)
                raw_metrics_list.append(metrics)
            except (ValueError, RuntimeError, FloatingPointError, AssertionError):
                logger.warning("Trial %d failed, skipping", i, exc_info=True)
                continue

        execution_time_sec = time.perf_counter() - start_time
        success_rate = len(raw_metrics_list) / config.n_trials if config.n_trials > 0 else 0.0
        
        # Compile statistics
        metrics_stats: dict[str, MetricStatistics] = {}
        if raw_metrics_list:
            keys = raw_metrics_list[0].keys()
            for key in keys:
                # Stack all successfully collected metrics
                values = np.array([m[key] for m in raw_metrics_list])
                metrics_stats[key] = compute_metric_statistics(values)

        # For the Robustness Score, we select 'end_effector_speed_final' if available
        cv_weighted = 0.0
        if "end_effector_speed_final" in metrics_stats:
            cv = metrics_stats["end_effector_speed_final"].cv
            if isinstance(cv, (float, np.number)) and cv >= 0:
                cv_weighted = float(cv)

        return PerturbationSummary(
            engine_name="Pendulum",
            config=config,
            robustness_score=compute_robustness_score(cv_weighted),
            metrics=metrics_stats,  # type: ignore
            success_rate=success_rate,
            execution_time_sec=execution_time_sec,
        )

# Maintain existing method for testing/GUI compatibility
def batch_perturb_and_simulate(
    base_coeffs: list[list[float]],
    config: PerturbationConfig,
    simulate_fn: SimulateFn,
    extract_fn: ExtractFn,
) -> list[dict]:
    """Run N perturbed simulations and collect results using old paradigm."""
    analyzer = PendulumPerturbationAnalyzer(simulate_fn, extract_fn)
    analyzer.set_base_torque_profile(base_coeffs)
    
    base_seed = config.seed if config.seed is not None else 0
    results = []
    
    for i in range(config.n_trials):
        trial_seed = base_seed + i
        perturbed = analyzer.perturb_torque(config, trial_seed)
        try:
            sim_result = simulate_fn(perturbed)
            metrics = extract_fn(sim_result)
            results.append(metrics)
        except Exception:
            continue
            
    return results
