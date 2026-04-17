from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.python.pendulum_simulator.perturbation_analysis import (
    perturb_torque_coeffs,
)
from src.shared.python.perturbation.config import (
    PerturbationConfig,
    PerturbationSummary,
)
from src.shared.python.perturbation.robustness_score import compute_robustness_score
from src.shared.python.perturbation.statistics import (
    MetricStatistics,
    compute_metric_statistics,
)

logger = logging.getLogger(__name__)

MANDATORY_METRICS: tuple[str, ...] = (
    "end_effector_position_final",
    "end_effector_velocity_final",
    "end_effector_speed_final",
    "peak_end_effector_speed",
    "total_energy_final",
    "joint_angles_final",
    "joint_velocities_final",
    "trajectory_rmse",
    "trajectory_max_deviation",
    "motion_duration",
)


@dataclass
class ComparisonReport:
    """Statistical comparison of two torque profiles."""

    winner: str
    confidence: float
    metric_comparisons: dict[str, Any] = field(default_factory=dict)
    pvalues: dict[str, float] = field(default_factory=dict)


class PerturbationAnalyzerBase(ABC):
    """Abstract base class for perturbation analyzers implementing common logic."""

    ENGINE_NAME: str = "base"
    _base_coeffs: list[list[float]] | None = None

    @abstractmethod
    def _simulate(self, coeffs: list[list[float]]) -> Any:
        pass

    @abstractmethod
    def extract_metrics(self, sim_result: object) -> dict[str, float | np.ndarray]:
        pass

    @abstractmethod
    def set_base_torque_profile(self, profile: object) -> None:
        pass

    def perturb_torque(
        self, config: PerturbationConfig, seed: int
    ) -> dict[str, list[list[float]]]:
        if self._base_coeffs is None:
            raise ValueError(
                "set_base_torque_profile() must be called before perturb_torque()"
            )
        perturbed = perturb_torque_coeffs(
            self._base_coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            seed=seed,
            perturb_mode=config.perturb_mode,
        )
        return {"coeffs": perturbed}

    def run_batch(self, config: PerturbationConfig) -> PerturbationSummary:
        if self._base_coeffs is None:
            raise ValueError(
                "set_base_torque_profile() must be called before run_batch()"
            )

        t_start = time.monotonic()
        base_seed = config.seed if config.seed is not None else 0

        scalar_metric_names = [
            m
            for m in MANDATORY_METRICS
            if m
            not in (
                "end_effector_position_final",
                "end_effector_velocity_final",
                "joint_angles_final",
                "joint_velocities_final",
            )
        ]

        metric_lists: dict[str, list[float]] = {m: [] for m in scalar_metric_names}
        n_success = 0

        for i in range(config.n_trials):
            perturbed = perturb_torque_coeffs(
                self._base_coeffs,
                noise_amplitude=config.noise_amplitude,
                noise_type=config.noise_type,
                seed=base_seed + i,
                perturb_mode=config.perturb_mode,
            )
            try:
                sim = self._simulate(perturbed)
                metrics = self.extract_metrics(sim)
                for m in scalar_metric_names:
                    v = metrics[m]
                    if isinstance(v, np.ndarray):
                        v = float(np.linalg.norm(v))
                    metric_lists[m].append(float(v))
                n_success += 1
            except Exception:
                logger.debug("Trial %d failed", i, exc_info=True)

        success_rate = n_success / config.n_trials if config.n_trials > 0 else 0.0

        if n_success == 0:
            logger.warning("All trials failed — returning zero-robustness summary")
            return PerturbationSummary(
                engine_name=self.ENGINE_NAME,
                config=config,
                robustness_score=0.0,
                metrics={},
                success_rate=0.0,
                execution_time_sec=time.monotonic() - t_start,
            )

        metric_stats: dict[str, MetricStatistics] = {}
        for m, values in metric_lists.items():
            if values:
                arr = np.array(values)
                metric_stats[m] = compute_metric_statistics(arr)

        cv_values = []
        for stats in metric_stats.values():
            _std = float(stats.std) if not isinstance(stats.std, float) else stats.std
            _mean = (
                float(stats.mean) if not isinstance(stats.mean, float) else stats.mean
            )
            if _std > 0 and abs(_mean) > 1e-12:
                cv_values.append(_std / abs(_mean))
            else:
                cv_values.append(0.0)
        cv_weighted = float(np.mean(cv_values)) if cv_values else 0.0
        rs = compute_robustness_score(cv_weighted)

        return PerturbationSummary(
            engine_name=self.ENGINE_NAME,
            config=config,
            robustness_score=rs,
            metrics=metric_stats,
            success_rate=success_rate,
            execution_time_sec=time.monotonic() - t_start,
        )

    def compare_profiles(
        self,
        profile_a: object,
        profile_b: object,
        config: PerturbationConfig,
        name_a: str = "A",
        name_b: str = "B",
    ) -> ComparisonReport:
        from scipy import stats as _stats

        metric_comparisons: dict[str, Any] = {}
        pvalues: dict[str, float] = {}
        a_wins = 0
        b_wins = 0

        scalar_metrics = [
            "end_effector_speed_final",
            "peak_end_effector_speed",
            "total_energy_final",
            "trajectory_rmse",
            "trajectory_max_deviation",
        ]

        base_seed = config.seed if config.seed is not None else 0

        def _collect_scalar(profile: object, metric: str) -> np.ndarray:
            self.set_base_torque_profile(profile)
            values = []
            for i in range(config.n_trials):
                perturbed = perturb_torque_coeffs(
                    self._base_coeffs,  # type: ignore[arg-type]
                    noise_amplitude=config.noise_amplitude,
                    noise_type=config.noise_type,
                    seed=base_seed + i,
                    perturb_mode=config.perturb_mode,
                )
                try:
                    sim = self._simulate(perturbed)
                    m_dict = self.extract_metrics(sim)
                    v = m_dict[metric]
                    if isinstance(v, np.ndarray):
                        v = float(np.linalg.norm(v))
                    values.append(float(v))
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Trial %d failed during profile comparison metric collection",
                        i,
                        exc_info=True,
                    )
            return np.array(values) if values else np.array([0.0])

        for metric in scalar_metrics:
            vals_a = _collect_scalar(profile_a, metric)
            vals_b = _collect_scalar(profile_b, metric)

            try:
                _stat, pval = _stats.mannwhitneyu(
                    vals_a, vals_b, alternative="two-sided"
                )
            except ValueError:
                pval = 1.0

            pvalues[metric] = float(pval)
            mean_a = float(np.mean(vals_a))
            mean_b = float(np.mean(vals_b))

            if metric in ("end_effector_speed_final", "peak_end_effector_speed"):
                winner_metric = "A" if mean_a > mean_b else "B"
            else:
                winner_metric = "A" if mean_a < mean_b else "B"

            metric_comparisons[metric] = {
                "mean_a": mean_a,
                "mean_b": mean_b,
                "winner": winner_metric,
                "pvalue": float(pval),
            }

            if winner_metric == "A":
                a_wins += 1
            else:
                b_wins += 1

        winner = name_a if a_wins >= b_wins else name_b
        confidence = 1.0 - float(np.median(list(pvalues.values())))
        confidence = max(0.0, min(1.0, confidence))

        return ComparisonReport(
            winner=winner,
            confidence=confidence,
            metric_comparisons=metric_comparisons,
            pvalues=pvalues,
        )
