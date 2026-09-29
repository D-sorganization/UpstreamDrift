from __future__ import annotations

import logging
import time
import warnings
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
    TrialFailure,
)
from src.shared.python.perturbation.robustness_score import compute_robustness_score
from src.shared.python.perturbation.statistics import (
    MetricStatistics,
    compute_metric_statistics,
)

logger = logging.getLogger(__name__)


def build_joint_polys(coeffs: list[list[float]], n_joints: int) -> list[np.ndarray]:
    """Build per-joint polynomial arrays for use with ``np.polyval``.

    Converts ascending-order coefficient lists ``[c0, c1, c2, ...]`` to
    descending order (reversed) as required by ``np.polyval``.  Missing joints
    are padded with a zero polynomial.

    Args:
        coeffs: Per-joint coefficient lists in ascending order.
        n_joints: Number of joints (DoF / actuators) in the model.

    Returns:
        List of length *n_joints* where each entry is an ``np.ndarray`` of
        reversed coefficients suitable for ``np.polyval``.
    """
    polys: list[np.ndarray] = []
    for j in range(n_joints):
        if j < len(coeffs):
            polys.append(np.array(coeffs[j][::-1]))
        else:
            polys.append(np.array([0.0]))
    return polys


def compute_ee_velocity_fd(ee_pos_arr: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
    """Compute end-effector velocity via first-order finite differences.

    Args:
        ee_pos_arr: Array of shape ``(N, 3)`` with end-effector positions.
        t_arr: Array of shape ``(N,)`` with corresponding timestamps.

    Returns:
        Array of shape ``(N, 3)`` with finite-difference velocities.
        The first element is always zero (no preceding sample).
    """
    ee_vel_arr = np.zeros_like(ee_pos_arr)
    for i in range(1, len(t_arr)):
        dt_i = max(t_arr[i] - t_arr[i - 1], 1e-12)
        ee_vel_arr[i] = (ee_pos_arr[i] - ee_pos_arr[i - 1]) / dt_i
    return ee_vel_arr


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
    failures: list[TrialFailure] = field(default_factory=list)


class PartialResultsWarning(UserWarning):
    """Warning emitted when perturbation analysis returns partial trial results."""


class PartialResultsError(RuntimeError):
    """Raised when partial perturbation results violate the configured threshold."""

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        success_rate: float,
        threshold: float,
        failures: list[TrialFailure],
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.success_rate = success_rate
        self.threshold = threshold
        self.failures = failures


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

    def _build_trial_failure(
        self,
        *,
        trial_index: int,
        seed: int,
        stage: str,
        exc: Exception,
    ) -> TrialFailure:
        return TrialFailure(
            trial_index=trial_index,
            seed=seed,
            stage=stage,
            error_type=type(exc).__name__,
            message=str(exc),
        )

    def _handle_partial_results(
        self,
        *,
        operation: str,
        config: PerturbationConfig,
        requested_trials: int,
        failures: list[TrialFailure],
    ) -> float:
        if requested_trials <= 0:
            return 0.0

        success_rate = (requested_trials - len(failures)) / requested_trials
        if not failures or success_rate >= config.min_success_rate:
            return success_rate

        message = (
            f"{operation} for {self.ENGINE_NAME} returned partial results: "
            f"{requested_trials - len(failures)}/{requested_trials} trials succeeded "
            f"(success_rate={success_rate:.3f}, "
            f"threshold={config.min_success_rate:.3f})"
        )
        logger.warning(message)
        if config.raise_on_partial_results:
            raise PartialResultsError(
                message,
                operation=operation,
                success_rate=success_rate,
                threshold=config.min_success_rate,
                failures=failures,
            )
        warnings.warn(message, PartialResultsWarning, stacklevel=3)
        return success_rate

    def _collect_batch_metrics(
        self,
        config: PerturbationConfig,
        metric_lists: dict[str, list[float]],
    ) -> list[TrialFailure]:
        if self._base_coeffs is None:
            raise ValueError(
                "set_base_torque_profile() must be called before _collect_batch_metrics()"
            )

        base_coeffs = self._base_coeffs
        base_seed = config.seed if config.seed is not None else 0
        failures: list[TrialFailure] = []

        for trial_index in range(config.n_trials):
            perturbed = perturb_torque_coeffs(
                base_coeffs,
                noise_amplitude=config.noise_amplitude,
                noise_type=config.noise_type,
                seed=base_seed + trial_index,
                perturb_mode=config.perturb_mode,
            )
            try:
                sim = self._simulate(perturbed)
                metrics = self.extract_metrics(sim)
                for metric_name in metric_lists:
                    metric_value = metrics[metric_name]
                    if isinstance(metric_value, np.ndarray):
                        metric_value = float(np.linalg.norm(metric_value))
                    metric_lists[metric_name].append(float(metric_value))
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    self._build_trial_failure(
                        trial_index=trial_index,
                        seed=base_seed + trial_index,
                        stage="run_batch",
                        exc=exc,
                    )
                )
                logger.debug("Trial %d failed", trial_index, exc_info=True)

        return failures

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
        failures = self._collect_batch_metrics(config, metric_lists)
        success_rate = self._handle_partial_results(
            operation="run_batch",
            config=config,
            requested_trials=config.n_trials,
            failures=failures,
        )

        if not any(metric_lists.values()):
            logger.warning("All trials failed — returning zero-robustness summary")
            return PerturbationSummary(
                engine_name=self.ENGINE_NAME,
                config=config,
                robustness_score=0.0,
                metrics={},
                success_rate=0.0,
                execution_time_sec=time.monotonic() - t_start,
                failures=failures,
            )

        metric_stats: dict[str, MetricStatistics] = {}
        for metric_name, values in metric_lists.items():
            if values:
                metric_stats[metric_name] = compute_metric_statistics(np.array(values))

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
            failures=failures,
        )

    def _collect_scalar_trials(
        self,
        *,
        profile: object,
        metric: str,
        profile_name: str,
        config: PerturbationConfig,
    ) -> tuple[np.ndarray, list[TrialFailure]]:
        self.set_base_torque_profile(profile)
        if self._base_coeffs is None:
            raise ValueError(
                "set_base_torque_profile() must populate _base_coeffs before compare_profiles()"
            )

        base_coeffs = self._base_coeffs
        values: list[float] = []
        failures: list[TrialFailure] = []
        base_seed = config.seed if config.seed is not None else 0

        for trial_index in range(config.n_trials):
            seed = base_seed + trial_index
            perturbed = perturb_torque_coeffs(
                base_coeffs,
                noise_amplitude=config.noise_amplitude,
                noise_type=config.noise_type,
                seed=seed,
                perturb_mode=config.perturb_mode,
            )
            try:
                sim = self._simulate(perturbed)
                metric_dict = self.extract_metrics(sim)
                metric_value = metric_dict[metric]
                if isinstance(metric_value, np.ndarray):
                    metric_value = float(np.linalg.norm(metric_value))
                values.append(float(metric_value))
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    self._build_trial_failure(
                        trial_index=trial_index,
                        seed=seed,
                        stage=f"compare_profiles:{profile_name}:{metric}",
                        exc=exc,
                    )
                )
                logger.debug(
                    "Trial %d failed during profile comparison metric collection",
                    trial_index,
                    exc_info=True,
                )

        return (np.array(values) if values else np.array([0.0])), failures

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

        failures: list[TrialFailure] = []

        for metric in scalar_metrics:
            vals_a, failures_a = self._collect_scalar_trials(
                profile=profile_a,
                metric=metric,
                profile_name="A",
                config=config,
            )
            vals_b, failures_b = self._collect_scalar_trials(
                profile=profile_b,
                metric=metric,
                profile_name="B",
                config=config,
            )
            failures.extend(failures_a)
            failures.extend(failures_b)

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
                "failures_a": len(failures_a),
                "failures_b": len(failures_b),
            }

            if winner_metric == "A":
                a_wins += 1
            else:
                b_wins += 1

        self._handle_partial_results(
            operation="compare_profiles",
            config=config,
            requested_trials=config.n_trials * len(scalar_metrics) * 2,
            failures=failures,
        )

        winner = name_a if a_wins >= b_wins else name_b
        confidence = 1.0 - float(np.median(list(pvalues.values())))
        confidence = max(0.0, min(1.0, confidence))

        return ComparisonReport(
            winner=winner,
            confidence=confidence,
            metric_comparisons=metric_comparisons,
            pvalues=pvalues,
            failures=failures,
        )
