# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Pendulum Perturbation Analyzer — reference implementation of the unified protocol.

Implements ``PerturbationAnalyzer`` (from ``src.shared.python.perturbation.config``)
for the driven double-pendulum golf model.  Serves as the gold-standard reference
that all other engine analyzers must match in API and metric naming.

Design by Contract
------------------
- ``set_base_torque_profile()`` must be called before ``run_batch()`` or
  ``perturb_torque()``.
- The ``run_batch()`` result always contains all mandatory metrics listed in
  ``MANDATORY_METRICS``.
- ``extract_metrics()`` always returns finite arrays for valid simulation results.

DRY
---
Delegates noise generation and coefficient perturbation to
``perturbation_analysis.py`` functions (which remain backward-compatible).
Uses the shared ``MetricStatistics``, ``compute_metric_statistics``, and
``compute_robustness_score`` utilities to avoid duplication with other engines.

See Also
--------
docs/perturbation_analysis_parity_guidelines.md for the full protocol spec.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from src.shared.python.perturbation.config import (
    PerturbationConfig,
    PerturbationSummary,
)
from src.shared.python.perturbation.robustness_score import compute_robustness_score
from src.shared.python.perturbation.statistics import (
    MetricStatistics,
    compute_metric_statistics,
)

from .perturbation_analysis import (
    generate_noise,
    perturb_torque_coeffs,
)
from .physics import TorqueFunc, forward_kinematics, total_energy
from .simulation import SimulationResult, run_simulation
from .torque_utils import make_polynomial_torque

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mandatory metric names (all engines must return these)
# ---------------------------------------------------------------------------

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

SCALAR_COMPARISON_METRICS: tuple[str, ...] = (
    "end_effector_speed_final",
    "peak_end_effector_speed",
    "total_energy_final",
    "trajectory_rmse",
    "trajectory_max_deviation",
)

HIGHER_IS_BETTER_METRICS: frozenset[str] = frozenset(
    ("end_effector_speed_final", "peak_end_effector_speed")
)


def _metric_winner(metric: str, mean_a: float, mean_b: float) -> str:
    if metric in HIGHER_IS_BETTER_METRICS:
        return "A" if mean_a > mean_b else "B"
    return "A" if mean_a < mean_b else "B"


def _overall_winner(a_wins: int, b_wins: int) -> tuple[str, float]:
    total = a_wins + b_wins
    if total == 0:
        return "tie", 0.5
    if a_wins > b_wins:
        return "A", a_wins / total
    if b_wins > a_wins:
        return "B", b_wins / total
    return "tie", 0.5


def _mann_whitney_pvalue(
    stats_module: Any,
    vals_a: np.ndarray,
    vals_b: np.ndarray,
) -> float:
    try:
        _stat, pval = stats_module.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    except ValueError:
        return 1.0
    return float(pval)


def _final_joint_arrays(result: SimulationResult) -> tuple[np.ndarray, np.ndarray]:
    last_idx = result.n_steps - 1
    angles = np.array([result.theta1[last_idx], result.phi[last_idx]])
    velocities = np.array([result.dtheta1[last_idx], result.dphi[last_idx]])
    return angles, velocities


def _tip_position(result: SimulationResult, index: int) -> np.ndarray:
    position = forward_kinematics(
        result.theta1[index], result.phi[index], result.params
    )
    return np.array(position["tip"], dtype=float)


def _final_tip_kinematics(
    result: SimulationResult,
    last_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    tip_pos_final = _tip_position(result, last_idx)
    if result.n_steps < 3:
        return tip_pos_final, np.zeros(2)

    dt = result.t[last_idx] - result.t[last_idx - 1]
    tip_prev = _tip_position(result, last_idx - 1)
    return tip_pos_final, (tip_pos_final - tip_prev) / max(dt, 1e-12)


def _peak_tip_speed(result: SimulationResult) -> float:
    speeds = [0.0]
    for index in range(1, result.n_steps):
        dt = result.t[index] - result.t[index - 1]
        tip_velocity = (
            _tip_position(result, index) - _tip_position(result, index - 1)
        ) / max(
            dt,
            1e-12,
        )
        speeds.append(float(np.linalg.norm(tip_velocity)))
    return float(max(speeds))


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


@dataclass
class ComparisonReport:
    """Result of comparing two torque profiles under perturbation."""

    profile_a_name: str
    profile_b_name: str
    winner: str  # 'A', 'B', or 'tie'
    confidence: float  # fraction of metrics in which winner leads
    metric_comparisons: dict[str, Any] = field(default_factory=dict)
    # Mann-Whitney U p-values for each scalar metric
    pvalues: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_a": self.profile_a_name,
            "profile_b": self.profile_b_name,
            "winner": self.winner,
            "confidence": self.confidence,
            "metric_comparisons": self.metric_comparisons,
            "pvalues": self.pvalues,
        }


# ---------------------------------------------------------------------------
# Perturbation modes (additive / multiplicative / both)
# ---------------------------------------------------------------------------


def _perturb_coeffs_by_mode(
    coeffs: list[list[float]],
    config: PerturbationConfig,
    seed: int,
) -> list[list[float]]:
    """Apply perturbation to polynomial torque coefficients according to mode.

    Supports additive, multiplicative, and both modes as defined in
    ``PerturbationConfig.perturb_mode``.

    Parameters
    ----------
    coeffs : list of lists — per-joint polynomial coefficients
    config : PerturbationConfig
    seed : int — per-trial seed for reproducibility

    Returns
    -------
    list of lists — perturbed coefficients (same shape as input)

    Design by Contract
    ------------------
    Pre:  config.perturb_mode in {'additive', 'multiplicative', 'both'}
    Post: output has same shape as input
    """
    total = sum(len(c) for c in coeffs)
    if total == 0:
        return [list(c) for c in coeffs]

    mode = config.perturb_mode

    if mode in ("additive", "both"):
        coeffs = perturb_torque_coeffs(
            coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            seed=seed,
        )

    if mode in ("multiplicative", "both"):
        # Multiplicative: coeff *= (1 + amplitude * noise)
        noise = generate_noise(
            config.noise_type, total, config.noise_amplitude, seed + 1
        )
        idx = 0
        result = []
        for joint_coeffs in coeffs:
            n = len(joint_coeffs)
            perturbed = [c * (1.0 + noise[idx + i]) for i, c in enumerate(joint_coeffs)]
            result.append(perturbed)
            idx += n
        coeffs = result

    return coeffs


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------


class PendulumPerturbationAnalyzer:
    """Reference implementation of PerturbationAnalyzer for the double pendulum.

    Usage::

        analyzer = PendulumPerturbationAnalyzer(params, t_end=1.0)
        analyzer.set_base_torque_profile({"coeffs": [[0.5, 0.1], [0.3, -0.05]]})
        summary = analyzer.run_batch(PerturbationConfig(n_trials=50))
        logger.info("RS: %s", summary.robustness_score)

    Design by Contract
    ------------------
    Pre:  ``set_base_torque_profile`` called before ``run_batch`` or
          ``perturb_torque``.
    Post: ``run_batch`` returns a ``PerturbationSummary`` with all
          ``MANDATORY_METRICS`` present.
    """

    ENGINE_NAME: str = "pendulum_double"

    def __init__(
        self,
        params: Any,  # PendulumParams
        t_end: float = 1.5,
        dt: float = 0.005,
        n_coeffs_per_joint: int = 4,
    ) -> None:
        self._params = params
        self._t_end = t_end
        self._dt = dt
        self._n_coeffs = n_coeffs_per_joint
        self._base_coeffs: list[list[float]] | None = None
        self._nominal_result: SimulationResult | None = None

    # ------------------------------------------------------------------
    # Protocol API
    # ------------------------------------------------------------------

    def set_base_torque_profile(self, profile: object) -> None:
        """Set the nominal torque polynomial coefficients.

        Parameters
        ----------
        profile : dict with key 'coeffs' : list[list[float]]
            Polynomial coefficients per joint, highest power first.
            Example: ``{"coeffs": [[c0, c1, c2], [d0, d1, d2]]}``

        Design by Contract
        ------------------
        Pre:  profile is a dict with 'coeffs' key.
        Post: self._base_coeffs is set; nominal simulation is cached.
        """
        if not isinstance(profile, dict):
            raise ValueError("profile must be a dict with 'coeffs' key")
        coeffs = profile["coeffs"]
        if not (isinstance(coeffs, list) and len(coeffs) >= 1):
            raise ValueError("profile['coeffs'] must be a non-empty list of lists")
        self._base_coeffs = [list(c) for c in coeffs]
        # Pre-run nominal to cache for trajectory RMSE
        self._nominal_result = self._simulate(self._base_coeffs)
        logger.debug(
            "Base torque profile set: %d joints, %d coeffs each",
            len(self._base_coeffs),
            len(self._base_coeffs[0]) if self._base_coeffs else 0,
        )

    def perturb_torque(self, config: PerturbationConfig, seed: int) -> object:
        """Apply perturbation to base profile, return perturbed coefficients dict.

        Parameters
        ----------
        config : PerturbationConfig
        seed : int — per-trial seed

        Returns
        -------
        dict with 'coeffs' key containing the perturbed coefficient lists

        Design by Contract
        ------------------
        Pre:  ``set_base_torque_profile`` has been called.
        Post: returned dict has same structure as the base profile.
        """
        if not (self._base_coeffs is not None):
            raise ValueError("Call set_base_torque_profile() before perturb_torque()")
        perturbed = _perturb_coeffs_by_mode(self._base_coeffs, config, seed)
        return {"coeffs": perturbed}

    def extract_metrics(self, sim_result: object) -> dict[str, float | np.ndarray]:
        """Extract mandatory metrics from a SimulationResult.

        Parameters
        ----------
        sim_result : SimulationResult

        Returns
        -------
        dict mapping each MANDATORY_METRIC name to its value.

        Design by Contract
        ------------------
        Pre:  sim_result is a non-None SimulationResult with >= 2 time steps.
        Post: all MANDATORY_METRICS present in output; all values are finite.
        """
        if not isinstance(sim_result, SimulationResult):
            raise TypeError(
                f"sim_result must be SimulationResult, got {type(sim_result)}"
            )
        if not (sim_result.n_steps >= 2):
            raise ValueError("Simulation must have >= 2 steps")

        result = sim_result
        last_idx = result.n_steps - 1
        joint_angles_final, joint_velocities_final = _final_joint_arrays(result)
        tip_pos_final, tip_vel_final = _final_tip_kinematics(result, last_idx)
        tip_speed_final = float(np.linalg.norm(tip_vel_final))
        peak_speed = _peak_tip_speed(result)
        total_energy_final = float(total_energy(result.states[last_idx], result.params))
        trajectory_rmse, trajectory_max_deviation = self._trajectory_deviation(result)

        metrics: dict[str, float | np.ndarray] = {
            "end_effector_position_final": tip_pos_final,
            "end_effector_velocity_final": tip_vel_final,
            "end_effector_speed_final": tip_speed_final,
            "peak_end_effector_speed": peak_speed,
            "total_energy_final": total_energy_final,
            "joint_angles_final": joint_angles_final,
            "joint_velocities_final": joint_velocities_final,
            "trajectory_rmse": trajectory_rmse,
            "trajectory_max_deviation": trajectory_max_deviation,
            "motion_duration": float(result.t[last_idx]),
        }

        if not (all(k in metrics for k in MANDATORY_METRICS)):
            raise ValueError(
                f"Missing mandatory metrics: {set(MANDATORY_METRICS) - set(metrics)}"
            )
        return metrics

    def _trajectory_deviation(self, result: SimulationResult) -> tuple[float, float]:
        if self._nominal_result is None:
            return 0.0, 0.0

        nominal = self._nominal_result
        n_compare = min(result.n_steps, nominal.n_steps)
        diff = result.states[:n_compare, :2] - nominal.states[:n_compare, :2]
        # ⚡ Bolt: np.einsum is ~2x faster than np.linalg.norm(..., axis=1) for calculating Euclidean distances along an axis
        deviations = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        rmse = float(np.sqrt(np.mean(deviations**2)))
        return rmse, float(np.max(deviations))

    def run_batch(self, config: PerturbationConfig) -> PerturbationSummary:
        """Run full Monte Carlo batch and return summary with all mandatory metrics.

        Parameters
        ----------
        config : PerturbationConfig

        Returns
        -------
        PerturbationSummary with robustness_score and per-metric MetricStatistics.

        Design by Contract
        ------------------
        Pre:  ``set_base_torque_profile`` has been called.
        Pre:  config.n_trials > 0
        Post: summary.metrics contains all MANDATORY_METRICS.
        Post: summary.robustness_score in [0.0, 1.0].
        """
        if not (self._base_coeffs is not None):
            raise TypeError("Call set_base_torque_profile() before run_batch()")
        if not (config.n_trials > 0):
            raise ValueError("DbC Blocked: Precondition failed.")

        base_seed = config.seed if config.seed is not None else 0
        t_start = time.monotonic()

        trial_results: list[dict[str, float | np.ndarray]] = []
        n_failed = 0

        for i in range(config.n_trials):
            seed = base_seed + i
            perturbed = _perturb_coeffs_by_mode(self._base_coeffs, config, seed)
            try:
                sim = self._simulate(perturbed)
                metrics = self.extract_metrics(sim)
                trial_results.append(metrics)
            except (AssertionError, ValueError, RuntimeError, FloatingPointError):
                logger.warning("Trial %d failed, skipping", i, exc_info=True)
                n_failed += 1

        elapsed = time.monotonic() - t_start
        success_rate = (
            len(trial_results) / config.n_trials if config.n_trials > 0 else 0.0
        )

        logger.info(
            "Batch complete: %d/%d succeeded in %.1fs",
            len(trial_results),
            config.n_trials,
            elapsed,
        )

        if not trial_results:
            # Return degenerate summary if all trials failed
            return PerturbationSummary(
                engine_name=self.ENGINE_NAME,
                config=config,
                robustness_score=0.0,
                metrics={},
                success_rate=0.0,
                execution_time_sec=elapsed,
            )

        # Aggregate per-metric statistics
        metric_stats: dict[str, MetricStatistics] = {}
        for metric_name in MANDATORY_METRICS:
            values = np.array(
                [
                    (
                        np.asarray(r[metric_name]).flatten()[0]
                        if np.asarray(r[metric_name]).ndim > 0
                        else r[metric_name]
                    )
                    for r in trial_results
                ]
            )
            metric_stats[metric_name] = compute_metric_statistics(values)

        # Robustness score — use tip speed CV as primary signal
        speed_stats = metric_stats["end_effector_speed_final"]
        cv_primary = float(
            np.mean(np.atleast_1d(speed_stats.cv))
            if np.isfinite(np.atleast_1d(speed_stats.cv)).all()
            else 1.0
        )
        robustness = compute_robustness_score(max(0.0, cv_primary))

        return PerturbationSummary(
            engine_name=self.ENGINE_NAME,
            config=config,
            robustness_score=robustness,
            metrics=metric_stats,
            success_rate=success_rate,
            execution_time_sec=elapsed,
        )

    # ------------------------------------------------------------------
    # Profile comparison
    # ------------------------------------------------------------------

    def compare_profiles(
        self,
        profile_a: object,
        profile_b: object,
        config: PerturbationConfig,
        name_a: str = "A",
        name_b: str = "B",
    ) -> ComparisonReport:
        """Compare two torque profiles under perturbation using statistical tests.

        Parameters
        ----------
        profile_a, profile_b : dicts with 'coeffs' key
        config : PerturbationConfig
        name_a, name_b : str — labels for the comparison report

        Returns
        -------
        ComparisonReport with winner, confidence, and per-metric statistics.

        Design by Contract
        ------------------
        Pre:  Both profiles are valid dicts with 'coeffs' key.
        Post: report.confidence in [0.0, 1.0].
        """
        from scipy import stats as _stats  # lazy import — optional dep

        metric_comparisons, pvalues, a_wins, b_wins = (
            self._build_scalar_metric_comparisons(
                _stats,
                profile_a,
                profile_b,
                config,
            )
        )
        winner, confidence = _overall_winner(a_wins, b_wins)

        # Restore profile A
        self.set_base_torque_profile(profile_a)

        return ComparisonReport(
            profile_a_name=name_a,
            profile_b_name=name_b,
            winner=winner,
            confidence=confidence,
            metric_comparisons=metric_comparisons,
            pvalues=pvalues,
        )

    def _build_scalar_metric_comparisons(
        self,
        stats_module: Any,
        profile_a: object,
        profile_b: object,
        config: PerturbationConfig,
    ) -> tuple[dict[str, Any], dict[str, float], int, int]:
        metric_comparisons: dict[str, Any] = {}
        pvalues: dict[str, float] = {}
        a_wins = 0
        b_wins = 0

        for metric in SCALAR_COMPARISON_METRICS:
            vals_a = self._collect_scalar_metric_values(profile_a, metric, config)
            vals_b = self._collect_scalar_metric_values(profile_b, metric, config)
            pval = _mann_whitney_pvalue(stats_module, vals_a, vals_b)
            mean_a = float(np.mean(vals_a))
            mean_b = float(np.mean(vals_b))
            winner_metric = _metric_winner(metric, mean_a, mean_b)

            pvalues[metric] = pval
            metric_comparisons[metric] = {
                "mean_a": mean_a,
                "mean_b": mean_b,
                "winner": winner_metric,
                "pvalue": pval,
            }
            a_wins += int(winner_metric == "A")
            b_wins += int(winner_metric == "B")

        return metric_comparisons, pvalues, a_wins, b_wins

    def _collect_scalar_metric_values(
        self,
        profile: object,
        metric: str,
        config: PerturbationConfig,
    ) -> np.ndarray:
        base_seed = config.seed if config.seed is not None else 0
        self.set_base_torque_profile(profile)
        values: list[float] = []
        for i in range(config.n_trials):
            perturbed = _perturb_coeffs_by_mode(
                self._base_coeffs,  # type: ignore[arg-type]
                config,
                base_seed + i,
            )
            try:
                sim = self._simulate(perturbed)
                value = self.extract_metrics(sim)[metric]
                if isinstance(value, np.ndarray):
                    value = float(np.linalg.norm(value))
                values.append(float(value))
            except (
                AssertionError,
                FloatingPointError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                continue
        return np.array(values) if values else np.array([0.0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate(self, coeffs: list[list[float]]) -> SimulationResult:
        """Run a double pendulum simulation with the given polynomial coefficients."""
        # make_polynomial_torque takes *coeffs_per_joint (one list per joint);
        # cast to TorqueFunc since mypy can't infer exact tuple length.
        torque_fn = cast(TorqueFunc, make_polynomial_torque(*coeffs))

        flat = [c for joint in coeffs for c in joint]
        n_per = len(coeffs[0]) if coeffs else 0

        initial_state = np.zeros(4)
        return run_simulation(
            params=self._params,
            initial_state=initial_state,
            t_end=self._t_end,
            torque_func=torque_fn,
            dt=self._dt,
            coeffs=flat,
            n_coeffs_per_joint=n_per,
        )
