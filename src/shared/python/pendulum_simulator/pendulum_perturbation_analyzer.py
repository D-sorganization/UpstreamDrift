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
        if not (isinstance(coeffs):
            raise ValueError(list) and len(coeffs) >= 1, ()
            "profile['coeffs'] must be a non-empty list of lists"
        )
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
            raise ValueError(
            "Call set_base_torque_profile() before perturb_torque()"
        )
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
            raise ValueError("f"sim_result must be SimulationResult, got {type(sim_result)}")
        if not (sim_result.n_steps >= 2):
            raise ValueError("Simulation must have >= 2 steps")

        result = sim_result
        last_idx = result.n_steps - 1

        # Joint angles and velocities at final time step
        joint_angles_final = np.array([result.theta1[last_idx], result.phi[last_idx]])
        joint_velocities_final = np.array(
            [result.dtheta1[last_idx], result.dphi[last_idx]]
        )

        # End-effector (tip) positions via FK at each time step
        pos_final = forward_kinematics(
            result.theta1[last_idx], result.phi[last_idx], result.params
        )
        tip_pos_final = np.array(pos_final["tip"], dtype=float)

        # End-effector velocities at final step (finite-difference of tip pos)
        if result.n_steps >= 3:
            dt = result.t[last_idx] - result.t[last_idx - 1]
            pos_prev = forward_kinematics(
                result.theta1[last_idx - 1],
                result.phi[last_idx - 1],
                result.params,
            )
            tip_prev = np.array(pos_prev["tip"], dtype=float)
            tip_vel_final = (tip_pos_final - tip_prev) / max(dt, 1e-12)
        else:
            tip_vel_final = np.zeros(2)

        tip_speed_final = float(np.linalg.norm(tip_vel_final))

        # Peak tip speed across all time steps
        speeds = []
        for i in range(result.n_steps):
            if i == 0:
                speeds.append(0.0)
                continue
            dt = result.t[i] - result.t[i - 1]
            pos_i = forward_kinematics(result.theta1[i], result.phi[i], result.params)
            pos_im1 = forward_kinematics(
                result.theta1[i - 1], result.phi[i - 1], result.params
            )
            v = (
                np.array(pos_i["tip"], dtype=float)
                - np.array(pos_im1["tip"], dtype=float)
            ) / max(dt, 1e-12)
            speeds.append(float(np.linalg.norm(v)))
        peak_speed = float(max(speeds))

        # Total energy at final step
        total_energy_final = float(total_energy(result.states[last_idx], result.params))

        # Trajectory RMSE vs nominal (if nominal available)
        trajectory_rmse = 0.0
        trajectory_max_deviation = 0.0
        if self._nominal_result is not None:
            nom = self._nominal_result
            n_compare = min(result.n_steps, nom.n_steps)
            deviations = np.linalg.norm(
                result.states[:n_compare, :2] - nom.states[:n_compare, :2], axis=1
            )
            trajectory_rmse = float(np.sqrt(np.mean(deviations**2)))
            trajectory_max_deviation = float(np.max(deviations))

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
            raise ValueError(
            "Call set_base_torque_profile() before run_batch()"
        )
        if not (config.n_trials > 0):
            raise ValueError('DbC Blocked: Precondition failed.')

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

        metric_comparisons: dict[str, Any] = {}
        pvalues: dict[str, float] = {}
        a_wins = 0
        b_wins = 0

        # Compare scalar metrics via Mann-Whitney U test
        scalar_metrics = [
            "end_effector_speed_final",
            "peak_end_effector_speed",
            "total_energy_final",
            "trajectory_rmse",
            "trajectory_max_deviation",
        ]

        base_seed = config.seed if config.seed is not None else 0

        # Re-run raw trials for statistical testing
        def _collect_scalar(profile: object, metric: str) -> np.ndarray:
            self.set_base_torque_profile(profile)
            values = []
            for i in range(config.n_trials):
                perturbed = _perturb_coeffs_by_mode(
                    self._base_coeffs,  # type: ignore[arg-type]
                    config,
                    base_seed + i,
                )
                try:
                    sim = self._simulate(perturbed)
                    m = self.extract_metrics(sim)
                    v = m[metric]
                    if isinstance(v, np.ndarray):
                        v = float(np.linalg.norm(v))
                    values.append(float(v))
                except Exception as e:  # noqa: BLE001
                    pass
            return np.array(values) if values else np.array([0.0])

        for metric in scalar_metrics:
            vals_a = _collect_scalar(profile_a, metric)
            vals_b = _collect_scalar(profile_b, metric)

            try:
                stat, pval = _stats.mannwhitneyu(
                    vals_a, vals_b, alternative="two-sided"
                )
            except ValueError:
                pval = 1.0

            pvalues[metric] = float(pval)
            mean_a = float(np.mean(vals_a))
            mean_b = float(np.mean(vals_b))

            # For most metrics lower is better (rmse, energy), for speed higher is better
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

        total = a_wins + b_wins
        if total == 0:
            winner, confidence = "tie", 0.5
        elif a_wins > b_wins:
            winner = "A"
            confidence = a_wins / total
        elif b_wins > a_wins:
            winner = "B"
            confidence = b_wins / total
        else:
            winner = "tie"
            confidence = 0.5

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
