"""Shared base class for engine perturbation analyzers.

``PerturbationAnalyzerBase`` captures the ~60–70 % of logic that is identical
across the Drake, MuJoCo, MyoSuite, OpenSim, and Pinocchio analyzer
implementations, eliminating the 3 603-line DRY violation tracked in #2273.

All five engine-specific analyzers inherit from this class and only need to
override:

* ``ENGINE_NAME`` — class variable string, e.g. ``"drake"``.
* ``_simulate()`` — run one forward simulation and return a *sim result*
  object whose interface is described by ``SimResultProtocol``.
* ``_get_q_traj()`` / ``_get_v_traj()`` — extract the position/velocity
  trajectory arrays from the engine-specific result type.
* ``_validate_sim_result_type()`` — type-check the sim result.

Everything else — ``set_base_torque_profile``, ``perturb_torque``,
``extract_metrics``, ``run_batch``, and ``compare_profiles`` — lives here.

Design by Contract
------------------
- ``set_base_torque_profile()`` must be called before ``run_batch()`` or
  ``perturb_torque()``.  Raises ``ValueError`` otherwise.
- ``run_batch()`` returns a ``PerturbationSummary`` containing all
  ``MANDATORY_METRICS`` as keys.
- ``extract_metrics()`` requires a sim-result with ``n_steps >= 2``.
- All returned metric values are finite.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

import numpy as np

from src.shared.python.perturbation.config import (
    PerturbationConfig,
    PerturbationSummary,
    TrialFailure,
)
from src.shared.python.perturbation.canonical_trial_executor import (
    TrialEvidenceCollector,
    VariationSampler,
    execute_serial_variation,
)
from src.shared.python.perturbation.noise import generate_noise
from src.shared.python.perturbation.robustness_score import compute_robustness_score
from src.shared.python.perturbation.statistics import (
    MetricStatistics,
    compute_metric_statistics,
)
from src.shared.python.perturbation.trial_evidence import CanonicalTrialEvidence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mandatory metric names — identical across all engine analyzers
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

# Metrics that are vectors — excluded from the scalar aggregation loop
_ARRAY_METRICS: frozenset[str] = frozenset(
    {
        "end_effector_position_final",
        "end_effector_velocity_final",
        "joint_angles_final",
        "joint_velocities_final",
    }
)


# ---------------------------------------------------------------------------
# Shared comparison report (same dataclass for all engines)
# ---------------------------------------------------------------------------


@dataclass
class ComparisonReport:
    """Statistical comparison of two torque profiles.

    Attributes
    ----------
    winner : str
        ``'A'`` or ``'B'``.
    confidence : float
        ``1 − median p-value`` across scalar metrics; clamped to ``[0, 1]``.
    metric_comparisons : dict
        Per-metric stats and winner label.
    pvalues : dict
        Per-metric Mann-Whitney U p-values.
    """

    winner: str
    confidence: float
    metric_comparisons: dict[str, Any] = field(default_factory=dict)
    pvalues: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalPerturbationBatch:
    """Canonical trial evidence plus a transient legacy compatibility view.

    ``records`` is the durable, provenance-bound authority. ``engine_results``
    and ``errors`` retain the exact in-process engine returns for host adapters
    and diagnostics; they are intentionally not persistence contracts. The
    legacy summary is a compatibility projection over only complete hit and
    no-impact trials and must not replace the canonical outcome records.
    """

    records: tuple[CanonicalTrialEvidence, ...]
    engine_results: tuple[object | None, ...]
    errors: tuple[Exception | None, ...]
    legacy_summary: PerturbationSummary

    def __post_init__(self) -> None:
        n_records = len(self.records)
        if n_records == 0:
            raise ValueError("canonical batch must contain at least one trial")
        if len(self.engine_results) != n_records or len(self.errors) != n_records:
            raise ValueError("canonical batch collections must have equal lengths")
        if any(
            record.trial_index != trial_index
            for trial_index, record in enumerate(self.records)
        ):
            raise ValueError("canonical batch records must be contiguous and ordered")
        if any(
            error is not None and not isinstance(error, Exception)
            for error in self.errors
        ):
            raise TypeError("canonical batch errors must be exceptions or None")
        legacy_summary = self.legacy_summary
        legacy_config = legacy_summary.config
        if legacy_config.n_trials != n_records:
            raise ValueError("legacy summary trial count must match canonical records")


# ---------------------------------------------------------------------------
# Lightweight protocol for sim-result objects
# ---------------------------------------------------------------------------


class SimResultProtocol:
    """Documentation-only interface for objects returned by ``_simulate()``.

    Subclasses are *not* required to inherit from this class; duck typing is
    used.  The object must expose the following attributes:

    ``t`` : ndarray, shape (n,)
        Time stamps.
    ``ee_pos_traj`` : ndarray, shape (n, 3)
        End-effector Cartesian position trajectory.
    ``ee_vel_traj`` : ndarray, shape (n, 3)
        End-effector velocity trajectory (finite-difference).
    ``kinetic_energy_traj`` : ndarray, shape (n,)
    ``potential_energy_traj`` : ndarray, shape (n,)
    ``n_steps`` : int
        ``len(t)``
    """


class _RetainingTrialCollector:
    """Delegate evidence mapping while retaining declared execution errors."""

    def __init__(
        self,
        delegate: TrialEvidenceCollector,
        errors: dict[int, Exception],
    ) -> None:
        self._delegate = delegate
        self._errors = errors

    def collect_success(
        self,
        trial_index: int,
        plan_seed: int,
        sampled_row: np.ndarray,
        result: object,
    ) -> CanonicalTrialEvidence:
        return self._delegate.collect_success(
            trial_index,
            plan_seed,
            sampled_row,
            result,
        )

    def collect_failure(
        self,
        trial_index: int,
        plan_seed: int,
        sampled_row: np.ndarray,
        error: Exception,
    ) -> CanonicalTrialEvidence:
        self._errors[trial_index] = error
        return self._delegate.collect_failure(
            trial_index,
            plan_seed,
            sampled_row,
            error,
        )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class PerturbationAnalyzerBase(ABC):
    """Abstract base class for engine-specific perturbation analyzers.

    Subclasses must set ``ENGINE_NAME`` and implement ``_simulate()``,
    ``_get_q_traj()``, ``_get_v_traj()``, and
    ``_validate_sim_result_type()``.  All shared protocol methods
    (``set_base_torque_profile``, ``perturb_torque``, ``extract_metrics``,
    ``run_batch``, ``compare_profiles``) are implemented here.

    Design by Contract
    ------------------
    Pre:  ``set_base_torque_profile`` must be called before ``run_batch`` or
          ``perturb_torque``.
    Post: ``run_batch`` returns a ``PerturbationSummary`` with all
          ``MANDATORY_METRICS`` present (scalar metrics only; array metrics
          summarised by norm).
    """

    ENGINE_NAME: str = "base"

    def __init__(self) -> None:
        self._base_coeffs: list[list[float]] | None = None
        self._nominal_result: Any = None

    # ------------------------------------------------------------------
    # Abstract hooks — subclasses provide engine-specific implementation
    # ------------------------------------------------------------------

    @abstractmethod
    def _simulate(self, coeffs: list[list[float]]) -> Any:
        """Run a single forward simulation with the given polynomial coeffs.

        Parameters
        ----------
        coeffs : list[list[float]]
            Per-joint/actuator coefficient lists.  ``coeffs[j][k]`` is the
            coefficient of ``t^k`` for joint ``j``.

        Returns
        -------
        Sim-result object compatible with ``SimResultProtocol``.
        """

    @abstractmethod
    def _get_q_traj(self, sim_result: Any) -> np.ndarray:
        """Return the joint-position trajectory array, shape (n, nq).

        Used by ``extract_metrics`` to compute trajectory RMSE vs nominal.
        """

    @abstractmethod
    def _get_v_traj(self, sim_result: Any) -> np.ndarray:
        """Return the joint-velocity trajectory array, shape (n, nv)."""

    @abstractmethod
    def _validate_sim_result_type(self, sim_result: object) -> None:
        """Raise ``ValueError`` if *sim_result* is not the expected type.

        Design by Contract
        ------------------
        Pre:  sim_result is any object.
        Post: raises ValueError with descriptive message on type mismatch.
        """

    # ------------------------------------------------------------------
    # Shared protocol methods
    # ------------------------------------------------------------------

    def set_base_torque_profile(self, profile: object) -> None:
        """Set the nominal torque polynomial coefficients.

        Parameters
        ----------
        profile : dict with ``'coeffs'`` key
            ``profile["coeffs"]`` is a list of per-joint coefficient lists.

        Design by Contract
        ------------------
        Pre:  profile is a dict with ``'coeffs'`` key; coeffs is non-empty.
        Post: ``self._base_coeffs`` is set; nominal simulation is cached.
        """
        if not isinstance(profile, dict):
            raise ValueError(f"profile must be a dict, got {type(profile)}")
        if "coeffs" not in profile:
            raise ValueError("'coeffs' key missing from profile")
        coeffs = profile["coeffs"]
        if not (isinstance(coeffs, list) and len(coeffs) > 0):
            raise ValueError("profile['coeffs'] must be a non-empty list")
        self._base_coeffs = coeffs
        self._nominal_result = self._simulate(coeffs)

    def perturb_torque(self, config: PerturbationConfig, seed: int) -> dict:
        """Apply noise to base coefficients and return a perturbed profile.

        Design by Contract
        ------------------
        Pre:  ``set_base_torque_profile`` has been called.
        Post: returned dict has ``'coeffs'`` with the same shape as base.
        """
        if self._base_coeffs is None:
            raise ValueError(
                "set_base_torque_profile() must be called before perturb_torque()"
            )
        perturbed = _perturb_coeffs(
            self._base_coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            seed=seed,
            perturb_mode=config.perturb_mode,
        )
        return {"coeffs": perturbed}

    def extract_metrics(self, sim_result: object) -> dict[str, float | np.ndarray]:
        """Extract all ``MANDATORY_METRICS`` from a simulation result.

        Parameters
        ----------
        sim_result : engine-specific sim-result object
            Must be compatible with ``SimResultProtocol``.

        Returns
        -------
        dict mapping metric name to scalar float or ndarray.

        Design by Contract
        ------------------
        Pre:  sim_result passes ``_validate_sim_result_type``; n_steps >= 2.
        Post: all ``MANDATORY_METRICS`` present; all values finite.
        """
        self._validate_sim_result_type(sim_result)
        if not (sim_result.n_steps >= 2):  # type: ignore[union-attr,attr-defined]
            raise ValueError("Simulation must have >= 2 steps")

        r = sim_result
        last = r.n_steps - 1  # type: ignore[union-attr,attr-defined]

        q_traj = self._get_q_traj(r)
        v_traj = self._get_v_traj(r)

        joint_angles_final = q_traj[last].copy()
        joint_velocities_final = v_traj[last].copy()
        ee_pos_final = r.ee_pos_traj[last].copy()  # type: ignore[union-attr,attr-defined]
        ee_vel_final = r.ee_vel_traj[last].copy()  # type: ignore[union-attr,attr-defined]
        ee_speed_final = float(math.hypot(*ee_vel_final))

        # ⚡ Bolt: np.einsum is ~3x faster than np.linalg.norm(..., axis=1) for computing euclidean norms
        speeds = np.sqrt(np.einsum("ij,ij->i", r.ee_vel_traj, r.ee_vel_traj))  # type: ignore[union-attr,attr-defined]
        peak_speed = float(np.max(speeds))

        total_energy_final = float(
            r.kinetic_energy_traj[last]  # type: ignore[union-attr,attr-defined]
            + r.potential_energy_traj[last]  # type: ignore[union-attr,attr-defined]
        )

        trajectory_rmse = 0.0
        trajectory_max_deviation = 0.0
        if self._nominal_result is not None:
            nom_q = self._get_q_traj(self._nominal_result)
            n_cmp = min(len(q_traj), len(nom_q))
            diff = q_traj[:n_cmp] - nom_q[:n_cmp]
            # ⚡ Bolt: np.einsum is ~3x faster than np.linalg.norm(..., axis=1)
            deviations = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            # ⚡ Bolt: np.vdot is ~2x faster than np.mean(deviations**2)
            trajectory_rmse = float(
                np.sqrt(np.vdot(deviations, deviations) / deviations.size)
            )
            trajectory_max_deviation = float(np.max(deviations))

        motion_duration = float(
            r.t[last] - r.t[0]  # type: ignore[union-attr,attr-defined]
        )

        return {
            "end_effector_position_final": ee_pos_final,
            "end_effector_velocity_final": ee_vel_final,
            "end_effector_speed_final": ee_speed_final,
            "peak_end_effector_speed": peak_speed,
            "total_energy_final": total_energy_final,
            "joint_angles_final": joint_angles_final,
            "joint_velocities_final": joint_velocities_final,
            "trajectory_rmse": trajectory_rmse,
            "trajectory_max_deviation": trajectory_max_deviation,
            "motion_duration": motion_duration,
        }

    def run_canonical_batch(
        self,
        *,
        plan: object,
        gateway: VariationSampler,
        collector: TrialEvidenceCollector,
        row_to_coeffs: Callable[[np.ndarray], list[list[float]]],
        compatibility_config: PerturbationConfig,
    ) -> CanonicalPerturbationBatch:
        """Execute a Tools-owned plan without disguising it as legacy noise.

        The caller injects the engine-specific mapping from one canonical
        sampled row to this analyzer's coefficient representation and the
        engine-specific evidence collector. Canonical records retain the plan
        and outcome authority. The returned ``PerturbationSummary`` exists only
        for legacy consumers and is computed from complete hit/no-impact rows.

        Design by Contract
        ------------------
        Pre: ``compatibility_config`` has the same trial count and seed as the
             canonical plan; no sampling provenance is inferred.
        Post: one ordered canonical record, transient engine-result slot, and
              error slot are retained for every sampled row.
        """
        n_runs = getattr(plan, "n_runs", None)
        plan_seed = getattr(plan, "seed", None)
        if type(n_runs) is not int or n_runs <= 0:
            raise ValueError("plan n_runs must be a positive integer")
        if type(plan_seed) is not int:
            raise ValueError("plan seed must be an integer")
        if compatibility_config.n_trials != n_runs:
            raise ValueError("compatibility_config n_trials must match plan n_runs")
        if compatibility_config.seed != plan_seed:
            raise ValueError("compatibility_config seed must match plan seed")
        if not callable(row_to_coeffs):
            raise TypeError("row_to_coeffs must be callable")

        t_start = time.monotonic()
        raw_results: dict[int, object] = {}
        errors: dict[int, Exception] = {}
        metric_rows: dict[int, dict[str, float | np.ndarray]] = {}
        next_trial_index = 0

        def run_row(sampled_row: np.ndarray) -> object:
            nonlocal next_trial_index
            trial_index = next_trial_index
            next_trial_index += 1
            result = self._simulate(row_to_coeffs(sampled_row))
            raw_results[trial_index] = result
            metric_rows[trial_index] = self.extract_metrics(result)
            return result

        retaining_collector = _RetainingTrialCollector(collector, errors)
        records = execute_serial_variation(
            plan,
            gateway,
            run_row,
            retaining_collector,
        )
        complete_indices = tuple(
            record.trial_index
            for record in records
            if record.outcome in {"hit", "no_impact"}
        )
        summary = _summarize_metric_rows(
            engine_name=self.ENGINE_NAME,
            config=compatibility_config,
            metric_rows=tuple(metric_rows[index] for index in complete_indices),
            execution_time_sec=time.monotonic() - t_start,
            failures=tuple(
                TrialFailure(
                    trial_index=record.trial_index,
                    seed=record.seed,
                    stage="canonical_execution",
                    error_type=(
                        type(errors[record.trial_index]).__name__
                        if record.trial_index in errors
                        else record.outcome
                    ),
                    message=record.failure_reason or record.outcome,
                )
                for record in records
                if record.outcome not in {"hit", "no_impact"}
            ),
        )
        return CanonicalPerturbationBatch(
            records=records,
            engine_results=tuple(raw_results.get(index) for index in range(n_runs)),
            errors=tuple(errors.get(index) for index in range(n_runs)),
            legacy_summary=summary,
        )

    def run_batch(self, config: PerturbationConfig) -> PerturbationSummary:
        """Run Monte Carlo perturbation analysis.

        Design by Contract
        ------------------
        Pre:  ``set_base_torque_profile`` has been called.
        Post: ``result.success_rate`` in ``[0, 1]``.
        Post: ``result.robustness_score`` in ``[0, 1]``.
        """
        if self._base_coeffs is None:
            raise ValueError(
                "set_base_torque_profile() must be called before run_batch()"
            )

        t_start = time.monotonic()
        base_seed = config.seed if config.seed is not None else 0

        metric_rows: list[dict[str, float | np.ndarray]] = []

        for i in range(config.n_trials):
            perturbed = _perturb_coeffs(
                self._base_coeffs,
                noise_amplitude=config.noise_amplitude,
                noise_type=config.noise_type,
                seed=base_seed + i,
                perturb_mode=config.perturb_mode,
            )
            try:
                sim = self._simulate(perturbed)
                metrics = self.extract_metrics(sim)
                metric_rows.append(metrics)
            except (ValueError, RuntimeError):
                logger.debug("Trial %d failed", i, exc_info=True)

        return _summarize_metric_rows(
            engine_name=self.ENGINE_NAME,
            config=config,
            execution_time_sec=time.monotonic() - t_start,
            metric_rows=tuple(metric_rows),
        )

    def compare_profiles(
        self,
        profile_a: object,
        profile_b: object,
        config: PerturbationConfig,
        name_a: str = "A",
        name_b: str = "B",
    ) -> ComparisonReport:
        """Compare two torque profiles via Mann-Whitney U test.

        Design by Contract
        ------------------
        Pre:  Both profiles are valid dicts with ``'coeffs'`` key.
        Post: ``report.confidence`` in ``[0.0, 1.0]``.
        """
        from scipy import stats as _stats  # noqa: PLC0415 — lazy import

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
            values: list[float] = []
            for i in range(config.n_trials):
                perturbed = _perturb_coeffs(
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
                        v = float(math.hypot(*v.flatten()))
                    values.append(float(v))
                except (ValueError, RuntimeError):
                    continue
            return np.array(values) if values else np.array([0.0])

        metric_comparisons: dict[str, Any] = {}
        pvalues: dict[str, float] = {}
        a_wins = 0
        b_wins = 0

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

            # Higher speed is better; lower error / energy is better
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _summarize_metric_rows(
    *,
    engine_name: str,
    config: PerturbationConfig,
    metric_rows: tuple[dict[str, float | np.ndarray], ...],
    execution_time_sec: float,
    failures: tuple[TrialFailure, ...] = (),
) -> PerturbationSummary:
    """Project successful raw metrics into the established summary type."""
    n_success = len(metric_rows)
    success_rate = n_success / config.n_trials
    if n_success == 0:
        logger.warning("All trials failed — returning zero-robustness summary")
        return PerturbationSummary(
            engine_name=engine_name,
            config=config,
            robustness_score=0.0,
            metrics={},
            success_rate=0.0,
            execution_time_sec=execution_time_sec,
            failures=list(failures),
        )

    scalar_metric_names = tuple(
        metric for metric in MANDATORY_METRICS if metric not in _ARRAY_METRICS
    )
    metric_lists: dict[str, list[float]] = {
        metric: [] for metric in scalar_metric_names
    }
    for row in metric_rows:
        missing = set(MANDATORY_METRICS).difference(row)
        if missing:
            raise ValueError(
                f"metric row is missing mandatory metrics: {sorted(missing)}"
            )
        for metric in scalar_metric_names:
            value = row[metric]
            if isinstance(value, np.ndarray):
                value = float(math.hypot(*value.flatten()))
            metric_lists[metric].append(float(value))

    metric_stats: dict[str, MetricStatistics] = {
        metric: compute_metric_statistics(np.array(values))
        for metric, values in metric_lists.items()
        if values
    }
    cv_values = _compute_cv_values(metric_stats)
    cv_weighted = float(np.mean(cv_values)) if cv_values else 0.0
    return PerturbationSummary(
        engine_name=engine_name,
        config=config,
        robustness_score=compute_robustness_score(cv_weighted),
        metrics=metric_stats,
        success_rate=success_rate,
        execution_time_sec=execution_time_sec,
        failures=list(failures),
    )


def _perturb_coeffs(
    coeffs: list[list[float]],
    noise_amplitude: float,
    noise_type: str = "white",
    seed: int | None = None,
    perturb_mode: str = "additive",
) -> list[list[float]]:
    """Perturb polynomial torque coefficients with noise.

    Each coefficient is independently perturbed according to *perturb_mode*:

    * ``'additive'``        — ``c' = c + noise``
    * ``'multiplicative'``  — ``c' = c * (1 + noise)``
    * ``'both'``            — additive then multiplicative

    Uses ``generate_noise`` from ``src.shared.python.perturbation.noise``
    to avoid the circular-import chain in ``pendulum_simulator``.

    Design by Contract
    ------------------
    Pre:  noise_amplitude >= 0
    Pre:  noise_type in {'white', 'pink', 'brown'}
    Pre:  perturb_mode in {'additive', 'multiplicative', 'both'}
    Post: output has same shape as input.
    """
    if noise_amplitude == 0.0:
        return [list(c) for c in coeffs]

    total = sum(len(c) for c in coeffs)
    if total == 0:
        return [list(c) for c in coeffs]

    noise = generate_noise(noise_type, total, noise_amplitude, seed)

    idx = 0
    result: list[list[float]] = []
    for joint_coeffs in coeffs:
        n = len(joint_coeffs)
        chunk = noise[idx : idx + n]
        if perturb_mode == "additive":
            perturbed = [c + chunk[i] for i, c in enumerate(joint_coeffs)]
        elif perturb_mode == "multiplicative":
            perturbed = [c * (1.0 + chunk[i]) for i, c in enumerate(joint_coeffs)]
        else:  # "both"
            perturbed = [
                c * (1.0 + chunk[i]) + chunk[i] for i, c in enumerate(joint_coeffs)
            ]
        result.append(perturbed)
        idx += n

    return result


def _compute_cv_values(
    metric_stats: dict[str, MetricStatistics],
) -> list[float]:
    """Compute coefficient-of-variation for each metric stat entry.

    Returns a list of non-negative floats (one per stat entry).

    Design by Contract
    ------------------
    Pre:  metric_stats values expose ``.std`` and ``.mean``.
    Post: returned list has the same length as metric_stats.
    """
    cv_values: list[float] = []
    for stats in metric_stats.values():
        std = float(stats.std) if not isinstance(stats.std, float) else stats.std
        mean = float(stats.mean) if not isinstance(stats.mean, float) else stats.mean
        if std > 0 and abs(mean) > 1e-12:
            cv_values.append(std / abs(mean))
        else:
            cv_values.append(0.0)
    return cv_values
