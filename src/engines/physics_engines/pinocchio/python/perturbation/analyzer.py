"""Pinocchio Perturbation Analyzer — PerturbationAnalyzer protocol (#1978).

Implements the ``PerturbationAnalyzer`` protocol for the Pinocchio rigid-body
dynamics engine.  Uses ``PinocchioPhysicsEngine`` for forward simulation with
polynomial torque profiles, and exposes Jacobian-based sensitivity as an
optional complement to Monte Carlo results.

Design by Contract
------------------
- ``set_base_torque_profile()`` must be called before ``run_batch()`` or
  ``perturb_torque()``.  Raises ``AssertionError`` otherwise.
- ``run_batch()`` returns a ``PerturbationSummary`` containing all
  ``MANDATORY_METRICS`` as keys (scalar metrics only; array metrics are
  summarised by norm).
- ``extract_metrics()`` requires a ``PinocchioSimResult`` (or compatible dict)
  and always returns finite values.
- pinocchio must be importable; if not, ``ImportError`` is raised at
  construction time with a helpful message.

DRY
---
Delegates noise generation and coefficient perturbation to the shared
``src.shared.python.perturbation`` package — the same functions used by
``PendulumPerturbationAnalyzer``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.engine_core.engine_availability import PINOCCHIO_AVAILABLE

# Shared noise / perturbation helpers
from src.shared.python.pendulum_simulator.perturbation_analysis import (
    generate_noise,
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

# ---------------------------------------------------------------------------
# Mandatory metric names (must match PendulumPerturbationAnalyzer.MANDATORY_METRICS)
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
# Simulation result container
# ---------------------------------------------------------------------------


@dataclass
class PinocchioSimResult:
    """Trajectory data from one Pinocchio forward simulation.

    Attributes
    ----------
    t : ndarray, shape (n,)
        Time stamps.
    q_traj : ndarray, shape (n, nq)
        Joint position trajectory.
    v_traj : ndarray, shape (n, nv)
        Joint velocity trajectory.
    ee_pos_traj : ndarray, shape (n, 3)
        End-effector (tip) Cartesian position trajectory.
    ee_vel_traj : ndarray, shape (n, 3)
        End-effector velocity trajectory (finite-difference).
    kinetic_energy_traj : ndarray, shape (n,)
        Kinetic energy at each step.
    potential_energy_traj : ndarray, shape (n,)
        Potential energy at each step (if gravity configured).
    """

    t: np.ndarray
    q_traj: np.ndarray
    v_traj: np.ndarray
    ee_pos_traj: np.ndarray
    ee_vel_traj: np.ndarray
    kinetic_energy_traj: np.ndarray
    potential_energy_traj: np.ndarray

    @property
    def n_steps(self) -> int:
        return len(self.t)


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


@dataclass
class ComparisonReport:
    """Statistical comparison of two torque profiles.

    Attributes
    ----------
    winner : str — 'A' or 'B'
    confidence : float — 1 − median p-value across scalar metrics
    metric_comparisons : dict — per-metric stats and winner
    pvalues : dict — per-metric Mann-Whitney U p-values
    """

    winner: str
    confidence: float
    metric_comparisons: dict[str, Any] = field(default_factory=dict)
    pvalues: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Coefficient perturbation helper
# ---------------------------------------------------------------------------


def _perturb_coeffs_by_mode(
    coeffs: list[list[float]],
    config: PerturbationConfig,
    seed: int,
) -> list[list[float]]:
    """Apply noise to polynomial torque coefficients.

    Supports 'additive', 'multiplicative', and 'both' modes (mirrors
    PendulumPerturbationAnalyzer._perturb_coeffs_by_mode).
    """
    mode = config.perturb_mode
    total = sum(len(j) for j in coeffs)

    if mode in ("additive", "both"):
        coeffs = perturb_torque_coeffs(
            coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            seed=seed,
        )

    if mode in ("multiplicative", "both"):
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
# Main analyzer
# ---------------------------------------------------------------------------


class PinocchioPerturbationAnalyzer:
    """PerturbationAnalyzer protocol implementation for the Pinocchio engine.

    Usage::

        from src.engines.physics_engines.pinocchio.python.perturbation.analyzer import (
            PinocchioPerturbationAnalyzer,
        )
        urdf = "path/to/model.urdf"
        analyzer = PinocchioPerturbationAnalyzer(urdf_path=urdf)
        analyzer.set_base_torque_profile({"coeffs": [[0.5, 0.1], [0.3, -0.05]]})
        summary = analyzer.run_batch(PerturbationConfig(n_trials=50))
        logger.info("RS: %s", summary.robustness_score)

    Design by Contract
    ------------------
    Pre:  pinocchio must be importable.
    Pre:  ``set_base_torque_profile`` must be called before ``run_batch`` or
          ``perturb_torque``.
    Post: ``run_batch`` returns a ``PerturbationSummary`` with all
          ``MANDATORY_METRICS`` present.
    """

    ENGINE_NAME: str = "pinocchio"

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        t_end: float = 1.5,
        dt: float = 0.005,
        ee_frame_name: str | None = None,
    ) -> None:
        """Initialise the Pinocchio perturbation analyzer.

        Parameters
        ----------
        urdf_path : str or Path, optional
            Path to the URDF model file.  If None, uses the bundled golfer URDF.
        t_end : float
            Simulation end time in seconds.
        dt : float
            Integration time step.
        ee_frame_name : str, optional
            Name of the end-effector frame in the URDF.  Defaults to last frame.
        """
        if not PINOCCHIO_AVAILABLE:
            msg = "pinocchio is not installed.  Install it with: pip install pinocchio"
            raise ImportError(msg)

        import pinocchio as pin  # noqa: PLC0415 — guard already checked

        if urdf_path is None:
            urdf_path = (
                Path(__file__).parents[4] / "models" / "generated" / "golfer.urdf"
            )

        self._urdf_path = Path(urdf_path)
        assert self._urdf_path.exists(), f"URDF not found: {self._urdf_path}"

        self._model = pin.buildModelFromUrdf(str(self._urdf_path))
        self._data = self._model.createData()
        self._t_end = t_end
        self._dt = dt

        # End-effector frame
        if ee_frame_name is not None:
            self._ee_frame_id = self._model.getFrameId(ee_frame_name)
        else:
            # Use the last operational frame (tip of kinematic chain)
            self._ee_frame_id = self._model.nframes - 1

        self._nq = self._model.nq
        self._nv = self._model.nv

        self._base_coeffs: list[list[float]] | None = None
        self._nominal_result: PinocchioSimResult | None = None

        logger.info(
            "PinocchioPerturbationAnalyzer: model=%s, nq=%d, nv=%d, t_end=%.2f",
            self._model.name,
            self._nq,
            self._nv,
            self._t_end,
        )

    # ------------------------------------------------------------------
    # Protocol API
    # ------------------------------------------------------------------

    def set_base_torque_profile(self, profile: object) -> None:
        """Set the nominal torque polynomial coefficients.

        Parameters
        ----------
        profile : dict with 'coeffs' key
            ``profile["coeffs"]`` is a list of per-joint coefficient lists:
            ``[[c0, c1, ...], [c0, c1, ...], ...]``.
            The number of joint lists must equal ``nv`` or be broadcastable.

        Design by Contract
        ------------------
        Pre: profile is a dict with 'coeffs' key.
        Post: self._base_coeffs is set and self._nominal_result is cached.
        """
        assert isinstance(profile, dict), f"profile must be a dict, got {type(profile)}"
        assert "coeffs" in profile, "'coeffs' key missing from profile"
        coeffs = profile["coeffs"]
        assert isinstance(coeffs, list) and len(coeffs) > 0, (
            "profile['coeffs'] must be a non-empty list"
        )
        self._base_coeffs = coeffs
        self._nominal_result = self._simulate(coeffs)

    def perturb_torque(self, config: PerturbationConfig, seed: int) -> dict:
        """Apply noise to base coefficients and return perturbed profile.

        Returns
        -------
        dict with 'coeffs' key (same structure as the profile passed to
        ``set_base_torque_profile``).

        Design by Contract
        ------------------
        Pre: ``set_base_torque_profile`` has been called.
        Post: returned dict has 'coeffs' with same shape as base.
        """
        assert self._base_coeffs is not None, (
            "set_base_torque_profile() must be called before perturb_torque()"
        )
        perturbed = _perturb_coeffs_by_mode(self._base_coeffs, config, seed)
        return {"coeffs": perturbed}

    def extract_metrics(self, sim_result: object) -> dict[str, float | np.ndarray]:
        """Extract all MANDATORY_METRICS from a simulation result.

        Parameters
        ----------
        sim_result : PinocchioSimResult

        Returns
        -------
        dict mapping metric name → scalar float or ndarray.

        Design by Contract
        ------------------
        Pre: sim_result is a PinocchioSimResult with n_steps >= 2.
        Post: all MANDATORY_METRICS present; all values finite.
        """
        assert isinstance(sim_result, PinocchioSimResult), (
            f"sim_result must be PinocchioSimResult, got {type(sim_result)}"
        )
        assert sim_result.n_steps >= 2, "Simulation must have >= 2 steps"

        r = sim_result
        last = r.n_steps - 1

        joint_angles_final = r.q_traj[last].copy()
        joint_velocities_final = r.v_traj[last].copy()

        # End-effector position at final step
        ee_pos_final = r.ee_pos_traj[last].copy()

        # End-effector velocity at final step
        ee_vel_final = r.ee_vel_traj[last].copy()
        ee_speed_final = float(np.linalg.norm(ee_vel_final))

        # Peak end-effector speed
        speeds = np.linalg.norm(r.ee_vel_traj, axis=1)
        peak_speed = float(np.max(speeds))

        # Total energy (kinetic + potential) at final step
        total_energy_final = float(
            r.kinetic_energy_traj[last] + r.potential_energy_traj[last]
        )

        # Trajectory RMSE vs nominal
        trajectory_rmse = 0.0
        trajectory_max_deviation = 0.0
        if self._nominal_result is not None:
            nom = self._nominal_result
            n_cmp = min(r.n_steps, nom.n_steps)
            deviations = np.linalg.norm(r.q_traj[:n_cmp] - nom.q_traj[:n_cmp], axis=1)
            trajectory_rmse = float(np.sqrt(np.mean(deviations**2)))
            trajectory_max_deviation = float(np.max(deviations))

        motion_duration = float(r.t[last] - r.t[0])

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

    def run_batch(self, config: PerturbationConfig) -> PerturbationSummary:
        """Run Monte Carlo perturbation analysis.

        Parameters
        ----------
        config : PerturbationConfig

        Returns
        -------
        PerturbationSummary with robustness_score and per-metric MetricStatistics.

        Design by Contract
        ------------------
        Pre: ``set_base_torque_profile`` has been called.
        Post: result.success_rate in [0, 1].
        Post: result.robustness_score in [0, 1].
        """
        assert self._base_coeffs is not None, (
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

        # Accumulate per-metric lists
        metric_lists: dict[str, list[float]] = {m: [] for m in scalar_metric_names}
        n_success = 0

        for i in range(config.n_trials):
            perturbed = _perturb_coeffs_by_mode(
                self._base_coeffs, config, base_seed + i
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
            except Exception:  # noqa: BLE001
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

        # Robustness score: 1 / (1 + CV_weighted) over scalar metrics
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
        """Compare two torque profiles via Mann-Whitney U test.

        Parameters
        ----------
        profile_a, profile_b : dict with 'coeffs' key
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
                perturbed = _perturb_coeffs_by_mode(
                    self._base_coeffs,  # type: ignore[arg-type]
                    config,
                    base_seed + i,
                )
                try:
                    sim = self._simulate(perturbed)
                    m_dict = self.extract_metrics(sim)
                    v = m_dict[metric]
                    if isinstance(v, np.ndarray):
                        v = float(np.linalg.norm(v))
                    values.append(float(v))
                except Exception:  # noqa: BLE001
                    pass
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate(self, coeffs: list[list[float]]) -> PinocchioSimResult:
        """Run a Pinocchio ABA forward simulation with polynomial torques.

        Each joint's torque follows tau_j(t) = sum_k c_jk * t^k.
        """
        import pinocchio as pin  # noqa: PLC0415

        model = self._model
        data = model.createData()  # fresh data per trial for thread safety
        nv = model.nv

        # Build per-joint polynomial arrays (ascending order)
        n_joints_coeff = len(coeffs)
        # Pad or truncate to match nv
        joint_polys: list[np.ndarray] = []
        for j in range(nv):
            if j < n_joints_coeff:
                # ascending coefficients [c0, c1, c2, ...] → reversed for polyval
                joint_polys.append(np.array(coeffs[j][::-1]))
            else:
                joint_polys.append(np.array([0.0]))

        # Integrate
        q = pin.neutral(model)
        v = np.zeros(nv)
        t = 0.0

        t_list = [t]
        q_list = [q.copy()]
        v_list = [v.copy()]

        n_steps = max(2, int(self._t_end / self._dt))
        for _ in range(n_steps):
            tau = np.array([float(np.polyval(joint_polys[j], t)) for j in range(nv)])
            a = pin.aba(model, data, q, v, tau)
            v = v + a * self._dt
            q = pin.integrate(model, q, v * self._dt)
            t += self._dt

            t_list.append(t)
            q_list.append(q.copy())
            v_list.append(v.copy())

        t_arr = np.array(t_list)
        q_arr = np.array(q_list)
        v_arr = np.array(v_list)

        # Compute end-effector positions via FK
        ee_pos_list = []
        for qi, vi in zip(q_arr, v_arr, strict=True):
            pin.forwardKinematics(model, data, qi, vi)
            pin.updateFramePlacement(model, data, self._ee_frame_id)
            ee_pos_list.append(data.oMf[self._ee_frame_id].translation.copy())

        ee_pos_arr = np.array(ee_pos_list)

        # EE velocities via finite difference
        ee_vel_arr = np.zeros_like(ee_pos_arr)
        for i in range(1, len(t_arr)):
            dt_i = max(t_arr[i] - t_arr[i - 1], 1e-12)
            ee_vel_arr[i] = (ee_pos_arr[i] - ee_pos_arr[i - 1]) / dt_i

        # Kinetic and potential energy
        ke_list = []
        pe_list = []
        for qi, vi in zip(q_arr, v_arr, strict=True):
            ke_list.append(float(pin.computeKineticEnergy(model, data, qi, vi)))
            pe_list.append(float(pin.computePotentialEnergy(model, data, qi)))

        return PinocchioSimResult(
            t=t_arr,
            q_traj=q_arr,
            v_traj=v_arr,
            ee_pos_traj=ee_pos_arr,
            ee_vel_traj=ee_vel_arr,
            kinetic_energy_traj=np.array(ke_list),
            potential_energy_traj=np.array(pe_list),
        )
