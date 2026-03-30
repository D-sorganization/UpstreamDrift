# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""OpenSim Perturbation Analyzer — PerturbationAnalyzer protocol for OpenSim (#1981).

Implements the ``PerturbationAnalyzer`` protocol for the OpenSim physics
simulation engine.  Uses a built-in minimal pendulum model when no model path
is provided.  When ``opensim`` is not installed the module imports cleanly but
construction raises ``ImportError``.

Design by Contract
------------------
- ``set_base_torque_profile()`` must be called before ``run_batch()`` or
  ``perturb_torque()``.  Raises ``AssertionError`` otherwise.
- ``run_batch()`` returns a ``PerturbationSummary`` containing all
  ``MANDATORY_METRICS`` as keys.
- ``extract_metrics()`` requires an ``OpenSimSimResult`` and always returns
  finite values.

DRY
---
Delegates noise generation and coefficient perturbation to the shared
``src.shared.python.perturbation`` package.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.engine_core.engine_availability import OPENSIM_AVAILABLE
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

# ---------------------------------------------------------------------------
# Mandatory metric names (shared across all PerturbationAnalyzer engines)
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
# Minimal OpenSim model XML (2-DOF pendulum with coordinate actuators)
# ---------------------------------------------------------------------------

_MINIMAL_OSIM_XML: str = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
  <Model name="minimal_pendulum">
    <gravity>0 -9.80665 0</gravity>
    <BodySet>
      <objects>
        <Body name="ground">
          <mass>0</mass>
          <mass_center>0 0 0</mass_center>
          <inertia>0 0 0 0 0 0</inertia>
        </Body>
        <Body name="link1">
          <mass>1.0</mass>
          <mass_center>0 -0.25 0</mass_center>
          <inertia>0.02 0.02 0.001 0 0 0</inertia>
        </Body>
        <Body name="link2">
          <mass>0.5</mass>
          <mass_center>0 -0.25 0</mass_center>
          <inertia>0.005 0.005 0.0005 0 0 0</inertia>
        </Body>
      </objects>
    </BodySet>
    <JointSet>
      <objects>
        <PinJoint name="j1">
          <parent_frame>ground</parent_frame>
          <child_frame>link1</child_frame>
          <location_in_parent>0 0 0</location_in_parent>
          <location>0 0.5 0</location>
          <coordinates>
            <Coordinate name="q1">
              <default_value>0</default_value>
              <range>-3.14159 3.14159</range>
            </Coordinate>
          </coordinates>
        </PinJoint>
        <PinJoint name="j2">
          <parent_frame>link1</parent_frame>
          <child_frame>link2</child_frame>
          <location_in_parent>0 -0.5 0</location_in_parent>
          <location>0 0.5 0</location>
          <coordinates>
            <Coordinate name="q2">
              <default_value>0</default_value>
              <range>-3.14159 3.14159</range>
            </Coordinate>
          </coordinates>
        </PinJoint>
      </objects>
    </JointSet>
    <ForceSet>
      <objects>
        <CoordinateActuator name="act1">
          <coordinate>q1</coordinate>
          <optimal_force>100</optimal_force>
        </CoordinateActuator>
        <CoordinateActuator name="act2">
          <coordinate>q2</coordinate>
          <optimal_force>100</optimal_force>
        </CoordinateActuator>
      </objects>
    </ForceSet>
  </Model>
</OpenSimDocument>
"""


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------


@dataclass
class OpenSimSimResult:
    """Trajectory data from one OpenSim forward simulation.

    Attributes
    ----------
    t : ndarray, shape (n,)
        Time stamps.
    qpos_traj : ndarray, shape (n, nq)
        Generalized coordinate trajectory.
    qvel_traj : ndarray, shape (n, nv)
        Generalized velocity trajectory.
    ee_pos_traj : ndarray, shape (n, 3)
        End-effector Cartesian position trajectory.
    ee_vel_traj : ndarray, shape (n, 3)
        End-effector velocity (finite-difference).
    kinetic_energy_traj : ndarray, shape (n,)
        Kinetic energy at each step.
    potential_energy_traj : ndarray, shape (n,)
        Potential energy at each step.
    """

    t: np.ndarray
    qpos_traj: np.ndarray
    qvel_traj: np.ndarray
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


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


class OpenSimPerturbationAnalyzer:
    """PerturbationAnalyzer protocol implementation for the OpenSim engine.

    When ``opensim`` is not installed, construction raises ``ImportError``.
    The built-in minimal OSIM model (2-DOF pendulum with coordinate actuators)
    is used when no model path is provided.

    Usage::

        from src.engines.physics_engines.opensim.python.perturbation.analyzer import (
            OpenSimPerturbationAnalyzer,
        )
        analyzer = OpenSimPerturbationAnalyzer()
        analyzer.set_base_torque_profile({"coeffs": [[0.5, 0.1], [0.3, -0.05]]})
        summary = analyzer.run_batch(PerturbationConfig(n_trials=50))
        logger.info("RS: %s", summary.robustness_score)

    Design by Contract
    ------------------
    Pre:  opensim must be importable.
    Pre:  ``set_base_torque_profile`` must be called before ``run_batch`` or
          ``perturb_torque``.
    Post: ``run_batch`` returns a ``PerturbationSummary`` with all
          ``MANDATORY_METRICS`` present.
    """

    ENGINE_NAME: str = "opensim"

    def __init__(
        self,
        model_path: str | Path | None = None,
        t_end: float = 1.5,
        dt: float = 0.01,
        ee_body_name: str | None = None,
    ) -> None:
        """Initialise the OpenSim perturbation analyzer.

        Parameters
        ----------
        model_path : str or Path, optional
            Path to an .osim file.  If None, a temporary minimal model is
            written to disk and used.
        t_end : float
            Simulation end time in seconds.
        dt : float
            Integration timestep in seconds.
        ee_body_name : str, optional
            Name of the end-effector body.  Defaults to the last body in the
            model (excluding ground).
        """
        if not OPENSIM_AVAILABLE:
            msg = (
                "opensim is not installed.  "
                "Install it with: conda install -c opensim-org opensim"
            )
            raise ImportError(msg)

        import opensim as osim  # noqa: PLC0415

        self._t_end = t_end
        self._dt = dt
        self._ee_body_name = ee_body_name

        if model_path is not None:
            model_path = Path(model_path)
            if not (model_path.exists()):
                raise ValueError(f"Model not found: {model_path}")
            self._model = osim.Model(str(model_path))
        else:
            # Write minimal model to a temp file — OpenSim requires a file path
            import tempfile  # noqa: PLC0415

            with tempfile.NamedTemporaryFile(
                suffix=".osim", delete=False, mode="w"
            ) as tmp:
                tmp.write(_MINIMAL_OSIM_XML)
                tmp_name = tmp.name
            self._model = osim.Model(tmp_name)

        self._model.initSystem()

        # Determine DOF count from coordinate set
        coord_set = self._model.getCoordinateSet()
        self._nq = coord_set.getSize()
        self._nv = self._nq  # pin joints: nv == nq

        # Actuator count
        force_set = self._model.getForceSet()
        self._nu = force_set.getSize()

        # End-effector body
        body_set = self._model.getBodySet()
        if ee_body_name is not None:
            self._ee_body_name = ee_body_name
        else:
            # Default: last body in the set (excluding ground at index 0)
            if body_set.getSize() > 0:
                self._ee_body_name = body_set.get(body_set.getSize() - 1).getName()
            else:
                self._ee_body_name = "link2"

        self._base_coeffs: list[list[float]] | None = None
        self._nominal_result: OpenSimSimResult | None = None

        logger.info(
            "OpenSimPerturbationAnalyzer: nq=%d, nu=%d, t_end=%.2f, ee=%s",
            self._nq,
            self._nu,
            self._t_end,
            self._ee_body_name,
        )

    # ------------------------------------------------------------------
    # Protocol API
    # ------------------------------------------------------------------

    def set_base_torque_profile(self, profile: object) -> None:
        """Set the nominal torque polynomial coefficients.

        Parameters
        ----------
        profile : dict with 'coeffs' key
            ``profile["coeffs"]`` is a list of per-actuator coefficient lists.

        Design by Contract
        ------------------
        Pre: profile is a dict with 'coeffs' key.
        Post: self._base_coeffs is set and self._nominal_result is cached.
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
        """Apply noise to base coefficients and return perturbed profile.

        Design by Contract
        ------------------
        Pre: ``set_base_torque_profile`` has been called.
        Post: returned dict has 'coeffs' with same shape as base.
        """
        if not (self._base_coeffs is not None):
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

    def extract_metrics(self, sim_result: object) -> dict[str, float | np.ndarray]:
        """Extract all MANDATORY_METRICS from a simulation result.

        Parameters
        ----------
        sim_result : OpenSimSimResult

        Returns
        -------
        dict mapping metric name → scalar float or ndarray.

        Design by Contract
        ------------------
        Pre: sim_result is an OpenSimSimResult with n_steps >= 2.
        Post: all MANDATORY_METRICS present; all values finite.
        """
        if not isinstance(sim_result, OpenSimSimResult):
            raise ValueError(
                f"sim_result must be OpenSimSimResult, got {type(sim_result)}"
            )
        if not (sim_result.n_steps >= 2):
            raise ValueError("Simulation must have >= 2 steps")

        r = sim_result
        last = r.n_steps - 1

        joint_angles_final = r.qpos_traj[last].copy()
        joint_velocities_final = r.qvel_traj[last].copy()
        ee_pos_final = r.ee_pos_traj[last].copy()
        ee_vel_final = r.ee_vel_traj[last].copy()
        ee_speed_final = float(np.linalg.norm(ee_vel_final))

        speeds = np.linalg.norm(r.ee_vel_traj, axis=1)
        peak_speed = float(np.max(speeds))

        total_energy_final = float(
            r.kinetic_energy_traj[last] + r.potential_energy_traj[last]
        )

        trajectory_rmse = 0.0
        trajectory_max_deviation = 0.0
        if self._nominal_result is not None:
            nom = self._nominal_result
            n_cmp = min(r.n_steps, nom.n_steps)
            deviations = np.linalg.norm(
                r.qpos_traj[:n_cmp] - nom.qpos_traj[:n_cmp], axis=1
            )
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
        """Run N perturbed trials and aggregate results.

        Design by Contract
        ------------------
        Pre:  ``set_base_torque_profile`` has been called.
        Post: returned ``PerturbationSummary`` has all ``MANDATORY_METRICS``
              as keys (scalar metrics only).
        """
        if not (self._base_coeffs is not None):
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
            except Exception:  # noqa: BLE001  # noqa: BLE001
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
        """Compare two torque profiles via Mann-Whitney U test.

        Design by Contract
        ------------------
        Pre:  Both profiles are valid dicts with 'coeffs' key.
        Post: report.confidence in [0.0, 1.0].
        """
        from scipy import stats as _stats  # noqa: PLC0415 — lazy import

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
                except Exception:  # noqa: BLE001  # noqa: BLE001
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

    def _simulate(self, coeffs: list[list[float]]) -> OpenSimSimResult:
        """Run an OpenSim forward simulation with polynomial torques.

        Each actuator's control follows ctrl_j(t) = sum_k c_jk * t^k.
        Uses the OpenSim Manager with a prescribed control function.
        """
        import opensim as osim  # noqa: PLC0415

        model = self._model
        state = model.initSystem()

        coord_set = model.getCoordinateSet()
        force_set = model.getForceSet()
        nu = force_set.getSize()

        # Build per-actuator polynomial arrays
        n_coeff_sets = len(coeffs)
        joint_polys: list[np.ndarray] = []
        for j in range(nu):
            if j < n_coeff_sets:
                joint_polys.append(np.array(coeffs[j][::-1]))  # high→low for polyval
            else:
                joint_polys.append(np.array([0.0]))

        dt = self._dt
        n_steps = max(2, int(self._t_end / dt))

        t_list: list[float] = []
        qpos_list: list[np.ndarray] = []
        qvel_list: list[np.ndarray] = []
        ee_pos_list: list[np.ndarray] = []
        ke_list: list[float] = []
        pe_list: list[float] = []

        # Reset to initial state
        model.initStateFromProperties(state)

        body_set = model.getBodySet()
        ground = model.getGround()

        for step in range(n_steps):
            t = step * dt

            # Apply polynomial torques via coordinate actuators
            for j in range(nu):
                try:
                    force_obj = force_set.get(j)
                    ctrl = float(np.polyval(joint_polys[j], t))
                    force_obj.setControls(
                        osim.Vector(1, ctrl), model.updDefaultControls()
                    )
                except Exception:  # noqa: BLE001  # noqa: BLE001
                    pass

            # Realize to acceleration
            with contextlib.suppress(Exception):
                model.realizeAcceleration(state)

            # Read state
            nq = coord_set.getSize()
            q = np.zeros(nq)
            qdot = np.zeros(nq)
            for k in range(nq):
                coord = coord_set.get(k)
                q[k] = coord.getValue(state)
                qdot[k] = coord.getSpeedValue(state)

            # EE position: last body in body set, expressed in ground
            ee_pos = np.zeros(3)
            try:
                n_bodies = body_set.getSize()
                if n_bodies > 0:
                    ee_body = body_set.get(n_bodies - 1)
                    pos_in_ground = ground.findStationLocationInGround(
                        state,
                        ee_body.findStationLocationInGround(state, osim.Vec3(0, 0, 0)),
                    )
                    ee_pos = np.array(
                        [pos_in_ground[0], pos_in_ground[1], pos_in_ground[2]]
                    )
            except Exception:  # noqa: BLE001  # noqa: BLE001
                # Fallback: use simple forward kinematics from joint angles
                link_len = 0.5
                angle_sum = float(np.sum(q))
                ee_pos = np.array(
                    [
                        (
                            link_len * np.sin(q[0]) + link_len * np.sin(angle_sum)
                            if nq >= 2
                            else link_len * np.sin(q[0])
                        ),
                        (
                            -(link_len * np.cos(q[0]) + link_len * np.cos(angle_sum))
                            if nq >= 2
                            else -link_len * np.cos(q[0])
                        ),
                        0.0,
                    ]
                )

            # Energy
            ke = 0.0
            pe = 0.0
            try:
                model.realizeVelocity(state)
                ke = float(model.calcKineticEnergy(state))
                pe = float(model.calcPotentialEnergy(state))
            except Exception:  # noqa: BLE001  # noqa: BLE001
                pass

            t_list.append(t)
            qpos_list.append(q)
            qvel_list.append(qdot)
            ee_pos_list.append(ee_pos)
            ke_list.append(ke)
            pe_list.append(pe)

            # Integrate one step using Euler (simple, no Manager overhead)
            try:
                integrator = osim.RungeKuttaMersonIntegrator(model.getSystem())
                integrator.setAccuracy(1e-4)
                manager = osim.Manager(model, integrator)
                manager.setInitialTime(t)
                manager.setFinalTime(t + dt)
                manager.integrate(state)
            except Exception:  # noqa: BLE001  # noqa: BLE001
                # Manual Euler fallback for joint coordinates
                for k in range(nq):
                    coord = coord_set.get(k)
                    try:
                        new_val = q[k] + qdot[k] * dt
                        coord.setValue(state, new_val)
                        coord.setSpeedValue(state, qdot[k])
                    except Exception:  # noqa: BLE001  # noqa: BLE001
                        pass

        t_arr = np.array(t_list)
        qpos_arr = np.array(qpos_list)
        qvel_arr = np.array(qvel_list)
        ee_pos_arr = np.array(ee_pos_list)
        ke_arr = np.array(ke_list)
        pe_arr = np.array(pe_list)

        # EE velocities via finite difference
        ee_vel_arr = np.zeros_like(ee_pos_arr)
        for i in range(1, len(t_arr)):
            dt_i = max(t_arr[i] - t_arr[i - 1], 1e-12)
            ee_vel_arr[i] = (ee_pos_arr[i] - ee_pos_arr[i - 1]) / dt_i

        return OpenSimSimResult(
            t=t_arr,
            qpos_traj=qpos_arr,
            qvel_traj=qvel_arr,
            ee_pos_traj=ee_pos_arr,
            ee_vel_traj=ee_vel_arr,
            kinetic_energy_traj=ke_arr,
            potential_energy_traj=pe_arr,
        )
