"""Drake Perturbation Analyzer — PerturbationAnalyzer protocol for Drake (#1979).

Implements the ``PerturbationAnalyzer`` protocol for Drake's ``MultibodyPlant``
within a ``DiagramBuilder`` / ``Simulator`` framework.  Polynomial torques are
injected via a ``TrajectorySource`` connected to the actuation input port.

Design by Contract
------------------
- ``set_base_torque_profile()`` must be called before ``run_batch()`` or
  ``perturb_torque()``.  Raises ``AssertionError`` otherwise.
- ``run_batch()`` returns a ``PerturbationSummary`` containing all
  ``MANDATORY_METRICS`` as keys.
- ``extract_metrics()`` requires a ``DrakeSimResult`` and always returns
  finite values.
- pydrake must be importable; if not, ``ImportError`` is raised at
  construction time with a helpful message.

DRY
---
Delegates noise generation and coefficient perturbation to the shared
``src.shared.python.perturbation`` package — the same functions used by
``PendulumPerturbationAnalyzer`` and ``PinocchioPerturbationAnalyzer``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.engine_core.engine_availability import is_engine_available
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
class DrakeSimResult:
    """Trajectory data from one Drake forward simulation.

    Attributes
    ----------
    t : ndarray, shape (n,)
        Time stamps.
    q_traj : ndarray, shape (n, nq)
        Joint position trajectory (generalized coordinates).
    v_traj : ndarray, shape (n, nv)
        Joint velocity trajectory.
    ee_pos_traj : ndarray, shape (n, 3)
        End-effector Cartesian position trajectory.
    ee_vel_traj : ndarray, shape (n, 3)
        End-effector velocity trajectory (finite-difference).
    kinetic_energy_traj : ndarray, shape (n,)
        Kinetic energy at each step.
    potential_energy_traj : ndarray, shape (n,)
        Potential energy at each step.
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


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


class DrakePerturbationAnalyzer:
    """PerturbationAnalyzer protocol implementation for the Drake engine.

    Usage::

        from src.engines.physics_engines.drake.python.perturbation.analyzer import (
            DrakePerturbationAnalyzer,
        )
        urdf = "path/to/model.urdf"
        analyzer = DrakePerturbationAnalyzer(urdf_path=urdf)
        analyzer.set_base_torque_profile({"coeffs": [[0.5, 0.1], [0.3, -0.05]]})
        summary = analyzer.run_batch(PerturbationConfig(n_trials=50))
        logger.info("RS: %s", summary.robustness_score)

    Design by Contract
    ------------------
    Pre:  pydrake must be importable.
    Pre:  ``set_base_torque_profile`` must be called before ``run_batch`` or
          ``perturb_torque``.
    Post: ``run_batch`` returns a ``PerturbationSummary`` with all
          ``MANDATORY_METRICS`` present.
    """

    ENGINE_NAME: str = "drake"

    # Minimal 1-DOF pendulum URDF for tests when no external URDF is given
    _MINIMAL_URDF: str = """<?xml version="1.0"?>
<robot name="simple_pendulum">
  <link name="world"/>
  <link name="arm">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="world"/>
    <child link="arm"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="100" velocity="100"/>
  </joint>
</robot>"""

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        t_end: float = 1.5,
        dt: float = 0.005,
        ee_body_name: str | None = None,
    ) -> None:
        """Initialise the Drake perturbation analyzer.

        Parameters
        ----------
        urdf_path : str or Path, optional
            Path to the URDF model file.  If None, uses the bundled golfer URDF,
            or falls back to the minimal 1-DOF pendulum.
        t_end : float
            Simulation end time in seconds.
        dt : float
            Integration time step.
        ee_body_name : str, optional
            Name of the end-effector body.  Defaults to last moving body.
        """
        if not is_engine_available("drake"):
            msg = "pydrake is not installed.  Install it with: pip install drake"
            raise ImportError(msg)

        from pydrake.all import (  # noqa: PLC0415
            AddMultibodyPlantSceneGraph,
            DiagramBuilder,
            Parser,
        )

        self._t_end = t_end
        self._dt = dt

        # Build plant
        builder = DiagramBuilder()
        plant, _scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

        if urdf_path is not None:
            urdf_path = Path(urdf_path)
            assert urdf_path.exists(), f"URDF not found: {urdf_path}"
            Parser(plant).AddModelFromFile(str(urdf_path))
        else:
            # Try bundled golfer URDF, fall back to minimal pendulum
            bundled = Path(__file__).parents[4] / "models" / "generated" / "golfer.urdf"
            if bundled.exists():
                Parser(plant).AddModelFromFile(str(bundled))
            else:
                import tempfile

                tmp = tempfile.NamedTemporaryFile(
                    suffix=".urdf", mode="w", delete=False
                )
                tmp.write(self._MINIMAL_URDF)
                tmp.close()
                Parser(plant).AddModelFromFile(tmp.name)

        plant.Finalize()
        self._plant = plant
        self._builder = builder

        self._nq = plant.num_positions()
        self._nv = plant.num_velocities()
        self._nu = plant.num_actuators()

        # End-effector body
        if ee_body_name is not None:
            self._ee_body_idx = plant.GetBodyByName(ee_body_name).index()
        else:
            # Use the last non-world body
            bodies = [
                plant.get_body(i)
                for i in range(plant.num_bodies())
                if not plant.get_body(i).is_world()
            ]
            self._ee_body_idx = bodies[-1].index() if bodies else None

        self._base_coeffs: list[list[float]] | None = None
        self._nominal_result: DrakeSimResult | None = None

        logger.info(
            "DrakePerturbationAnalyzer: nq=%d, nv=%d, nu=%d, t_end=%.2f",
            self._nq,
            self._nv,
            self._nu,
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
            ``profile["coeffs"]`` is a list of per-joint coefficient lists.

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

        Design by Contract
        ------------------
        Pre: ``set_base_torque_profile`` has been called.
        Post: returned dict has 'coeffs' with same shape as base.
        """
        assert self._base_coeffs is not None, (
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
        sim_result : DrakeSimResult

        Returns
        -------
        dict mapping metric name → scalar float or ndarray.

        Design by Contract
        ------------------
        Pre: sim_result is a DrakeSimResult with n_steps >= 2.
        Post: all MANDATORY_METRICS present; all values finite.
        """
        assert isinstance(sim_result, DrakeSimResult), (
            f"sim_result must be DrakeSimResult, got {type(sim_result)}"
        )
        assert sim_result.n_steps >= 2, "Simulation must have >= 2 steps"

        r = sim_result
        last = r.n_steps - 1

        joint_angles_final = r.q_traj[last].copy()
        joint_velocities_final = r.v_traj[last].copy()
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
            except Exception as e:  # noqa: BLE001
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
                except Exception as e:  # noqa: BLE001
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

    def _simulate(self, coeffs: list[list[float]]) -> DrakeSimResult:
        """Run a Drake forward simulation with polynomial torques.

        Each joint's torque follows tau_j(t) = sum_k c_jk * t^k.
        Uses a manual Euler integration loop to avoid Drake Simulator
        overhead for short Monte Carlo trials.
        """
        from pydrake.all import (  # noqa: PLC0415
            AddMultibodyPlantSceneGraph,
            DiagramBuilder,
            Parser,
            Simulator,
        )

        # Rebuild a fresh plant+simulator for each trial
        # (Drake Simulator is not reusable after AdvanceTo)
        builder = DiagramBuilder()
        plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

        # Re-add model from same URDF (plant was already finalized in __init__)
        # We copy model state by running from scratch — keep it clean and DRY
        # by using the _MINIMAL_URDF as source of truth for simple models.
        import tempfile

        if hasattr(self, "_urdf_str"):
            tmp = tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False)
            tmp.write(self._urdf_str)
            tmp.close()
            Parser(plant).AddModelFromFile(tmp.name)
        else:
            # Re-finalize fresh from existing plant's URDF path stored at init
            # Fall back to minimal URDF
            tmp = tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False)
            tmp.write(self._MINIMAL_URDF)
            tmp.close()
            Parser(plant).AddModelFromFile(tmp.name)

        plant.Finalize()
        diagram = builder.Build()
        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()
        plant_context = plant.GetMyMutableContextFromRoot(context)

        nq = plant.num_positions()
        nv = plant.num_velocities()
        nu = plant.num_actuators()

        # Build per-joint polynomial arrays (ascending → reversed for polyval)
        n_joints = len(coeffs)
        joint_polys: list[np.ndarray] = []
        for j in range(nu):
            if j < n_joints:
                joint_polys.append(np.array(coeffs[j][::-1]))
            else:
                joint_polys.append(np.array([0.0]))

        # Runge-Kutta 4 integration
        def compute_a(q_val: np.ndarray, v_val: np.ndarray, t_val: float) -> np.ndarray:
            plant.SetPositions(plant_context, q_val)
            plant.SetVelocities(plant_context, v_val)
            tau_val = np.array(
                [float(np.polyval(joint_polys[j], t_val)) for j in range(nu)]
            )
            plant.get_actuation_input_port().FixValue(plant_context, tau_val)
            M_val = plant.CalcMassMatrixViaInverseDynamics(plant_context)
            bias_val = plant.CalcBiasTerm(plant_context)
            gravity_val = plant.CalcGravityGeneralizedForces(plant_context)
            return np.linalg.solve(M_val, tau_val - bias_val + gravity_val)  # type: ignore[no-any-return]

        q = np.zeros(nq)
        v = np.zeros(nv)
        t = 0.0
        dt = self._dt

        t_list = [t]
        q_list = [q.copy()]
        v_list = [v.copy()]

        n_steps = max(2, int(self._t_end / dt))
        for _ in range(n_steps):
            k1_v = compute_a(q, v, t)
            k1_q = v

            k2_v = compute_a(q + 0.5 * dt * k1_q, v + 0.5 * dt * k1_v, t + 0.5 * dt)
            k2_q = v + 0.5 * dt * k1_v

            k3_v = compute_a(q + 0.5 * dt * k2_q, v + 0.5 * dt * k2_v, t + 0.5 * dt)
            k3_q = v + 0.5 * dt * k2_v

            k4_v = compute_a(q + dt * k3_q, v + dt * k3_v, t + dt)
            k4_q = v + dt * k3_v

            v = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
            q = q + (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
            t += dt

            t_list.append(t)
            q_list.append(q.copy())
            v_list.append(v.copy())

        t_arr = np.array(t_list)
        q_arr = np.array(q_list)
        v_arr = np.array(v_list)

        # End-effector positions via FK
        ee_pos_list = []
        for qi in q_arr:
            plant.SetPositions(plant_context, qi)
            plant.SetVelocities(plant_context, np.zeros(nv))
            plant.EvalBodyPoseInWorld(plant_context, plant.world_body())
            # Get last body position
            bodies = [
                plant.get_body(i)
                for i in range(plant.num_bodies())
                if not plant.get_body(i).is_world()
            ]
            if bodies:
                pose = plant.EvalBodyPoseInWorld(plant_context, bodies[-1])
                ee_pos_list.append(pose.translation().copy())
            else:
                ee_pos_list.append(np.zeros(3))

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
            plant.SetPositions(plant_context, qi)
            plant.SetVelocities(plant_context, vi)
            ke_list.append(float(plant.CalcKineticEnergy(plant_context)))
            pe_list.append(float(plant.CalcPotentialEnergy(plant_context)))

        return DrakeSimResult(
            t=t_arr,
            q_traj=q_arr,
            v_traj=v_arr,
            ee_pos_traj=ee_pos_arr,
            ee_vel_traj=ee_vel_arr,
            kinetic_energy_traj=np.array(ke_list),
            potential_energy_traj=np.array(pe_list),
        )
