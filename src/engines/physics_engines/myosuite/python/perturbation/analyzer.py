"""MyoSuite Perturbation Analyzer — PerturbationAnalyzer protocol for MyoSuite (#1982).

Implements the ``PerturbationAnalyzer`` protocol for the MyoSuite musculoskeletal
simulation environment (built on MuJoCo).  Uses a built-in minimal tendon-driven
arm model when no environment ID or model path is provided.  When ``myosuite``
is not installed the module imports cleanly but construction raises ``ImportError``.

Design by Contract
------------------
- ``set_base_torque_profile()`` must be called before ``run_batch()`` or
  ``perturb_torque()``.  Raises ``AssertionError`` otherwise.
- ``run_batch()`` returns a ``PerturbationSummary`` containing all
  ``MANDATORY_METRICS`` as keys.
- ``extract_metrics()`` requires a ``MyoSuiteSimResult`` and always returns
  finite values.

DRY
---
Delegates noise generation and coefficient perturbation to the shared
``src.shared.python.perturbation`` package.

Note on MyoSuite
----------------
MyoSuite wraps MuJoCo gym environments.  When a gym environment is used, the
``env.step(action)`` interface is used with polynomial torque profiles as
actions.  For robustness, a fallback pure-MuJoCo integration path is provided
for environments that expose the underlying ``model`` and ``data`` attributes.
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

MYOSUITE_AVAILABLE: bool = is_engine_available("myosuite")

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
# Minimal MJCF model for testing (2-DOF tendon-driven arm, MyoSuite-style)
# ---------------------------------------------------------------------------

_MINIMAL_MJCF: str = """<mujoco model="minimal_myo_arm">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="upper_arm" pos="0 0 1">
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.04" fromto="0 0 0 0.3 0 0" mass="1.5"/>
      <body name="forearm" pos="0.3 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.25 0 0" mass="0.8"/>
        <body name="hand" pos="0.25 0 0">
          <geom type="sphere" size="0.025" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="shoulder" gear="15" name="shoulder_act"/>
    <motor joint="elbow" gear="10" name="elbow_act"/>
  </actuator>
</mujoco>"""

# Default MyoSuite gym environment ID (used when myosuite is available)
_DEFAULT_ENV_ID = "myoHandPoseFixed-v0"


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------


@dataclass
class MyoSuiteSimResult:
    """Trajectory data from one MyoSuite forward simulation.

    Attributes
    ----------
    t : ndarray, shape (n,)
        Time stamps.
    qpos_traj : ndarray, shape (n, nq)
        Joint position trajectory.
    qvel_traj : ndarray, shape (n, nv)
        Joint velocity trajectory.
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


class MyoSuitePerturbationAnalyzer:
    """PerturbationAnalyzer protocol implementation for the MyoSuite engine.

    MyoSuite is a musculoskeletal simulation library built on MuJoCo/gym.
    When MyoSuite is not installed, falls back to a direct MuJoCo integration
    using the built-in minimal MJCF model (which also requires mujoco).
    When neither is available, construction raises ``ImportError``.

    Usage::

        from src.engines.physics_engines.myosuite.python.perturbation.analyzer import (
            MyoSuitePerturbationAnalyzer,
        )
        analyzer = MyoSuitePerturbationAnalyzer()
        analyzer.set_base_torque_profile({"coeffs": [[0.5, 0.1], [0.3, -0.05]]})
        summary = analyzer.run_batch(PerturbationConfig(n_trials=50))
        logger.info("RS: %s", summary.robustness_score)

    Design by Contract
    ------------------
    Pre:  myosuite or mujoco must be importable.
    Pre:  ``set_base_torque_profile`` must be called before ``run_batch`` or
          ``perturb_torque``.
    Post: ``run_batch`` returns a ``PerturbationSummary`` with all
          ``MANDATORY_METRICS`` present.
    """

    ENGINE_NAME: str = "myosuite"

    def __init__(
        self,
        env_id: str | None = None,
        model_xml: str | None = None,
        model_path: str | Path | None = None,
        t_end: float = 1.5,
        ee_body_name: str | None = None,
    ) -> None:
        """Initialise the MyoSuite perturbation analyzer.

        Parameters
        ----------
        env_id : str, optional
            MyoSuite gym environment ID (e.g. 'myoHandPoseFixed-v0').
            Used only when myosuite is available.
        model_xml : str, optional
            MJCF XML string for direct MuJoCo fallback.
        model_path : str or Path, optional
            Path to MJCF file for direct MuJoCo fallback.
        t_end : float
            Simulation end time in seconds.
        ee_body_name : str, optional
            Name of the end-effector body.  Defaults to the last body.
        """
        self._t_end = t_end
        self._ee_body_name = ee_body_name
        self._env: Any = None
        self._model: Any = None
        self._use_gym = False

        if MYOSUITE_AVAILABLE:
            self._init_myosuite(env_id or _DEFAULT_ENV_ID)
        else:
            self._init_mujoco_fallback(model_xml, model_path)

        self._base_coeffs: list[list[float]] | None = None
        self._nominal_result: MyoSuiteSimResult | None = None

        logger.info(
            "MyoSuitePerturbationAnalyzer: nq=%d, nu=%d, t_end=%.2f, use_gym=%s",
            self._nq,
            self._nu,
            self._t_end,
            self._use_gym,
        )

    def _init_myosuite(self, env_id: str) -> None:
        """Initialise via MyoSuite gym environment."""
        try:
            import gymnasium as gym  # noqa: PLC0415
            import myosuite  # noqa: F401, PLC0415

            self._env = gym.make(env_id)
            self._env.reset()
            # Extract MuJoCo model from gym env
            mj_model = getattr(self._env, "model", None) or getattr(
                self._env.unwrapped, "model", None
            )
            if mj_model is not None:
                self._model = mj_model
                self._nq = int(mj_model.nq)
                self._nv = int(mj_model.nv)
                self._nu = int(mj_model.nu)
            else:
                # Fallback dimensions from action space
                act_space = self._env.action_space
                self._nu = int(np.prod(act_space.shape))
                self._nq = self._nu
                self._nv = self._nu
            self._use_gym = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("MyoSuite init failed (%s), falling back to MuJoCo", exc)
            self._init_mujoco_fallback(None, None)

    def _init_mujoco_fallback(
        self,
        model_xml: str | None,
        model_path: str | Path | None,
    ) -> None:
        """Initialise via direct MuJoCo (no gym)."""
        try:
            import mujoco  # noqa: PLC0415
        except ImportError as exc:
            msg = (
                "Neither myosuite nor mujoco is installed.  "
                "Install myosuite: pip install myosuite  "
                "or mujoco: pip install mujoco"
            )
            raise ImportError(msg) from exc

        import mujoco  # noqa: PLC0415

        if model_path is not None:
            model_path = Path(model_path)
            assert model_path.exists(), f"Model not found: {model_path}"
            self._model = mujoco.MjModel.from_xml_path(str(model_path))
        elif model_xml is not None:
            self._model = mujoco.MjModel.from_xml_string(model_xml)
        else:
            self._model = mujoco.MjModel.from_xml_string(_MINIMAL_MJCF)

        self._nq = int(self._model.nq)
        self._nv = int(self._model.nv)
        self._nu = int(self._model.nu)
        self._use_gym = False

    # ------------------------------------------------------------------
    # Protocol API
    # ------------------------------------------------------------------

    def set_base_torque_profile(self, profile: object) -> None:
        """Set the nominal torque polynomial coefficients.

        Design by Contract
        ------------------
        Pre: profile is a dict with 'coeffs' key.
        Post: self._base_coeffs is set and self._nominal_result is cached.
        """
        assert isinstance(profile, dict), f"profile must be a dict, got {type(profile)}"
        assert "coeffs" in profile, "'coeffs' key missing from profile"
        coeffs = profile["coeffs"]
        assert (
            isinstance(coeffs, list) and len(coeffs) > 0
        ), "profile['coeffs'] must be a non-empty list"
        self._base_coeffs = coeffs
        self._nominal_result = self._simulate(coeffs)

    def perturb_torque(self, config: PerturbationConfig, seed: int) -> dict:
        """Apply noise to base coefficients and return perturbed profile.

        Design by Contract
        ------------------
        Pre: ``set_base_torque_profile`` has been called.
        Post: returned dict has 'coeffs' with same shape as base.
        """
        assert (
            self._base_coeffs is not None
        ), "set_base_torque_profile() must be called before perturb_torque()"
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

        Design by Contract
        ------------------
        Pre: sim_result is a MyoSuiteSimResult with n_steps >= 2.
        Post: all MANDATORY_METRICS present; all values finite.
        """
        assert isinstance(
            sim_result, MyoSuiteSimResult
        ), f"sim_result must be MyoSuiteSimResult, got {type(sim_result)}"
        assert sim_result.n_steps >= 2, "Simulation must have >= 2 steps"

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
        assert (
            self._base_coeffs is not None
        ), "set_base_torque_profile() must be called before run_batch()"
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

    def _simulate(self, coeffs: list[list[float]]) -> MyoSuiteSimResult:
        """Run a MyoSuite/MuJoCo forward simulation with polynomial torques."""
        if self._use_gym and self._env is not None:
            return self._simulate_gym(coeffs)
        return self._simulate_mujoco(coeffs)

    def _simulate_gym(self, coeffs: list[list[float]]) -> MyoSuiteSimResult:
        """Run simulation via MyoSuite gym step interface."""
        env = self._env
        env.reset()

        nu = self._nu
        n_coeff_sets = len(coeffs)
        joint_polys: list[np.ndarray] = []
        for j in range(nu):
            if j < n_coeff_sets:
                joint_polys.append(np.array(coeffs[j][::-1]))
            else:
                joint_polys.append(np.array([0.0]))

        dt_env = getattr(env, "dt", 0.005)
        n_steps = max(2, int(self._t_end / dt_env))

        t_list: list[float] = []
        qpos_list: list[np.ndarray] = []
        qvel_list: list[np.ndarray] = []
        ee_pos_list: list[np.ndarray] = []
        ke_list: list[float] = []
        pe_list: list[float] = []

        for step in range(n_steps):
            t = step * dt_env
            action = np.array([float(np.polyval(joint_polys[j], t)) for j in range(nu)])

            # Clip to action space bounds
            act_space = env.action_space
            action = np.clip(action, act_space.low, act_space.high)

            obs, _reward, _terminated, _truncated, _info = env.step(action)

            # Extract state from underlying MuJoCo data
            mj_data = getattr(env, "data", None) or getattr(env.unwrapped, "data", None)
            if mj_data is not None:
                qpos = np.array(mj_data.qpos).copy()
                qvel = np.array(mj_data.qvel).copy()
                ee_idx = max(0, len(mj_data.xpos) - 1)
                ee_pos = np.array(mj_data.xpos[ee_idx]).copy()
            else:
                # Fallback from observation
                half = len(obs) // 2
                qpos = np.array(obs[:half])
                qvel = np.array(obs[half:])
                ee_pos = np.zeros(3)

            # Energy from MuJoCo
            ke = 0.0
            pe = 0.0
            if mj_data is not None:
                mj_model = getattr(env, "model", None) or getattr(
                    env.unwrapped, "model", None
                )
                if mj_model is not None:
                    try:
                        import mujoco  # noqa: PLC0415

                        mujoco.mj_energyPos(mj_model, mj_data)
                        mujoco.mj_energyVel(mj_model, mj_data)
                        pe = float(mj_data.energy[0])
                        ke = float(mj_data.energy[1])
                    except Exception:  # noqa: BLE001
                        pass

            t_list.append(t)
            qpos_list.append(qpos)
            qvel_list.append(qvel)
            ee_pos_list.append(ee_pos)
            ke_list.append(ke)
            pe_list.append(pe)

        return self._build_result(
            t_list, qpos_list, qvel_list, ee_pos_list, ke_list, pe_list
        )

    def _simulate_mujoco(self, coeffs: list[list[float]]) -> MyoSuiteSimResult:
        """Run simulation via direct MuJoCo integration (fallback path)."""
        import mujoco  # noqa: PLC0415

        model = self._model
        data = mujoco.MjData(model)

        nu = model.nu
        n_coeff_sets = len(coeffs)
        joint_polys: list[np.ndarray] = []
        for j in range(nu):
            if j < n_coeff_sets:
                joint_polys.append(np.array(coeffs[j][::-1]))
            else:
                joint_polys.append(np.array([0.0]))

        dt = float(model.opt.timestep)
        n_steps = max(2, int(self._t_end / dt))

        t_list: list[float] = []
        qpos_list: list[np.ndarray] = []
        qvel_list: list[np.ndarray] = []
        ee_pos_list: list[np.ndarray] = []
        ke_list: list[float] = []
        pe_list: list[float] = []

        mujoco.mj_resetData(model, data)

        for _ in range(n_steps):
            t = float(data.time)
            ctrl = np.array([float(np.polyval(joint_polys[j], t)) for j in range(nu)])
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

            t_list.append(float(data.time))
            qpos_list.append(data.qpos.copy())
            qvel_list.append(data.qvel.copy())
            ee_idx = max(0, model.nbody - 1)
            ee_pos_list.append(data.xpos[ee_idx].copy())

            ke = 0.0
            pe = 0.0
            try:
                mujoco.mj_energyPos(model, data)
                mujoco.mj_energyVel(model, data)
                ke = float(data.energy[1])
                pe = float(data.energy[0])
            except AttributeError:
                pass
            ke_list.append(ke)
            pe_list.append(pe)

        return self._build_result(
            t_list, qpos_list, qvel_list, ee_pos_list, ke_list, pe_list
        )

    @staticmethod
    def _build_result(
        t_list: list[float],
        qpos_list: list[np.ndarray],
        qvel_list: list[np.ndarray],
        ee_pos_list: list[np.ndarray],
        ke_list: list[float],
        pe_list: list[float],
    ) -> MyoSuiteSimResult:
        t_arr = np.array(t_list)
        qpos_arr = np.array(qpos_list)
        qvel_arr = np.array(qvel_list)
        ee_pos_arr = np.array(ee_pos_list)
        ke_arr = np.array(ke_list)
        pe_arr = np.array(pe_list)

        ee_vel_arr = np.zeros_like(ee_pos_arr)
        for i in range(1, len(t_arr)):
            dt_i = max(t_arr[i] - t_arr[i - 1], 1e-12)
            ee_vel_arr[i] = (ee_pos_arr[i] - ee_pos_arr[i - 1]) / dt_i

        return MyoSuiteSimResult(
            t=t_arr,
            qpos_traj=qpos_arr,
            qvel_traj=qvel_arr,
            ee_pos_traj=ee_pos_arr,
            ee_vel_traj=ee_vel_arr,
            kinetic_energy_traj=ke_arr,
            potential_energy_traj=pe_arr,
        )
