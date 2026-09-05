"""Cross-engine perturbation consistency runner.

Addresses GH2021: Severe divergence in cross-engine perturbation consistency (CV > 1.0).

Root causes of divergence:
- Mismatched integration timesteps across engines (0.001, 0.002, 0.01 defaults)
- No standardised physical parameter set for cross-engine comparison
- No unified runner that applies identical noise to all engines

This module provides ``CrossEnginePerturbationRunner`` which:
- Normalises the integration timestep (every engine is stepped with the same dt)
- Resets every engine to the same initial state before each trial
- Applies identical additive noise to all engines within a trial
- Collects unified metrics and reports coefficient of variation (CV) per metric

Design by Contract
------------------
- config.dt > 0, config.t_end > 0, config.t_end > config.dt
- config.noise_amplitude >= 0
- At least one engine must be registered before run_comparison
- CV computation is safe for zero-mean metrics (returns 0.0)

DRY
---
Noise generation delegates to ``perturb_torque_profile`` in perturbation_analysis.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.shared.python.pendulum_simulator.perturbation_analysis import (
    perturb_torque_profile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine protocol (minimal subset used by the runner)
# ---------------------------------------------------------------------------


@runtime_checkable
class SteppableEngine(Protocol):
    """Minimal protocol: an engine that can be reset, controlled, and stepped."""

    def reset(self) -> None:
        """Reset to initial state."""
        ...

    def set_control(self, u: np.ndarray) -> None:
        """Set control inputs."""
        ...

    def step(self, dt: float | None = None) -> None:
        """Advance simulation by dt."""
        ...

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (positions, velocities)."""
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CrossEngineSimConfig:
    """Configuration for cross-engine perturbation comparison.

    Attributes
    ----------
    t_end : float
        Total simulation duration in seconds. Default 1.5 s (per GH2021 diagnostics).
    dt : float
        Integration timestep used for *all* engines. Default 0.01 s.
        This overrides each engine's internal default to ensure a fair comparison.
    noise_amplitude : float
        Standard deviation of additive Gaussian torque noise. Default 0.1 N·m.
    n_trials : int
        Number of Monte Carlo trials per engine. Default 10.
    seed : int
        Base random seed. Trial i uses seed+i for reproducibility.

    Design by Contract
    ------------------
    Post-init: t_end > dt > 0, noise_amplitude >= 0, n_trials > 0
    """

    t_end: float = 1.5
    dt: float = 0.01
    noise_amplitude: float = 0.1
    n_trials: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.t_end <= 0:
            raise ValueError(f"t_end must be positive, got {self.t_end}")
        if self.t_end <= self.dt:
            raise ValueError(
                f"t_end ({self.t_end}) must be greater than dt ({self.dt})"
            )
        if self.noise_amplitude < 0:
            raise ValueError(
                f"noise_amplitude must be non-negative, got {self.noise_amplitude}"
            )
        if self.n_trials <= 0:
            raise ValueError(f"n_trials must be positive, got {self.n_trials}")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class EngineTrialMetrics:
    """Metrics collected from a single engine trial.

    Attributes
    ----------
    total_energy_final : float
        Total mechanical energy at end of simulation in Joules (J).
        Computed using the engine's mass matrix / inertia tensor M(q)
        (0.5 * v^T M(q) v) and potential energy where available.
    end_effector_speed_final : float
        Cartesian end-effector speed at end of simulation in meters per second (m/s).
    peak_end_effector_speed : float
        Maximum Cartesian end-effector speed observed across all timesteps (m/s).
    trajectory_q : np.ndarray
        Position trajectory, shape (n_steps, n_dof).
    trajectory_v : np.ndarray
        Velocity trajectory, shape (n_steps, n_dof).
    """

    total_energy_final: float
    end_effector_speed_final: float
    peak_end_effector_speed: float
    trajectory_q: np.ndarray
    trajectory_v: np.ndarray


@dataclass
class CrossEngineRunResult:
    """Aggregated result across trials for a single engine.

    Attributes
    ----------
    engine_name : str
    metrics_per_trial : list of EngineTrialMetrics
    mean_total_energy_final : float
    std_total_energy_final : float
    mean_end_effector_speed_final : float
    std_end_effector_speed_final : float
    mean_peak_end_effector_speed : float
    std_peak_end_effector_speed : float
    """

    engine_name: str
    metrics_per_trial: list[EngineTrialMetrics] = field(default_factory=list)
    mean_total_energy_final: float = 0.0
    std_total_energy_final: float = 0.0
    mean_end_effector_speed_final: float = 0.0
    std_end_effector_speed_final: float = 0.0
    mean_peak_end_effector_speed: float = 0.0
    std_peak_end_effector_speed: float = 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class CrossEnginePerturbationRunner:
    """Run identical perturbation trials across multiple physics engines.

    Solves the divergence described in GH2021 by ensuring every engine:
    1. Is reset to the same zero initial state before each trial
    2. Receives the same perturbed torque profile (seeded per trial)
    3. Is integrated with the same dt (overriding internal defaults)

    Usage
    -----
    >>> config = CrossEngineSimConfig(t_end=1.5, dt=0.01)
    >>> runner = CrossEnginePerturbationRunner(config)
    >>> runner.register_engine("pendulum", pendulum_engine)
    >>> runner.register_engine("mujoco", mujoco_engine)
    >>> results = runner.run_comparison(base_torque_profile)
    >>> cv = runner.compute_cv_summary(results)

    Design by Contract
    ------------------
    Pre:  config is a valid CrossEngineSimConfig
    Pre:  register_engine called at least once before run_comparison
    Post: run_comparison returns one result per registered engine
    Post: compute_cv_summary returns CV >= 0 for each metric key
    """

    def __init__(self, config: CrossEngineSimConfig) -> None:
        if not isinstance(config, CrossEngineSimConfig):
            raise TypeError(
                f"config must be a CrossEngineSimConfig, got {type(config).__name__}"
            )
        self._config = config
        self._engines: dict[str, SteppableEngine] = {}

    @property
    def config(self) -> CrossEngineSimConfig:
        """Return the runner configuration."""
        return self._config

    def register_engine(self, name: str, engine: SteppableEngine) -> None:
        """Register a physics engine under the given name.

        Parameters
        ----------
        name : str — unique identifier for this engine
        engine : SteppableEngine — engine implementing reset/set_control/step/get_state

        Raises
        ------
        ValueError : if name is empty or already registered
        TypeError  : if engine does not implement SteppableEngine
        """
        if not name:
            raise ValueError("Engine name must be non-empty")
        if name in self._engines:
            raise ValueError(f"Engine '{name}' is already registered")
        if not isinstance(engine, SteppableEngine):
            raise TypeError(
                f"engine must implement SteppableEngine protocol, got {type(engine).__name__}"
            )
        self._engines[name] = engine
        logger.debug("Registered engine: %s", name)

    def run_comparison(
        self,
        base_torque_profile: np.ndarray,
    ) -> dict[str, CrossEngineRunResult]:
        """Run identical perturbed trials on all registered engines.

        Parameters
        ----------
        base_torque_profile : np.ndarray, shape (n_steps, n_actuators) or (n_steps,)
            Nominal torque profile. n_steps must equal round(t_end / dt).
            If 1-D, it is broadcast to all actuators.

        Returns
        -------
        dict mapping engine_name → CrossEngineRunResult

        Raises
        ------
        RuntimeError : if no engines are registered
        ValueError   : if base_torque_profile shape is incompatible with config
        """
        if not self._engines:
            raise RuntimeError(
                "No engines registered. Call register_engine() before run_comparison()."
            )
        expected_steps = round(self._config.t_end / self._config.dt)
        profile = np.atleast_1d(np.asarray(base_torque_profile, dtype=float))
        profile_2d: np.ndarray
        if profile.ndim == 1:
            # Shape (n_steps,) — same torque for all DOF; reshape to (n_steps, 1)
            profile_2d = profile.reshape(-1, 1)
        elif profile.ndim == 2:
            profile_2d = profile
        else:
            raise ValueError(
                f"base_torque_profile must be 1-D or 2-D, got {profile.ndim}-D"
            )
        if profile_2d.shape[0] != expected_steps:
            raise ValueError(
                f"base_torque_profile has {profile_2d.shape[0]} steps, "
                f"expected {expected_steps} (t_end={self._config.t_end}, dt={self._config.dt})"
            )

        n_actuators = profile_2d.shape[1]
        results: dict[str, CrossEngineRunResult] = {
            name: CrossEngineRunResult(engine_name=name) for name in self._engines
        }

        for trial_idx in range(self._config.n_trials):
            trial_seed = self._config.seed + trial_idx
            # Generate identical perturbed profiles for this trial (same seed)
            perturbed_columns = [
                perturb_torque_profile(
                    profile_2d[:, col],
                    noise_amplitude=self._config.noise_amplitude,
                    noise_type="additive",
                    seed=trial_seed * n_actuators + col,
                )
                for col in range(n_actuators)
            ]
            perturbed_profile = np.column_stack(perturbed_columns)  # (n_steps, n_act)

            for name, engine in self._engines.items():
                metrics = self._run_single_trial(engine, perturbed_profile, trial_idx)
                results[name].metrics_per_trial.append(metrics)

        # Aggregate statistics per engine
        for result in results.values():
            self._aggregate(result)

        return results

    def _run_single_trial(
        self,
        engine: SteppableEngine,
        perturbed_profile: np.ndarray,
        trial_idx: int,
    ) -> EngineTrialMetrics:
        """Run one trial on a single engine and collect metrics.

        Parameters
        ----------
        engine : SteppableEngine
        perturbed_profile : np.ndarray, shape (n_steps, n_actuators)
        trial_idx : int — used for logging only

        Returns
        -------
        EngineTrialMetrics
        """
        engine.reset()
        n_steps = perturbed_profile.shape[0]

        traj_q: list[np.ndarray] = []
        traj_v: list[np.ndarray] = []
        peak_speed = 0.0

        for step_i in range(n_steps):
            control = perturbed_profile[step_i, :]
            # Broadcast single-channel profiles to 2-DOF (shoulder + wrist)
            if len(control) < 2:
                control = np.full(2, control[0])
            engine.set_control(control)
            engine.step(self._config.dt)
            q, v = engine.get_state()
            traj_q.append(q.copy())
            traj_v.append(v.copy())
            speed = self._compute_end_effector_speed(engine, q, v)
            if speed > peak_speed:
                peak_speed = speed

        q_arr = np.array(traj_q)
        v_arr = np.array(traj_v)
        final_q = q_arr[-1] if len(q_arr) > 0 else np.zeros(1)
        final_v = v_arr[-1] if len(v_arr) > 0 else np.zeros(1)
        final_speed = self._compute_end_effector_speed(engine, final_q, final_v)
        total_energy = self._compute_trial_energy(engine, final_q, final_v)

        logger.debug(
            "Trial %d: final_speed=%.4f, peak_speed=%.4f, energy=%.4f",
            trial_idx,
            final_speed,
            peak_speed,
            total_energy,
        )

        return EngineTrialMetrics(
            total_energy_final=total_energy,
            end_effector_speed_final=final_speed,
            peak_end_effector_speed=peak_speed,
            trajectory_q=q_arr,
            trajectory_v=v_arr,
        )

    @staticmethod
    def _eval_native_total_energy(
        engine: Any,
        q: np.ndarray,
        v: np.ndarray,
    ) -> float | None:
        """Query engine-native total energy methods if available."""
        for method_name in ("get_total_energy", "total_energy", "compute_total_energy"):
            fn = getattr(engine, method_name, None)
            if callable(fn):
                try:
                    return float(fn())
                except TypeError:
                    try:
                        return float(fn(q, v))
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    pass
        return None

    @staticmethod
    def _eval_kinetic_potential_methods(
        engine: Any,
        q: np.ndarray,
        v: np.ndarray,
    ) -> tuple[float | None, float | None]:
        """Query separate kinetic and potential energy methods if available."""
        ke: float | None = None
        for ke_name in (
            "get_kinetic_energy",
            "compute_kinetic_energy",
            "kinetic_energy",
        ):
            fn = getattr(engine, ke_name, None)
            if callable(fn):
                try:
                    ke = float(fn())
                    break
                except TypeError:
                    try:
                        ke = float(fn(q, v))
                        break
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    pass

        pe: float | None = None
        for pe_name in (
            "get_potential_energy",
            "compute_potential_energy",
            "potential_energy",
        ):
            fn = getattr(engine, pe_name, None)
            if callable(fn):
                try:
                    pe = float(fn())
                    break
                except TypeError:
                    try:
                        pe = float(fn(q))
                        break
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    pass

        return ke, pe

    @staticmethod
    def _eval_mass_or_inertia_energy(
        engine: Any,
        q: np.ndarray,
        v: np.ndarray,
    ) -> float | None:
        """Evaluate kinetic energy from mass matrix, inertia tensor, or mass."""
        mass_matrix = None
        if hasattr(engine, "get_mass_matrix") and callable(engine.get_mass_matrix):
            try:
                mass_matrix = engine.get_mass_matrix(q)
            except Exception:  # noqa: BLE001
                pass
        elif hasattr(engine, "mass_matrix"):
            mass_matrix = getattr(engine, "mass_matrix", None)

        v_vec = np.asarray(v, dtype=float)
        if mass_matrix is not None:
            M = np.asarray(mass_matrix, dtype=float)
            if M.ndim == 2 and M.shape[0] == len(v_vec) and M.shape[1] == len(v_vec):
                return 0.5 * float(v_vec @ M @ v_vec)

        inertia = getattr(engine, "inertia", None)
        if inertia is not None:
            if np.isscalar(inertia):
                i_val = float(np.asarray(inertia).item())
                return 0.5 * i_val * float(np.dot(v_vec, v_vec))
            i_arr = np.asarray(inertia, dtype=float)
            if i_arr.ndim == 1 and len(i_arr) == len(v_vec):
                return 0.5 * float(np.sum(i_arr * (v_vec**2)))
            if i_arr.ndim == 2 and i_arr.shape == (len(v_vec), len(v_vec)):
                return 0.5 * float(v_vec @ i_arr @ v_vec)

        mass = getattr(engine, "mass", None)
        lengths = getattr(engine, "lengths", getattr(engine, "link_lengths", None))
        if (
            mass is not None
            and lengths is not None
            and np.isscalar(mass)
            and len(lengths) == 1
            and len(v) == 1
        ):
            i_val = float(np.asarray(mass).item()) * (float(lengths[0]) ** 2)
            return 0.5 * i_val * (float(v[0]) ** 2)

        if mass is not None:
            if np.isscalar(mass):
                m_val = float(np.asarray(mass).item())
                return 0.5 * m_val * float(np.dot(v_vec, v_vec))
            m_arr = np.asarray(mass, dtype=float)
            if m_arr.ndim == 1 and len(m_arr) == len(v_vec):
                return 0.5 * float(np.sum(m_arr * (v_vec**2)))
            if m_arr.ndim == 2 and m_arr.shape == (len(v_vec), len(v_vec)):
                return 0.5 * float(v_vec @ m_arr @ v_vec)

        return None

    @staticmethod
    def _compute_trial_energy(
        engine: Any,
        q: np.ndarray,
        v: np.ndarray,
    ) -> float:
        """Compute total mechanical energy in Joules (J).

        Queries engine-native energy methods if available (e.g.
        ``get_total_energy()``, ``total_energy()``, or combined kinetic +
        potential energy). Otherwise evaluates kinetic energy 0.5 * v^T M(q) v
        using the engine's mass matrix, inertia, or mass properties, plus
        potential energy if available.
        """
        native_te = CrossEnginePerturbationRunner._eval_native_total_energy(
            engine, q, v
        )
        if native_te is not None:
            return native_te

        ke, pe = CrossEnginePerturbationRunner._eval_kinetic_potential_methods(
            engine, q, v
        )
        pe_val = pe if pe is not None else 0.0
        if ke is not None:
            return ke + pe_val

        matrix_ke = CrossEnginePerturbationRunner._eval_mass_or_inertia_energy(
            engine, q, v
        )
        if matrix_ke is not None:
            return matrix_ke + pe_val

        v_vec = np.asarray(v, dtype=float)
        default_mass = float(getattr(engine, "default_mass", 1.0))
        return 0.5 * default_mass * float(np.dot(v_vec, v_vec)) + pe_val

    @staticmethod
    def _compute_end_effector_speed(
        engine: Any,
        q: np.ndarray,
        v: np.ndarray,
    ) -> float:
        """Compute Cartesian end-effector speed in meters per second (m/s).

        Queries engine-native end-effector kinematics / Jacobians if available.
        Falls back to link-length forward kinematics for planar chains, or
        to the generalized velocity norm if no spatial geometry is defined.
        """
        # 1. Native end-effector speed or velocity
        for name in ("get_end_effector_speed", "end_effector_speed"):
            fn = getattr(engine, name, None)
            if callable(fn):
                try:
                    return float(fn())
                except TypeError:
                    try:
                        return float(fn(q, v))
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    pass

        for name in (
            "get_end_effector_velocity",
            "compute_end_effector_velocity",
            "end_effector_velocity",
        ):
            fn = getattr(engine, name, None)
            if callable(fn):
                try:
                    vel = fn()
                    return float(np.linalg.norm(vel))
                except TypeError:
                    try:
                        vel = fn(q, v)
                        return float(np.linalg.norm(vel))
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    pass

        # 2. Jacobian: v_ee = J[:3] @ v (or J @ v if 2D/3D)
        if hasattr(engine, "get_jacobian") and callable(engine.get_jacobian):
            try:
                J = np.asarray(engine.get_jacobian(q), dtype=float)
                v_vec = np.asarray(v, dtype=float)
                if J.shape[0] == 6:
                    order = getattr(engine, "spatial_order", None)
                    vel = (
                        J[3:, :] @ v_vec
                        if order == "angular_linear"
                        else J[:3, :] @ v_vec
                    )
                else:
                    vel = J @ v_vec
                return float(np.linalg.norm(vel))
            except Exception:  # noqa: BLE001
                pass

        # 3. Planar kinematic chain lengths
        lengths = getattr(engine, "lengths", getattr(engine, "link_lengths", None))
        if lengths is not None:
            try:
                l_arr = np.asarray(lengths, dtype=float)
                q_arr = np.asarray(q, dtype=float)
                v_arr = np.asarray(v, dtype=float)
                if len(l_arr) == 1 and len(q_arr) >= 1 and len(v_arr) >= 1:
                    return float(l_arr[0] * abs(v_arr[0]))
                if len(l_arr) >= 2 and len(q_arr) >= 2 and len(v_arr) >= 2:
                    vx = -l_arr[0] * np.sin(q_arr[0]) * v_arr[0] - l_arr[1] * np.sin(
                        q_arr[0] + q_arr[1]
                    ) * (v_arr[0] + v_arr[1])
                    vy = l_arr[0] * np.cos(q_arr[0]) * v_arr[0] + l_arr[1] * np.cos(
                        q_arr[0] + q_arr[1]
                    ) * (v_arr[0] + v_arr[1])
                    return float(np.hypot(vx, vy))
            except Exception:  # noqa: BLE001
                pass

        # 4. Fallback norm of generalized velocities
        return float(np.linalg.norm(v))

    @staticmethod
    def _aggregate(result: CrossEngineRunResult) -> None:
        """Compute mean/std statistics from trial metrics in-place."""
        if not result.metrics_per_trial:
            return
        energies = np.array([m.total_energy_final for m in result.metrics_per_trial])
        speeds = np.array(
            [m.end_effector_speed_final for m in result.metrics_per_trial]
        )
        peaks = np.array([m.peak_end_effector_speed for m in result.metrics_per_trial])

        result.mean_total_energy_final = float(np.mean(energies))
        result.std_total_energy_final = float(np.std(energies))
        result.mean_end_effector_speed_final = float(np.mean(speeds))
        result.std_end_effector_speed_final = float(np.std(speeds))
        result.mean_peak_end_effector_speed = float(np.mean(peaks))
        result.std_peak_end_effector_speed = float(np.std(peaks))

    def compute_cv_summary(
        self,
        results: dict[str, CrossEngineRunResult],
    ) -> dict[str, float]:
        """Compute coefficient of variation (CV) across engines per metric.

        CV = std(metric_means) / mean(metric_means) across all registered engines.
        A CV close to 0 indicates engines agree; CV > 1.0 indicates severe divergence.

        Parameters
        ----------
        results : dict[str, CrossEngineRunResult]
            Output of run_comparison().

        Returns
        -------
        dict with keys:
            'cv_total_energy_final',
            'cv_end_effector_speed_final',
            'cv_peak_end_effector_speed'

        Raises
        ------
        ValueError : if results is empty
        """
        if not results:
            raise ValueError("results must be non-empty")

        def _cv(values: np.ndarray) -> float:
            """Compute CV safely (returns 0.0 when mean is near zero)."""
            m = float(np.mean(values))
            s = float(np.std(values))
            if abs(m) < 1e-12:
                return 0.0
            return s / abs(m)

        energies = np.array([r.mean_total_energy_final for r in results.values()])
        speeds = np.array([r.mean_end_effector_speed_final for r in results.values()])
        peaks = np.array([r.mean_peak_end_effector_speed for r in results.values()])

        summary = {
            "cv_total_energy_final": _cv(energies),
            "cv_end_effector_speed_final": _cv(speeds),
            "cv_peak_end_effector_speed": _cv(peaks),
        }

        for key, cv in summary.items():
            if cv > 1.0:
                logger.warning(
                    "High CV detected for %s: %.2f — engines may have mismatched "
                    "parameters (check dt, initial state, and physical constants)",
                    key,
                    cv,
                )
            else:
                logger.info("CV %s: %.4f", key, cv)

        return summary

    def compute_trajectory_rmse(
        self,
        results: dict[str, CrossEngineRunResult],
        baseline_engine: str,
    ) -> dict[str, float]:
        """Compute trajectory RMSE of each engine versus the baseline engine.

        Parameters
        ----------
        results : dict[str, CrossEngineRunResult]
        baseline_engine : str — name of the reference engine

        Returns
        -------
        dict mapping engine_name → mean RMSE across trials (vs baseline)

        Raises
        ------
        ValueError : if baseline_engine is not in results
        """
        if baseline_engine not in results:
            raise ValueError(
                f"Baseline engine '{baseline_engine}' not found in results. "
                f"Available: {list(results.keys())}"
            )
        baseline_result = results[baseline_engine]
        rmse_dict: dict[str, float] = {}

        for name, result in results.items():
            if name == baseline_engine:
                rmse_dict[name] = 0.0
                continue
            trial_rmses: list[float] = []
            n_trials = min(
                len(result.metrics_per_trial), len(baseline_result.metrics_per_trial)
            )
            for i in range(n_trials):
                q_base = baseline_result.metrics_per_trial[i].trajectory_q
                q_cmp = result.metrics_per_trial[i].trajectory_q
                min_steps = min(q_base.shape[0], q_cmp.shape[0])
                min_dof = min(q_base.shape[1], q_cmp.shape[1]) if q_base.ndim > 1 else 1
                if q_base.ndim > 1 and q_cmp.ndim > 1:
                    diff = q_base[:min_steps, :min_dof] - q_cmp[:min_steps, :min_dof]
                else:
                    diff = q_base[:min_steps] - q_cmp[:min_steps]
                # ⚡ Bolt: np.vdot is ~2x faster than np.mean(diff**2) and avoids temporary array allocations
                rmse = float(np.sqrt(np.vdot(diff, diff) / diff.size))
                trial_rmses.append(rmse)
            rmse_dict[name] = (
                float(np.mean(trial_rmses)) if trial_rmses else float("nan")
            )

        return rmse_dict
