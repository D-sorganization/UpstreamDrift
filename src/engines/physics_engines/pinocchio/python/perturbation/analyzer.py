# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.  # noqa: E501
# It requires domain-aware structural extraction to isolate its internal classes appropriately.  # noqa: E501

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.shared.python.engine_core.engine_availability import PINOCCHIO_AVAILABLE

# Shared noise / perturbation helpers
from src.shared.python.perturbation.analyzer_base import (  # noqa: F401  re-exported for test imports
    MANDATORY_METRICS,
    ComparisonReport,  # noqa: F401
    PerturbationAnalyzerBase,
    build_joint_polys,
    compute_ee_velocity_fd,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mandatory metric names (must match PendulumPerturbationAnalyzer.MANDATORY_METRICS)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Coefficient perturbation helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


class PinocchioPerturbationAnalyzer(PerturbationAnalyzerBase):
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
        if not (self._urdf_path.exists()):
            raise ValueError(f"URDF not found: {self._urdf_path}")

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
        if not isinstance(profile, dict):
            raise ValueError(f"profile must be a dict, got {type(profile)}")
        if "coeffs" not in profile:
            raise ValueError("'coeffs' key missing from profile")
        coeffs = profile["coeffs"]
        if not (isinstance(coeffs, list) and len(coeffs) > 0):
            raise ValueError("profile['coeffs'] must be a non-empty list")
        self._base_coeffs = coeffs
        self._nominal_result = self._simulate(coeffs)

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
        if not isinstance(sim_result, PinocchioSimResult):
            raise ValueError(
                f"sim_result must be PinocchioSimResult, got {type(sim_result)}"
            )  # noqa: E501
        if not (sim_result.n_steps >= 2):
            raise ValueError("Simulation must have >= 2 steps")

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
        # ⚡ Bolt: np.einsum is ~2-3x faster than np.sum(..., axis=1) for computing
        # sum of squares when finding max
        sq_speeds = np.einsum("ij,ij->i", r.ee_vel_traj, r.ee_vel_traj)
        peak_speed = float(np.sqrt(np.max(sq_speeds)))

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
            # ⚡ Bolt: np.einsum is ~2-3x faster than np.sum(..., axis=1) for computing
            # sum of squares
            diffs = r.q_traj[:n_cmp] - nom.q_traj[:n_cmp]
            sq_deviations = np.einsum("ij,ij->i", diffs, diffs)
            trajectory_rmse = float(np.sqrt(np.mean(sq_deviations)))
            trajectory_max_deviation = float(np.sqrt(np.max(sq_deviations)))

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

    def _simulate(self, coeffs: list[list[float]]) -> PinocchioSimResult:
        """Run a Pinocchio ABA forward simulation with polynomial torques.

        Each joint's torque follows tau_j(t) = sum_k c_jk * t^k.
        """
        import pinocchio as pin  # noqa: PLC0415

        model = self._model
        data = model.createData()  # fresh data per trial for thread safety
        nv = model.nv

        # Build per-joint polynomial arrays (ascending → reversed for polyval)
        joint_polys = build_joint_polys(coeffs, nv)

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
        ee_vel_arr = compute_ee_velocity_fd(ee_pos_arr, t_arr)

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
