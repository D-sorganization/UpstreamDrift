"""Pinocchio Perturbation Analyzer — PerturbationAnalyzer protocol (#1978).

Implements the ``PerturbationAnalyzer`` protocol for the Pinocchio rigid-body
dynamics engine.  Uses ``PinocchioPhysicsEngine`` for forward simulation with
polynomial torque profiles, and exposes Jacobian-based sensitivity as an
optional complement to Monte Carlo results.

Inherits from ``PerturbationAnalyzerBase`` (see #2273) which provides the
shared ``set_base_torque_profile``, ``perturb_torque``, ``extract_metrics``,
``run_batch``, and ``compare_profiles`` implementations.

Design by Contract
------------------
- ``set_base_torque_profile()`` must be called before ``run_batch()`` or
  ``perturb_torque()``.  Raises ``ValueError`` otherwise.
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
from src.shared.python.perturbation.perturbation_base import (
    MANDATORY_METRICS,
    ComparisonReport,
    PerturbationAnalyzerBase,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PinocchioPerturbationAnalyzer",
    "PinocchioSimResult",
    "ComparisonReport",
    "MANDATORY_METRICS",
]


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
        super().__init__()

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

        logger.info(
            "PinocchioPerturbationAnalyzer: model=%s, nq=%d, nv=%d, t_end=%.2f",
            self._model.name,
            self._nq,
            self._nv,
            self._t_end,
        )

    # ------------------------------------------------------------------
    # Base-class abstract method implementations
    # ------------------------------------------------------------------

    def _get_q_traj(self, sim_result: PinocchioSimResult) -> np.ndarray:
        return sim_result.q_traj

    def _get_v_traj(self, sim_result: PinocchioSimResult) -> np.ndarray:
        return sim_result.v_traj

    def _validate_sim_result_type(self, sim_result: object) -> None:
        if not isinstance(sim_result, PinocchioSimResult):
            raise ValueError(
                f"sim_result must be PinocchioSimResult, got {type(sim_result)}"
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
