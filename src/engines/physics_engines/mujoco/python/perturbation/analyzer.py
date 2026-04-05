# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.  # noqa: E501
# It requires domain-aware structural extraction to isolate its internal classes appropriately.  # noqa: E501

"""MuJoCo Perturbation Analyzer — PerturbationAnalyzer protocol for MuJoCo (#1980).

Implements the ``PerturbationAnalyzer`` protocol for the MuJoCo physics
simulation engine.  Uses ``mujoco.MjModel`` + ``mujoco.MjData`` for forward
simulation with polynomial torque profiles injected via ``data.ctrl``.

Design by Contract
------------------
- ``set_base_torque_profile()`` must be called before ``run_batch()`` or
  ``perturb_torque()``.  Raises ``AssertionError`` otherwise.
- ``run_batch()`` returns a ``PerturbationSummary`` containing all
  ``MANDATORY_METRICS`` as keys.
- ``extract_metrics()`` requires a ``MuJoCoSimResult`` and always returns
  finite values.
- mujoco must be importable; if not, ``ImportError`` is raised at
  construction time with a helpful message.

DRY
---
Delegates noise generation and coefficient perturbation to the shared
``src.shared.python.perturbation`` package.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.shared.python.engine_core.engine_availability import MUJOCO_AVAILABLE
from src.shared.python.perturbation.analyzer_base import (
    MANDATORY_METRICS,  # noqa: F401  re-exported for test imports
    ComparisonReport,  # noqa: F401
    PerturbationAnalyzerBase,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mandatory metric names (must match PendulumPerturbationAnalyzer.MANDATORY_METRICS)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Minimal MJCF model for testing (2-DOF pendulum with 2 actuators)
# ---------------------------------------------------------------------------

_MINIMAL_MJCF: str = """<mujoco model="minimal_pendulum">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1"/>
      <body name="link2" pos="0.5 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0.5 0 0" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" gear="1"/>
    <motor joint="j2" gear="1"/>
  </actuator>
</mujoco>"""


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------


@dataclass
class MuJoCoSimResult:
    """Trajectory data from one MuJoCo forward simulation.

    Attributes
    ----------
    t : ndarray, shape (n,)
        Time stamps.
    qpos_traj : ndarray, shape (n, nq)
        Joint position trajectory (generalized coordinates).
    qvel_traj : ndarray, shape (n, nv)
        Joint velocity trajectory.
    ee_pos_traj : ndarray, shape (n, 3)
        End-effector Cartesian position trajectory (last body xpos).
    ee_vel_traj : ndarray, shape (n, 3)
        End-effector velocity trajectory (finite-difference).
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


# ---------------------------------------------------------------------------
# Coefficient perturbation helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


class MuJoCoPerturbationAnalyzer(PerturbationAnalyzerBase):
    """PerturbationAnalyzer protocol implementation for the MuJoCo engine.

    Usage::

        from src.engines.physics_engines.mujoco.python.perturbation.analyzer import (
            MuJoCoPerturbationAnalyzer,
        )
        analyzer = MuJoCoPerturbationAnalyzer()  # uses _MINIMAL_MJCF
        analyzer.set_base_torque_profile({"coeffs": [[0.5, 0.1], [0.3, -0.05]]})
        summary = analyzer.run_batch(PerturbationConfig(n_trials=50))
        logger.info("RS: %s", summary.robustness_score)

    Design by Contract
    ------------------
    Pre:  mujoco must be importable.
    Pre:  ``set_base_torque_profile`` must be called before ``run_batch`` or
          ``perturb_torque``.
    Post: ``run_batch`` returns a ``PerturbationSummary`` with all
          ``MANDATORY_METRICS`` present.
    """

    ENGINE_NAME: str = "mujoco"

    def __init__(
        self,
        model_xml: str | None = None,
        model_path: str | Path | None = None,
        t_end: float = 1.5,
        ee_body_name: str | None = None,
    ) -> None:
        """Initialise the MuJoCo perturbation analyzer.

        Parameters
        ----------
        model_xml : str, optional
            MJCF XML string.  If None, uses bundled ``_MINIMAL_MJCF``.
        model_path : str or Path, optional
            Path to MJCF XML file.  Overrides ``model_xml`` if provided.
        t_end : float
            Simulation end time in seconds.
        ee_body_name : str, optional
            Name of the end-effector body in the MJCF model.  Defaults to
            the last non-world body.
        """
        if not MUJOCO_AVAILABLE:
            msg = "mujoco is not installed.  Install it with: pip install mujoco"
            raise ImportError(msg)

        import mujoco  # noqa: PLC0415

        if model_path is not None:
            model_path = Path(model_path)
            if not (model_path.exists()):
                raise ValueError(f"Model not found: {model_path}")
            self._model = mujoco.MjModel.from_xml_path(str(model_path))
        elif model_xml is not None:
            self._model = mujoco.MjModel.from_xml_string(model_xml)
        else:
            self._model = mujoco.MjModel.from_xml_string(_MINIMAL_MJCF)

        self._t_end = t_end
        self._nq = self._model.nq
        self._nv = self._model.nv
        self._nu = self._model.nu

        # End-effector body index
        if ee_body_name is not None:
            self._ee_body_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name
            )
            if not (self._ee_body_id >= 0):
                raise ValueError(f"Body '{ee_body_name}' not found in model")
        else:
            # Use the last non-world body (body 0 is always world in MuJoCo)
            self._ee_body_id = max(0, self._model.nbody - 1)

        self._base_coeffs: list[list[float]] | None = None
        self._nominal_result: MuJoCoSimResult | None = None

        logger.info(
            "MuJoCoPerturbationAnalyzer: nq=%d, nv=%d, nu=%d, t_end=%.2f",
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

    def extract_metrics(self, sim_result: object) -> dict[str, float | np.ndarray]:
        """Extract all MANDATORY_METRICS from a simulation result.

        Parameters
        ----------
        sim_result : MuJoCoSimResult

        Returns
        -------
        dict mapping metric name → scalar float or ndarray.

        Design by Contract
        ------------------
        Pre: sim_result is a MuJoCoSimResult with n_steps >= 2.
        Post: all MANDATORY_METRICS present; all values finite.
        """
        if not isinstance(sim_result, MuJoCoSimResult):
            raise ValueError(
                f"sim_result must be MuJoCoSimResult, got {type(sim_result)}"
            )  # noqa: E501
        if not (sim_result.n_steps >= 2):
            raise ValueError("Simulation must have >= 2 steps")

        r = sim_result
        last = r.n_steps - 1

        joint_angles_final = r.qpos_traj[last].copy()
        joint_velocities_final = r.qvel_traj[last].copy()
        ee_pos_final = r.ee_pos_traj[last].copy()
        ee_vel_final = r.ee_vel_traj[last].copy()
        ee_speed_final = float(np.linalg.norm(ee_vel_final))

        # ⚡ Bolt: Explicit element-wise sum of squares is faster than
        # np.linalg.norm(..., axis=1) when finding max
        sq_speeds = np.sum(r.ee_vel_traj**2, axis=1)
        peak_speed = float(np.sqrt(np.max(sq_speeds)))

        total_energy_final = float(
            r.kinetic_energy_traj[last] + r.potential_energy_traj[last]
        )

        trajectory_rmse = 0.0
        trajectory_max_deviation = 0.0
        if self._nominal_result is not None:
            nom = self._nominal_result
            n_cmp = min(r.n_steps, nom.n_steps)
            # ⚡ Bolt: Explicit element-wise sum of squares is faster than
            # np.linalg.norm(..., axis=1)
            sq_deviations = np.sum(
                (r.qpos_traj[:n_cmp] - nom.qpos_traj[:n_cmp]) ** 2, axis=1
            )
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

    def _simulate(self, coeffs: list[list[float]]) -> MuJoCoSimResult:
        """Run a MuJoCo forward simulation with polynomial torques.

        Each actuator's control follows ctrl_j(t) = sum_k c_jk * t^k.
        """
        import mujoco  # noqa: PLC0415

        model = self._model
        data = mujoco.MjData(model)

        nu = model.nu

        # Build per-actuator polynomial arrays (ascending → reversed for polyval)
        n_actuators_coeff = len(coeffs)
        joint_polys: list[np.ndarray] = []
        for j in range(nu):
            if j < n_actuators_coeff:
                joint_polys.append(np.array(coeffs[j][::-1]))
            else:
                joint_polys.append(np.array([0.0]))

        dt = model.opt.timestep
        n_steps = max(2, int(self._t_end / dt))

        t_list = []
        qpos_list = []
        qvel_list = []
        ee_pos_list = []
        ke_list = []
        pe_list = []

        mujoco.mj_resetData(model, data)

        for _ in range(n_steps):
            t = float(data.time)
            ctrl = np.array([float(np.polyval(joint_polys[j], t)) for j in range(nu)])
            data.ctrl[:] = ctrl

            mujoco.mj_step(model, data)

            t_list.append(float(data.time))
            qpos_list.append(data.qpos.copy())
            qvel_list.append(data.qvel.copy())
            ee_pos_list.append(data.xpos[self._ee_body_id].copy())

            # Kinetic energy via MuJoCo
            ke = float(mujoco.mj_getTotalmass(model) * 0.0)  # placeholder
            try:
                mujoco.mj_energyPos(model, data)
                mujoco.mj_energyVel(model, data)
                ke = float(data.energy[1])
                pe = float(data.energy[0])
            except AttributeError:
                # Older MuJoCo API
                ke = 0.0
                pe = 0.0
            ke_list.append(ke)
            pe_list.append(pe)

        t_arr = np.array(t_list)
        qpos_arr = np.array(qpos_list)
        qvel_arr = np.array(qvel_list)
        ee_pos_arr = np.array(ee_pos_list)

        # EE velocities via finite difference
        ee_vel_arr = np.zeros_like(ee_pos_arr)
        for i in range(1, len(t_arr)):
            dt_i = max(t_arr[i] - t_arr[i - 1], 1e-12)
            ee_vel_arr[i] = (ee_pos_arr[i] - ee_pos_arr[i - 1]) / dt_i

        return MuJoCoSimResult(
            t=t_arr,
            qpos_traj=qpos_arr,
            qvel_traj=qvel_arr,
            ee_pos_traj=ee_pos_arr,
            ee_vel_traj=ee_vel_arr,
            kinetic_energy_traj=np.array(ke_list),
            potential_energy_traj=np.array(pe_list),
        )
