# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""OpenSim Perturbation Analyzer — PerturbationAnalyzer protocol for OpenSim (#1981).

Implements the ``PerturbationAnalyzer`` protocol for the OpenSim physics
simulation engine.  Uses a built-in minimal pendulum model when no model path
is provided.  When ``opensim`` is not installed the module imports cleanly but
construction raises ``ImportError``.


Inherits from ``PerturbationAnalyzerBase`` (see #2273) which provides the
shared ``set_base_torque_profile``, ``perturb_torque``, ``extract_metrics``,
``run_batch``, and ``compare_profiles`` implementations.

Design by Contract
------------------
- ``set_base_torque_profile()`` must be called before ``run_batch()`` or
  ``perturb_torque()``.  Raises ``ValueError`` otherwise.
- ``run_batch()`` returns a ``PerturbationSummary`` containing all
  ``MANDATORY_METRICS`` as keys.
- ``extract_metrics()`` requires an ``OpenSimSimResult`` and always returns
  finite values.

DRY
---
Delegates noise generation and coefficient perturbation to the shared
``src.shared.python.perturbation`` package — the same functions used by
``PendulumPerturbationAnalyzer``.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.perturbation.perturbation_base import (
    MANDATORY_METRICS,
    ComparisonReport,
    PerturbationAnalyzerBase,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OpenSimPerturbationAnalyzer",
    "OpenSimSimResult",
    "ComparisonReport",
    "MANDATORY_METRICS",
]

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


@dataclass
class _SimulationTrajectory:
    """Mutable trajectory samples collected during one OpenSim simulation."""

    t: list[float]
    qpos: list[np.ndarray]
    qvel: list[np.ndarray]
    ee_pos: list[np.ndarray]
    kinetic_energy: list[float]
    potential_energy: list[float]

    @classmethod
    def empty(cls) -> _SimulationTrajectory:
        return cls([], [], [], [], [], [])

    def append(
        self,
        *,
        t: float,
        q: np.ndarray,
        qdot: np.ndarray,
        ee_pos: np.ndarray,
        kinetic_energy: float,
        potential_energy: float,
    ) -> None:
        self.t.append(t)
        self.qpos.append(q)
        self.qvel.append(qdot)
        self.ee_pos.append(ee_pos)
        self.kinetic_energy.append(kinetic_energy)
        self.potential_energy.append(potential_energy)


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


class OpenSimPerturbationAnalyzer(PerturbationAnalyzerBase):
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
        super().__init__()
        try:
            import opensim as osim  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "opensim not found. Please install it with 'pip install opensim'."
            ) from e

        if model_path is None:
            # Create a temporary file with the minimal XML
            tmp_path = Path("minimal_pendulum.osim")
            if not tmp_path.exists():
                tmp_path.write_text(_MINIMAL_OSIM_XML, encoding="utf-8")
            model_path = tmp_path

        self._model = osim.Model(str(model_path))
        self._model.finalizeFromProperties()
        self._nq = self._model.getCoordinateSet().getSize()
        self._nu = self._model.getForceSet().getSize()
        self._t_end = t_end
        self._dt = dt

        if ee_body_name:
            self._ee_body_name = ee_body_name
        else:
            body_set = self._model.getBodySet()
            if body_set.getSize() > 0:
                self._ee_body_name = body_set.get(body_set.getSize() - 1).getName()
            else:
                self._ee_body_name = "ground"

        logger.info(
            "OpenSimPerturbationAnalyzer: nq=%d, nu=%d, t_end=%.2f, ee=%s",
            self._nq,
            self._nu,
            self._t_end,
            self._ee_body_name,
        )

    # ------------------------------------------------------------------
    # Base-class abstract method implementations
    # ------------------------------------------------------------------

    def _get_q_traj(self, sim_result: OpenSimSimResult) -> np.ndarray:
        return sim_result.qpos_traj

    def _get_v_traj(self, sim_result: OpenSimSimResult) -> np.ndarray:
        return sim_result.qvel_traj

    def _validate_sim_result_type(self, sim_result: object) -> None:
        if not isinstance(sim_result, OpenSimSimResult):
            raise ValueError(
                f"sim_result must be OpenSimSimResult, got {type(sim_result)}"
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

        dt = self._dt
        n_steps = max(2, int(self._t_end / dt))
        body_set = model.getBodySet()
        ground = model.getGround()
        joint_polys = self._build_joint_polys(coeffs, nu)
        trajectory = _SimulationTrajectory.empty()

        model.initStateFromProperties(state)
        manager = self._build_reusable_manager(osim, model)

        for step in range(n_steps):
            t = step * dt

            self._apply_polynomial_controls(osim, model, force_set, joint_polys, t, nu)
            with contextlib.suppress(ValueError, RuntimeError, TypeError):
                model.realizeAcceleration(state)

            q, qdot = self._read_coordinate_state(coord_set, state)
            ee_pos = self._end_effector_position(osim, state, body_set, ground, q)
            ke, pe = self._read_energy(model, state)
            trajectory.append(
                t=t,
                q=q,
                qdot=qdot,
                ee_pos=ee_pos,
                kinetic_energy=ke,
                potential_energy=pe,
            )
            self._integrate_step(manager, state, coord_set, q, qdot, t, dt)

        return self._build_sim_result(trajectory)

    def _build_joint_polys(
        self, coeffs: list[list[float]], nu: int
    ) -> list[np.ndarray]:
        """Return per-actuator polynomial arrays ordered for ``np.polyval``."""
        n_coeff_sets = len(coeffs)
        return [
            np.array(coeffs[j][::-1]) if j < n_coeff_sets else np.array([0.0])
            for j in range(nu)
        ]

    def _build_reusable_manager(self, osim: Any, model: Any) -> Any | None:
        """Build the hoisted OpenSim Manager, or ``None`` for Euler fallback."""
        with contextlib.suppress(ValueError, RuntimeError, TypeError):
            integrator = osim.RungeKuttaMersonIntegrator(model.getSystem())
            integrator.setAccuracy(1e-4)
            return osim.Manager(model, integrator)
        return None

    def _apply_polynomial_controls(
        self,
        osim: Any,
        model: Any,
        force_set: Any,
        joint_polys: list[np.ndarray],
        t: float,
        nu: int,
    ) -> None:
        """Apply polynomial torques via OpenSim coordinate actuators."""
        for j in range(nu):
            try:
                force_obj = force_set.get(j)
                ctrl = float(np.polyval(joint_polys[j], t))
                force_obj.setControls(osim.Vector(1, ctrl), model.updDefaultControls())
            except (ValueError, RuntimeError, TypeError):  # noqa: BLE001
                pass

    def _read_coordinate_state(
        self, coord_set: Any, state: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read generalized coordinates and speeds from the OpenSim state."""
        nq = coord_set.getSize()
        q = np.zeros(nq)
        qdot = np.zeros(nq)
        for k in range(nq):
            coord = coord_set.get(k)
            q[k] = coord.getValue(state)
            qdot[k] = coord.getSpeedValue(state)
        return q, qdot

    def _end_effector_position(
        self, osim: Any, state: Any, body_set: Any, ground: Any, q: np.ndarray
    ) -> np.ndarray:
        """Return end-effector position, falling back to planar kinematics."""
        try:
            n_bodies = body_set.getSize()
            if n_bodies > 0:
                ee_body = body_set.get(n_bodies - 1)
                pos_in_ground = ground.findStationLocationInGround(
                    state,
                    ee_body.findStationLocationInGround(state, osim.Vec3(0, 0, 0)),
                )
                return np.array([pos_in_ground[0], pos_in_ground[1], pos_in_ground[2]])
        except (ValueError, RuntimeError, TypeError):  # noqa: BLE001
            pass
        return self._fallback_end_effector_position(q)

    def _fallback_end_effector_position(self, q: np.ndarray) -> np.ndarray:
        """Approximate end-effector position from joint angles."""
        nq = len(q)
        if nq == 0:
            return np.zeros(3)
        link_len = 0.5
        angle_sum = float(np.sum(q))
        x_pos = (
            link_len * np.sin(q[0]) + link_len * np.sin(angle_sum)
            if nq >= 2
            else link_len * np.sin(q[0])
        )
        y_pos = (
            -(link_len * np.cos(q[0]) + link_len * np.cos(angle_sum))
            if nq >= 2
            else -link_len * np.cos(q[0])
        )
        return np.array([x_pos, y_pos, 0.0])

    def _read_energy(self, model: Any, state: Any) -> tuple[float, float]:
        """Return kinetic and potential energy if OpenSim can realize velocity."""
        try:
            model.realizeVelocity(state)
            return float(model.calcKineticEnergy(state)), float(
                model.calcPotentialEnergy(state)
            )
        except (ValueError, RuntimeError, TypeError):  # noqa: BLE001
            return 0.0, 0.0

    def _integrate_step(
        self,
        manager: Any | None,
        state: Any,
        coord_set: Any,
        q: np.ndarray,
        qdot: np.ndarray,
        t: float,
        dt: float,
    ) -> None:
        """Integrate one step with the hoisted Manager or Euler fallback."""
        try:
            if manager is None:
                raise RuntimeError("OpenSim Manager unavailable")
            manager.setInitialTime(t)
            manager.setFinalTime(t + dt)
            manager.integrate(state)
        except (ValueError, RuntimeError, TypeError):  # noqa: BLE001
            self._integrate_step_euler(coord_set, state, q, qdot, dt)

    def _integrate_step_euler(
        self,
        coord_set: Any,
        state: Any,
        q: np.ndarray,
        qdot: np.ndarray,
        dt: float,
    ) -> None:
        """Manual Euler fallback for joint coordinates."""
        for k in range(len(q)):
            coord = coord_set.get(k)
            try:
                coord.setValue(state, q[k] + qdot[k] * dt)
                coord.setSpeedValue(state, qdot[k])
            except (ValueError, RuntimeError, TypeError):  # noqa: BLE001
                pass

    def _build_sim_result(self, trajectory: _SimulationTrajectory) -> OpenSimSimResult:
        """Pack collected trajectory samples into ``OpenSimSimResult``."""
        t_arr = np.array(trajectory.t)
        qpos_arr = np.array(trajectory.qpos)
        qvel_arr = np.array(trajectory.qvel)
        ee_pos_arr = np.array(trajectory.ee_pos)
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
            kinetic_energy_traj=np.array(trajectory.kinetic_energy),
            potential_energy_traj=np.array(trajectory.potential_energy),
        )
