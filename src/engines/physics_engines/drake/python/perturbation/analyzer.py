"""Drake Perturbation Analyzer — PerturbationAnalyzer protocol for Drake (#1979).

Implements the ``PerturbationAnalyzer`` protocol for Drake's ``MultibodyPlant``
within a ``DiagramBuilder`` / ``Simulator`` framework.  Polynomial torques are
injected via a ``TrajectorySource`` connected to the actuation input port.

Inherits from ``PerturbationAnalyzerBase`` (see #2273) which provides the
shared ``set_base_torque_profile``, ``perturb_torque``, ``extract_metrics``,
``run_batch``, and ``compare_profiles`` implementations.

Design by Contract
------------------
- ``set_base_torque_profile()`` must be called before ``run_batch()`` or
  ``perturb_torque()``.  Raises ``ValueError`` otherwise.
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.shared.python.engine_core.engine_availability import is_engine_available
from src.shared.python.perturbation.perturbation_base import (
    MANDATORY_METRICS,
    ComparisonReport,
    PerturbationAnalyzerBase,
)

logger = logging.getLogger(__name__)

# Re-export so existing imports of MANDATORY_METRICS from this module keep working
__all__ = [
    "DrakePerturbationAnalyzer",
    "DrakeSimResult",
    "ComparisonReport",
    "MANDATORY_METRICS",
]


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
# Main analyzer
# ---------------------------------------------------------------------------


class DrakePerturbationAnalyzer(PerturbationAnalyzerBase):
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
  <transmission name="shoulder_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="shoulder">
      <hardwareInterface>EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="shoulder_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>
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
        super().__init__()

        if not is_engine_available("drake"):
            msg = "pydrake is not installed.  Install it with: pip install drake"
            raise ImportError(msg)

        from pydrake.all import (  # noqa: PLC0415
            AddMultibodyPlantSceneGraph,
            DiagramBuilder,
            Parser,
        )
        from pydrake.multibody.tree import BodyIndex  # noqa: PLC0415

        self._t_end = t_end
        self._dt = dt

        # Build plant
        builder = DiagramBuilder()
        plant, _scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

        if urdf_path is not None:
            urdf_path = Path(urdf_path)
            if not (urdf_path.exists()):
                raise ValueError(f"URDF not found: {urdf_path}")
            Parser(plant).AddModels(str(urdf_path))
        else:
            # Try bundled golfer URDF, fall back to minimal pendulum
            bundled = Path(__file__).parents[4] / "models" / "generated" / "golfer.urdf"
            if bundled.exists():
                Parser(plant).AddModels(str(bundled))
            else:
                import tempfile

                tmp = tempfile.NamedTemporaryFile(
                    suffix=".urdf", mode="w", delete=False
                )
                tmp.write(self._MINIMAL_URDF)
                tmp.close()
                Parser(plant).AddModels(tmp.name)

        plant.Finalize()
        self._plant = plant
        self._builder = builder

        self._nq = plant.num_positions()
        self._nv = plant.num_velocities()
        self._nu = plant.num_actuators()

        # End-effector body (BodyIndex wrapper required by Drake API)
        if ee_body_name is not None:
            self._ee_body_idx = plant.GetBodyByName(ee_body_name).index()
        else:
            # Use the last non-world body
            world_idx = plant.world_body().index()
            bodies = [
                plant.get_body(BodyIndex(i))
                for i in range(plant.num_bodies())
                if BodyIndex(i) != world_idx
            ]
            self._ee_body_idx = bodies[-1].index() if bodies else None

        logger.info(
            "DrakePerturbationAnalyzer: nq=%d, nv=%d, nu=%d, t_end=%.2f",
            self._nq,
            self._nv,
            self._nu,
            self._t_end,
        )

    # ------------------------------------------------------------------
    # Base-class abstract method implementations
    # ------------------------------------------------------------------

    def _get_q_traj(self, sim_result: DrakeSimResult) -> np.ndarray:
        return sim_result.q_traj

    def _get_v_traj(self, sim_result: DrakeSimResult) -> np.ndarray:
        return sim_result.v_traj

    def _validate_sim_result_type(self, sim_result: object) -> None:
        if not isinstance(sim_result, DrakeSimResult):
            raise ValueError(
                f"sim_result must be DrakeSimResult, got {type(sim_result)}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate(self, coeffs: list[list[float]]) -> DrakeSimResult:
        """Run a Drake forward simulation with polynomial torques.

        Each joint's torque follows tau_j(t) = sum_k c_jk * t^k.
        Uses a 4th-order Runge-Kutta (RK4) integration loop to avoid
        Drake Simulator overhead for short Monte Carlo trials while
        maintaining energy stability (fixes issue #2121).
        """
        from pydrake.all import (  # noqa: PLC0415
            AddMultibodyPlantSceneGraph,
            DiagramBuilder,
            Parser,
            Simulator,
        )
        from pydrake.multibody.tree import BodyIndex  # noqa: PLC0415

        # Rebuild a fresh plant+simulator for each trial
        # (Drake Simulator is not reusable after AdvanceTo)
        builder = DiagramBuilder()
        plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

        import tempfile

        if hasattr(self, "_urdf_str"):
            tmp = tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False)
            tmp.write(self._urdf_str)
            tmp.close()
            Parser(plant).AddModels(tmp.name)
        else:
            # Fall back to minimal URDF
            tmp = tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False)
            tmp.write(self._MINIMAL_URDF)
            tmp.close()
            Parser(plant).AddModels(tmp.name)

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
            m_val = plant.CalcMassMatrixViaInverseDynamics(plant_context)
            bias_val = plant.CalcBiasTerm(plant_context)
            gravity_val = plant.CalcGravityGeneralizedForces(plant_context)
            return np.linalg.solve(  # type: ignore[no-any-return]
                m_val, tau_val - bias_val + gravity_val
            )

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
        world_idx = plant.world_body().index()
        non_world_bodies = [
            plant.get_body(BodyIndex(i))
            for i in range(plant.num_bodies())
            if BodyIndex(i) != world_idx
        ]
        ee_pos_list = []
        for qi in q_arr:
            plant.SetPositions(plant_context, qi)
            plant.SetVelocities(plant_context, np.zeros(nv))
            if non_world_bodies:
                pose = plant.EvalBodyPoseInWorld(plant_context, non_world_bodies[-1])
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
