"""Power flow and inter-segment energy transfer (Guideline E3 - Required).

This module implements power flow analysis per project design guidelines Section E3:
"Power transfer between segments (not just system energy). Work decomposition
aligned with drift/control/constraint components."

Reference: docs/assessments/project_design_guidelines.qmd Section E3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    import mujoco

logger = get_logger(__name__)


@dataclass
class PowerFlowResult:
    """Result of power flow analysis for a single timestep.

    Per Guideline E3, tracks power transfer between segments and work
    decomposition into drift/control components.

    Attributes:
        joint_powers: Instantaneous power at each joint [nv] (Watts)
        joint_work_drift: Work from drift components [nv] (Joules)
        joint_work_control: Work from control components [nv] (Joules)
        joint_work_total: Total work [nv] (Joules)
        segment_kinetic_energy: Kinetic energy per segment [nbody] (Joules)
        segment_potential_energy: Potential energy per segment [nbody] (Joules)
        total_mechanical_energy: Sum of KE + PE (Joules)
        power_in: Power input from actuators (Watts)
        power_dissipation: Power dissipated by damping (Watts)
        energy_conservation_residual: |dE/dt - P_in + P_diss| for validation,
            where P_in is the *total* joint power ``tau . qvel`` (not the
            positive-only ``power_in`` field) and dE/dt is derived from the
            equations of motion. ~0 for a self-consistent (qacc, tau) pair.
    """

    joint_powers: np.ndarray
    joint_work_drift: np.ndarray
    joint_work_control: np.ndarray
    joint_work_total: np.ndarray
    segment_kinetic_energy: np.ndarray
    segment_potential_energy: np.ndarray
    total_mechanical_energy: float
    power_in: float
    power_dissipation: float
    energy_conservation_residual: float


@dataclass
class InterSegmentTransfer:
    """Inter-segment power transfer analysis.

    Tracks how power flows from parent to child segments through joints.

    Attributes:
        segment_name: Name of the segment
        parent_name: Name of parent segment (or "world")
        power_from_parent: Power received from parent (Watts)
        power_to_children: Power sent to children (Watts)
        power_generation: Power generated internally (actuation) (Watts)
        power_dissipation: Power dissipated (damping/friction) (Watts)
        net_power_balance: Should equal zero for validation (Watts)
    """

    segment_name: str
    parent_name: str
    power_from_parent: float
    power_to_children: float
    power_generation: float
    power_dissipation: float
    net_power_balance: float


class PowerFlowAnalyzer:
    """Analyze power flow and energy transfer in golf swing (Guideline E3).

    This is a REQUIRED feature per project design guidelines Section E3.
    Implements:
    - Joint-level power (torque × angular velocity)
    - Work decomposition (drift vs control contributions)
    - Inter-segment power transfer
    - Energy conservation validation

    Example:
        >>> model = mujoco.MjModel.from_xml_path("humanoid.xml")
        >>> analyzer = PowerFlowAnalyzer(model)
        >>>
        >>> # Analyze single timestep
        >>> result = analyzer.compute_power_flow(qpos, qvel, qacc, tau, dt=0.01)
        >>> print(f"Joint powers: {result.joint_powers}")
        >>> print(f"Total mechanical energy: {result.total_mechanical_energy}")
        >>>
        >>> # Analyze trajectory
        >>> trajectory_results = analyzer.analyze_trajectory(
        ...     times, qpos_traj, qvel_traj, qacc_traj, tau_traj
        ... )
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        """Initialize power flow analyzer.

        Args:
            model: MuJoCo model
        """
        if model is None:
            raise ValueError("model must be provided")
        self.model = model

        # Thread-safe data structure for computations
        import mujoco

        self._data = mujoco.MjData(model)

        # Cache per-DOF damping/dof-address once, so the per-step power
        # dissipation loop does not rebuild arrays on every call.
        # Only DOFs with positive damping contribute.
        nv = int(model.nv)
        dof_damping = np.asarray(model.dof_damping, dtype=np.float64)
        active = dof_damping > 0
        self._diss_damping = dof_damping[active]
        self._diss_dofadr = np.arange(nv, dtype=np.int64)[active]

    def _compute_work_decomposition(
        self,
        tau: np.ndarray,
        qvel: np.ndarray,
        dt: float,
        tau_drift: np.ndarray | None,
        tau_control: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if tau is None:
            raise ValueError("tau must be provided")
        if tau_drift is not None:
            joint_work_drift = tau_drift * qvel * dt
        else:
            joint_work_drift = np.zeros_like(tau)

        if tau_control is not None:
            joint_work_control = tau_control * qvel * dt
        else:
            joint_work_control = np.zeros_like(tau)

        joint_work_total = tau * qvel * dt
        return joint_work_drift, joint_work_control, joint_work_total

    def _compute_segment_energies(
        self, qvel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-body kinetic and gravitational potential energy.

        Both halves are evaluated at the body centre of mass, and the
        rotational term is contracted in the body's *inertial* frame because
        ``model.body(i).inertia`` is the diagonal principal inertia expressed
        in that frame while ``mj_jacBodyCom`` returns a world-frame angular
        Jacobian. Summed over bodies these reproduce ``mjData.energy``
        (``mj_energyPos`` / ``mj_energyVel``) exactly for gravity-only models.
        """
        if qvel is None:
            raise ValueError("qvel must be provided")
        import mujoco

        segment_ke = np.zeros(self.model.nbody)
        segment_pe = np.zeros(self.model.nbody)

        # Full gravity vector: do not assume it points along -Z.
        gravity = np.asarray(self.model.opt.gravity, dtype=np.float64)

        for i in range(self.model.nbody):
            body = self.model.body(i)

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBodyCom(self.model, self._data, jacp, jacr, i)

            com_vel = jacp @ qvel
            ang_vel_world = jacr @ qvel

            mass = body.mass[0]
            inertia = np.asarray(body.inertia, dtype=np.float64)

            # R maps inertial-frame -> world, so R.T maps world -> inertial.
            rot = np.asarray(self._data.ximat[i], dtype=np.float64).reshape(3, 3)
            ang_vel_body = rot.T @ ang_vel_world

            ke_linear = 0.5 * mass * np.dot(com_vel, com_vel)
            # ⚡ Bolt: np.vdot avoids temporary allocations for element-wise squares
            ke_rotational = 0.5 * np.vdot(inertia, ang_vel_body * ang_vel_body)
            segment_ke[i] = ke_linear + ke_rotational

            # xipos is the body COM; xpos is the body frame origin.
            com_pos_world = np.asarray(self._data.xipos[i], dtype=np.float64)
            segment_pe[i] = -mass * float(np.dot(gravity, com_pos_world))

        return segment_ke, segment_pe

    def _compute_energy_conservation_residual(
        self,
        qvel: np.ndarray,
        qacc: np.ndarray,
        tau: np.ndarray,
        power_dissipation: float,
    ) -> float:
        """Residual ``|dE/dt - P_in + P_diss|`` for the supplied state.

        ``dE/dt`` is obtained from the equations of motion rather than by
        finite-differencing: with ``M(q) qacc + c(q, qvel) = tau_total`` and the
        skew-symmetry identity ``0.5 v' Mdot v = v' C v``, the mechanical energy
        rate is ``qvel . (M qacc + qfrc_bias)``. ``P_in`` is the total joint
        power ``tau . qvel`` of the supplied (actuator) torques, so the residual
        measures how much power the supplied ``qacc`` implies that the supplied
        ``tau`` minus damping does not account for. It is ~0 for a consistent
        ``(qacc, tau)`` pair and grows with any inconsistency, unmodelled
        constraint work, or non-gravitational potential.
        """
        import mujoco

        qm_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, qm_full, self._data.qM)
        de_dt = float(qvel @ (qm_full @ np.asarray(qacc, dtype=np.float64)))
        de_dt += float(qvel @ np.asarray(self._data.qfrc_bias, dtype=np.float64))

        power_joint_total = float(np.dot(tau, qvel))
        return abs(de_dt - power_joint_total + power_dissipation)

    def _compute_power_dissipation(self, qvel: np.ndarray) -> float:
        if qvel is None:
            raise ValueError("qvel must be provided")
        if self._diss_dofadr.size == 0:
            return 0.0
        # Vectorized over the precomputed damped joints; identical to summing
        # damping * qvel[dofadr]^2 over each joint with positive damping.
        v = np.asarray(qvel)[self._diss_dofadr]
        return float(np.dot(self._diss_damping, v * v))

    @precondition(
        lambda self, qpos, qvel, qacc, tau, dt=0.01, **kw: dt > 0,
        "Timestep must be positive",
    )
    @precondition(
        lambda self, qpos, qvel, qacc, tau, **kw: len(qvel) == len(tau),
        "Velocity and torque arrays must have the same length",
    )
    def compute_power_flow(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        tau: np.ndarray,
        dt: float = 0.01,
        tau_drift: np.ndarray | None = None,
        tau_control: np.ndarray | None = None,
    ) -> PowerFlowResult:
        """Compute power flow at a single timestep.

        Per Guideline E3, decomposes work into drift and control components
        and validates energy conservation.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            qacc: Joint accelerations [nv]
            tau: Joint torques [nv]
            dt: Timestep for work calculation [s]
            tau_drift: Drift torque components [nv] (optional)
            tau_control: Control torque components [nv] (optional)

        Returns:
            PowerFlowResult with complete power flow analysis
        """
        if qpos is None:
            raise ValueError("qpos must be provided")
        import mujoco

        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        self._data.qacc[:] = qacc
        mujoco.mj_forward(self.model, self._data)

        joint_powers = tau * qvel

        joint_work_drift, joint_work_control, joint_work_total = (
            self._compute_work_decomposition(tau, qvel, dt, tau_drift, tau_control)
        )  # noqa: E501

        segment_ke, segment_pe = self._compute_segment_energies(qvel)
        total_me = float(segment_ke.sum() + segment_pe.sum())  # ⚡ Bolt: ndarray.sum() is ~2x faster than np.sum() since it skips array conversion checks

        power_in = float(np.sum(np.maximum(joint_powers, 0)))
        power_diss = self._compute_power_dissipation(qvel)
        residual = self._compute_energy_conservation_residual(
            qvel, qacc, tau, float(power_diss)
        )

        return PowerFlowResult(
            joint_powers=joint_powers,
            joint_work_drift=joint_work_drift,
            joint_work_control=joint_work_control,
            joint_work_total=joint_work_total,
            segment_kinetic_energy=segment_ke,
            segment_potential_energy=segment_pe,
            total_mechanical_energy=total_me,
            power_in=power_in,
            power_dissipation=float(power_diss),
            energy_conservation_residual=residual,
        )

    @precondition(
        lambda self, times, qpos_traj, qvel_traj, qacc_traj, tau_traj: len(times) > 0,
        "Time array must be non-empty",
    )
    @precondition(
        lambda self, times, qpos_traj, qvel_traj, qacc_traj, tau_traj: (
            len(times)
            == len(qpos_traj)
            == len(qvel_traj)
            == len(qacc_traj)
            == len(tau_traj)  # noqa: E501
        ),
        "All trajectory arrays must have the same length",
    )
    def analyze_trajectory(
        self,
        times: np.ndarray,
        qpos_traj: np.ndarray,
        qvel_traj: np.ndarray,
        qacc_traj: np.ndarray,
        tau_traj: np.ndarray,
    ) -> list[PowerFlowResult]:
        """Analyze power flow over entire trajectory.

        Args:
            times: Time array [N]
            qpos_traj: Position trajectory [N × nv]
            qvel_traj: Velocity trajectory [N × nv]
            qacc_traj: Acceleration trajectory [N × nv]
            tau_traj: Torque trajectory [N × nv]

        Returns:
            List of PowerFlowResult for each timestep
        """
        if times is None:
            raise ValueError("times must be provided")
        results = []

        for i in range(len(times)):
            dt = times[i] - times[i - 1] if i > 0 else 0.01

            result = self.compute_power_flow(
                qpos_traj[i],
                qvel_traj[i],
                qacc_traj[i],
                tau_traj[i],
                dt=dt,
            )
            results.append(result)

        return results

    def compute_inter_segment_transfer(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        tau: np.ndarray,
    ) -> list[InterSegmentTransfer]:
        """Compute power transfer between segments.

        Per Guideline E3, tracks how power flows from parent to child
        through joints.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            tau: Joint torques [nv]

        Returns:
            List of InterSegmentTransfer for each body
        """
        if qpos is None:
            raise ValueError("qpos must be provided")
        import mujoco

        # Set state
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel

        mujoco.mj_forward(self.model, self._data)

        transfers = []

        for i in range(self.model.nbody):
            body = self.model.body(i)
            parent_id = body.parentid[0]
            parent_name = "world" if parent_id == 0 else self.model.body(parent_id).name

            power_from_parent, power_generation = self._compute_body_joint_power(
                i, tau, qvel
            )  # noqa: E501
            power_to_children = self._compute_child_joint_power(i, tau, qvel)
            power_diss = self._compute_joint_dissipation(i, qvel)

            net_balance = (
                power_from_parent - power_to_children - power_generation + power_diss
            )  # noqa: E501

            transfers.append(
                InterSegmentTransfer(
                    segment_name=body.name,
                    parent_name=parent_name,
                    power_from_parent=power_from_parent,
                    power_to_children=power_to_children,
                    power_generation=power_generation,
                    power_dissipation=power_diss,
                    net_power_balance=net_balance,
                )
            )

        return transfers

    def _compute_body_joint_power(
        self, body_id: int, tau: np.ndarray, qvel: np.ndarray
    ) -> tuple[float, float]:
        """Compute power from parent and generation at joints owned by a body.

        Args:
            body_id: MuJoCo body index.
            tau: Joint torques [nv].
            qvel: Joint velocities [nv].

        Returns:
            Tuple of (power_from_parent, power_generation).
        """
        if body_id is None:
            raise ValueError("body_id must be provided")
        power_from_parent = 0.0
        power_generation = 0.0
        for i in range(self.model.nv):
            j = self.model.dof_jntid[i]
            if self.model.jnt_bodyid[j] == body_id:
                joint_power = tau[i] * qvel[i]
                power_from_parent += joint_power
                if abs(tau[i]) > 1e-6:
                    power_generation += joint_power
        return power_from_parent, power_generation

    def _compute_child_joint_power(
        self, body_id: int, tau: np.ndarray, qvel: np.ndarray
    ) -> float:  # noqa: E501
        """Compute total power transferred to child bodies through their joints.

        Args:
            body_id: MuJoCo body index.
            tau: Joint torques [nv].
            qvel: Joint velocities [nv].

        Returns:
            Total power flowing to children.
        """
        if body_id is None:
            raise ValueError("body_id must be provided")
        power_to_children = 0.0
        for i in range(self.model.nv):
            j = self.model.dof_jntid[i]
            child_body_id = self.model.jnt_bodyid[j]
            if child_body_id > 0:
                child_parent_id = self.model.body_parentid[child_body_id]
                if child_parent_id == body_id:
                    power_to_children += tau[i] * qvel[i]
        return power_to_children

    def _compute_joint_dissipation(self, body_id: int, qvel: np.ndarray) -> float:
        """Compute power dissipation from joint damping for a body.

        Args:
            body_id: MuJoCo body index.
            qvel: Joint velocities [nv].

        Returns:
            Total dissipated power at this body's joints.
        """
        if body_id is None:
            raise ValueError("body_id must be provided")
        power_diss = 0.0
        for i in range(self.model.nv):
            j = self.model.dof_jntid[i]
            if self.model.jnt_bodyid[j] == body_id and self.model.dof_damping[i] > 0:
                power_diss += self.model.dof_damping[i] * (qvel[i] * qvel[i])
        return power_diss

    @precondition(
        lambda self, times, results, joint_idx=0: joint_idx >= 0,
        "Joint index must be non-negative",
    )
    @precondition(
        lambda self, times, results, joint_idx=0: len(results) > 0,
        "Results list must be non-empty",
    )
    def plot_power_flow(
        self,
        times: np.ndarray,
        results: list[PowerFlowResult],
        joint_idx: int = 0,
    ) -> None:
        """Plot power flow analysis for a single joint.

        Args:
            times: Time array [N]
            results: Power flow results for trajectory
            joint_idx: Joint index to plot
        """
        if times is None:
            raise ValueError("times must be provided")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available - cannot plot power flow")
            return

        # Extract data
        power = np.array([r.joint_powers[joint_idx] for r in results])
        work_total = np.cumsum([r.joint_work_total[joint_idx] for r in results])
        total_energy = np.array([r.total_mechanical_energy for r in results])

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Instantaneous power
        axes[0].plot(times, power, "b-", linewidth=2)
        axes[0].axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        axes[0].fill_between(
            times,
            0,
            power,
            where=(power >= 0),
            alpha=0.3,
            color="green",
            label="Positive (generation)",
        )
        axes[0].fill_between(
            times,
            0,
            power,
            where=(power < 0),
            alpha=0.3,
            color="red",
            label="Negative (absorption)",
        )
        axes[0].set_ylabel("Power [W]")
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_title(f"Joint {joint_idx} Power Flow")

        # Cumulative work
        axes[1].plot(times, work_total, "g-", linewidth=2)
        axes[1].set_ylabel("Cumulative Work [J]")
        axes[1].grid(True)

        # Total mechanical energy
        axes[2].plot(times, total_energy, "k-", linewidth=2)
        axes[2].set_ylabel("Total ME [J]")
        axes[2].set_xlabel("Time [s]")
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()
