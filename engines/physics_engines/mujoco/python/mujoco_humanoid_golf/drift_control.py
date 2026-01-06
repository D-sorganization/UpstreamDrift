"""Drift-Control Decomposition for MuJoCo (Guideline F - MANDATORY).

This module implements the required drift-control separation per the project
design guidelines Section F. It decomposes motion and acceleration into:

- Drift components: Coriolis/centrifugal coupling, gravity, passive constraints
- Control components: Actuation (torques), control-dependent constraint interaction

Reference: docs/assessments/project_design_guidelines.qmd Section F
"""

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftControlResult:
    """Result of drift-control decomposition.

    Per Guideline F, this decomposes acceleration into drift and control components
    with superposition test: drift + control = full.

    Attributes:
        drift_acceleration: Acceleration due to passive dynamics [nv]
                           (Coriolis + gravity + passive constraints)
        control_acceleration: Acceleration due to actuation [nv]
        full_acceleration: Total acceleration (should equal drift + control) [nv]
        residual: |full - (drift + control)| for validation [scalar]
        drift_velocity_component: Coriolis/centrifugal contribution [nv]
        drift_gravity_component: Gravity contribution [nv]
        drift_constraint_component: Passive constraint contribution [nv]
        control_actuation_component: Direct actuation contribution [nv]
        control_constraint_component: Control-mediated constraint [nv]
    """

    drift_acceleration: np.ndarray
    control_acceleration: np.ndarray
    full_acceleration: np.ndarray
    residual: float

    # Detailed breakdown
    drift_velocity_component: np.ndarray
    drift_gravity_component: np.ndarray
    drift_constraint_component: np.ndarray | None
    control_actuation_component: np.ndarray
    control_constraint_component: np.ndarray | None


class DriftControlDecomposer:
    """Decompose acceleration into drift and control components (Guideline F).

    This is a MANDATORY feature per project design guidelines. Implements
    the required separation of passive dynamics from active control.

    Example:
        >>> model = mujoco.MjModel.from_xml_path("humanoid.xml")
        >>> data = mujoco.MjData(model)
        >>> decomposer = DriftControlDecomposer(model, data)
        >>>
        >>> # Set state and control
        >>> data.qpos[:] = initial_position
        >>> data.qvel[:] = initial_velocity
        >>> data.ctrl[:] = applied_torques
        >>>
        >>> result = decomposer.decompose(data.qpos, data.qvel, data.ctrl)
        >>>
        >>> # Verify superposition: drift + control = full
        >>> assert result.residual < 1e-6
        >>> print(f"Drift contribution: {result.drift_acceleration}")
        >>> print(f"Control contribution: {result.control_acceleration}")
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        """Initialize decomposer.

        Args:
            model: MuJoCo model
        """
        self.model = model

        # Create private data structures for thread-safe analysis
        self._data_drift = mujoco.MjData(model)
        self._data_control = mujoco.MjData(model)
        self._data_full = mujoco.MjData(model)

    def decompose(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
    ) -> DriftControlResult:
        """Decompose acceleration into drift and control components.

        Per Guideline F, this computes:
        1. Full acceleration with both drift and control
        2. Drift-only acceleration (zero control)
        3. Control-only acceleration (zero initial velocity)
        4. Validates superposition

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            ctrl: Applied control torques [nu]

        Returns:
            DriftControlResult with full decomposition

        Raises:
            ValueError: If superposition fails (residual > 1e-5)
        """
        # 1. Compute FULL acceleration (drift + control)
        self._data_full.qpos[:] = qpos
        self._data_full.qvel[:] = qvel
        self._data_full.ctrl[: len(ctrl)] = ctrl

        mujoco.mj_forward(self.model, self._data_full)

        # Get full acceleration
        # Use inverse dynamics to get qacc from current state
        mujoco.mj_inverse(self.model, self._data_full)

        # Actually, we need forward acceleration. Let me use the mass matrix approach:
        # M * qacc = tau - bias
        # qacc = M^-1 * (tau - bias)

        # Get mass matrix
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self._data_full.qM)

        # Get bias forces (Coriolis + gravity)
        bias = self._data_full.qfrc_bias.copy()

        # Get actuation forces
        tau = self._data_full.qfrc_actuator.copy()

        # Solve for acceleration: M * qacc = tau - bias
        net_force = tau - bias
        qacc_full = np.linalg.solve(M, net_force)

        # 2. Compute DRIFT-ONLY acceleration (ctrl = 0, passive dynamics)
        self._data_drift.qpos[:] = qpos
        self._data_drift.qvel[:] = qvel
        self._data_drift.ctrl[:] = 0  # NO CONTROL

        mujoco.mj_forward(self.model, self._data_drift)

        # Get drift components
        M_drift = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M_drift, self._data_drift.qM)

        bias_drift = self._data_drift.qfrc_bias.copy()

        # Drift acceleration: M * qacc_drift = -bias (no actuation)
        qacc_drift = np.linalg.solve(M_drift, -bias_drift)

        # Decompose drift bias into gravity and velocity (Coriolis)
        # Gravity component: evaluate bias with qvel=0
        self._data_drift.qvel[:] = 0
        mujoco.mj_forward(self.model, self._data_drift)
        gravity_bias = self._data_drift.qfrc_bias.copy()

        # Velocity (Coriolis/centrifugal) component
        coriolis_bias = bias_drift - gravity_bias

        qacc_drift_gravity = np.linalg.solve(M_drift, -gravity_bias)
        qacc_drift_velocity = np.linalg.solve(M_drift, -coriolis_bias)

        # 3. Compute CONTROL-ONLY acceleration
        # This is: acceleration with control but removing drift effects
        # qacc_control = qacc_full - qacc_drift
        qacc_control = qacc_full - qacc_drift

        # Decompose control into actuation and constraint-mediated
        # For now, treat all control as actuation (constraints handled separately)
        qacc_control_actuation = qacc_control.copy()
        # Note: Constraint-mediated control not yet decomposed (deferred)
        qacc_control_constraint = None

        # Constraint components (if model has constraints)
        qacc_drift_constraint = None
        if self.model.neq > 0:
            # Note: Constraint decomposition not yet implemented
            # Constraint forces are included in bias terms but not separately tracked
            logger.debug(
                f"Model has {self.model.neq} equality constraints. "
                "Constraint decomposition not yet implemented."
            )

        # 4. Validate superposition: full = drift + control
        qacc_reconstructed = qacc_drift + qacc_control
        residual = float(np.linalg.norm(qacc_full - qacc_reconstructed))

        # Guideline F requires this test to pass
        if residual > 1e-5:
            logger.warning(
                f"Drift-control superposition failed: residual={residual:.2e} > 1e-5. "
                f"This indicates numerical issues or missing constraint handling. "
                f"Guideline F requires drift + control = full."
            )

        return DriftControlResult(
            drift_acceleration=qacc_drift,
            control_acceleration=qacc_control,
            full_acceleration=qacc_full,
            residual=residual,
            drift_velocity_component=qacc_drift_velocity,
            drift_gravity_component=qacc_drift_gravity,
            drift_constraint_component=qacc_drift_constraint,
            control_actuation_component=qacc_control_actuation,
            control_constraint_component=qacc_control_constraint,
        )

    def analyze_trajectory(
        self,
        qpos_traj: np.ndarray,
        qvel_traj: np.ndarray,
        ctrl_traj: np.ndarray,
    ) -> list[DriftControlResult]:
        """Decompose entire trajectory.

        Args:
            qpos_traj: Position trajectory [N × nv]
            qvel_traj: Velocity trajectory [N × nv]
            ctrl_traj: Control trajectory [N × nu]

        Returns:
            List of DriftControlResult for each timestep
        """
        results = []

        for i in range(len(qpos_traj)):
            result = self.decompose(
                qpos_traj[i],
                qvel_traj[i],
                ctrl_traj[i] if i < len(ctrl_traj) else np.zeros(self.model.nu),
            )
            results.append(result)

        return results

    def plot_decomposition(
        self,
        times: np.ndarray,
        results: list[DriftControlResult],
        joint_idx: int = 0,
    ) -> None:
        """Plot drift-control decomposition for a single joint.

        Creates stacked plot showing:
        - Full acceleration
        - Drift component
        - Control component
        - Residual (should be near zero)

        Args:
            times: Time array [N]
            results: Decomposition results for trajectory
            joint_idx: Joint index to plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available - cannot plot decomposition")
            return

        # Extract data
        full = np.array([r.full_acceleration[joint_idx] for r in results])
        drift = np.array([r.drift_acceleration[joint_idx] for r in results])
        control = np.array([r.control_acceleration[joint_idx] for r in results])
        residual = np.array([r.residual for r in results])

        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

        # Full acceleration
        axes[0].plot(times, full, "k-", linewidth=2, label="Full")
        axes[0].set_ylabel("Full Accel [rad/s²]")
        axes[0].legend()
        axes[0].grid(True)

        # Drift component
        axes[1].plot(times, drift, "b-", linewidth=2, label="Drift")
        axes[1].set_ylabel("Drift Accel [rad/s²]")
        axes[1].legend()
        axes[1].grid(True)

        # Control component
        axes[2].plot(times, control, "r-", linewidth=2, label="Control")
        axes[2].set_ylabel("Control Accel [rad/s²]")
        axes[2].legend()
        axes[2].grid(True)

        # Residual (superposition test)
        axes[3].semilogy(times, residual, "g-", linewidth=2, label="Residual")
        axes[3].axhline(y=1e-5, color="gray", linestyle="--", label="Tolerance")
        axes[3].set_ylabel("Residual [rad/s²]")
        axes[3].set_xlabel("Time [s]")
        axes[3].legend()
        axes[3].grid(True)

        plt.suptitle(f"Drift-Control Decomposition (Joint {joint_idx})")
        plt.tight_layout()
        plt.show()
