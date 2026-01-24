"""Counterfactual analysis (ZTCF & ZVCF) for MuJoCo (Guideline G - MANDATORY).

This module implements the required counterfactual experiments per project design
guidelines Section G:

- ZTCF (Zero-Torque Counterfactual): What happens with no actuation?
- ZVCF (Zero-Velocity Counterfactual): What happens with no momentum?

These are MANDATORY features for causal interpretation of golf swing dynamics.

Reference: docs/assessments/project_design_guidelines.qmd Section G
"""

from src.shared.python.logging_config import get_logger
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

logger = get_logger(__name__)


@dataclass
class CounterfactualResult:
    """Result of counterfactual experiment.

    Per Guideline G, this captures the difference between observed motion
    and counterfactual motion to infer causal effects.

    Attributes:
        type: Counterfactual type ('ztcf' or 'zvcf')
        observed_acceleration: Actual acceleration [nv]
        counterfactual_acceleration: Acceleration under counterfactual [nv]
        delta_acceleration: Difference (causal attribution) [nv]
        observed_position: Actual position at next timestep [nv]
        counterfactual_position: Position under counterfactual [nv]
        delta_position: Spatial consequence of intervention [nv]
        torque_attributed_effect: Effect attributed to torques (ZTCF) [nv]
        velocity_attributed_effect: Effect attributed to momentum (ZVCF) [nv]
    """

    type: Literal["ztcf", "zvcf"]
    observed_acceleration: np.ndarray
    counterfactual_acceleration: np.ndarray
    delta_acceleration: np.ndarray
    observed_position: np.ndarray | None
    counterfactual_position: np.ndarray | None
    delta_position: np.ndarray | None
    torque_attributed_effect: np.ndarray | None
    velocity_attributed_effect: np.ndarray | None


class CounterfactualAnalyzer:
    """Perform ZTCF and ZVCF counterfactual experiments (Guideline G).

    This is a MANDATORY feature per project design guidelines Section G.
    Enables causal interpretation by answering:
    - ZTCF: "What motion is due to actuation vs passive dynamics?"
    - ZVCF: "What motion is due to momentum vs gravitational/constraint forces?"

    Example (ZTCF):
        >>> model = mujoco.MjModel.from_xml_path("humanoid.xml")
        >>> analyzer = CounterfactualAnalyzer(model)
        >>>
        >>> # Observed state with control
        >>> qpos = np.array([...])
        >>> qvel = np.array([...])
        >>> ctrl = np.array([...])  # Applied torques
        >>>
        >>> result = analyzer.ztcf(qpos, qvel, ctrl)
        >>>
        >>> # Delta shows what torques contributed
        >>> print(f"Torque attribution: {result.delta_acceleration}")
        >>> # Positive delta means torques accelerated motion
        >>> # Negative delta means torques opposed motion

    Example (ZVCF):
        >>> result = analyzer.zvcf(qpos, qvel)
        >>>
        >>> # Delta shows what momentum contributed
        >>> print(f"Momentum attribution: {result.delta_acceleration}")
        >>> # Shows Coriolis/centrifugal effects
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        """Initialize counterfactual analyzer.

        Args:
            model: MuJoCo model
        """
        self.model = model

        # Thread-safe data structures
        self._data_observed = mujoco.MjData(model)
        self._data_counterfactual = mujoco.MjData(model)

    def ztcf(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
        dt: float = 0.01,
        compute_trajectories: bool = False,
    ) -> CounterfactualResult:
        """Zero-Torque Counterfactual (ZTCF) - Guideline G1.

        Compares observed motion (with control) to counterfactual motion
        (with zero control). The difference reveals the causal effect of
        actuation.

        Per Guideline G1:
        "Zero applied torques while preserving state. Simulate passive
        evolution under drift/constraints. Compute delta vs observed motion
        and infer torque-attributed effects."

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            ctrl: Applied control torques [nu]
            dt: Timestep for trajectory prediction [s]
            compute_trajectories: If True, integrate forward one step

        Returns:
            CounterfactualResult with torque attribution
        """
        # 1. Compute OBSERVED acceleration (with control)
        self._data_observed.qpos[:] = qpos
        self._data_observed.qvel[:] = qvel
        self._data_observed.ctrl[: len(ctrl)] = ctrl

        mujoco.mj_forward(self.model, self._data_observed)

        # Get mass matrix and forces
        M_obs = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M_obs, self._data_observed.qM)

        bias_obs = self._data_observed.qfrc_bias.copy()
        tau_obs = self._data_observed.qfrc_actuator.copy()

        # Observed acceleration: M * qacc = tau - bias
        qacc_observed = np.linalg.solve(M_obs, tau_obs - bias_obs)

        # 2. Compute COUNTERFACTUAL acceleration (zero control)
        self._data_counterfactual.qpos[:] = qpos
        self._data_counterfactual.qvel[:] = qvel
        self._data_counterfactual.ctrl[:] = 0  # ZERO TORQUE

        mujoco.mj_forward(self.model, self._data_counterfactual)

        M_cf = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M_cf, self._data_counterfactual.qM)

        bias_cf = self._data_counterfactual.qfrc_bias.copy()

        # Counterfactual acceleration: M * qacc = -bias (no actuation)
        qacc_counterfactual = np.linalg.solve(M_cf, -bias_cf)

        # 3. Compute DELTA (causal attribution to torques)
        delta_qacc = qacc_observed - qacc_counterfactual

        # 4. Optionally compute position deltas via forward integration
        qpos_obs = None
        qpos_cf = None
        delta_qpos = None

        if compute_trajectories:
            # Simple Euler integration for one step
            qpos_obs = qpos + qvel * dt + 0.5 * qacc_observed * dt**2
            qpos_cf = qpos + qvel * dt + 0.5 * qacc_counterfactual * dt**2
            delta_qpos = qpos_obs - qpos_cf

        return CounterfactualResult(
            type="ztcf",
            observed_acceleration=qacc_observed,
            counterfactual_acceleration=qacc_counterfactual,
            delta_acceleration=delta_qacc,
            observed_position=qpos_obs,
            counterfactual_position=qpos_cf,
            delta_position=delta_qpos,
            torque_attributed_effect=delta_qacc,  # ZTCF attributes to torques
            velocity_attributed_effect=None,
        )

    def zvcf(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        dt: float = 0.01,
        compute_trajectories: bool = False,
    ) -> CounterfactualResult:
        """Zero-Velocity Counterfactual (ZVCF) - Guideline G2.

        Compares observed motion (with momentum) to counterfactual motion
        (from rest). The difference reveals the causal effect of velocity-
        dependent forces (Coriolis/centrifugal).

        Per Guideline G2:
        "Zero joint velocities while preserving configuration. Isolate
        acceleration/constraint/gravity-driven motion from momentum effects."

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            dt: Timestep for trajectory prediction [s]
            compute_trajectories: If True, integrate forward one step

        Returns:
            CounterfactualResult with velocity attribution
        """
        # 1. Compute OBSERVED acceleration (with velocity)
        self._data_observed.qpos[:] = qpos
        self._data_observed.qvel[:] = qvel
        self._data_observed.ctrl[:] = 0  # No control for clean comparison

        mujoco.mj_forward(self.model, self._data_observed)

        M_obs = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M_obs, self._data_observed.qM)

        bias_obs = self._data_observed.qfrc_bias.copy()

        qacc_observed = np.linalg.solve(M_obs, -bias_obs)

        # 2. Compute COUNTERFACTUAL acceleration (zero velocity)
        self._data_counterfactual.qpos[:] = qpos
        self._data_counterfactual.qvel[:] = 0  # ZERO VELOCITY
        self._data_counterfactual.ctrl[:] = 0

        mujoco.mj_forward(self.model, self._data_counterfactual)

        M_cf = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M_cf, self._data_counterfactual.qM)

        bias_cf = self._data_counterfactual.qfrc_bias.copy()

        qacc_counterfactual = np.linalg.solve(M_cf, -bias_cf)

        # 3. Compute DELTA (causal attribution to velocity)
        delta_qacc = qacc_observed - qacc_counterfactual

        # 4. Optionally compute position deltas
        qpos_obs = None
        qpos_cf = None
        delta_qpos = None

        if compute_trajectories:
            # Observed: continues with momentum
            qpos_obs = qpos + qvel * dt + 0.5 * qacc_observed * dt**2

            # Counterfactual: starts from rest
            # (zero initial velocity, only driven by gravity/constraints)
            qpos_cf = qpos + 0.5 * qacc_counterfactual * dt**2

            delta_qpos = qpos_obs - qpos_cf

        return CounterfactualResult(
            type="zvcf",
            observed_acceleration=qacc_observed,
            counterfactual_acceleration=qacc_counterfactual,
            delta_acceleration=delta_qacc,
            observed_position=qpos_obs,
            counterfactual_position=qpos_cf,
            delta_position=delta_qpos,
            torque_attributed_effect=None,
            velocity_attributed_effect=delta_qacc,  # ZVCF attributes to velocity
        )

    def analyze_trajectory_ztcf(
        self,
        qpos_traj: np.ndarray,
        qvel_traj: np.ndarray,
        ctrl_traj: np.ndarray,
    ) -> list[CounterfactualResult]:
        """Perform ZTCF analysis on entire trajectory.

        Args:
            qpos_traj: Position trajectory [N × nv]
            qvel_traj: Velocity trajectory [N × nv]
            ctrl_traj: Control trajectory [N × nu]

        Returns:
            List of CounterfactualResult (ZTCF) for each timestep
        """
        results = []

        for i in range(len(qpos_traj)):
            result = self.ztcf(qpos_traj[i], qvel_traj[i], ctrl_traj[i])
            results.append(result)

        return results

    def analyze_trajectory_zvcf(
        self,
        qpos_traj: np.ndarray,
        qvel_traj: np.ndarray,
    ) -> list[CounterfactualResult]:
        """Perform ZVCF analysis on entire trajectory.

        Args:
            qpos_traj: Position trajectory [N × nv]
            qvel_traj: Velocity trajectory [N × nv]

        Returns:
            List of CounterfactualResult (ZVCF) for each timestep
        """
        results = []

        for i in range(len(qpos_traj)):
            result = self.zvcf(qpos_traj[i], qvel_traj[i])
            results.append(result)

        return results

    def plot_counterfactual_comparison(
        self,
        times: np.ndarray,
        results: list[CounterfactualResult],
        joint_idx: int = 0,
    ) -> None:
        """Plot counterfactual comparison for a single joint.

        Creates comparison plot showing:
        - Observed acceleration
        - Counterfactual acceleration
        - Delta (causal attribution)

        Args:
            times: Time array [N]
            results: Counterfactual results for trajectory
            joint_idx: Joint index to plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available - cannot plot counterfactual")
            return

        cf_type = results[0].type

        # Extract data
        observed = np.array([r.observed_acceleration[joint_idx] for r in results])
        counterfactual = np.array(
            [r.counterfactual_acceleration[joint_idx] for r in results]
        )
        delta = np.array([r.delta_acceleration[joint_idx] for r in results])

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Observed
        axes[0].plot(times, observed, "k-", linewidth=2, label="Observed")
        axes[0].set_ylabel("Observed Accel [rad/s²]")
        axes[0].legend()
        axes[0].grid(True)

        # Counterfactual
        axes[1].plot(times, counterfactual, "b--", linewidth=2, label="Counterfactual")
        axes[1].set_ylabel("Counterfactual Accel [rad/s²]")
        axes[1].legend()
        axes[1].grid(True)

        # Delta (causal attribution)
        axes[2].plot(times, delta, "r-", linewidth=2, label="Delta (Attribution)")
        axes[2].axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        axes[2].set_ylabel("Delta Accel [rad/s²]")
        axes[2].set_xlabel("Time [s]")
        axes[2].legend()
        axes[2].grid(True)

        title = (
            "ZTCF: Torque Attribution"
            if cf_type == "ztcf"
            else "ZVCF: Velocity Attribution"
        )
        plt.suptitle(f"{title} (Joint {joint_idx})")
        plt.tight_layout()
        plt.show()
