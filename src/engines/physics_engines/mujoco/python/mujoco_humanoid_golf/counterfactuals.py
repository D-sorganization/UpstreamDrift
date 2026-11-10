"""Counterfactual analysis (ZTCF & ZVCF) for MuJoCo (Guideline G - MANDATORY).

This module implements the required counterfactual experiments per project design
guidelines Section G:

- ZTCF (Zero-Torque Counterfactual): What happens with no actuation?
- ZVCF (Zero-Velocity Counterfactual): the control component M(q)^-1 tau
  (torques on, velocity and gravity off); ZTCF is the complementary drift.

These are MANDATORY features for causal interpretation of golf swing dynamics.

Reference: docs/assessments/project_design_guidelines.qmd Section G
"""

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from src.shared.python.engine_core.counterfactual_contract import (
    COUNTERFACTUAL_DEFINITION_VERSION,
)
from src.shared.python.engine_core.mujoco_compat import full_mass_matrix
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CounterfactualResult:
    """Result of counterfactual experiment.

    Per Guideline G, this captures the difference between observed motion
    and counterfactual motion to infer causal effects.

    Attributes:
        type: Counterfactual type ('ztcf', 'zvcf' or 'zero_velocity_zero_torque')
        observed_acceleration: Actual acceleration [nv]
        counterfactual_acceleration: Acceleration under counterfactual [nv]
        delta_acceleration: Difference (causal attribution) [nv]
        observed_position: Actual position at next timestep [nv]
        counterfactual_position: Position under counterfactual [nv]
        delta_position: Spatial consequence of intervention [nv]
        torque_attributed_effect: Effect attributed to torques (ZTCF delta, ZVCF control) [nv]
        velocity_attributed_effect: Velocity effect (zero_velocity_zero_torque) [nv]
    """

    type: Literal["ztcf", "zvcf", "zero_velocity_zero_torque"]
    observed_acceleration: np.ndarray
    counterfactual_acceleration: np.ndarray
    delta_acceleration: np.ndarray
    observed_position: np.ndarray | None
    counterfactual_position: np.ndarray | None
    delta_position: np.ndarray | None
    torque_attributed_effect: np.ndarray | None
    velocity_attributed_effect: np.ndarray | None
    definition_version: str = COUNTERFACTUAL_DEFINITION_VERSION


class CounterfactualAnalyzer:
    """Perform ZTCF and ZVCF counterfactual experiments (Guideline G).

    This is a MANDATORY feature per project design guidelines Section G.
    Enables causal interpretation by answering:
    - ZTCF: "What motion is due to actuation vs passive dynamics?"
    - ZVCF: "What motion is the control component M(q)^-1 tau, with velocity
      and gravity off?" (ZTCF is the complementary drift; #14495)

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
        >>> result = analyzer.zvcf(qpos, qvel, ctrl)
        >>>
        >>> # The counterfactual is the control component M^-1 tau
        >>> print(f"Control component: {result.counterfactual_acceleration}")
        >>> # Delta is the drift (observed minus control)
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        """Initialize counterfactual analyzer.

        Args:
            model: MuJoCo model
        """
        if model is None:
            raise ValueError("model must be provided")
        self.model = model

        # Thread-safe data structures
        self._data_observed = mujoco.MjData(model)
        self._data_counterfactual = mujoco.MjData(model)

    def _forward_acceleration(
        self,
        data: mujoco.MjData,
        qpos: np.ndarray,
        qvel: np.ndarray | float,
        ctrl: np.ndarray | float,
        *,
        include_actuation: bool,
    ) -> np.ndarray:
        """Load ``(qpos, qvel, ctrl)`` into ``data`` and solve M qacc = f.

        ``f`` is ``qfrc_actuator - qfrc_bias`` when ``include_actuation`` and
        ``-qfrc_bias`` otherwise.
        """
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, data)
        mass = full_mass_matrix(mujoco, self.model, data)
        force = -data.qfrc_bias
        if include_actuation:
            force = data.qfrc_actuator - data.qfrc_bias
        return np.linalg.solve(mass, force)

    def _validated_observed_acceleration(
        self, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray
    ) -> np.ndarray:
        """Validate the observed state and return its acceleration.

        Observed acceleration solves ``M * qacc = tau - bias``.

        Preconditions:
            ``qpos`` is not None and ``len(ctrl) == model.nu``.

        Raises:
            ValueError: If a precondition is violated.
        """
        if qpos is None:
            raise ValueError("qpos must be provided")
        if len(ctrl) != self.model.nu:
            raise ValueError(f"ctrl must have length {self.model.nu}, got {len(ctrl)}")
        return self._forward_acceleration(
            self._data_observed, qpos, qvel, ctrl, include_actuation=True
        )

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
        qacc_observed = self._validated_observed_acceleration(qpos, qvel, ctrl)

        # 2. Compute COUNTERFACTUAL acceleration (zero control)
        # Counterfactual acceleration: M * qacc = -bias (ZERO TORQUE)
        qacc_counterfactual = self._forward_acceleration(
            self._data_counterfactual, qpos, qvel, 0, include_actuation=False
        )

        # 3. Compute DELTA (causal attribution to torques)
        delta_qacc = qacc_observed - qacc_counterfactual

        # 4. Optionally compute position deltas via forward integration
        qpos_obs = None
        qpos_cf = None
        delta_qpos = None

        if compute_trajectories:
            # Simple Euler integration for one step
            qpos_obs = qpos.copy()
            mujoco.mj_integratePos(
                self.model, qpos_obs, qvel + 0.5 * qacc_observed * dt, dt
            )
            qpos_cf = qpos.copy()
            mujoco.mj_integratePos(
                self.model, qpos_cf, qvel + 0.5 * qacc_counterfactual * dt, dt
            )
            delta_qpos = np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(self.model, delta_qpos, 1.0, qpos_cf, qpos_obs)

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

    def control_component_at(self, qpos: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
        """SPEC §4 ZVCF control acceleration ``M(q)⁻¹ τ`` at ``q`` (#14943)."""
        if qpos is None or ctrl is None:
            raise ValueError("qpos and ctrl must be provided")
        return self._control_acceleration(
            self._data_counterfactual, qpos, np.asarray(ctrl, dtype=float)
        )

    def _control_acceleration(
        self, data: mujoco.MjData, qpos: np.ndarray, ctrl: np.ndarray
    ) -> np.ndarray:
        """Return ``M(q)^-1 tau``: torques on, velocity and gravity off."""
        data.qpos[:] = qpos
        data.qvel[:] = 0
        data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, data)
        mass = full_mass_matrix(mujoco, self.model, data)
        return np.linalg.solve(mass, data.qfrc_actuator)

    def zvcf(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
        dt: float = 0.01,
        compute_trajectories: bool = False,
    ) -> CounterfactualResult:
        """Zero-Velocity Counterfactual (ZVCF) - Guideline G2.

        ZVCF is the *control component* of the acceleration at a fixed state
        (owner decision, #14495): the applied torques with velocity and gravity
        turned off, ``qddot_zvcf = M(q)^-1 tau``. It matches
        ``simulation_backends.ztcf_zvcf.zvcf_acceleration``. The complementary
        ZTCF is the drift, so ``ztcf + zvcf`` is the observed acceleration with
        gravity counted once. The identity is pointwise only; integrated
        counterfactual trajectories do not generally sum.

        Here ``counterfactual_acceleration`` is the ZVCF control component,
        ``delta_acceleration`` is observed minus ZVCF (the drift), and
        ``torque_attributed_effect`` carries the control component.

        The former gravity-only behaviour is
        :meth:`zero_velocity_zero_torque`.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv] (observed state only)
            ctrl: Applied control torques [nu]
            dt: Timestep for trajectory prediction [s]
            compute_trajectories: If True, integrate forward one step

        Returns:
            CounterfactualResult with the control component
        """
        qacc_observed = self._validated_observed_acceleration(qpos, qvel, ctrl)
        qacc_counterfactual = self._control_acceleration(
            self._data_counterfactual, qpos, ctrl
        )
        delta_qacc = qacc_observed - qacc_counterfactual

        qpos_obs = None
        qpos_cf = None
        delta_qpos = None

        if compute_trajectories:
            qpos_obs = qpos.copy()
            mujoco.mj_integratePos(
                self.model, qpos_obs, qvel + 0.5 * qacc_observed * dt, dt
            )
            # Counterfactual starts from rest and is driven by torques only.
            qpos_cf = qpos.copy()
            mujoco.mj_integratePos(
                self.model, qpos_cf, 0.5 * qacc_counterfactual * dt, dt
            )
            delta_qpos = np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(self.model, delta_qpos, 1.0, qpos_cf, qpos_obs)

        return CounterfactualResult(
            type="zvcf",
            observed_acceleration=qacc_observed,
            counterfactual_acceleration=qacc_counterfactual,
            delta_acceleration=delta_qacc,
            observed_position=qpos_obs,
            counterfactual_position=qpos_cf,
            delta_position=delta_qpos,
            torque_attributed_effect=qacc_counterfactual,
            velocity_attributed_effect=None,
        )

    def zero_velocity_zero_torque(
        self, qpos: np.ndarray, qvel: np.ndarray
    ) -> CounterfactualResult:
        """Gravity-only comparison: the former ``zvcf()`` behaviour (#11553).

        Compares the observed unforced acceleration at ``(qpos, qvel)`` with
        the acceleration at ``qpos`` with velocity and control both zero. The
        difference is the velocity-dependent (Coriolis/centrifugal) effect.
        """
        if qpos is None:
            raise ValueError("qpos must be provided")
        qacc_observed = self._forward_acceleration(
            self._data_observed, qpos, qvel, 0, include_actuation=False
        )
        qacc_counterfactual = self._forward_acceleration(
            self._data_counterfactual, qpos, 0, 0, include_actuation=False
        )
        delta_qacc = qacc_observed - qacc_counterfactual
        return CounterfactualResult(
            type="zero_velocity_zero_torque",
            observed_acceleration=qacc_observed,
            counterfactual_acceleration=qacc_counterfactual,
            delta_acceleration=delta_qacc,
            observed_position=None,
            counterfactual_position=None,
            delta_position=None,
            torque_attributed_effect=None,
            velocity_attributed_effect=delta_qacc,
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
        if qpos_traj is None:
            raise ValueError("qpos_traj must be provided")
        results = []

        for i in range(len(qpos_traj)):
            result = self.ztcf(qpos_traj[i], qvel_traj[i], ctrl_traj[i])
            results.append(result)

        return results

    def analyze_trajectory_zvcf(
        self,
        qpos_traj: np.ndarray,
        qvel_traj: np.ndarray,
        ctrl_traj: np.ndarray,
    ) -> list[CounterfactualResult]:
        """Perform ZVCF analysis on entire trajectory.

        Args:
            qpos_traj: Position trajectory [N × nv]
            qvel_traj: Velocity trajectory [N × nv]
            ctrl_traj: Control trajectory [N × nu]

        Returns:
            List of CounterfactualResult (ZVCF) for each timestep
        """
        if qpos_traj is None:
            raise ValueError("qpos_traj must be provided")
        results = []

        for i in range(len(qpos_traj)):
            result = self.zvcf(qpos_traj[i], qvel_traj[i], ctrl_traj[i])
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
        if times is None:
            raise ValueError("times must be provided")
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
        )  # noqa: E501
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
            else "ZVCF: Control Component"
        )  # noqa: E501
        plt.suptitle(f"{title} (Joint {joint_idx})")
        plt.tight_layout()
        plt.show()
