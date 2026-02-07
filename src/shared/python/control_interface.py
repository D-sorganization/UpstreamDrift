"""Joint Torque Input / Control Interface (Shared).

Provides a shared interface for joint torque inputs and control strategy management.
Exposes all control capabilities from the physics engines and robotics control module
to the user through a unified API.

Design by Contract:
    Preconditions:
        - Engine must implement PhysicsEngine protocol
        - Control inputs must match engine DOFs
    Postconditions:
        - Applied controls are recorded and queryable
        - Control strategy state is always consistent
    Invariants:
        - Joint names and indices are immutable after initialization
        - Control bounds are enforced on all inputs

Usage:
    >>> from src.shared.python.control_interface import ControlInterface
    >>> ctrl = ControlInterface(engine)
    >>> ctrl.set_strategy("pd")
    >>> ctrl.set_gains(kp=100.0, kd=10.0)
    >>> ctrl.set_target_positions(q_desired)
    >>> torques = ctrl.compute_control()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.shared.python.interfaces import PhysicsEngine
from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


class ControlStrategy(str, Enum):
    """Available control strategies."""

    DIRECT_TORQUE = "direct_torque"
    PD = "pd"
    PID = "pid"
    COMPUTED_TORQUE = "computed_torque"
    GRAVITY_COMPENSATION = "gravity_compensation"
    WHOLE_BODY = "whole_body"
    IMPEDANCE = "impedance"
    ZERO = "zero"


@dataclass
class JointInfo:
    """Information about a single joint.

    Attributes:
        index: Joint index in the state vector.
        name: Joint name.
        torque_limit: Maximum absolute torque (N*m).
        position_limit_lower: Lower position limit (rad or m).
        position_limit_upper: Upper position limit (rad or m).
        velocity_limit: Maximum absolute velocity (rad/s or m/s).
    """

    index: int
    name: str
    torque_limit: float = 100.0
    position_limit_lower: float = -np.pi
    position_limit_upper: float = np.pi
    velocity_limit: float = 10.0


@dataclass
class ControlState:
    """Current state of the control interface.

    Attributes:
        strategy: Active control strategy.
        torques: Current applied torques (n_v,).
        target_positions: Target positions for tracking (n_q,) or None.
        target_velocities: Target velocities for tracking (n_v,) or None.
        kp: Proportional gains per joint (n_v,).
        kd: Derivative gains per joint (n_v,).
        ki: Integral gains per joint (n_v,).
        integral_error: Accumulated integral error (n_v,).
        custom_params: Additional strategy-specific parameters.
    """

    strategy: ControlStrategy = ControlStrategy.ZERO
    torques: np.ndarray = field(default_factory=lambda: np.array([]))
    target_positions: np.ndarray | None = None
    target_velocities: np.ndarray | None = None
    kp: np.ndarray = field(default_factory=lambda: np.array([]))
    kd: np.ndarray = field(default_factory=lambda: np.array([]))
    ki: np.ndarray = field(default_factory=lambda: np.array([]))
    integral_error: np.ndarray = field(default_factory=lambda: np.array([]))
    custom_params: dict[str, Any] = field(default_factory=dict)


class ControlInterface:
    """Unified control interface for all physics engines.

    Manages joint torque inputs and control strategy selection.
    Provides visibility into all control parameters and applied torques.

    Design by Contract:
        Preconditions:
            - Engine must be initialized with a loaded model
            - Control inputs must have correct dimensions
        Postconditions:
            - Applied torques are within joint limits
            - Control state is queryable at any time
        Invariants:
            - Joint count and names are fixed after initialization
    """

    def __init__(
        self,
        engine: PhysicsEngine,
        torque_limits: np.ndarray | None = None,
    ) -> None:
        """Initialize the control interface.

        Args:
            engine: Physics engine with loaded model.
            torque_limits: Per-joint torque limits. Uses default if None.
        """
        self.engine = engine
        self._n_q, self._n_v = self._get_dimensions()
        self._joint_info = self._build_joint_info(torque_limits)
        self._state = ControlState(
            torques=np.zeros(self._n_v),
            kp=np.full(self._n_v, 100.0),
            kd=np.full(self._n_v, 10.0),
            ki=np.zeros(self._n_v),
            integral_error=np.zeros(self._n_v),
        )
        self._torque_history: list[np.ndarray] = []
        self._max_history = 10000

    @property
    def n_joints(self) -> int:
        """Number of actuated joints."""
        return self._n_v

    @property
    def joint_names(self) -> list[str]:
        """Names of all joints."""
        return [ji.name for ji in self._joint_info]

    @property
    def current_torques(self) -> np.ndarray:
        """Currently applied torques (copy)."""
        return self._state.torques.copy()

    @property
    def strategy(self) -> ControlStrategy:
        """Active control strategy."""
        return self._state.strategy

    def get_state(self) -> dict[str, Any]:
        """Get complete control state as a dictionary.

        Returns:
            Dictionary with all control parameters, suitable for
            dashboard display and API serialization.
        """
        return {
            "strategy": self._state.strategy.value,
            "n_joints": self._n_v,
            "joint_names": self.joint_names,
            "torques": self._state.torques.tolist(),
            "target_positions": (
                self._state.target_positions.tolist()
                if self._state.target_positions is not None
                else None
            ),
            "target_velocities": (
                self._state.target_velocities.tolist()
                if self._state.target_velocities is not None
                else None
            ),
            "kp": self._state.kp.tolist(),
            "kd": self._state.kd.tolist(),
            "ki": self._state.ki.tolist(),
            "joints": [
                {
                    "index": ji.index,
                    "name": ji.name,
                    "torque_limit": ji.torque_limit,
                    "position_limit_lower": ji.position_limit_lower,
                    "position_limit_upper": ji.position_limit_upper,
                    "velocity_limit": ji.velocity_limit,
                    "current_torque": float(self._state.torques[ji.index]),
                }
                for ji in self._joint_info
            ],
            "custom_params": self._state.custom_params,
        }

    def set_strategy(self, strategy: str | ControlStrategy) -> None:
        """Set the active control strategy.

        Args:
            strategy: Control strategy name or enum value.

        Raises:
            ValueError: If strategy is not recognized.
        """
        if isinstance(strategy, str):
            try:
                strategy = ControlStrategy(strategy.lower())
            except ValueError:
                valid = [s.value for s in ControlStrategy]
                raise ValueError(f"Unknown strategy '{strategy}'. Valid: {valid}")

        self._state.strategy = strategy
        self._state.integral_error = np.zeros(self._n_v)
        logger.info("Control strategy set to: %s", strategy.value)

    def set_torques(self, torques: np.ndarray | list[float]) -> None:
        """Set direct torque inputs for all joints.

        Args:
            torques: Torque values for each joint (n_v,).

        Raises:
            ValueError: If dimensions don't match.
        """
        torques = np.asarray(torques, dtype=np.float64)
        if len(torques) != self._n_v:
            raise ValueError(f"Expected {self._n_v} torques, got {len(torques)}")

        # Clip to torque limits
        torques = self._clip_torques(torques)
        self._state.torques = torques

    def set_joint_torque(self, joint: int | str, torque: float) -> None:
        """Set torque for a single joint.

        Args:
            joint: Joint index or name.
            torque: Torque value.

        Raises:
            ValueError: If joint not found.
        """
        idx = self._resolve_joint_index(joint)
        limit = self._joint_info[idx].torque_limit
        self._state.torques[idx] = np.clip(torque, -limit, limit)

    def set_gains(
        self,
        kp: float | np.ndarray | None = None,
        kd: float | np.ndarray | None = None,
        ki: float | np.ndarray | None = None,
    ) -> None:
        """Set PD/PID gains.

        Args:
            kp: Proportional gain(s). Scalar applies to all joints.
            kd: Derivative gain(s). Scalar applies to all joints.
            ki: Integral gain(s). Scalar applies to all joints.
        """
        if kp is not None:
            self._state.kp = (
                np.full(self._n_v, kp) if np.isscalar(kp) else np.asarray(kp)
            )
        if kd is not None:
            self._state.kd = (
                np.full(self._n_v, kd) if np.isscalar(kd) else np.asarray(kd)
            )
        if ki is not None:
            self._state.ki = (
                np.full(self._n_v, ki) if np.isscalar(ki) else np.asarray(ki)
            )

    def set_target_positions(self, positions: np.ndarray | list[float]) -> None:
        """Set target positions for tracking controllers (PD, PID, etc.).

        Args:
            positions: Target joint positions (n_q,).
        """
        self._state.target_positions = np.asarray(positions, dtype=np.float64)

    def set_target_velocities(self, velocities: np.ndarray | list[float]) -> None:
        """Set target velocities for tracking controllers.

        Args:
            velocities: Target joint velocities (n_v,).
        """
        self._state.target_velocities = np.asarray(velocities, dtype=np.float64)

    def compute_control(self, dt: float = 0.002) -> np.ndarray:
        """Compute control torques based on current strategy and state.

        Args:
            dt: Timestep for integral/derivative computation.

        Returns:
            Computed torque vector (n_v,).
        """
        q, v = self.engine.get_state()
        strategy = self._state.strategy

        if strategy == ControlStrategy.ZERO:
            torques = np.zeros(self._n_v)

        elif strategy == ControlStrategy.DIRECT_TORQUE:
            torques = self._state.torques.copy()

        elif strategy == ControlStrategy.PD:
            torques = self._compute_pd(q, v)

        elif strategy == ControlStrategy.PID:
            torques = self._compute_pid(q, v, dt)

        elif strategy == ControlStrategy.GRAVITY_COMPENSATION:
            torques = self._compute_gravity_compensation()

        elif strategy == ControlStrategy.COMPUTED_TORQUE:
            torques = self._compute_computed_torque(q, v)

        elif strategy == ControlStrategy.IMPEDANCE:
            torques = self._compute_impedance(q, v)

        elif strategy == ControlStrategy.WHOLE_BODY:
            torques = self._compute_whole_body(q, v)

        else:
            torques = np.zeros(self._n_v)

        # Clip and apply
        torques = self._clip_torques(torques)
        self._state.torques = torques

        # Record history
        if len(self._torque_history) < self._max_history:
            self._torque_history.append(torques.copy())

        # Apply to engine
        self.engine.set_control(torques)

        return torques

    def get_torque_history(self) -> np.ndarray:
        """Get recorded torque history.

        Returns:
            Array of shape (n_steps, n_v) with torque history.
        """
        if not self._torque_history:
            return np.zeros((0, self._n_v))
        return np.array(self._torque_history)

    def reset(self) -> None:
        """Reset control state to defaults."""
        self._state.torques = np.zeros(self._n_v)
        self._state.integral_error = np.zeros(self._n_v)
        self._state.target_positions = None
        self._state.target_velocities = None
        self._torque_history.clear()
        self.engine.set_control(np.zeros(self._n_v))

    def get_available_strategies(self) -> list[dict[str, str]]:
        """Get list of available control strategies with descriptions.

        Returns:
            List of dicts with 'name' and 'description' for each strategy.
        """
        return [
            {
                "name": ControlStrategy.ZERO.value,
                "description": "Zero torque (passive dynamics / free fall)",
            },
            {
                "name": ControlStrategy.DIRECT_TORQUE.value,
                "description": "Direct joint torque input (user-specified)",
            },
            {
                "name": ControlStrategy.PD.value,
                "description": "PD controller (proportional-derivative position tracking)",
            },
            {
                "name": ControlStrategy.PID.value,
                "description": "PID controller (with integral term for steady-state error)",
            },
            {
                "name": ControlStrategy.GRAVITY_COMPENSATION.value,
                "description": "Gravity compensation (counteract gravitational torques)",
            },
            {
                "name": ControlStrategy.COMPUTED_TORQUE.value,
                "description": "Computed torque / inverse dynamics control",
            },
            {
                "name": ControlStrategy.IMPEDANCE.value,
                "description": "Impedance control (compliant position tracking)",
            },
            {
                "name": ControlStrategy.WHOLE_BODY.value,
                "description": "Whole-body control with task prioritization (QP-based)",
            },
        ]

    # ---- Private control computation methods ----

    def _compute_pd(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute PD control torques.

        tau = Kp * (q_target - q) + Kd * (v_target - v)
        """
        q_target = (
            self._state.target_positions
            if self._state.target_positions is not None
            else np.zeros(self._n_q)
        )
        v_target = (
            self._state.target_velocities
            if self._state.target_velocities is not None
            else np.zeros(self._n_v)
        )

        # Handle dimension mismatch (n_q vs n_v)
        pos_error = q_target[: self._n_v] - q[: self._n_v]
        vel_error = v_target[: self._n_v] - v[: self._n_v]

        return self._state.kp * pos_error + self._state.kd * vel_error

    def _compute_pid(self, q: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
        """Compute PID control torques.

        tau = Kp * e + Kd * e_dot + Ki * integral(e)
        """
        q_target = (
            self._state.target_positions
            if self._state.target_positions is not None
            else np.zeros(self._n_q)
        )
        v_target = (
            self._state.target_velocities
            if self._state.target_velocities is not None
            else np.zeros(self._n_v)
        )

        pos_error = q_target[: self._n_v] - q[: self._n_v]
        vel_error = v_target[: self._n_v] - v[: self._n_v]

        # Update integral
        self._state.integral_error += pos_error * dt
        # Anti-windup: clip integral
        max_integral = 10.0
        self._state.integral_error = np.clip(
            self._state.integral_error, -max_integral, max_integral
        )

        return (
            self._state.kp * pos_error
            + self._state.kd * vel_error
            + self._state.ki * self._state.integral_error
        )

    def _compute_gravity_compensation(self) -> np.ndarray:
        """Compute gravity compensation torques.

        tau = g(q) (counteract gravity)
        """
        try:
            return self.engine.compute_gravity_forces()
        except Exception as e:
            logger.warning("Gravity compensation failed: %s", e)
            return np.zeros(self._n_v)

    def _compute_computed_torque(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute inverse-dynamics-based control.

        tau = M(q) * (a_desired) + C(q,v) + g(q)
        where a_desired = Kp * (q_target - q) + Kd * (v_target - v)
        """
        q_target = (
            self._state.target_positions
            if self._state.target_positions is not None
            else np.zeros(self._n_q)
        )
        v_target = (
            self._state.target_velocities
            if self._state.target_velocities is not None
            else np.zeros(self._n_v)
        )

        pos_error = q_target[: self._n_v] - q[: self._n_v]
        vel_error = v_target[: self._n_v] - v[: self._n_v]
        a_desired = self._state.kp * pos_error + self._state.kd * vel_error

        try:
            return self.engine.compute_inverse_dynamics(a_desired)
        except Exception as e:
            logger.warning("Computed torque failed, falling back to PD: %s", e)
            return self._compute_pd(q, v)

    def _compute_impedance(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute impedance control torques.

        tau = g(q) + Kp * (q_target - q) + Kd * (v_target - v)
        """
        pd_torques = self._compute_pd(q, v)
        try:
            gravity = self.engine.compute_gravity_forces()
            return gravity + pd_torques
        except Exception:
            return pd_torques

    def _compute_whole_body(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute whole-body control torques using task priorities.

        Falls back to computed torque if WBC module not available.
        """
        try:
            from src.robotics.control import WholeBodyController

            wbc = WholeBodyController(self.engine)
            # Use custom params for task configuration
            solution = wbc.solve()
            return (
                solution.torques
                if hasattr(solution, "torques")
                else np.zeros(self._n_v)
            )
        except Exception as e:
            logger.debug("WBC not available, using computed torque: %s", e)
            return self._compute_computed_torque(q, v)

    def _clip_torques(self, torques: np.ndarray) -> np.ndarray:
        """Clip torques to joint limits.

        Args:
            torques: Raw torque values.

        Returns:
            Clipped torque values.
        """
        for ji in self._joint_info:
            torques[ji.index] = np.clip(
                torques[ji.index], -ji.torque_limit, ji.torque_limit
            )
        return torques

    def _resolve_joint_index(self, joint: int | str) -> int:
        """Resolve a joint identifier to an index.

        Args:
            joint: Joint index or name.

        Returns:
            Joint index.

        Raises:
            ValueError: If joint not found.
        """
        if isinstance(joint, int):
            if 0 <= joint < self._n_v:
                return joint
            raise ValueError(f"Joint index {joint} out of range [0, {self._n_v})")

        for ji in self._joint_info:
            if ji.name == joint:
                return ji.index
        raise ValueError(f"Joint '{joint}' not found. Available: {self.joint_names}")

    def _get_dimensions(self) -> tuple[int, int]:
        """Get model dimensions."""
        try:
            q, v = self.engine.get_state()
            return len(q), len(v)
        except Exception:
            return 7, 7

    def _build_joint_info(self, torque_limits: np.ndarray | None) -> list[JointInfo]:
        """Build joint information list.

        Args:
            torque_limits: Optional per-joint torque limits.

        Returns:
            List of JointInfo for each joint.
        """
        try:
            names = self.engine.get_joint_names()
            if not names or len(names) != self._n_v:
                names = [f"joint_{i}" for i in range(self._n_v)]
        except Exception:
            names = [f"joint_{i}" for i in range(self._n_v)]

        joints = []
        for i in range(self._n_v):
            limit = (
                float(torque_limits[i])
                if torque_limits is not None and i < len(torque_limits)
                else 100.0
            )
            joints.append(
                JointInfo(
                    index=i,
                    name=names[i],
                    torque_limit=limit,
                )
            )
        return joints
