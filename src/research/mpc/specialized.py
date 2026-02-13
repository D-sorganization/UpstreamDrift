"""Specialized MPC implementations for robotics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.research.mpc.controller import (
    CostFunction,
    ModelPredictiveController,
    MPCResult,
)
from src.shared.python.core.constants import GRAVITY

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.engines.protocols import PhysicsEngineProtocol


@dataclass
class CentroidalState:
    """Centroidal dynamics state.

    Attributes:
        com_position: Center of mass position [x, y, z].
        com_velocity: Center of mass velocity [vx, vy, vz].
        angular_momentum: Angular momentum [Lx, Ly, Lz].
        contact_forces: Contact forces per contact point.
    """

    com_position: NDArray[np.floating]
    com_velocity: NDArray[np.floating]
    angular_momentum: NDArray[np.floating]
    contact_forces: dict[str, NDArray[np.floating]]


class CentroidalMPC(ModelPredictiveController):
    """MPC using centroidal dynamics for locomotion.

    Uses simplified centroidal dynamics model that captures
    the robot's center of mass and angular momentum, suitable
    for fast locomotion planning.

    The state is: [com_x, com_y, com_z, com_vx, com_vy, com_vz, Lx, Ly, Lz]
    Controls are contact forces at each foot.
    """

    def __init__(
        self,
        model: PhysicsEngineProtocol,
        horizon: int = 30,
        dt: float = 0.02,
        n_contacts: int = 2,
    ) -> None:
        """Initialize centroidal MPC.

        Args:
            model: Physics engine.
            horizon: Prediction horizon.
            dt: Timestep.
            n_contacts: Number of contact points (feet).
        """
        super().__init__(model, horizon, dt)

        # Centroidal state dimension: 9 (com pos, vel, ang momentum)
        self._n_x = 9

        # Control: 3D force per contact
        self._n_contacts = n_contacts
        self._n_u = 3 * n_contacts

        # Physical parameters
        self._mass = 50.0  # kg
        self._gravity = np.array([0, 0, -GRAVITY])

        # Contact positions (updated from model)
        self._contact_positions: list[NDArray[np.floating]] = [
            np.array([0.1, 0.1, 0.0]),
            np.array([0.1, -0.1, 0.0]),
        ]

        # Friction cone constraint
        self._friction_coef = 0.5

        # Default cost
        self._setup_default_cost()

    def _setup_default_cost(self) -> None:
        """Setup default locomotion cost function."""
        # State cost: track reference CoM position and velocity
        Q = np.diag(
            [
                100,
                100,
                100,  # CoM position
                10,
                10,
                10,  # CoM velocity
                1,
                1,
                1,  # Angular momentum
            ]
        )

        # Control cost: minimize contact forces
        R = np.eye(self._n_u) * 0.001

        # Terminal cost
        P = Q * 10

        self._cost = CostFunction(Q=Q, R=R, P=P)

    def set_mass(self, mass: float) -> None:
        """Set robot mass.

        Args:
            mass: Robot mass in kg.
        """
        self._mass = mass

    def update_contact_positions(
        self,
        positions: list[NDArray[np.floating]],
    ) -> None:
        """Update contact point positions.

        Args:
            positions: List of contact positions in world frame.
        """
        self._contact_positions = positions

    def _dynamics(
        self,
        x: NDArray[np.floating],
        u: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Centroidal dynamics.

        Args:
            x: State [com_pos, com_vel, L].
            u: Contact forces [f1_x, f1_y, f1_z, f2_x, ...].

        Returns:
            Next state.
        """
        com = x[:3]
        com_vel = x[3:6]
        L = x[6:9]

        # Sum of contact forces
        total_force = np.zeros(3)
        total_moment = np.zeros(3)

        for i in range(self._n_contacts):
            force = u[i * 3 : (i + 1) * 3]
            total_force += force

            # Moment about CoM
            r = self._contact_positions[i] - com
            total_moment += np.cross(r, force)

        # Linear acceleration
        com_acc = (total_force / self._mass) + self._gravity

        # Angular momentum rate
        L_dot = total_moment

        # Euler integration
        com_new = com + com_vel * self.dt
        com_vel_new = com_vel + com_acc * self.dt
        L_new = L + L_dot * self.dt

        return np.concatenate([com_new, com_vel_new, L_new])

    def add_friction_cone_constraints(self) -> None:
        """Add friction cone constraints for contact forces."""
        from src.research.mpc.controller import Constraint

        # For each contact, add linearized friction cone
        for i in range(self._n_contacts):
            # f_z >= 0 (normal force positive)
            A_normal = np.zeros((1, self._n_u))
            A_normal[0, i * 3 + 2] = -1  # -f_z <= 0

            self.add_constraint(
                Constraint(
                    B=A_normal,
                    ub=np.zeros(1),
                )
            )

            # |f_x| <= mu * f_z, |f_y| <= mu * f_z
            # Linearized: -mu*f_z <= f_x <= mu*f_z
            for j in range(2):  # x and y
                A_friction = np.zeros((2, self._n_u))
                A_friction[0, i * 3 + j] = 1
                A_friction[0, i * 3 + 2] = -self._friction_coef
                A_friction[1, i * 3 + j] = -1
                A_friction[1, i * 3 + 2] = -self._friction_coef

                self.add_constraint(
                    Constraint(
                        B=A_friction,
                        ub=np.zeros(2),
                    )
                )

    def set_gait_reference(
        self,
        target_velocity: NDArray[np.floating],
        target_height: float = 0.9,
    ) -> None:
        """Set reference trajectory for walking gait.

        Args:
            target_velocity: Desired CoM velocity [vx, vy].
            target_height: Desired CoM height.
        """
        # Build reference trajectory
        x_ref = np.zeros((self.horizon + 1, self._n_x))

        # Linear CoM trajectory at target velocity
        com = np.array([0, 0, target_height])
        vel = np.array([target_velocity[0], target_velocity[1], 0])

        for k in range(self.horizon + 1):
            x_ref[k, :3] = com + vel * k * self.dt
            x_ref[k, 3:6] = vel
            x_ref[k, 6:9] = 0  # Zero angular momentum

        assert self._cost is not None, "Cost function not initialized"
        self._cost.x_ref = x_ref


class WholeBodyMPC(ModelPredictiveController):
    """MPC using full rigid-body dynamics.

    Uses the complete robot dynamics model for precise
    manipulation and whole-body control tasks.
    """

    def __init__(
        self,
        model: PhysicsEngineProtocol,
        horizon: int = 10,
        dt: float = 0.01,
    ) -> None:
        """Initialize whole-body MPC.

        Args:
            model: Physics engine with full dynamics.
            horizon: Prediction horizon.
            dt: Timestep.
        """
        super().__init__(model, horizon, dt)

        # Setup default costs for whole-body control
        self._setup_default_cost()

        # Task-space tracking
        self._end_effector_targets: dict[str, NDArray[np.floating]] = {}

    def _setup_default_cost(self) -> None:
        """Setup default whole-body cost function."""
        # State cost: track joint positions and velocities
        Q_pos = np.eye(self._n_x // 2) * 10
        Q_vel = np.eye(self._n_x // 2) * 1
        Q = np.block(
            [
                [Q_pos, np.zeros((self._n_x // 2, self._n_x // 2))],
                [np.zeros((self._n_x // 2, self._n_x // 2)), Q_vel],
            ]
        )

        # Control cost: minimize torques
        R = np.eye(self._n_u) * 0.001

        # Terminal cost
        P = Q * 10

        self._cost = CostFunction(Q=Q, R=R, P=P)

    def set_end_effector_target(
        self,
        ee_name: str,
        target_pose: NDArray[np.floating],
    ) -> None:
        """Set target pose for an end-effector.

        Args:
            ee_name: End-effector name.
            target_pose: Target pose [x, y, z, qw, qx, qy, qz].
        """
        self._end_effector_targets[ee_name] = target_pose

    def set_joint_targets(
        self,
        target_positions: NDArray[np.floating],
        target_velocities: NDArray[np.floating] | None = None,
    ) -> None:
        """Set target joint positions and velocities.

        Args:
            target_positions: Target joint positions.
            target_velocities: Target joint velocities (zeros if None).
        """
        n_q = self._n_x // 2

        if target_velocities is None:
            target_velocities = np.zeros(n_q)

        x_ref = np.concatenate([target_positions, target_velocities])
        assert self._cost is not None, "Cost function not initialized"
        self._cost.x_ref = x_ref

    def add_joint_limit_constraints(
        self,
        lower_limits: NDArray[np.floating],
        upper_limits: NDArray[np.floating],
    ) -> None:
        """Add joint position limit constraints.

        Args:
            lower_limits: Lower joint limits.
            upper_limits: Upper joint limits.
        """
        from src.research.mpc.controller import Constraint

        n_q = self._n_x // 2

        # Position constraints: lb <= q <= ub
        A = np.zeros((n_q, self._n_x))
        A[:, :n_q] = np.eye(n_q)

        self.add_constraint(
            Constraint(
                A=A,
                lb=lower_limits,
                ub=upper_limits,
            )
        )

    def add_torque_limit_constraints(
        self,
        torque_limits: NDArray[np.floating],
    ) -> None:
        """Add joint torque limit constraints.

        Args:
            torque_limits: Maximum torque magnitudes.
        """
        from src.research.mpc.controller import Constraint

        B = np.eye(self._n_u)

        self.add_constraint(
            Constraint(
                B=B,
                lb=-torque_limits,
                ub=torque_limits,
            )
        )

    def solve_with_ee_tracking(
        self,
        initial_state: NDArray[np.floating],
    ) -> MPCResult:
        """Solve MPC with end-effector tracking.

        Args:
            initial_state: Initial robot state.

        Returns:
            MPC solution result.
        """
        # Convert EE targets to joint targets via IK
        if self._end_effector_targets and hasattr(self.model, "solve_ik"):
            # Use first EE target
            ee_name = list(self._end_effector_targets.keys())[0]
            target_pose = self._end_effector_targets[ee_name]

            q_target, success = self.model.solve_ik(ee_name, target_pose)
            if success:
                self.set_joint_targets(q_target)

        return self.solve(initial_state)
