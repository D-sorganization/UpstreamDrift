"""Dual-hand inverse kinematics solver for golf swing motion.

This module implements IK solving with both hands as end-effectors that must
track positions on a golf club grip as it moves through the swing trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.engine_availability import PINOCCHIO_AVAILABLE

if TYPE_CHECKING:
    from numpy.typing import NDArray

if PINOCCHIO_AVAILABLE:
    import pinocchio as pin
else:
    pin = None  # type: ignore[assignment]


try:
    import pink
    from pink import Configuration
    from pink.tasks import FrameTask, PostureTask

    PINK_AVAILABLE = True
except ImportError:
    PINK_AVAILABLE = False

from motion_training.club_trajectory_parser import ClubFrame, ClubTrajectory


@dataclass
class IKSolverSettings:
    """Settings for the IK solver."""

    solver: str = "quadprog"
    damping: float = 1e-6
    dt: float = 0.01
    max_iterations: int = 100
    position_tolerance: float = 1e-3
    orientation_tolerance: float = 1e-2

    # Task weights
    left_hand_position_weight: float = 10.0
    left_hand_orientation_weight: float = 5.0
    right_hand_position_weight: float = 10.0
    right_hand_orientation_weight: float = 5.0
    posture_weight: float = 1e-3

    # Hand grip offsets (meters, along grip Z-axis)
    left_hand_offset: float = 0.04  # Lead hand (bottom)
    right_hand_offset: float = -0.04  # Trail hand (top)


@dataclass
class IKResult:
    """Result of IK solving for a single frame."""

    q: NDArray[np.float64]  # Joint configuration
    left_hand_error: float  # Position error for left hand
    right_hand_error: float  # Position error for right hand
    converged: bool
    iterations: int


@dataclass
class TrajectoryIKResult:
    """Result of IK solving for entire trajectory."""

    configurations: list[NDArray[np.float64]] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    left_hand_errors: list[float] = field(default_factory=list)
    right_hand_errors: list[float] = field(default_factory=list)
    convergence_rate: float = 0.0

    @property
    def q_trajectory(self) -> NDArray[np.float64]:
        """Return NxQ array of joint configurations."""
        return np.array(self.configurations)


class DualHandIKSolver:
    """Inverse kinematics solver for dual-hand golf swing tracking.

    Uses Pink (task-based IK for Pinocchio) to solve for body configurations
    that place both hands at the correct positions on the club grip.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        left_hand_frame: str = "hand_left_tip",
        right_hand_frame: str = "hand_right_tip",
        settings: IKSolverSettings | None = None,
    ) -> None:
        """Initialize the IK solver.

        Args:
            urdf_path: Path to the golfer URDF (should be without club attached)
            left_hand_frame: Name of the left hand end-effector frame
            right_hand_frame: Name of the right hand end-effector frame
            settings: Solver configuration
        """
        if not PINOCCHIO_AVAILABLE:
            raise ImportError("Pinocchio required. Install with: pip install pin")
        if not PINK_AVAILABLE:
            raise ImportError("Pink required. Install with: pip install pink")

        self.urdf_path = Path(urdf_path)
        self.settings = settings or IKSolverSettings()

        # Load model
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()

        # Load geometry models for visualization
        self.visual_model = pin.buildGeomFromUrdf(
            self.model,
            str(self.urdf_path),
            pin.GeometryType.VISUAL,
        )
        self.collision_model = pin.buildGeomFromUrdf(
            self.model,
            str(self.urdf_path),
            pin.GeometryType.COLLISION,
        )

        # Verify frames exist
        self.left_hand_frame = left_hand_frame
        self.right_hand_frame = right_hand_frame

        if not self.model.existFrame(left_hand_frame):
            raise ValueError(f"Frame '{left_hand_frame}' not found in model")
        if not self.model.existFrame(right_hand_frame):
            raise ValueError(f"Frame '{right_hand_frame}' not found in model")

        self.left_hand_frame_id = self.model.getFrameId(left_hand_frame)
        self.right_hand_frame_id = self.model.getFrameId(right_hand_frame)

        # Create Pink tasks
        self._setup_tasks()

        # Reference configuration (neutral pose)
        self.q_ref = pin.neutral(self.model)

    def _setup_tasks(self) -> None:
        """Setup Pink IK tasks."""
        s = self.settings

        # Left hand frame task
        self.left_hand_task = FrameTask(
            self.left_hand_frame,
            position_cost=s.left_hand_position_weight,
            orientation_cost=s.left_hand_orientation_weight,
        )
        self.left_hand_task.lm_damping = s.damping

        # Right hand frame task
        self.right_hand_task = FrameTask(
            self.right_hand_frame,
            position_cost=s.right_hand_position_weight,
            orientation_cost=s.right_hand_orientation_weight,
        )
        self.right_hand_task.lm_damping = s.damping

        # Posture regularization task
        self.posture_task = PostureTask(cost=s.posture_weight)

    def compute_hand_targets(
        self,
        frame: ClubFrame,
    ) -> tuple[pin.SE3, pin.SE3]:
        """Compute target SE3 poses for both hands from club frame.

        Args:
            frame: Club trajectory frame with grip position/orientation

        Returns:
            Tuple of (left_hand_target, right_hand_target) as SE3 transforms
        """
        s = self.settings

        # Get grip orientation as rotation matrix
        R = frame.grip_rotation

        # Get grip Z-axis (along shaft)
        grip_z = R[:, 2]

        # Compute hand positions
        left_pos = frame.grip_position + s.left_hand_offset * grip_z
        right_pos = frame.grip_position + s.right_hand_offset * grip_z

        # Create SE3 targets
        # Hand orientation: align hand Z with grip Z (pointing down shaft)
        left_target = pin.SE3(R, left_pos)
        right_target = pin.SE3(R, right_pos)

        return left_target, right_target

    def solve_frame(
        self,
        frame: ClubFrame,
        q_init: NDArray[np.float64] | None = None,
    ) -> IKResult:
        """Solve IK for a single club frame.

        Args:
            frame: Club trajectory frame
            q_init: Initial joint configuration (uses reference if None)

        Returns:
            IKResult with solved configuration and error metrics
        """
        if q_init is None:
            q_init = self.q_ref.copy()

        # Compute hand targets
        left_target, right_target = self.compute_hand_targets(frame)

        # Set task targets
        self.left_hand_task.set_target(left_target)
        self.right_hand_task.set_target(right_target)
        self.posture_task.set_target(self.q_ref)

        tasks = [self.left_hand_task, self.right_hand_task, self.posture_task]

        # Create Pink configuration
        configuration = Configuration(self.model, self.data, q_init)

        q = q_init.copy()
        s = self.settings

        for iteration in range(s.max_iterations):
            # Solve differential IK
            velocity = pink.solve_ik(
                configuration,
                tasks,
                s.dt,
                solver=s.solver,
                damping=s.damping,
            )

            # Integrate
            q = pin.integrate(self.model, q, velocity * s.dt)

            # Update configuration
            configuration = Configuration(self.model, self.data, q)

            # Compute errors
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            left_current = self.data.oMf[self.left_hand_frame_id]
            right_current = self.data.oMf[self.right_hand_frame_id]

            left_error = np.linalg.norm(
                left_current.translation - left_target.translation
            )
            right_error = np.linalg.norm(
                right_current.translation - right_target.translation
            )

            # Check convergence
            if left_error < s.position_tolerance and right_error < s.position_tolerance:
                return IKResult(
                    q=q,
                    left_hand_error=left_error,
                    right_hand_error=right_error,
                    converged=True,
                    iterations=iteration + 1,
                )

        # Return best result even if not converged
        return IKResult(
            q=q,
            left_hand_error=left_error,
            right_hand_error=right_error,
            converged=False,
            iterations=s.max_iterations,
        )

    def solve_trajectory(
        self,
        trajectory: ClubTrajectory,
        q_init: NDArray[np.float64] | None = None,
        verbose: bool = False,
    ) -> TrajectoryIKResult:
        """Solve IK for entire trajectory.

        Args:
            trajectory: Club trajectory to track
            q_init: Initial configuration for first frame
            verbose: Print progress information

        Returns:
            TrajectoryIKResult with all solved configurations
        """
        result = TrajectoryIKResult()

        q = q_init if q_init is not None else self.q_ref.copy()
        num_converged = 0

        for i, frame in enumerate(trajectory.frames):
            ik_result = self.solve_frame(frame, q)

            result.configurations.append(ik_result.q)
            result.times.append(frame.time)
            result.left_hand_errors.append(ik_result.left_hand_error)
            result.right_hand_errors.append(ik_result.right_hand_error)

            if ik_result.converged:
                num_converged += 1

            # Use this configuration as starting point for next frame
            q = ik_result.q

            if verbose and (i + 1) % 50 == 0:
                print(
                    f"Frame {i + 1}/{trajectory.num_frames}: "
                    f"L={ik_result.left_hand_error:.4f}, "
                    f"R={ik_result.right_hand_error:.4f}, "
                    f"converged={ik_result.converged}"
                )

        result.convergence_rate = (
            num_converged / trajectory.num_frames if trajectory.num_frames > 0 else 0.0
        )

        if verbose:
            print("\nTrajectory IK complete:")
            print(f"  Frames: {trajectory.num_frames}")
            print(f"  Convergence rate: {result.convergence_rate * 100:.1f}%")
            print(f"  Mean left error: {np.mean(result.left_hand_errors):.4f} m")
            print(f"  Mean right error: {np.mean(result.right_hand_errors):.4f} m")

        return result

    def get_current_hand_positions(
        self,
        q: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get current hand positions for a configuration.

        Args:
            q: Joint configuration

        Returns:
            Tuple of (left_hand_pos, right_hand_pos)
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        left_pos = self.data.oMf[self.left_hand_frame_id].translation.copy()
        right_pos = self.data.oMf[self.right_hand_frame_id].translation.copy()

        return left_pos, right_pos


class DualHandIKSolverFallback:
    """Fallback IK solver using pure Pinocchio when Pink is not available.

    Uses damped least-squares (Levenberg-Marquardt) IK with Jacobian
    pseudo-inverse for solving dual end-effector constraints.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        left_hand_frame: str = "hand_left_tip",
        right_hand_frame: str = "hand_right_tip",
        settings: IKSolverSettings | None = None,
    ) -> None:
        """Initialize fallback IK solver."""
        if not PINOCCHIO_AVAILABLE:
            raise ImportError("Pinocchio required")

        self.urdf_path = Path(urdf_path)
        self.settings = settings or IKSolverSettings()

        # Load model
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()

        self.left_hand_frame = left_hand_frame
        self.right_hand_frame = right_hand_frame
        self.left_hand_frame_id = self.model.getFrameId(left_hand_frame)
        self.right_hand_frame_id = self.model.getFrameId(right_hand_frame)

        self.q_ref = pin.neutral(self.model)

    def solve_frame(
        self,
        frame: ClubFrame,
        q_init: NDArray[np.float64] | None = None,
    ) -> IKResult:
        """Solve IK using damped least-squares."""
        if q_init is None:
            q_init = self.q_ref.copy()

        s = self.settings
        q = q_init.copy()

        # Compute targets
        R = frame.grip_rotation
        grip_z = R[:, 2]
        left_target = frame.grip_position + s.left_hand_offset * grip_z
        right_target = frame.grip_position + s.right_hand_offset * grip_z

        damping = 1e-3

        for iteration in range(s.max_iterations):
            # Forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            pin.computeJointJacobians(self.model, self.data, q)

            # Get current positions
            left_current = self.data.oMf[self.left_hand_frame_id].translation
            right_current = self.data.oMf[self.right_hand_frame_id].translation

            # Compute errors
            left_error_vec = left_target - left_current
            right_error_vec = right_target - right_current

            left_error = np.linalg.norm(left_error_vec)
            right_error = np.linalg.norm(right_error_vec)

            if left_error < s.position_tolerance and right_error < s.position_tolerance:
                return IKResult(
                    q=q,
                    left_hand_error=left_error,
                    right_hand_error=right_error,
                    converged=True,
                    iterations=iteration + 1,
                )

            # Get Jacobians (position only, 3xnv)
            J_left = pin.getFrameJacobian(
                self.model, self.data, self.left_hand_frame_id, pin.LOCAL_WORLD_ALIGNED
            )[:3, :]
            J_right = pin.getFrameJacobian(
                self.model, self.data, self.right_hand_frame_id, pin.LOCAL_WORLD_ALIGNED
            )[:3, :]

            # Stack for dual end-effector
            J = np.vstack([J_left, J_right])
            error = np.concatenate([left_error_vec, right_error_vec])

            # Damped least-squares solution
            JJT = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, error)

            # Limit step size
            max_step = 0.1
            step_norm = np.linalg.norm(dq)
            if step_norm > max_step:
                dq = dq * max_step / step_norm

            # Update configuration
            q = pin.integrate(self.model, q, dq)

        return IKResult(
            q=q,
            left_hand_error=left_error,
            right_hand_error=right_error,
            converged=False,
            iterations=s.max_iterations,
        )

    def solve_trajectory(
        self,
        trajectory: ClubTrajectory,
        q_init: NDArray[np.float64] | None = None,
        verbose: bool = False,
    ) -> TrajectoryIKResult:
        """Solve IK for entire trajectory."""
        result = TrajectoryIKResult()

        q = q_init if q_init is not None else self.q_ref.copy()
        num_converged = 0

        for i, frame in enumerate(trajectory.frames):
            ik_result = self.solve_frame(frame, q)

            result.configurations.append(ik_result.q)
            result.times.append(frame.time)
            result.left_hand_errors.append(ik_result.left_hand_error)
            result.right_hand_errors.append(ik_result.right_hand_error)

            if ik_result.converged:
                num_converged += 1

            q = ik_result.q

            if verbose and (i + 1) % 50 == 0:
                print(f"Frame {i + 1}/{trajectory.num_frames}")

        result.convergence_rate = (
            num_converged / trajectory.num_frames if trajectory.num_frames > 0 else 0.0
        )
        return result


def create_ik_solver(
    urdf_path: str | Path,
    left_hand_frame: str = "hand_left_tip",
    right_hand_frame: str = "hand_right_tip",
    settings: IKSolverSettings | None = None,
) -> DualHandIKSolver | DualHandIKSolverFallback:
    """Factory function to create IK solver (Pink or fallback).

    Args:
        urdf_path: Path to golfer URDF
        left_hand_frame: Left hand end-effector frame name
        right_hand_frame: Right hand end-effector frame name
        settings: Solver settings

    Returns:
        IK solver instance (Pink-based or fallback)
    """
    if PINK_AVAILABLE:
        return DualHandIKSolver(urdf_path, left_hand_frame, right_hand_frame, settings)
    else:
        print("Pink not available, using fallback damped least-squares solver")
        return DualHandIKSolverFallback(
            urdf_path, left_hand_frame, right_hand_frame, settings
        )
