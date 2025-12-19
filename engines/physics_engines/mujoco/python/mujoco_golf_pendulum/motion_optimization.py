"""Motion optimization and trajectory planning for golf swing.

This module provides advanced optimization tools for generating optimal
golf swing trajectories, including:
- Direct trajectory optimization
- Optimal control synthesis
- Multi-objective optimization
- Biomechanical constraint satisfaction
- Club head speed maximization
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution, minimize


@dataclass
class OptimizationObjectives:
    """Objectives for trajectory optimization."""

    maximize_club_speed: bool = True  # Maximize club head speed at impact
    minimize_energy: bool = True  # Minimize energy expenditure
    minimize_jerk: bool = True  # Minimize jerk (smoothness)
    minimize_torque: bool = True  # Minimize joint torques
    target_ball_position: np.ndarray | None = None  # Hit specific target

    # Weights for multi-objective optimization
    weight_speed: float = 10.0
    weight_energy: float = 1.0
    weight_jerk: float = 0.5
    weight_torque: float = 0.1
    weight_accuracy: float = 5.0


@dataclass
class OptimizationConstraints:
    """Constraints for trajectory optimization."""

    joint_position_limits: bool = True  # Respect joint limits
    joint_velocity_limits: bool = True  # Respect velocity limits
    joint_torque_limits: bool = True  # Respect torque limits
    collision_avoidance: bool = False  # Avoid self-collisions
    maintain_grip: bool = True  # Keep hands on club
    balance_constraint: bool = False  # Maintain balance (COM over support)

    # Limit values (if not from model)
    max_joint_velocity: np.ndarray | None = None
    max_joint_torque: np.ndarray | None = None


@dataclass
class OptimizationResult:
    """Result of trajectory optimization."""

    success: bool
    optimal_trajectory: np.ndarray  # [num_steps x nv] positions
    optimal_velocities: np.ndarray  # [num_steps x nv] velocities
    optimal_controls: np.ndarray  # [num_steps x nu] control torques
    objective_value: float
    num_iterations: int
    computation_time: float
    peak_club_speed: float
    final_club_position: np.ndarray


class SwingOptimizer:
    """Optimizer for golf swing trajectories.

    This class implements state-of-the-art trajectory optimization
    techniques for synthesizing optimal golf swings.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        objectives: OptimizationObjectives | None = None,
        constraints: OptimizationConstraints | None = None,
    ) -> None:
        """Initialize swing optimizer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            objectives: Optimization objectives
            constraints: Optimization constraints
        """
        self.model = model
        self.data = data

        if objectives is None:
            self.objectives = OptimizationObjectives()
        else:
            self.objectives = objectives

        if constraints is None:
            self.constraints = OptimizationConstraints()
        else:
            self.constraints = constraints

        # Find important bodies
        self.club_head_id = self._find_body_id("club_head")
        self.ball_id = self._find_body_id("ball")

        # Trajectory parameterization
        self.num_knot_points = 10  # Number of waypoints
        self.swing_duration = 1.5  # Total swing time [s]

    def _find_body_id(self, name_pattern: str) -> int | None:
        """Find body ID by name pattern."""
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and name_pattern.lower() in body_name.lower():
                return i
        return None

    def optimize_trajectory(
        self,
        initial_guess: np.ndarray | None = None,
        method: str = "SLSQP",
    ) -> OptimizationResult:
        """Optimize golf swing trajectory.

        This uses direct trajectory optimization with collocation.

        Args:
            initial_guess: Initial trajectory guess [num_knots x nv]
            method: Optimization method ("SLSQP", "differential_evolution", etc.)

        Returns:
            OptimizationResult with optimal trajectory
        """
        start_time = time.time()

        # Generate initial guess if not provided
        if initial_guess is None:
            initial_guess = self._generate_initial_guess()

        # Flatten for optimization (decision variables)
        x0 = initial_guess.flatten()

        # Bounds (joint limits)
        bounds = self._compute_bounds()

        # Constraints
        constraints_list = self._setup_constraints()

        # Objective function
        def objective(x) -> float:
            """Docstring for objective."""
            return self._evaluate_objective(x)

        # Optimize
        if method == "differential_evolution":
            result = differential_evolution(
                objective,
                bounds,
                maxiter=100,
                popsize=15,
                atol=1e-3,
                tol=1e-3,
            )
        else:
            result = minimize(
                objective,
                x0,
                method=method,
                bounds=bounds,
                constraints=constraints_list,
                options={"maxiter": 200, "disp": True},
            )

        # Extract optimal trajectory
        optimal_trajectory = result.x.reshape(self.num_knot_points, self.model.nv)

        # Simulate and compute velocities/controls
        velocities, controls, metrics = self._simulate_trajectory(optimal_trajectory)

        computation_time = time.time() - start_time

        return OptimizationResult(
            success=result.success,
            optimal_trajectory=optimal_trajectory,
            optimal_velocities=velocities,
            optimal_controls=controls,
            objective_value=result.fun,
            num_iterations=result.nit if hasattr(result, "nit") else 0,
            computation_time=computation_time,
            peak_club_speed=metrics["peak_club_speed"],
            final_club_position=metrics["final_club_position"],
        )

    def _generate_initial_guess(self) -> np.ndarray:
        """Generate initial trajectory guess.

        Uses a simple strategy: interpolate from address to finish.

        Returns:
            Initial trajectory [num_knots x nv]
        """
        # Define key poses
        address_pose = np.zeros(self.model.nv)  # Start position
        backswing_pose = np.zeros(self.model.nv)
        downswing_pose = np.zeros(self.model.nv)
        impact_pose = np.zeros(self.model.nv)
        followthrough_pose = np.zeros(self.model.nv)

        # Set reasonable values for key joints (example for upper body model)
        # This should be customized based on the specific model

        # Backswing: rotate shoulders, lift arms
        if self.model.nv >= 10:
            backswing_pose[0] = -1.5  # Shoulder rotation
            backswing_pose[1] = 0.5  # Left shoulder swing
            backswing_pose[2] = 1.5  # Left shoulder lift

        # Downswing: transition phase
        downswing_pose = (backswing_pose + impact_pose) / 2

        # Impact: arms extended, shoulders rotated through
        if self.model.nv >= 10:
            impact_pose[0] = 1.0  # Shoulder rotation
            impact_pose[3] = -0.5  # Left elbow extension

        # Follow-through: full rotation
        if self.model.nv >= 10:
            followthrough_pose[0] = 1.8

        # Interpolate between key poses
        key_poses = np.array(
            [
                address_pose,
                address_pose,  # Pause at address
                backswing_pose,
                downswing_pose,
                impact_pose,
                followthrough_pose,
                followthrough_pose,  # Hold finish
                followthrough_pose,
                followthrough_pose,
                followthrough_pose,
            ],
        )

        return key_poses[: self.num_knot_points]

    def _compute_bounds(self) -> list[tuple[float, float]]:
        """Compute optimization bounds from joint limits.

        Returns:
            List of (min, max) tuples for each decision variable
        """
        bounds = []

        for _knot in range(self.num_knot_points):
            for joint_idx in range(self.model.njnt):
                if (
                    self.constraints.joint_position_limits
                    and self.model.jnt_limited[joint_idx]
                ):
                    q_min = self.model.jnt_range[joint_idx, 0]
                    q_max = self.model.jnt_range[joint_idx, 1]
                else:
                    q_min = -np.pi
                    q_max = np.pi

                bounds.append((q_min, q_max))

            # Add bounds for any extra DOFs (freejoint, etc.)
            for _ in range(self.model.nv - self.model.njnt):
                bounds.append((-10.0, 10.0))

        return bounds

    def _setup_constraints(self) -> list:
        """Setup optimization constraints.

        Returns:
            List of constraint dictionaries for scipy.optimize
        """
        constraints = []

        # Velocity limits
        if self.constraints.joint_velocity_limits:

            def velocity_constraint(x) -> np.ndarray:
                """Docstring for velocity_constraint."""
                trajectory = x.reshape(self.num_knot_points, self.model.nv)
                dt = self.swing_duration / (self.num_knot_points - 1)

                # Finite difference velocities
                velocities = np.diff(trajectory, axis=0) / dt

                max_vel = self.constraints.max_joint_velocity
                if max_vel is None:
                    max_vel = np.ones(self.model.nv) * 10.0  # rad/s

                # Constraint: |v| <= v_max
                # Formulate as: v_max - |v| >= 0
                violations = max_vel - np.abs(velocities)
                return violations.flatten()

            constraints.append({"type": "ineq", "fun": velocity_constraint})

        return constraints

    def _evaluate_objective(self, x: np.ndarray) -> float:
        """Evaluate objective function.

        Args:
            x: Decision variables (flattened trajectory)

        Returns:
            Objective value (to minimize)
        """
        trajectory = x.reshape(self.num_knot_points, self.model.nv)

        # Simulate trajectory to get metrics
        _, controls, metrics = self._simulate_trajectory(trajectory)

        objective = 0.0

        # Club head speed (maximize = minimize negative)
        if self.objectives.maximize_club_speed:
            objective -= self.objectives.weight_speed * metrics["peak_club_speed"]

        # Energy (minimize)
        if self.objectives.minimize_energy:
            total_energy = metrics["total_energy"]
            objective += self.objectives.weight_energy * total_energy

        # Jerk (minimize)
        if self.objectives.minimize_jerk:
            jerk = self._compute_jerk(trajectory)
            objective += self.objectives.weight_jerk * jerk

        # Torque (minimize)
        if self.objectives.minimize_torque:
            total_torque = np.sum(np.abs(controls))
            objective += self.objectives.weight_torque * total_torque

        # Accuracy (hit target)
        if self.objectives.target_ball_position is not None:
            distance_error = float(
                np.linalg.norm(
                    metrics["final_club_position"]
                    - self.objectives.target_ball_position,
                ),
            )
            objective += self.objectives.weight_accuracy * distance_error

        return objective

    def _simulate_trajectory(
        self,
        trajectory: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Simulate a trajectory and extract metrics.

        Args:
            trajectory: Joint trajectory [num_knots x nv]

        Returns:
            Tuple of (velocities, controls, metrics_dict)
        """
        # Interpolate trajectory to simulation timesteps
        dt = self.model.opt.timestep
        num_steps = int(self.swing_duration / dt)

        knot_times = np.linspace(0, self.swing_duration, self.num_knot_points)
        sim_times = np.linspace(0, self.swing_duration, num_steps)

        # Cubic spline interpolation
        trajectory_interp = np.zeros((num_steps, self.model.nv))
        for dof in range(self.model.nv):
            spline = CubicSpline(knot_times, trajectory[:, dof])
            trajectory_interp[:, dof] = spline(sim_times)

        # Simulate
        velocities = np.zeros((num_steps, self.model.nv))
        controls = np.zeros((num_steps, self.model.nu))
        club_speeds = []
        club_positions = []

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        for step in range(num_steps):
            # Set desired position
            self.data.qpos[:] = trajectory_interp[step]

            # Simple PD control to track trajectory
            if step < num_steps - 1:
                desired_vel = (
                    trajectory_interp[step + 1] - trajectory_interp[step]
                ) / dt
            else:
                desired_vel = np.zeros(self.model.nv)

            # PD gains
            kp = 100.0
            kd = 20.0

            pos_error = trajectory_interp[step] - self.data.qpos
            vel_error = desired_vel - self.data.qvel

            ctrl = kp * pos_error + kd * vel_error
            # Limit torques to reasonable range
            max_torque = 100.0
            ctrl = np.clip(ctrl, -max_torque, max_torque)

            self.data.ctrl[:] = ctrl[: self.model.nu]

            # Step simulation
            mujoco.mj_step(self.model, self.data)

            # Record
            velocities[step] = self.data.qvel.copy()
            controls[step] = self.data.ctrl.copy()

            # Club head speed
            if self.club_head_id is not None:
                # MuJoCo 3.3+ may require reshaped arrays
                try:
                    jacp = np.zeros((3, self.model.nv))
                    jacr = np.zeros((3, self.model.nv))
                    mujoco.mj_jacBody(
                        self.model, self.data, jacp, jacr, self.club_head_id
                    )
                except TypeError:
                    # Fallback to flat array approach
                    jacp_flat = np.zeros(3 * self.model.nv)
                    jacr_flat = np.zeros(3 * self.model.nv)
                    mujoco.mj_jacBody(
                        self.model,
                        self.data,
                        jacp_flat,
                        jacr_flat,
                        self.club_head_id,
                    )
                    jacp = jacp_flat.reshape(3, self.model.nv)
                vel = jacp @ self.data.qvel
                club_speed = np.linalg.norm(vel)
                club_speeds.append(club_speed)

                club_positions.append(self.data.xpos[self.club_head_id].copy())

        # Compute metrics
        peak_club_speed = (
            float(max(float(s) for s in club_speeds)) if club_speeds else 0.0
        )
        total_energy = np.sum(np.abs(controls) * np.abs(velocities[:, : self.model.nu]))
        final_club_position = club_positions[-1] if club_positions else np.zeros(3)

        metrics = {
            "peak_club_speed": peak_club_speed,
            "total_energy": total_energy,
            "final_club_position": final_club_position,
        }

        return velocities, controls, metrics

    def _compute_jerk(self, trajectory: np.ndarray) -> float:
        """Compute total jerk (third derivative of position).

        Args:
            trajectory: Joint trajectory [num_knots x nv]

        Returns:
            Total jerk magnitude
        """
        dt = self.swing_duration / (self.num_knot_points - 1)

        # Second derivative (acceleration)
        accel = np.diff(trajectory, n=2, axis=0) / dt**2

        # Third derivative (jerk)
        jerk = np.diff(accel, axis=0) / dt

        return float(np.sum(np.abs(jerk)))

    def optimize_swing_for_speed(
        self,
        target_speed: float = 50.0,  # m/s (professional level)
    ) -> OptimizationResult:
        """Optimize swing specifically for maximum club head speed.

        Args:
            target_speed: Target club head speed [m/s]

        Returns:
            OptimizationResult with speed-optimized trajectory
        """
        # Set objectives for pure speed
        objectives = OptimizationObjectives(
            maximize_club_speed=True,
            minimize_energy=False,
            minimize_jerk=True,
            minimize_torque=False,
            weight_speed=100.0,
            weight_jerk=1.0,
        )

        old_objectives = self.objectives
        self.objectives = objectives

        result = self.optimize_trajectory()

        self.objectives = old_objectives

        return result

    def optimize_swing_for_accuracy(
        self,
        target_position: np.ndarray,
    ) -> OptimizationResult:
        """Optimize swing for accuracy (hitting specific target).

        Args:
            target_position: Target position [3] in world frame

        Returns:
            OptimizationResult with accuracy-optimized trajectory
        """
        objectives = OptimizationObjectives(
            maximize_club_speed=True,
            minimize_energy=False,
            minimize_jerk=True,
            minimize_torque=False,
            target_ball_position=target_position,
            weight_speed=10.0,
            weight_accuracy=100.0,
            weight_jerk=1.0,
        )

        old_objectives = self.objectives
        self.objectives = objectives

        result = self.optimize_trajectory()

        self.objectives = old_objectives

        return result

    def generate_library_of_swings(
        self,
        num_swings: int = 10,
        variation: str = "speed",  # "speed", "accuracy", "style"
    ) -> list[OptimizationResult]:
        """Generate a library of different swing styles.

        Args:
            num_swings: Number of swings to generate
            variation: Type of variation

        Returns:
            List of OptimizationResult for different swings
        """
        swings = []

        if variation == "speed":
            # Vary target speeds
            speeds = np.linspace(30.0, 55.0, num_swings)
            for speed in speeds:
                result = self.optimize_swing_for_speed(target_speed=speed)
                swings.append(result)

        elif variation == "accuracy":
            # Vary target positions
            base_pos = np.array([2.0, 0.0, 0.0])
            for i in range(num_swings):
                offset = np.array([0, (i - num_swings / 2) * 0.2, 0])
                target = base_pos + offset
                result = self.optimize_swing_for_accuracy(target_position=target)
                swings.append(result)

        return swings


class MotionPrimitiveLibrary:
    """Library of motion primitives for golf swing composition.

    This stores and retrieves pre-computed motion primitives that can be
    combined to create new swings.
    """

    def __init__(self) -> None:
        """Initialize empty library."""
        self.primitives: dict[str, np.ndarray] = {}
        self.metadata: dict[str, dict] = {}

    def add_primitive(
        self,
        name: str,
        trajectory: np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        """Add a motion primitive to library.

        Args:
            name: Primitive name
            trajectory: Joint trajectory
            metadata: Additional metadata
        """
        self.primitives[name] = trajectory
        self.metadata[name] = metadata if metadata is not None else {}

    def get_primitive(self, name: str) -> np.ndarray | None:
        """Get primitive by name.

        Args:
            name: Primitive name

        Returns:
            Trajectory or None if not found
        """
        return self.primitives.get(name)

    def blend_primitives(
        self,
        names: list[str],
        weights: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Blend multiple primitives.

        Args:
            names: List of primitive names
            weights: Blending weights (default: equal)

        Returns:
            Blended trajectory
        """
        if weights is None:
            weights = np.ones(len(names)) / len(names)

        # Get primitives
        primitives = [
            self.primitives[name] for name in names if name in self.primitives
        ]

        if not primitives:
            return None

        # Ensure same length
        min_len = min(p.shape[0] for p in primitives)
        primitives = [p[:min_len] for p in primitives]

        # Weighted sum
        blended = np.zeros_like(primitives[0])
        for prim, weight in zip(primitives, weights, strict=False):
            blended += weight * prim

        return blended

    def save_library(self, filename: str) -> None:
        """Save library to file.

        Args:
            filename: Output filename (.npz)
        """
        # Convert metadata to a format np.savez can handle
        metadata_str = json.dumps(self.metadata)
        # Save primitives and metadata separately
        # Use dict() to avoid type issues with ** unpacking
        save_dict: dict[str, Any] = dict(self.primitives)
        save_dict["metadata"] = metadata_str
        np.savez(filename, **save_dict)  # type: ignore[arg-type]

    def load_library(self, filename: str) -> None:
        """Load library from file.

        Args:
            filename: Input filename (.npz)
        """
        data = np.load(filename, allow_pickle=True)

        for key in data:
            if key == "metadata":
                # Metadata is stored as JSON string, need to deserialize it
                metadata_value = data[key]
                if isinstance(metadata_value, str):
                    self.metadata = json.loads(metadata_value)
                else:
                    # Fallback for numpy array containing string
                    self.metadata = json.loads(metadata_value.item())
            else:
                self.primitives[key] = data[key]
