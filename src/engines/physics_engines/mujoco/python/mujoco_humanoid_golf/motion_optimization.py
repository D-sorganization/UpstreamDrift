# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.  # noqa: E501
# It requires domain-aware structural extraction to isolate its internal classes appropriately.  # noqa: E501

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

import time

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution, minimize

from ._motion_opt_simulation import (
    evaluate_objective,
    setup_constraints,
    simulate_trajectory,
)
from ._motion_opt_trajectory import compute_bounds, generate_initial_guess
from ._motion_opt_types import (
    OptimizationConstraints,
    OptimizationObjectives,
    OptimizationResult,
)

__all__ = [
    "MotionPrimitiveLibrary",
    "OptimizationConstraints",
    "OptimizationObjectives",
    "OptimizationResult",
    "SwingOptimizer",
]

from ._motion_primitive_library import MotionPrimitiveLibrary


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
        if not (model is not None):
            raise ValueError("model must be provided")
        self.model = model
        self.data = data

        self.objectives = (
            objectives if objectives is not None else OptimizationObjectives()
        )
        self.constraints = (
            constraints if constraints is not None else OptimizationConstraints()
        )

        self.club_head_id = self._find_body_id("club_head")
        self.ball_id = self._find_body_id("ball")

        self.num_knot_points = 10
        self.swing_duration = 1.5

    def _find_body_id(self, name_pattern: str) -> int | None:
        """Find body ID by name pattern."""
        if not (name_pattern is not None):
            raise ValueError("name_pattern must be provided")
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
        if not (method is not None):
            raise ValueError("method must be provided")
        start_time = time.time()

        if initial_guess is None:
            initial_guess = generate_initial_guess(self.model, self.num_knot_points)

        x0 = initial_guess.flatten()
        bounds = compute_bounds(self.model, self.constraints, self.num_knot_points)
        constraints_list = setup_constraints(
            self.model, self.constraints, self.num_knot_points, self.swing_duration
        )

        def objective(x: np.ndarray) -> float:
            return evaluate_objective(
                x,
                self.model,
                self.data,
                self.objectives,
                self.club_head_id,
                self.swing_duration,
                self.num_knot_points,
            )

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

        optimal_trajectory = result.x.reshape(self.num_knot_points, self.model.nv)
        velocities, controls, metrics = simulate_trajectory(
            self.model,
            self.data,
            optimal_trajectory,
            self.club_head_id,
            self.swing_duration,
            self.num_knot_points,
        )

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
        # Build per-knot joint bounds
        joint_bounds = [
            (
                (float(self.model.jnt_range[j, 0]), float(self.model.jnt_range[j, 1]))
                if self.constraints.joint_position_limits and self.model.jnt_limited[j]
                else (-np.pi, np.pi)
            )
            for j in range(self.model.njnt)
        ]
        # Extra DOFs (freejoint, etc.)
        extra_bounds = [(-10.0, 10.0)] * (self.model.nv - self.model.njnt)
        knot_bounds = joint_bounds + extra_bounds

        # Repeat for each knot point
        return knot_bounds * self.num_knot_points

    def _setup_constraints(self) -> list:
        """Setup optimization constraints.

        Returns:
            List of constraint dictionaries for scipy.optimize
        """
        constraints = []

        # Velocity limits
        if self.constraints.joint_velocity_limits:

            def velocity_constraint(x: np.ndarray) -> np.ndarray:
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
                return np.asarray(violations.flatten())

            constraints.append({"type": "ineq", "fun": velocity_constraint})

        return constraints

    def _evaluate_objective(self, x: np.ndarray) -> float:
        """Evaluate objective function.

        Args:
            x: Decision variables (flattened trajectory)

        Returns:
            Objective value (to minimize)
        """
        if x is None:
            raise ValueError("x must be provided")
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
            diff = metrics["final_club_position"] - self.objectives.target_ball_position
            distance_error = float(np.sqrt(np.vdot(diff, diff)))
            objective += self.objectives.weight_accuracy * distance_error

        return objective

    def _interpolate_trajectory(
        self, trajectory: np.ndarray
    ) -> tuple[np.ndarray, float, int]:  # noqa: E501
        if trajectory is None:
            raise ValueError("trajectory must be provided")
        dt = self.model.opt.timestep
        num_steps = int(self.swing_duration / dt)

        knot_times = np.linspace(0, self.swing_duration, self.num_knot_points)
        sim_times = np.linspace(0, self.swing_duration, num_steps)

        trajectory_interp = np.zeros((num_steps, self.model.nv))
        for dof in range(self.model.nv):
            spline = CubicSpline(knot_times, trajectory[:, dof])
            trajectory_interp[:, dof] = spline(sim_times)

        return trajectory_interp, dt, num_steps

    def _detect_jacobian_api(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        jacp_flat = np.zeros(3 * self.model.nv)
        jacr_flat = np.zeros(3 * self.model.nv)
        use_flat_jac = False

        if self.club_head_id is not None:
            try:
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.club_head_id)
            except TypeError:
                use_flat_jac = True

        return jacp, jacr, jacp_flat, jacr_flat, use_flat_jac

    def _compute_club_speed(
        self,
        jacp: np.ndarray,
        jacr: np.ndarray,
        jacp_flat: np.ndarray,
        jacr_flat: np.ndarray,
        use_flat_jac: bool,
    ) -> float:
        if jacp is None:
            raise ValueError("jacp must be provided")
        if use_flat_jac:
            mujoco.mj_jacBody(
                self.model,
                self.data,
                jacp_flat,
                jacr_flat,
                self.club_head_id,
            )
            jacp[:] = jacp_flat.reshape(3, self.model.nv)
        else:
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.club_head_id)

        vel = jacp @ self.data.qvel
        return float(np.linalg.norm(vel))

    def _collect_simulation_metrics(
        self,
        club_speeds: list,
        club_positions: list,
        controls: np.ndarray,
        velocities: np.ndarray,
    ) -> dict:
        if club_speeds is None:
            raise ValueError("club_speeds must be provided")
        peak_club_speed = (
            float(max(float(s) for s in club_speeds)) if club_speeds else 0.0
        )  # noqa: E501
        total_energy = np.vdot(np.abs(controls), np.abs(velocities[:, : self.model.nu]))
        final_club_position = club_positions[-1] if club_positions else np.zeros(3)

        return {
            "peak_club_speed": peak_club_speed,
            "total_energy": total_energy,
            "final_club_position": final_club_position,
        }

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
        if trajectory is None:
            raise ValueError("trajectory must be provided")
        trajectory_interp, dt, num_steps = self._interpolate_trajectory(trajectory)

        velocities = np.zeros((num_steps, self.model.nv))
        controls = np.zeros((num_steps, self.model.nu))
        club_speeds: list[float] = []
        club_positions: list[np.ndarray] = []

        mujoco.mj_resetData(self.model, self.data)
        jacp, jacr, jacp_flat, jacr_flat, use_flat_jac = self._detect_jacobian_api()

        for step in range(num_steps):
            self.data.qpos[:] = trajectory_interp[step]

            if step < num_steps - 1:
                desired_vel = (
                    trajectory_interp[step + 1] - trajectory_interp[step]
                ) / dt  # noqa: E501
            else:
                desired_vel = np.zeros(self.model.nv)

            kp = 100.0
            kd = 20.0
            pos_error = trajectory_interp[step] - self.data.qpos
            vel_error = desired_vel - self.data.qvel
            ctrl = np.clip(kp * pos_error + kd * vel_error, -100.0, 100.0)
            self.data.ctrl[:] = ctrl[: self.model.nu]

            mujoco.mj_step(self.model, self.data)

            velocities[step] = self.data.qvel.copy()
            controls[step] = self.data.ctrl.copy()

            if self.club_head_id is not None:
                club_speeds.append(
                    self._compute_club_speed(
                        jacp,
                        jacr,
                        jacp_flat,
                        jacr_flat,
                        use_flat_jac,
                    )
                )
                club_positions.append(self.data.xpos[self.club_head_id].copy())

        metrics = self._collect_simulation_metrics(
            club_speeds, club_positions, controls, velocities
        )
        return velocities, controls, metrics

    def _compute_jerk(self, trajectory: np.ndarray) -> float:
        """Compute total jerk (third derivative of position).

        Args:
            trajectory: Joint trajectory [num_knots x nv]

        Returns:
            Total jerk magnitude
        """
        if trajectory is None:
            raise ValueError("trajectory must be provided")
        dt = self.swing_duration / (self.num_knot_points - 1)

        # Second derivative (acceleration)
        accel = np.diff(trajectory, n=2, axis=0) / dt**2

        # Third derivative (jerk)
        jerk = np.diff(accel, axis=0) / dt

        return float(np.sum(np.abs(jerk)))

    def optimize_swing_for_speed(
        self,
        target_speed: float = 50.0,
    ) -> OptimizationResult:
        """Optimize swing specifically for maximum club head speed.

        Args:
            target_speed: Target club head speed [m/s]

        Returns:
            OptimizationResult with speed-optimized trajectory
        """
        if target_speed is None:
            raise ValueError("target_speed must be provided")
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
        if target_position is None:
            raise ValueError("target_position must be provided")
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
        variation: str = "speed",
    ) -> list[OptimizationResult]:
        """Generate a library of different swing styles.

        Args:
            num_swings: Number of swings to generate
            variation: Type of variation

        Returns:
            List of OptimizationResult for different swings
        """
        if num_swings is None:
            raise ValueError("num_swings must be provided")
        swings = []

        if variation == "speed":
            speeds = np.linspace(30.0, 55.0, num_swings)
            return [
                self.optimize_swing_for_speed(target_speed=float(speed))
                for speed in speeds
            ]

        elif variation == "accuracy":
            base_pos = np.array([2.0, 0.0, 0.0])
            return [
                self.optimize_swing_for_accuracy(
                    target_position=base_pos
                    + np.array([0, (i - num_swings / 2) * 0.2, 0])
                )
                for i in range(num_swings)
            ]

        return swings
