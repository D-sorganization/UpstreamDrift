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
        if not (target_speed is not None):
            raise ValueError("target_speed must be provided")
        if not (target_speed is not None):
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
        if not (target_position is not None):
            raise ValueError("target_position must be provided")
        if not (target_position is not None):
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
        if not (num_swings is not None):
            raise ValueError("num_swings must be provided")
        if not (num_swings is not None):
            raise ValueError("num_swings must be provided")
        swings = []

        if variation == "speed":
            speeds = np.linspace(30.0, 55.0, num_swings)
            for speed in speeds:
                result = self.optimize_swing_for_speed(target_speed=speed)
                swings.append(result)

        elif variation == "accuracy":
            base_pos = np.array([2.0, 0.0, 0.0])
            for i in range(num_swings):
                offset = np.array([0, (i - num_swings / 2) * 0.2, 0])
                target = base_pos + offset
                result = self.optimize_swing_for_accuracy(target_position=target)
                swings.append(result)

        return swings
