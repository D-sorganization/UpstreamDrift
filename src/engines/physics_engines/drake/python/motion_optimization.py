"""Motion Optimization for Drake Golf Engine.

This module provides motion optimization capabilities for Drake-based golf swing
simulations, matching the functionality available in the MuJoCo engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from shared.python.core import setup_logging
except ImportError as e:
    raise ImportError(
        "Failed to import shared modules. Ensure shared.python is in PYTHONPATH."
    ) from e

logger = setup_logging(__name__)

#: Default revolute joint limit used by the standard joint-limit constraint
#: when the trajectory carries joint angles (rad). Matches the canonical
#: humanoid URDFs' ``[-pi, pi]`` revolute range (#7052).
DEFAULT_JOINT_LIMIT_RAD: float = float(np.pi)

#: Canonical fraction of the swing at which impact occurs (peak clubhead
#: speed). Used by the impact-timing constraint (#7052).
IMPACT_FRACTION: float = 0.9

#: Standard gravity (m/s^2) for the ballistic carry-distance estimate (#7052).
_GRAVITY_M_S2: float = 9.81


@dataclass
class OptimizationObjective:
    """Defines an optimization objective for golf swing motion."""

    name: str
    weight: float
    target_value: float | None = None
    cost_function: Callable[[np.ndarray], float] | None = None


@dataclass
class OptimizationConstraint:
    """Defines a constraint for golf swing optimization."""

    name: str
    constraint_type: str  # 'equality', 'inequality', 'bounds'
    lower_bound: float | None = None
    upper_bound: float | None = None
    constraint_function: Callable[[np.ndarray], float] | None = None


@dataclass
class OptimizationResult:
    """Results from golf swing motion optimization."""

    success: bool
    optimal_trajectory: np.ndarray
    optimal_cost: float
    iterations: int
    convergence_message: str
    objective_values: dict[str, float]
    constraint_violations: dict[str, float]


class DrakeMotionOptimizer:
    """Motion optimization for Drake golf swing simulations."""

    def __init__(self) -> None:
        """Initialize the Drake motion optimizer."""
        self.logger = logger
        self.objectives: list[OptimizationObjective] = []
        self.constraints: list[OptimizationConstraint] = []

    def add_objective(
        self,
        name: str,
        weight: float,
        cost_function: Callable[[np.ndarray], float],
        target_value: float | None = None,
    ) -> None:
        """Add an optimization objective.

        Args:
            name: Name of the objective
            weight: Weight in the total cost function
            cost_function: Function that computes cost from trajectory
            target_value: Optional target value for the objective
        """
        if name is None:
            raise ValueError("name must be provided")
        objective = OptimizationObjective(
            name=name,
            weight=weight,
            target_value=target_value,
            cost_function=cost_function,
        )
        self.objectives.append(objective)
        self.logger.info(f"Added optimization objective: {name} (weight={weight})")

    def add_constraint(
        self,
        name: str,
        constraint_type: str,
        constraint_function: Callable[[np.ndarray], float],
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ) -> None:
        """Add an optimization constraint.

        Args:
            name: Name of the constraint
            constraint_type: Type of constraint ('equality', 'inequality', 'bounds')
            constraint_function: Function that evaluates the constraint
            lower_bound: Lower bound for inequality/bounds constraints
            upper_bound: Upper bound for inequality/bounds constraints
        """
        if name is None:
            raise ValueError("name must be provided")
        constraint = OptimizationConstraint(
            name=name,
            constraint_type=constraint_type,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            constraint_function=constraint_function,
        )
        self.constraints.append(constraint)
        self.logger.info(f"Added optimization constraint: {name} ({constraint_type})")

    def setup_standard_golf_objectives(self) -> None:
        """Set up standard golf swing optimization objectives."""

        # Ball speed objective
        def ball_speed_cost(trajectory: np.ndarray) -> float:
            """Negative peak inter-sample speed (minimized → maximizes speed)."""
            # ⚡ Bolt: Computing max of np.einsum first, then sqrt,
            # is much faster than np.max(np.linalg.norm(..., axis=1))
            diff = np.diff(trajectory, axis=0)
            return float(-np.sqrt(np.max(np.einsum("ij,ij->i", diff, diff))))

        self.add_objective(
            name="ball_speed",
            weight=1.0,
            cost_function=ball_speed_cost,
            target_value=45.0,  # m/s target ball speed
        )

        # Accuracy objective (minimize lateral deviation)
        def accuracy_cost(trajectory: np.ndarray) -> float:
            """Lateral (y-axis) deviation of the final position from the line."""
            final_position = trajectory[-1]
            return float(abs(final_position[1]))  # y-deviation from target line

        self.add_objective(
            name="accuracy", weight=0.8, cost_function=accuracy_cost, target_value=0.0
        )

        # Smoothness objective
        def smoothness_cost(trajectory: np.ndarray) -> float:
            """Compute trajectory jerk as a smoothness penalty."""
            # Compute trajectory smoothness via second derivatives
            if len(trajectory) < 3:
                return 0.0
            second_derivatives = np.diff(trajectory, n=2, axis=0)
            # ⚡ Bolt: np.sqrt(np.einsum(...)) avoids temp arrays,
            # and is much faster than np.linalg.norm(..., axis=1)
            return float(
                np.sum(
                    np.sqrt(
                        np.einsum("ij,ij->i", second_derivatives, second_derivatives)
                    )
                )
            )

        self.add_objective(name="smoothness", weight=0.5, cost_function=smoothness_cost)

    def setup_standard_golf_constraints(self) -> None:
        """Set up standard golf swing optimization constraints."""

        # Joint angle limits (#7052). The trajectory is a (N, n_joints) array
        # of joint angles (rad); the constraint returns the total magnitude by
        # which any sample exceeds the default revolute limits [-pi, pi]. A
        # value of 0 means every joint stays in range; the optimizer treats
        # ``upper_bound=0`` as "no violation allowed".
        def joint_angle_constraint(trajectory: np.ndarray) -> float:
            """Return the summed joint-angle limit violation (rad)."""
            traj = np.asarray(trajectory, dtype=float)
            over = np.clip(np.abs(traj) - DEFAULT_JOINT_LIMIT_RAD, 0.0, None)
            return float(np.sum(over))

        self.add_constraint(
            name="joint_limits",
            constraint_type="inequality",
            constraint_function=joint_angle_constraint,
            upper_bound=0.0,
        )

        # Impact-timing constraint (#7052). Impact is defined as the frame of
        # peak inter-sample speed; this returns the (signed) fractional
        # deviation of that frame from the canonical impact fraction so the
        # equality solver can drive it to zero.
        def impact_timing_constraint(trajectory: np.ndarray) -> float:
            """Return peak-speed-frame deviation from the impact fraction."""
            return self._impact_timing_deviation(trajectory, IMPACT_FRACTION)

        self.add_constraint(
            name="impact_timing",
            constraint_type="equality",
            constraint_function=impact_timing_constraint,
        )

    @staticmethod
    def _impact_timing_deviation(
        trajectory: np.ndarray, impact_fraction: float
    ) -> float:
        """Signed deviation of the peak-speed frame from ``impact_fraction``.

        Impact is the frame of maximum inter-sample speed. The returned value
        is ``peak_fraction - impact_fraction`` in ``[-impact, 1-impact]``; an
        equality solver drives it to zero. Degenerate trajectories (< 2
        samples) report no deviation.
        """
        traj = np.asarray(trajectory, dtype=float)
        n = traj.shape[0]
        if n < 2:
            return 0.0
        diff = np.diff(traj, axis=0)
        speeds = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        if not np.any(speeds > 0.0):
            return 0.0
        peak_idx = int(np.argmax(speeds))
        # diff[k] spans samples k..k+1; place the event at the later sample.
        peak_fraction = float(peak_idx + 1) / float(n - 1)
        return peak_fraction - impact_fraction

    @staticmethod
    def _ballistic_carry_distance(trajectory: np.ndarray) -> float:
        """Flat-ground projectile carry from the final launch velocity (m).

        Uses the last inter-sample displacement of a ``(N, 3)`` world-position
        trajectory as the (unit-time) launch velocity ``[vx, vy, vz]`` and
        returns the range ``2*vx*vz/g`` for the vertical-plane components.
        Returns 0 for fewer than two samples or a non-positive launch.
        """
        traj = np.asarray(trajectory, dtype=float)
        if traj.shape[0] < 2 or traj.shape[1] < 3:
            return 0.0
        launch = traj[-1] - traj[-2]
        v_x = float(launch[0])
        v_z = float(launch[2])
        if v_x <= 0.0 or v_z <= 0.0:
            return 0.0
        return 2.0 * v_x * v_z / _GRAVITY_M_S2

    def _build_total_cost_function(
        self, traj_shape: tuple
    ) -> Callable[[np.ndarray], float]:  # noqa: E501
        def total_cost(x: np.ndarray) -> float:
            """Combined weighted objective function."""
            traj = x.reshape(traj_shape)
            cost = 0.0
            for obj in self.objectives:
                if obj.cost_function is not None:
                    cost += obj.weight * obj.cost_function(traj)
            return cost

        return total_cost

    def _build_scipy_constraints(self, traj_shape: tuple) -> list[dict]:
        if traj_shape is None:
            raise ValueError("traj_shape must be provided")
        scipy_constraints: list[dict] = []
        for con in self.constraints:
            if con.constraint_function is None:
                continue
            if con.constraint_type == "equality":
                scipy_constraints.append(
                    {
                        "type": "eq",
                        "fun": lambda x, c=con: c.constraint_function(
                            x.reshape(traj_shape)
                        ),  # noqa: E501
                    }
                )
            elif con.constraint_type == "inequality":
                if con.upper_bound is not None:
                    scipy_constraints.append(
                        {
                            "type": "ineq",
                            "fun": lambda x, c=con: (
                                c.upper_bound
                                - c.constraint_function(
                                    x.reshape(traj_shape)
                                )  # noqa: E501
                            ),
                        }
                    )
                if con.lower_bound is not None:
                    scipy_constraints.append(
                        {
                            "type": "ineq",
                            "fun": lambda x, c=con: (
                                c.constraint_function(x.reshape(traj_shape))
                                - c.lower_bound  # noqa: E501
                            ),
                        }
                    )
        return scipy_constraints

    def _evaluate_objectives(self, optimal_trajectory: np.ndarray) -> dict[str, float]:
        if optimal_trajectory is None:
            raise ValueError("optimal_trajectory must be provided")
        objective_values = {}
        for obj in self.objectives:
            if obj.cost_function is not None:
                objective_values[obj.name] = obj.cost_function(optimal_trajectory)
        return objective_values

    def _evaluate_constraint_violations(
        self, optimal_trajectory: np.ndarray, tolerance: float
    ) -> tuple[dict[str, float], bool]:
        if optimal_trajectory is None:
            raise ValueError("optimal_trajectory must be provided")
        constraint_violations = {}
        all_satisfied = True
        for con in self.constraints:
            if con.constraint_function is not None:
                val = con.constraint_function(optimal_trajectory)
                violation = 0.0
                if con.lower_bound is not None and val < con.lower_bound:
                    violation = con.lower_bound - val
                elif con.upper_bound is not None and val > con.upper_bound:
                    violation = val - con.upper_bound
                constraint_violations[con.name] = violation
                if violation > tolerance:
                    all_satisfied = False
        return constraint_violations, all_satisfied

    def _build_optimization_result(
        self,
        opt_result: Any,
        optimal_trajectory: np.ndarray,
        objective_values: dict[str, float],
        constraint_violations: dict[str, float],
        all_satisfied: bool,
    ) -> OptimizationResult:
        success = opt_result.success and all_satisfied
        return OptimizationResult(
            success=success,
            optimal_trajectory=optimal_trajectory,
            optimal_cost=float(opt_result.fun),
            iterations=opt_result.nit if hasattr(opt_result, "nit") else 0,
            convergence_message=(
                opt_result.message
                if hasattr(opt_result, "message")
                else str(opt_result.get("message", "Unknown"))
            ),
            objective_values=objective_values,
            constraint_violations=constraint_violations,
        )

    def optimize_trajectory(
        self,
        initial_trajectory: np.ndarray,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> OptimizationResult:
        """Optimize golf swing trajectory using SLSQP.

        Formulates the trajectory optimization as a nonlinear program:

        .. math::
            \\min_{x} \\sum_i w_i \\cdot f_i(x)

        subject to:

        .. math::
            g_j(x) \\leq 0 \\quad \\text{(inequality constraints)}
            h_k(x) = 0 \\quad \\text{(equality constraints)}

        where :math:`x` is the flattened trajectory, :math:`f_i` are the
        weighted objective functions, and :math:`g_j, h_k` are constraint
        functions.

        Args:
            initial_trajectory: Initial guess for trajectory (N, dim)
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance

        Returns:
            OptimizationResult with optimization results
        """
        if initial_trajectory is None:
            raise ValueError("initial_trajectory must be provided")
        from scipy.optimize import minimize as scipy_minimize

        self.logger.info(
            f"Starting trajectory optimization with {len(self.objectives)} objectives "
            f"and {len(self.constraints)} constraints"
        )

        traj_shape = initial_trajectory.shape
        x0 = initial_trajectory.flatten()

        total_cost = self._build_total_cost_function(traj_shape)
        scipy_constraints = self._build_scipy_constraints(traj_shape)

        opt_result = scipy_minimize(
            total_cost,
            x0,
            method="SLSQP",
            constraints=scipy_constraints,
            options={
                "maxiter": max_iterations,
                "ftol": tolerance,
                "disp": False,
            },
        )

        optimal_trajectory = opt_result.x.reshape(traj_shape)
        objective_values = self._evaluate_objectives(optimal_trajectory)
        constraint_violations, all_satisfied = self._evaluate_constraint_violations(
            optimal_trajectory, tolerance
        )

        result = self._build_optimization_result(
            opt_result,
            optimal_trajectory,
            objective_values,
            constraint_violations,
            all_satisfied,
        )

        self.logger.info(
            f"Trajectory optimization complete. Cost: {result.optimal_cost:.4f}. "
            f"Success: {result.success}. Iterations: {result.iterations}"
        )

        return result

    def optimize_for_distance(
        self, initial_trajectory: np.ndarray, target_distance: float = 250.0
    ) -> OptimizationResult:
        """Optimize trajectory for maximum distance.

        Args:
            initial_trajectory: Initial trajectory guess
            target_distance: Target carry distance (meters)

        Returns:
            OptimizationResult optimized for distance
        """
        # Clear existing objectives and add distance-specific ones
        if initial_trajectory is None:
            raise ValueError("initial_trajectory must be provided")
        self.objectives.clear()

        def distance_cost(trajectory: np.ndarray) -> float:
            """Negative ballistic carry distance (minimized → maximizes carry).

            The launch velocity is the final inter-sample displacement of the
            clubhead/ball trajectory (``(N, 3)`` world positions). Carry is the
            flat-ground projectile range ``2*vx*vz/g`` using the launch's
            horizontal (x) and vertical (z) components. Returned negative so
            SLSQP *maximizes* carry; degenerate trajectories cost 0.
            """
            return -self._ballistic_carry_distance(trajectory)

        self.add_objective(
            name="carry_distance",
            weight=1.0,
            cost_function=distance_cost,
            target_value=target_distance,
        )

        return self.optimize_trajectory(initial_trajectory)

    def optimize_for_accuracy(
        self, initial_trajectory: np.ndarray, target_point: np.ndarray
    ) -> OptimizationResult:
        """Optimize trajectory for accuracy to target.

        Args:
            initial_trajectory: Initial trajectory guess
            target_point: Target point (x, y, z) coordinates

        Returns:
            OptimizationResult optimized for accuracy
        """
        # Clear existing objectives and add accuracy-specific ones
        if initial_trajectory is None:
            raise ValueError("initial_trajectory must be provided")
        self.objectives.clear()

        def accuracy_cost(trajectory: np.ndarray) -> float:
            """Euclidean distance from the final position to the target point."""
            final_position = trajectory[-1]
            # ⚡ Bolt: np.dot is faster than np.linalg.norm for small 1D arrays
            diff = final_position - target_point
            return float(np.sqrt(np.dot(diff, diff)))

        self.add_objective(
            name="target_accuracy",
            weight=1.0,
            cost_function=accuracy_cost,
            target_value=0.0,
        )

        return self.optimize_trajectory(initial_trajectory)

    def export_optimization_results(
        self, result: OptimizationResult, output_path: str
    ) -> None:  # noqa: E501
        """Export optimization results for analysis.

        Args:
            result: Optimization results to export
            output_path: Path to save results
        """
        if result is None:
            raise ValueError("result must be provided")
        import json
        from pathlib import Path

        export_data = {
            "optimization_result": {
                "success": result.success,
                "optimal_cost": result.optimal_cost,
                "iterations": result.iterations,
                "convergence_message": result.convergence_message,
                "objective_values": result.objective_values,
                "constraint_violations": result.constraint_violations,
            },
            "trajectory": {
                "positions": result.optimal_trajectory.tolist(),
                "num_points": len(result.optimal_trajectory),
            },
            "engine": "drake",
            "optimization_setup": {
                "num_objectives": len(self.objectives),
                "num_constraints": len(self.constraints),
                "objective_names": [obj.name for obj in self.objectives],
                "constraint_names": [con.name for con in self.constraints],
            },
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Optimization results exported to {output_path}")
