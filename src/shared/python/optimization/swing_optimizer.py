# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
Swing Optimizer Module

Multi-objective trajectory optimization for the golf swing using forward dynamics.
This is the core differentiating feature - generating optimal swings rather than
just analyzing existing ones.

Approach:
1. Define the golfer model (anthropometrics, strength limits, flexibility)
2. Define optimization objectives (clubhead speed, accuracy, injury risk)
3. Define constraints (joint limits, force limits, kinematic feasibility)
4. Solve trajectory optimization using direct collocation
5. Return optimal joint trajectories and predicted outcomes

This uses the Drake engine for trajectory optimization when available,
falling back to scipy.optimize for simpler optimization problems.

References:
- Sharp (2009) Kinetic Constrained Optimization of the Golf Swing Hub Path
- Nesbit & Serrano (2005) Work and Power Analysis of the Golf Swing
- MacKenzie (2012) Understanding the role of shaft stiffness
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import optimize

from src.shared.python.core.contracts import (
    ContractChecker,
    invariant,
    postcondition,
    precondition,
)
from src.shared.python.optimization._swing_constraints import build_constraints
from src.shared.python.optimization._swing_kinematics import (
    generate_initial_guess,
    get_bounds,
    trajectory_to_vector,
    vector_to_trajectory,
)
from src.shared.python.optimization._swing_models import (
    ClubModel,
    GolferModel,
    OptimizationConfig,
    OptimizationConstraint,
    OptimizationObjective,
    OptimizationResult,
    SwingTrajectory,
)
from src.shared.python.optimization._swing_objectives import (
    compute_metrics,
    compute_objective,
)

__all__ = [
    "ClubModel",
    "GolferModel",
    "OptimizationConfig",
    "OptimizationConstraint",
    "OptimizationObjective",
    "OptimizationResult",
    "SwingOptimizer",
    "SwingTrajectory",
    "create_example_optimization",
]


@invariant(
    lambda self: self.config.n_nodes >= 2,
    "Optimization must have at least 2 collocation nodes",
)
@invariant(
    lambda self: self.config.swing_duration > 0,
    "Swing duration must be positive",
)
class SwingOptimizer(ContractChecker):
    """
    Multi-objective swing trajectory optimizer.

    This optimizer uses forward dynamics to find optimal swing trajectories
    that maximize performance while respecting biomechanical constraints.
    It can optimize for multiple objectives simultaneously (Pareto optimization).

    Design by Contract:
        Invariants:
            - golfer model has positive mass and height
            - club model has positive total length
            - config has at least one objective
            - joint_limits and torque_limits cover all joints

    Example:
        >>> golfer = GolferModel(height=1.80, mass=80.0)
        >>> club = ClubModel(total_length=1.15)
        >>> config = OptimizationConfig(objectives={
        ...     OptimizationObjective.CLUBHEAD_VELOCITY: 1.0,
        ...     OptimizationObjective.INJURY_RISK: 0.5,
        ... })
        >>> optimizer = SwingOptimizer(golfer, club, config)
        >>> result = optimizer.optimize()
        >>> print(f"Optimal clubhead speed: {result.predicted_clubhead_speed:.1f} m/s")
    """

    JOINTS = [
        "hip_rotation",
        "trunk_rotation",
        "shoulder_horizontal",
        "shoulder_vertical",
        "elbow_flexion",
        "wrist_cock",
        "wrist_rotation",
    ]

    def __init__(
        self,
        golfer: GolferModel,
        club: ClubModel,
        config: OptimizationConfig | None = None,
    ) -> None:
        """
        Initialize the swing optimizer.

        Args:
            golfer: Golfer physical model
            club: Golf club model
            config: Optimization configuration (uses defaults if not provided)
        """
        if golfer is None:
            raise ValueError("golfer must be provided")
        self.golfer = golfer
        self.club = club
        self.config = config or OptimizationConfig()

        self._setup_model()

    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        """Define class invariants for SwingOptimizer."""
        return [
            (
                lambda: self.golfer.mass > 0 and self.golfer.height > 0,
                "Golfer model must have positive mass and height",
            ),
            (
                lambda: self.club.total_length > 0,
                "Club model must have positive total length",
            ),
            (
                lambda: len(self.config.objectives) > 0,
                "Optimization config must have at least one objective",
            ),
            (
                lambda: all(j in self.joint_limits for j in self.JOINTS),
                "joint_limits must cover all joints",
            ),
            (
                lambda: all(j in self.torque_limits for j in self.JOINTS),
                "torque_limits must cover all joints",
            ),
        ]

    def _setup_model(self) -> None:
        """Set up the biomechanical model parameters."""
        self.total_lever = self.golfer.arm_length + self.club.total_length

        arm_mass = self.golfer.mass * self.golfer.arm_mass_ratio
        self.system_moi = (
            arm_mass * self.golfer.arm_length**2 / 3
            + self.club.club_moi
            + self.club.total_mass * self.total_lever**2
        )

        self.joint_limits = {
            "hip_rotation": self.golfer.hip_rom,
            "trunk_rotation": self.golfer.trunk_rotation_rom,
            "shoulder_horizontal": self.golfer.shoulder_rom,
            "shoulder_vertical": (-1.5, 1.5),
            "elbow_flexion": self.golfer.elbow_rom,
            "wrist_cock": self.golfer.wrist_rom,
            "wrist_rotation": (-1.0, 1.0),
        }

        self.torque_limits = {
            "hip_rotation": self.golfer.max_hip_torque,
            "trunk_rotation": self.golfer.max_trunk_torque,
            "shoulder_horizontal": self.golfer.max_shoulder_torque,
            "shoulder_vertical": self.golfer.max_shoulder_torque,
            "elbow_flexion": self.golfer.max_elbow_torque,
            "wrist_cock": self.golfer.max_wrist_torque,
            "wrist_rotation": self.golfer.max_wrist_torque,
        }

    @postcondition(
        lambda result: result is not None and isinstance(result.success, bool),
        "Optimization must return a valid result with success status",
    )
    def optimize(
        self,
        initial_swing: SwingTrajectory | None = None,
        callback: Callable[[int, float], None] | None = None,
    ) -> OptimizationResult:
        """
        Run the optimization to find optimal swing trajectory.

        Args:
            initial_swing: Optional initial swing to start from (warm start)
            callback: Optional callback function(iteration, objective_value)

        Returns:
            OptimizationResult with optimal trajectory and metrics
        """
        import time

        start_time = time.time()

        x0 = self._prepare_initial_guess(initial_swing)
        result, iteration_count = self._run_scipy_optimization(x0, callback)

        computation_time = time.time() - start_time

        if result.success:
            return self._build_success_result(result, iteration_count, computation_time)
        return self._build_failure_result(result, iteration_count, computation_time)

    def _prepare_initial_guess(
        self, initial_swing: SwingTrajectory | None
    ) -> np.ndarray:
        """Build the initial decision-variable vector."""
        if initial_swing is not None:
            return trajectory_to_vector(initial_swing)
        return generate_initial_guess(self.golfer, self.config, self.joint_limits)

    def _run_scipy_optimization(
        self,
        x0: np.ndarray,
        callback: Callable[[int, float], None] | None,
    ) -> tuple[Any, int]:
        """Execute the scipy minimization and return raw result + iterations."""
        if x0 is None:
            raise ValueError("x0 must be provided")
        bounds = get_bounds(self.golfer, self.config, self.joint_limits)
        constraints = build_constraints(
            self.config,
            self.golfer,
            self.club,
            self.torque_limits,
            self.system_moi,
        )

        def objective(x: np.ndarray) -> float:
            """Evaluate the optimization objective for a given parameter vector."""
            return compute_objective(
                x,
                self.config,
                self.torque_limits,
                self.golfer,
                self.club,
                self.system_moi,
            )

        iteration_count = [0]

        def scipy_callback(xk: np.ndarray) -> None:
            """Increment iteration count and invoke the user callback."""
            iteration_count[0] += 1
            if callback:
                obj_val = objective(xk)
                callback(iteration_count[0], obj_val)

        result = optimize.minimize(
            objective,
            x0,
            method=self.config.solver,
            bounds=bounds,
            constraints=constraints,
            callback=scipy_callback,
            options={
                "maxiter": self.config.max_iterations,
                "ftol": self.config.tolerance,
            },
        )
        return result, iteration_count[0]

    def _build_success_result(
        self,
        result: Any,
        iterations: int,
        computation_time: float,
    ) -> OptimizationResult:
        """Extract trajectory and metrics from a successful optimization."""
        if iterations is None:
            raise ValueError("iterations must be provided")
        trajectory = vector_to_trajectory(
            result.x, self.config, self.golfer, self.club, self.system_moi
        )
        metrics = compute_metrics(trajectory, self.club, self.torque_limits)

        return OptimizationResult(
            success=True,
            message=result.message,
            trajectory=trajectory,
            predicted_clubhead_speed=metrics["clubhead_speed"],
            predicted_ball_speed=metrics["ball_speed"],
            predicted_carry_distance=metrics["carry_distance"],
            predicted_launch_angle=metrics["launch_angle"],
            predicted_spin_rate=metrics["spin_rate"],
            peak_spinal_compression=metrics["spinal_compression"],
            peak_spinal_shear=metrics["spinal_shear"],
            injury_risk_score=metrics["injury_risk"],
            objective_value=result.fun,
            iterations=iterations,
            computation_time=computation_time,
        )

    @staticmethod
    def _build_failure_result(
        result: Any,
        iterations: int,
        computation_time: float,
    ) -> OptimizationResult:
        """Build an OptimizationResult for a failed optimization."""
        return OptimizationResult(
            success=False,
            message=f"Optimization failed: {result.message}",
            iterations=iterations,
            computation_time=computation_time,
        )

    @precondition(
        lambda self, n_points=10: n_points > 0,
        "Number of Pareto points must be positive",
    )
    @postcondition(
        lambda result: (
            result is not None and isinstance(result, list) and len(result) > 0
        ),
        "Pareto optimization must return at least one result",
    )
    def optimize_pareto(
        self,
        n_points: int = 10,
    ) -> list[OptimizationResult]:
        """
        Generate Pareto-optimal solutions for multi-objective optimization.

        Args:
            n_points: Number of Pareto points to generate

        Returns:
            List of OptimizationResults representing the Pareto frontier
        """
        if n_points is None:
            raise ValueError("n_points must be provided")
        results = []

        objectives = list(self.config.objectives.keys())
        if len(objectives) < 2:
            return [self.optimize()]

        weights = np.linspace(0, 1, n_points)

        original_weights = self.config.objectives.copy()

        for w in weights:
            self.config.objectives[objectives[0]] = w
            self.config.objectives[objectives[1]] = 1 - w

            result = self.optimize()
            results.append(result)

        self.config.objectives = original_weights

        return results

    def _compute_objective(self, x: np.ndarray) -> float:
        """Compute the weighted objective function."""
        return compute_objective(
            x, self.config, self.torque_limits, self.golfer, self.club, self.system_moi
        )

    def _compute_injury_risk(self, trajectory: SwingTrajectory) -> float:
        """Compute simplified injury risk score (0-100)."""
        from src.shared.python.optimization._swing_objectives import compute_injury_risk

        return compute_injury_risk(trajectory, self.torque_limits)

    def _compute_energy_cost(self, trajectory: SwingTrajectory) -> float:
        """Compute metabolic energy cost of the swing."""
        from src.shared.python.optimization._swing_objectives import compute_energy_cost

        return compute_energy_cost(trajectory)

    def _compute_metrics(self, trajectory: SwingTrajectory) -> dict:
        """Compute all metrics for a trajectory."""
        return compute_metrics(trajectory, self.club, self.torque_limits)

    def _generate_initial_guess(self) -> np.ndarray:
        """Generate an initial guess for the optimization."""
        return generate_initial_guess(self.golfer, self.config, self.joint_limits)

    def _trajectory_to_vector(self, trajectory: SwingTrajectory) -> np.ndarray:
        """Convert a SwingTrajectory to optimization vector."""
        return trajectory_to_vector(trajectory)

    def _vector_to_trajectory(self, x: np.ndarray) -> SwingTrajectory:
        """Convert optimization vector to SwingTrajectory."""
        return vector_to_trajectory(
            x, self.config, self.golfer, self.club, self.system_moi
        )

    def _compute_clubhead_trajectory(
        self,
        joint_angles: dict[str, np.ndarray],
        time: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute clubhead position and velocity from joint angles."""
        from src.shared.python.optimization._swing_kinematics import (
            compute_clubhead_trajectory,
        )

        return compute_clubhead_trajectory(joint_angles, time, self.golfer, self.club)

    def _get_bounds(self) -> list[tuple[float, float]]:
        """Get optimization bounds for all variables."""
        return get_bounds(self.golfer, self.config, self.joint_limits)

    def _build_constraints(self) -> list[dict]:
        """Build scipy constraint dictionaries."""
        return build_constraints(
            self.config,
            self.golfer,
            self.club,
            self.torque_limits,
            self.system_moi,
        )

    def _torque_constraint(self, x: np.ndarray) -> np.ndarray:
        """Constraint: torques must be within limits."""
        from src.shared.python.optimization._swing_constraints import torque_constraint

        return torque_constraint(
            x, self.config, self.golfer, self.club, self.torque_limits, self.system_moi
        )

    def _kinematic_sequence_constraint(self, x: np.ndarray) -> np.ndarray:
        """Constraint: enforce proximal-to-distal sequencing."""
        from src.shared.python.optimization._swing_constraints import (
            kinematic_sequence_constraint,
        )

        return kinematic_sequence_constraint(
            x, self.config, self.golfer, self.club, self.system_moi
        )


def create_example_optimization() -> tuple[SwingOptimizer, OptimizationResult]:
    """Create an example optimization for testing and demonstration."""
    golfer = GolferModel(
        height=1.80,
        mass=80.0,
        arm_length=0.62,
        max_shoulder_torque=120.0,
        max_trunk_torque=250.0,
    )
    club = ClubModel(
        total_length=1.15,
        head_mass=0.20,
        loft_angle=10.5,
    )
    config = OptimizationConfig(
        objectives={
            OptimizationObjective.CLUBHEAD_VELOCITY: 1.0,
            OptimizationObjective.INJURY_RISK: 0.3,
        },
        constraints=[
            OptimizationConstraint.JOINT_LIMITS,
            OptimizationConstraint.TORQUE_LIMITS,
        ],
        n_nodes=30,
        max_iterations=100,
    )
    optimizer = SwingOptimizer(golfer, club, config)
    result = optimizer.optimize()
    return optimizer, result


if __name__ == "__main__":
    optimizer, result = create_example_optimization()
    if result.success:
        pass
