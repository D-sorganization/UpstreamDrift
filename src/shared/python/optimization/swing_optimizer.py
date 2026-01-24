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
from dataclasses import dataclass, field
from enum import Enum
from typing import cast

import numpy as np
from scipy import optimize

from src.shared.python.constants import GRAVITY_M_S2


class OptimizationObjective(Enum):
    """Available optimization objectives."""

    CLUBHEAD_VELOCITY = "clubhead_velocity"  # Maximize clubhead speed at impact
    BALL_DISTANCE = "ball_distance"  # Maximize carry distance
    ACCURACY = "accuracy"  # Minimize lateral deviation
    ENERGY_EFFICIENCY = "energy_efficiency"  # Minimize metabolic cost
    INJURY_RISK = "injury_risk"  # Minimize spinal/joint loading
    CONSISTENCY = "consistency"  # Minimize sensitivity to perturbations


class OptimizationConstraint(Enum):
    """Available optimization constraints."""

    JOINT_LIMITS = "joint_limits"  # Stay within ROM
    TORQUE_LIMITS = "torque_limits"  # Stay within strength limits
    VELOCITY_LIMITS = "velocity_limits"  # Max angular velocities
    CONTACT_CONSTRAINTS = "contact_constraints"  # Maintain ground contact
    KINEMATIC_CHAIN = "kinematic_chain"  # Proximal-to-distal sequencing


@dataclass
class GolferModel:
    """Model of a golfer's physical characteristics."""

    # Anthropometrics
    height: float = 1.75  # meters
    mass: float = 75.0  # kg
    arm_length: float = 0.60  # meters (shoulder to wrist)
    trunk_length: float = 0.50  # meters

    # Segment masses (as fraction of body mass)
    arm_mass_ratio: float = 0.05
    trunk_mass_ratio: float = 0.43

    # Joint limits (radians)
    shoulder_rom: tuple[float, float] = (-2.5, 2.5)
    elbow_rom: tuple[float, float] = (0.0, 2.4)
    wrist_rom: tuple[float, float] = (-1.2, 1.2)
    hip_rom: tuple[float, float] = (-0.8, 0.8)
    trunk_rotation_rom: tuple[float, float] = (-1.5, 1.5)

    # Strength limits (Nm)
    max_shoulder_torque: float = 100.0
    max_elbow_torque: float = 60.0
    max_wrist_torque: float = 20.0
    max_hip_torque: float = 150.0
    max_trunk_torque: float = 200.0

    # Flexibility (how much past "normal" ROM)
    flexibility_factor: float = 1.0  # 1.0 = average, 1.2 = very flexible


@dataclass
class ClubModel:
    """Model of a golf club."""

    # Club properties
    total_length: float = 1.15  # meters (driver)
    shaft_length: float = 1.05  # meters
    head_mass: float = 0.20  # kg
    shaft_mass: float = 0.07  # kg
    grip_mass: float = 0.05  # kg

    # Shaft properties
    shaft_flex: str = "regular"  # stiff, regular, senior, ladies
    kick_point: str = "mid"  # low, mid, high

    # Head properties
    loft_angle: float = 10.5  # degrees (driver)
    face_angle: float = 0.0  # degrees (square)
    lie_angle: float = 56.0  # degrees

    @property
    def total_mass(self) -> float:
        return self.head_mass + self.shaft_mass + self.grip_mass

    @property
    def club_moi(self) -> float:
        """Moment of inertia about grip end."""
        # Simplified calculation
        return (
            self.head_mass * self.total_length**2
            + self.shaft_mass * (self.shaft_length / 2) ** 2
        )


@dataclass
class OptimizationConfig:
    """Configuration for the optimization problem."""

    # Objectives and weights
    objectives: dict[OptimizationObjective, float] = field(
        default_factory=lambda: {
            OptimizationObjective.CLUBHEAD_VELOCITY: 1.0,
            OptimizationObjective.INJURY_RISK: 0.3,
        }
    )

    # Constraints
    constraints: list[OptimizationConstraint] = field(
        default_factory=lambda: [
            OptimizationConstraint.JOINT_LIMITS,
            OptimizationConstraint.TORQUE_LIMITS,
        ]
    )

    # Time discretization
    n_nodes: int = 50  # Number of collocation nodes
    swing_duration: float = 1.2  # Total swing time (seconds)
    backswing_fraction: float = 0.4  # Fraction of time in backswing

    # Solver settings
    max_iterations: int = 500
    tolerance: float = 1e-6
    solver: str = "SLSQP"  # scipy solver


@dataclass
class SwingTrajectory:
    """Represents a complete swing trajectory."""

    time: np.ndarray  # Time points
    joint_angles: dict[str, np.ndarray]  # Joint angle trajectories
    joint_velocities: dict[str, np.ndarray]  # Joint velocity trajectories
    joint_torques: dict[str, np.ndarray]  # Joint torque trajectories

    # Clubhead trajectory
    clubhead_position: np.ndarray  # [n_frames, 3] xyz positions
    clubhead_velocity: np.ndarray  # [n_frames, 3] xyz velocities

    # Key metrics
    impact_speed: float = 0.0  # m/s
    impact_time: float = 0.0  # s


@dataclass
class OptimizationResult:
    """Results from swing optimization."""

    # Success status
    success: bool
    message: str

    # Optimal trajectory
    trajectory: SwingTrajectory | None = None

    # Predicted outcomes
    predicted_clubhead_speed: float = 0.0  # m/s
    predicted_ball_speed: float = 0.0  # m/s
    predicted_carry_distance: float = 0.0  # meters
    predicted_launch_angle: float = 0.0  # degrees
    predicted_spin_rate: float = 0.0  # rpm

    # Injury risk metrics
    peak_spinal_compression: float = 0.0  # x body weight
    peak_spinal_shear: float = 0.0  # x body weight
    injury_risk_score: float = 0.0  # 0-100

    # Optimization metrics
    objective_value: float = 0.0
    iterations: int = 0
    computation_time: float = 0.0  # seconds

    # Comparison to input swing (if provided)
    speed_improvement: float = 0.0  # m/s
    risk_reduction: float = 0.0  # percentage


class SwingOptimizer:
    """
    Multi-objective swing trajectory optimizer.

    This optimizer uses forward dynamics to find optimal swing trajectories
    that maximize performance while respecting biomechanical constraints.
    It can optimize for multiple objectives simultaneously (Pareto optimization).

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

    # Joint names in the model
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
        self.golfer = golfer
        self.club = club
        self.config = config or OptimizationConfig()

        # Derived quantities
        self._setup_model()

    def _setup_model(self) -> None:
        """Set up the biomechanical model parameters."""
        # Arm + club length
        self.total_lever = self.golfer.arm_length + self.club.total_length

        # Combined moment of inertia (simplified 2D model)
        arm_mass = self.golfer.mass * self.golfer.arm_mass_ratio
        self.system_moi = (
            arm_mass * self.golfer.arm_length**2 / 3  # Arm about shoulder
            + self.club.club_moi  # Club about wrist
            + self.club.total_mass * self.total_lever**2  # Club about shoulder
        )

        # Joint limit arrays
        self.joint_limits = {
            "hip_rotation": self.golfer.hip_rom,
            "trunk_rotation": self.golfer.trunk_rotation_rom,
            "shoulder_horizontal": self.golfer.shoulder_rom,
            "shoulder_vertical": (-1.5, 1.5),
            "elbow_flexion": self.golfer.elbow_rom,
            "wrist_cock": self.golfer.wrist_rom,
            "wrist_rotation": (-1.0, 1.0),
        }

        # Torque limit arrays
        self.torque_limits = {
            "hip_rotation": self.golfer.max_hip_torque,
            "trunk_rotation": self.golfer.max_trunk_torque,
            "shoulder_horizontal": self.golfer.max_shoulder_torque,
            "shoulder_vertical": self.golfer.max_shoulder_torque,
            "elbow_flexion": self.golfer.max_elbow_torque,
            "wrist_cock": self.golfer.max_wrist_torque,
            "wrist_rotation": self.golfer.max_wrist_torque,
        }

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

        # Set up the optimization problem
        n_joints = len(self.JOINTS)
        n_nodes = self.config.n_nodes
        n_joints * n_nodes * 2  # angles and velocities

        # Initial guess
        if initial_swing is not None:
            x0 = self._trajectory_to_vector(initial_swing)
        else:
            x0 = self._generate_initial_guess()

        # Bounds
        bounds = self._get_bounds()

        # Constraints
        constraints = self._build_constraints()

        # Objective function
        def objective(x: np.ndarray) -> float:
            return self._compute_objective(x)

        # Run optimization
        iteration_count = [0]

        def scipy_callback(xk: np.ndarray) -> None:
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

        computation_time = time.time() - start_time

        # Extract results
        if result.success:
            trajectory = self._vector_to_trajectory(result.x)
            metrics = self._compute_metrics(trajectory)

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
                iterations=iteration_count[0],
                computation_time=computation_time,
            )
        else:
            return OptimizationResult(
                success=False,
                message=f"Optimization failed: {result.message}",
                iterations=iteration_count[0],
                computation_time=computation_time,
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
        results = []

        # Vary weights between objectives
        objectives = list(self.config.objectives.keys())
        if len(objectives) < 2:
            # Single objective, just return one solution
            return [self.optimize()]

        # Generate weight combinations
        weights = np.linspace(0, 1, n_points)

        original_weights = self.config.objectives.copy()

        for w in weights:
            # Interpolate weights
            self.config.objectives[objectives[0]] = w
            self.config.objectives[objectives[1]] = 1 - w

            result = self.optimize()
            results.append(result)

        # Restore original weights
        self.config.objectives = original_weights

        return results

    def _generate_initial_guess(self) -> np.ndarray:
        """Generate an initial guess for the optimization."""
        n_joints = len(self.JOINTS)
        n_nodes = self.config.n_nodes
        t = np.linspace(0, self.config.swing_duration, n_nodes)

        # Time of top of backswing
        t_top = self.config.swing_duration * self.config.backswing_fraction

        # Generate smooth trajectories for each joint
        angles = np.zeros((n_joints, n_nodes))
        velocities = np.zeros((n_joints, n_nodes))

        for i, joint in enumerate(self.JOINTS):
            lo, hi = self.joint_limits[joint]
            mid = (lo + hi) / 2
            amp = (hi - lo) / 4  # Use 1/4 of ROM

            # Sinusoidal pattern: go to backswing position, then through to finish
            for j, tj in enumerate(t):
                if tj <= t_top:
                    # Backswing phase
                    phase = np.pi * tj / t_top
                    angles[i, j] = mid + amp * np.sin(phase)
                else:
                    # Downswing phase
                    phase = np.pi * (tj - t_top) / (self.config.swing_duration - t_top)
                    angles[i, j] = mid + amp * np.sin(np.pi - phase)

            # Compute velocities from angles
            dt = t[1] - t[0]
            velocities[i, :] = np.gradient(angles[i, :], dt)

        # Flatten to vector
        x = np.concatenate([angles.flatten(), velocities.flatten()])
        return cast(np.ndarray, x)

    def _trajectory_to_vector(self, trajectory: SwingTrajectory) -> np.ndarray:
        """Convert a SwingTrajectory to optimization vector."""
        angles = np.array([trajectory.joint_angles[j] for j in self.JOINTS])
        velocities = np.array([trajectory.joint_velocities[j] for j in self.JOINTS])
        return np.concatenate([angles.flatten(), velocities.flatten()])

    def _vector_to_trajectory(self, x: np.ndarray) -> SwingTrajectory:
        """Convert optimization vector to SwingTrajectory."""
        n_joints = len(self.JOINTS)
        n_nodes = self.config.n_nodes

        angles = x[: n_joints * n_nodes].reshape(n_joints, n_nodes)
        velocities = x[n_joints * n_nodes :].reshape(n_joints, n_nodes)

        t = np.linspace(0, self.config.swing_duration, n_nodes)

        joint_angles = {self.JOINTS[i]: angles[i] for i in range(n_joints)}
        joint_velocities = {self.JOINTS[i]: velocities[i] for i in range(n_joints)}

        # Compute torques from dynamics (simplified)
        joint_torques = {}
        dt = t[1] - t[0]
        for i, joint in enumerate(self.JOINTS):
            accel = np.gradient(velocities[i], dt)
            # Simplified: torque = I * alpha
            joint_torques[joint] = self.system_moi * accel * 0.1

        # Compute clubhead trajectory
        clubhead_pos, clubhead_vel = self._compute_clubhead_trajectory(joint_angles, t)

        # Find impact (maximum clubhead velocity)
        speed = np.linalg.norm(clubhead_vel, axis=1)
        impact_idx = np.argmax(speed)

        return SwingTrajectory(
            time=t,
            joint_angles=joint_angles,
            joint_velocities=joint_velocities,
            joint_torques=joint_torques,
            clubhead_position=clubhead_pos,
            clubhead_velocity=clubhead_vel,
            impact_speed=speed[impact_idx],
            impact_time=t[impact_idx],
        )

    def _compute_clubhead_trajectory(
        self,
        joint_angles: dict[str, np.ndarray],
        time: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute clubhead position and velocity from joint angles."""
        n_frames = len(time)
        position = np.zeros((n_frames, 3))
        velocity = np.zeros((n_frames, 3))

        # Simplified 2D kinematic model (can be extended to full 3D)
        arm_length = self.golfer.arm_length
        club_length = self.club.total_length

        for i in range(n_frames):
            # Get relevant angles
            trunk_rot = joint_angles.get("trunk_rotation", np.zeros(n_frames))[i]
            shoulder_h = joint_angles.get("shoulder_horizontal", np.zeros(n_frames))[i]
            joint_angles.get("shoulder_vertical", np.zeros(n_frames))[i]
            joint_angles.get("elbow_flexion", np.zeros(n_frames))[i]
            wrist = joint_angles.get("wrist_cock", np.zeros(n_frames))[i]

            # Forward kinematics (simplified)
            total_angle = trunk_rot + shoulder_h + wrist

            # Position in swing plane
            position[i, 0] = (arm_length + club_length) * np.sin(
                total_angle
            )  # x (forward)
            position[i, 1] = 0  # y (lateral, simplified)
            position[i, 2] = (arm_length + club_length) * np.cos(
                total_angle
            ) - club_length  # z (vertical)

        # Compute velocity from position
        dt = time[1] - time[0] if len(time) > 1 else 0.001
        for dim in range(3):
            velocity[:, dim] = np.gradient(position[:, dim], dt)

        return position, velocity

    def _get_bounds(self) -> list[tuple[float, float]]:
        """Get optimization bounds for all variables."""
        bounds = []

        # Angle bounds
        for joint in self.JOINTS:
            lo, hi = self.joint_limits[joint]
            flex = self.golfer.flexibility_factor
            for _ in range(self.config.n_nodes):
                bounds.append((lo * flex, hi * flex))

        # Velocity bounds (generous limits)
        max_vel = 30.0  # rad/s (very fast)
        for _ in range(len(self.JOINTS) * self.config.n_nodes):
            bounds.append((-max_vel, max_vel))

        return bounds

    def _build_constraints(self) -> list[dict]:
        """Build scipy constraint dictionaries."""
        constraints = []

        if OptimizationConstraint.TORQUE_LIMITS in self.config.constraints:
            constraints.append(
                {"type": "ineq", "fun": lambda x: self._torque_constraint(x)}
            )

        if OptimizationConstraint.KINEMATIC_CHAIN in self.config.constraints:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x: self._kinematic_sequence_constraint(x),
                }
            )

        return constraints

    def _torque_constraint(self, x: np.ndarray) -> np.ndarray:
        """Constraint: torques must be within limits."""
        trajectory = self._vector_to_trajectory(x)
        violations = []

        for joint in self.JOINTS:
            torque = trajectory.joint_torques[joint]
            limit = self.torque_limits[joint]
            # Constraint: limit - |torque| >= 0
            violations.extend(limit - np.abs(torque))

        return np.array(violations)

    def _kinematic_sequence_constraint(self, x: np.ndarray) -> np.ndarray:
        """Constraint: enforce proximal-to-distal sequencing."""
        trajectory = self._vector_to_trajectory(x)

        # During downswing, peak velocities should occur in order:
        # hip -> trunk -> shoulder -> wrist
        sequence = [
            "hip_rotation",
            "trunk_rotation",
            "shoulder_horizontal",
            "wrist_cock",
        ]

        # Find time of peak velocity for each
        peak_times = []
        for joint in sequence:
            if joint in trajectory.joint_velocities:
                vel = np.abs(trajectory.joint_velocities[joint])
                peak_idx = np.argmax(vel)
                peak_times.append(trajectory.time[peak_idx])

        # Constraint: each peak should be after the previous
        # t_hip < t_trunk < t_shoulder < t_wrist
        violations = []
        for i in range(len(peak_times) - 1):
            # Constraint: t[i+1] - t[i] >= 0
            violations.append(peak_times[i + 1] - peak_times[i])

        return np.array(violations)

    def _compute_objective(self, x: np.ndarray) -> float:
        """Compute the weighted objective function."""
        trajectory = self._vector_to_trajectory(x)
        objective = 0.0

        # Clubhead velocity objective (maximize = minimize negative)
        if OptimizationObjective.CLUBHEAD_VELOCITY in self.config.objectives:
            weight = self.config.objectives[OptimizationObjective.CLUBHEAD_VELOCITY]
            speed = trajectory.impact_speed
            # Negative because we minimize, and we want to maximize speed
            objective -= weight * speed / 50.0  # Normalize to ~1

        # Injury risk objective (minimize)
        if OptimizationObjective.INJURY_RISK in self.config.objectives:
            weight = self.config.objectives[OptimizationObjective.INJURY_RISK]
            risk = self._compute_injury_risk(trajectory)
            objective += weight * risk / 100.0  # Normalize to ~1

        # Energy efficiency objective (minimize)
        if OptimizationObjective.ENERGY_EFFICIENCY in self.config.objectives:
            weight = self.config.objectives[OptimizationObjective.ENERGY_EFFICIENCY]
            energy = self._compute_energy_cost(trajectory)
            objective += weight * energy / 1000.0  # Normalize to ~1

        return objective

    def _compute_injury_risk(self, trajectory: SwingTrajectory) -> float:
        """Compute simplified injury risk score (0-100)."""
        risk = 0.0

        # Check joint velocities (high velocities = higher risk)
        for _joint, vel in trajectory.joint_velocities.items():
            max_vel = np.max(np.abs(vel))
            if max_vel > 20:  # rad/s
                risk += 10

        # Check torques
        for joint, torque in trajectory.joint_torques.items():
            max_torque = np.max(np.abs(torque))
            limit = self.torque_limits.get(joint, 100)
            if max_torque > 0.8 * limit:
                risk += 15

        # Trunk rotation risk (high rotation = higher spinal load)
        trunk_rot = trajectory.joint_angles.get("trunk_rotation", np.zeros(1))
        max_rotation = np.max(np.abs(trunk_rot))
        if max_rotation > 1.2:  # ~70 degrees
            risk += 20

        return min(risk, 100)

    def _compute_energy_cost(self, trajectory: SwingTrajectory) -> float:
        """Compute metabolic energy cost of the swing."""
        total_work = 0.0
        dt = (
            trajectory.time[1] - trajectory.time[0]
            if len(trajectory.time) > 1
            else 0.001
        )

        for joint in self.JOINTS:
            if (
                joint in trajectory.joint_torques
                and joint in trajectory.joint_velocities
            ):
                torque = trajectory.joint_torques[joint]
                velocity = trajectory.joint_velocities[joint]
                power = torque * velocity
                # Handle NumPy 2.0 deprecation of trapz
                if hasattr(np, "trapezoid"):
                    work = np.trapezoid(np.abs(power), dx=dt)
                else:
                    trapz_func = getattr(np, "trapz")  # noqa: B009
                    work = trapz_func(np.abs(power), dx=dt)
                total_work += work

        return total_work

    def _compute_metrics(self, trajectory: SwingTrajectory) -> dict:
        """Compute all metrics for a trajectory."""
        # Clubhead speed at impact
        clubhead_speed = trajectory.impact_speed

        # Ball speed (simplified: 1.5x clubhead speed for driver)
        smash_factor = 1.50 if self.club.loft_angle < 15 else 1.35
        ball_speed = clubhead_speed * smash_factor

        # Carry distance (simplified model)
        launch_angle = self.club.loft_angle  # Simplified
        carry = (ball_speed**2 * np.sin(2 * np.radians(launch_angle))) / GRAVITY_M_S2
        carry *= 0.9  # Air resistance reduction factor

        # Spin rate (simplified)
        spin_rate = 2500 if self.club.loft_angle < 15 else 6000

        # Injury metrics
        injury_risk = self._compute_injury_risk(trajectory)

        # Spinal loads (simplified)
        trunk_vel = trajectory.joint_velocities.get("trunk_rotation", np.zeros(1))
        max_trunk_vel = np.max(np.abs(trunk_vel))
        spinal_compression = 4.0 + max_trunk_vel * 0.2  # x body weight
        spinal_shear = 0.3 + max_trunk_vel * 0.05

        return {
            "clubhead_speed": clubhead_speed,
            "ball_speed": ball_speed,
            "carry_distance": carry,
            "launch_angle": launch_angle,
            "spin_rate": spin_rate,
            "spinal_compression": spinal_compression,
            "spinal_shear": spinal_shear,
            "injury_risk": injury_risk,
        }


def create_example_optimization() -> tuple[SwingOptimizer, OptimizationResult]:
    """Create an example optimization for testing and demonstration."""
    # Create golfer and club models
    golfer = GolferModel(
        height=1.80,
        mass=80.0,
        arm_length=0.62,
        max_shoulder_torque=120.0,
        max_trunk_torque=250.0,
    )

    club = ClubModel(
        total_length=1.15,  # Driver
        head_mass=0.20,
        loft_angle=10.5,
    )

    # Configure optimization
    config = OptimizationConfig(
        objectives={
            OptimizationObjective.CLUBHEAD_VELOCITY: 1.0,
            OptimizationObjective.INJURY_RISK: 0.3,
        },
        constraints=[
            OptimizationConstraint.JOINT_LIMITS,
            OptimizationConstraint.TORQUE_LIMITS,
        ],
        n_nodes=30,  # Fewer nodes for faster example
        max_iterations=100,
    )

    # Create optimizer and run
    optimizer = SwingOptimizer(golfer, club, config)

    result = optimizer.optimize()

    return optimizer, result


if __name__ == "__main__":
    optimizer, result = create_example_optimization()

    if result.success:
        pass
