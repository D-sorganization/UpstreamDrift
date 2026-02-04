"""System Identification for sim-to-real transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.engines.protocols import PhysicsEngineProtocol
    from src.learning.imitation.dataset import Demonstration


@dataclass
class IdentificationResult:
    """Result of system identification.

    Attributes:
        identified_params: Dictionary of identified parameters.
        residual_error: Final optimization residual.
        iterations: Number of optimization iterations.
        converged: Whether optimization converged.
    """

    identified_params: dict[str, float | NDArray[np.floating]]
    residual_error: float
    iterations: int
    converged: bool


class SystemIdentifier:
    """Identify real robot parameters from data.

    Uses optimization to find simulation parameters that best
    match observed real robot behavior.

    Attributes:
        model: Physics engine model.
        param_bounds: Parameter bounds for optimization.
    """

    def __init__(
        self,
        model: PhysicsEngineProtocol,
        param_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize system identifier.

        Args:
            model: Physics engine to tune.
            param_bounds: Bounds for each parameter.
        """
        self.model = model
        self.param_bounds = param_bounds or self._default_bounds()
        self._nominal_params = self._get_current_params()

    def _default_bounds(self) -> dict[str, tuple[float, float]]:
        """Get default parameter bounds.

        Returns:
            Dictionary of parameter bounds.
        """
        return {
            "mass_scale": (0.5, 2.0),
            "friction_scale": (0.2, 3.0),
            "damping_scale": (0.5, 2.0),
            "motor_scale": (0.5, 1.5),
            "com_offset_x": (-0.05, 0.05),
            "com_offset_y": (-0.05, 0.05),
            "com_offset_z": (-0.05, 0.05),
        }

    def _get_current_params(self) -> dict[str, Any]:
        """Get current model parameters.

        Returns:
            Dictionary of current parameters.
        """
        params = {}

        if hasattr(self.model, "get_link_masses"):
            params["masses"] = self.model.get_link_masses().copy()

        if hasattr(self.model, "get_joint_damping"):
            params["damping"] = self.model.get_joint_damping().copy()

        if hasattr(self.model, "get_friction_coefficients"):
            params["friction"] = self.model.get_friction_coefficients().copy()

        if hasattr(self.model, "get_motor_strength"):
            params["motor"] = self.model.get_motor_strength().copy()

        return params

    def _apply_params(self, param_vector: NDArray[np.floating]) -> None:
        """Apply parameter vector to model.

        Args:
            param_vector: Flattened parameter vector.
        """
        idx = 0
        param_names = list(self.param_bounds.keys())

        for name in param_names:
            if name == "mass_scale":
                if "masses" in self._nominal_params:
                    scale = param_vector[idx]
                    masses = self._nominal_params["masses"] * scale
                    if hasattr(self.model, "set_link_masses"):
                        self.model.set_link_masses(masses)
                idx += 1

            elif name == "friction_scale":
                if "friction" in self._nominal_params:
                    scale = param_vector[idx]
                    friction = self._nominal_params["friction"] * scale
                    if hasattr(self.model, "set_friction_coefficients"):
                        self.model.set_friction_coefficients(friction)
                idx += 1

            elif name == "damping_scale":
                if "damping" in self._nominal_params:
                    scale = param_vector[idx]
                    damping = self._nominal_params["damping"] * scale
                    if hasattr(self.model, "set_joint_damping"):
                        self.model.set_joint_damping(damping)
                idx += 1

            elif name == "motor_scale":
                if "motor" in self._nominal_params:
                    scale = param_vector[idx]
                    motor = self._nominal_params["motor"] * scale
                    if hasattr(self.model, "set_motor_strength"):
                        self.model.set_motor_strength(motor)
                idx += 1

            elif name.startswith("com_offset"):
                idx += 1  # Placeholder for CoM offset handling

    def _simulate_trajectory(
        self,
        initial_state: NDArray[np.floating],
        actions: NDArray[np.floating],
        dt: float,
    ) -> NDArray[np.floating]:
        """Simulate a trajectory with current parameters.

        Args:
            initial_state: Initial joint positions and velocities.
            actions: Action sequence to apply.
            dt: Timestep.

        Returns:
            Simulated state trajectory.
        """
        n_steps = len(actions)
        n_q = len(initial_state) // 2
        states = [initial_state.copy()]

        # Set initial state
        q0 = initial_state[:n_q]
        v0 = initial_state[n_q:]

        if hasattr(self.model, "set_joint_positions"):
            self.model.set_joint_positions(q0)
        if hasattr(self.model, "set_joint_velocities"):
            self.model.set_joint_velocities(v0)

        for i in range(n_steps):
            # Apply action
            if hasattr(self.model, "set_joint_torques"):
                self.model.set_joint_torques(actions[i])

            # Step simulation
            if hasattr(self.model, "step"):
                self.model.step(dt)

            # Record state
            if hasattr(self.model, "get_joint_positions"):
                q = self.model.get_joint_positions()
            else:
                q = np.zeros(n_q)

            if hasattr(self.model, "get_joint_velocities"):
                v = self.model.get_joint_velocities()
            else:
                v = np.zeros(n_q)

            states.append(np.concatenate([q, v]))

        return np.array(states)

    def _compute_trajectory_error(
        self,
        sim_trajectory: NDArray[np.floating],
        real_trajectory: NDArray[np.floating],
        weights: NDArray[np.floating] | None = None,
    ) -> float:
        """Compute error between simulated and real trajectory.

        Args:
            sim_trajectory: Simulated state trajectory.
            real_trajectory: Real robot state trajectory.
            weights: State component weights.

        Returns:
            Weighted mean squared error.
        """
        # Ensure same length
        n = min(len(sim_trajectory), len(real_trajectory))
        sim = sim_trajectory[:n]
        real = real_trajectory[:n]

        diff = sim - real

        if weights is not None:
            diff = diff * weights

        return float(np.mean(diff**2))

    def identify_from_trajectories(
        self,
        trajectories: list[Demonstration],
        params_to_identify: list[str] | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> IdentificationResult:
        """Identify parameters from real robot trajectories.

        Uses gradient-free optimization to find parameters
        that minimize trajectory prediction error.

        Args:
            trajectories: List of real robot demonstrations.
            params_to_identify: Which parameters to identify.
            max_iterations: Maximum optimization iterations.
            tolerance: Convergence tolerance.

        Returns:
            Identification result.
        """
        if params_to_identify is None:
            params_to_identify = list(self.param_bounds.keys())

        # Initialize parameter vector
        n_params = len(params_to_identify)
        param_vector = np.ones(n_params)  # Start at nominal (scale = 1)

        # Get bounds
        lower_bounds = np.array([self.param_bounds[p][0] for p in params_to_identify])
        upper_bounds = np.array([self.param_bounds[p][1] for p in params_to_identify])

        def objective(params: NDArray[np.floating]) -> float:
            """Compute total error over all trajectories."""
            self._apply_params(params)
            total_error = 0.0

            for demo in trajectories:
                if demo.actions is None:
                    continue

                # Build initial state
                initial_state = np.concatenate([
                    demo.joint_positions[0],
                    demo.joint_velocities[0],
                ])

                # Build real trajectory
                real_traj = np.concatenate([
                    demo.joint_positions,
                    demo.joint_velocities,
                ], axis=1)

                # Compute dt from timestamps
                dt = float(np.mean(np.diff(demo.timestamps)))

                # Simulate
                sim_traj = self._simulate_trajectory(
                    initial_state, demo.actions, dt
                )

                # Compute error
                total_error += self._compute_trajectory_error(sim_traj, real_traj)

            return total_error / len(trajectories)

        # Simple gradient-free optimization (coordinate descent)
        best_params = param_vector.copy()
        best_error = objective(best_params)
        converged = False

        for iteration in range(max_iterations):
            improved = False

            for i in range(n_params):
                # Try small perturbations
                for delta in [0.1, -0.1, 0.05, -0.05, 0.01, -0.01]:
                    test_params = best_params.copy()
                    test_params[i] = np.clip(
                        test_params[i] + delta,
                        lower_bounds[i],
                        upper_bounds[i],
                    )

                    error = objective(test_params)
                    if error < best_error - tolerance:
                        best_error = error
                        best_params = test_params.copy()
                        improved = True

            if not improved:
                converged = True
                break

        # Build result dictionary
        identified = {}
        for i, name in enumerate(params_to_identify):
            identified[name] = float(best_params[i])

        return IdentificationResult(
            identified_params=identified,
            residual_error=best_error,
            iterations=iteration + 1,
            converged=converged,
        )

    def compute_reality_gap(
        self,
        sim_trajectory: NDArray[np.floating],
        real_trajectory: NDArray[np.floating],
    ) -> dict[str, float]:
        """Quantify the sim-to-real gap.

        Args:
            sim_trajectory: Simulated state trajectory.
            real_trajectory: Real robot trajectory.

        Returns:
            Dictionary of gap metrics.
        """
        n = min(len(sim_trajectory), len(real_trajectory))
        sim = sim_trajectory[:n]
        real = real_trajectory[:n]

        diff = sim - real
        n_dof = sim.shape[1] // 2

        metrics = {
            "total_mse": float(np.mean(diff**2)),
            "position_mse": float(np.mean(diff[:, :n_dof] ** 2)),
            "velocity_mse": float(np.mean(diff[:, n_dof:] ** 2)),
            "max_position_error": float(np.max(np.abs(diff[:, :n_dof]))),
            "max_velocity_error": float(np.max(np.abs(diff[:, n_dof:]))),
            "mean_position_error": float(np.mean(np.abs(diff[:, :n_dof]))),
            "mean_velocity_error": float(np.mean(np.abs(diff[:, n_dof:]))),
            "trajectory_length": n,
        }

        # Per-joint errors
        for j in range(n_dof):
            metrics[f"joint_{j}_position_mse"] = float(np.mean(diff[:, j] ** 2))
            metrics[f"joint_{j}_velocity_mse"] = float(
                np.mean(diff[:, n_dof + j] ** 2)
            )

        return metrics

    def validate_identification(
        self,
        test_trajectories: list[Demonstration],
        identified_params: dict[str, float | NDArray[np.floating]],
    ) -> dict[str, float]:
        """Validate identified parameters on test data.

        Args:
            test_trajectories: Held-out test demonstrations.
            identified_params: Previously identified parameters.

        Returns:
            Validation metrics.
        """
        # Apply identified parameters
        param_vector = np.array([
            identified_params.get(name, 1.0)
            for name in self.param_bounds.keys()
        ])
        self._apply_params(param_vector)

        # Compute errors on test set
        errors = []
        for demo in test_trajectories:
            if demo.actions is None:
                continue

            initial_state = np.concatenate([
                demo.joint_positions[0],
                demo.joint_velocities[0],
            ])

            real_traj = np.concatenate([
                demo.joint_positions,
                demo.joint_velocities,
            ], axis=1)

            dt = float(np.mean(np.diff(demo.timestamps)))
            sim_traj = self._simulate_trajectory(initial_state, demo.actions, dt)

            error = self._compute_trajectory_error(sim_traj, real_traj)
            errors.append(error)

        return {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "max_error": float(np.max(errors)),
            "min_error": float(np.min(errors)),
            "n_trajectories": len(errors),
        }
