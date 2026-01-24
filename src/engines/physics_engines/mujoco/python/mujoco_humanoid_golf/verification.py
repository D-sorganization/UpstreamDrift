"""
Scientific Verification Engine for Golf Modeling Suite.

This module implements Phase 2 of the Jan 2026 Roadmap: "Scientific Verification".
It provides rigorous runtime checks for physics correctness, focusing on:
1. Energy Conservation (The First Law of Thermodynamics)
2. Derivative Correctness (Analytical vs Numerical)

These verifiers are designed to be run alongside simulations to detect 'phantom forces',
integration errors, or model definition flaws.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from src.shared.python.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)


@dataclass
class EnergyState:
    """Snapshot of system energy state."""

    time: float
    kinetic: float
    potential: float
    total: float
    mechanical_work: float = 0.0


class EnergyMonitor:
    """Monitors energy conservation to detect physics hallucinations.

    The fundamental check is:
        Delta_Energy == Work_Done_By_Actuators - Dissipated_Energy

    If this equality is violated (beyond integration error), the physics
    simulation is 'leaking' energy, often due to:
    - Bad collision parameters (soft contacts)
    - Unstable integration (timestep too large)
    - User code modifying state (teleportation)
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.history: list[EnergyState] = []
        self.cumulative_work = 0.0
        self.prev_time = data.time

    def reset(self) -> None:
        """Reset history and accumulators."""
        self.history.clear()
        self.cumulative_work = 0.0
        self.prev_time = self.data.time

    def record_step(self, control_torques: np.ndarray | None = None) -> None:
        """Record energy state after a simulation step.

        Args:
            control_torques: The torques (tau) applied during this step.
                           Used to compute mechanical work (Power = tau * velocity).
        """
        # 1. Snapshot Energy (MuJoCo computes this efficiently)
        mujoco.mj_energyPos(self.model, self.data)
        mujoco.mj_energyVel(self.model, self.data)

        pe = self.data.energy[0]  # Potential
        ke = self.data.energy[1]  # Kinetic
        total = pe + ke

        # 2. Compute Mechanical Work Input (Power * dt)
        # Power = Force * Velocity
        # We use simple Euler integration for monitoring: P = tau . qvel
        dt = self.data.time - self.prev_time

        if control_torques is not None and dt > 0:
            # Power = tau * qvel
            # Note: This is an approximation. MuJoCo uses semi-implicit integration.
            # For strict monitoring, we'd need work done by constraints too.
            power = np.dot(control_torques, self.data.qvel)
            work_increment = power * dt
            self.cumulative_work += work_increment

        # 3. Store State
        state = EnergyState(
            time=self.data.time,
            kinetic=ke,
            potential=pe,
            total=total,
            mechanical_work=self.cumulative_work,
        )
        self.history.append(state)
        self.prev_time = self.data.time

    def check_conservation(self, tolerance: float = 1.0) -> tuple[bool, float]:
        """Verify strict energy conservation.

        Returns:
            passed (bool): True if energy drift is within tolerance.
            drift (float): Energy error magnitude (Joules).
        """
        if not self.history:
            return True, 0.0

        initial = self.history[0]
        current = self.history[-1]

        # Theoretical Energy = Initial Total + Work Input
        predicted_total = initial.total + (
            current.mechanical_work - initial.mechanical_work
        )

        # Actual Energy
        actual_total = current.total

        # Drift = Actual - Predicted
        drift = abs(actual_total - predicted_total)

        # For a conservative system (no friction/damping), drift should be ~0.
        # With damping, Actual < Predicted is expected (dissipation).
        # Actual > Predicted is a CRITICAL FAILURE (Energy creation).

        passed = drift < tolerance
        if not passed:
            logger.warning(
                f"Energy Conservation Violation! Drift: {drift:.4f} J. "
                f"(Predicted: {predicted_total:.2f}, Actual: {actual_total:.2f})"
            )

        return passed, drift


class JacobianTester:
    """Verifies Analytical Jacobians against Numerical Differences.

    This detects:
    1. Incorrect joint definitions in URDF.
    2. Broken compilation of kinematic chains.
    3. 'Observer Effect' where reading data modifies it.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        # Use a private MjData to avoid side effects (Phase 1 Fix)
        self.data = mujoco.MjData(model)

    def check_body_jacobian(
        self, body_name: str, qpos: np.ndarray, epsilon: float = 1e-6
    ) -> float:
        """Compare analytical body Jacobian vs finite difference.

        Args:
            body_name: Name of the body to test.
            qpos: Joint configuration to test at.
            epsilon: Finite difference perturbation step.

        Returns:
            max_error: Maximum element-wise difference between methods.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found.")

        # 1. Setup State
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        # 2. Analytical Jacobian (Reference)
        jacp_analytical = np.zeros((3, self.model.nv))
        jacr_analytical = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(
            self.model, self.data, jacp_analytical, jacr_analytical, body_id
        )

        # 3. Finite Difference Jacobian
        jacp_fd = np.zeros((3, self.model.nv))

        for i in range(self.model.nv):
            # Perturb +
            self.data.qpos[i] += epsilon
            mujoco.mj_kinematics(self.model, self.data)
            pos_plus = self.data.xpos[body_id].copy()

            # Perturb -
            self.data.qpos[i] -= 2 * epsilon
            mujoco.mj_kinematics(self.model, self.data)
            pos_minus = self.data.xpos[body_id].copy()

            # Central Difference: f'(x) ~ (f(x+h) - f(x-h)) / 2h
            jacp_fd[:, i] = (pos_plus - pos_minus) / (2 * epsilon)

            # Reset
            self.data.qpos[i] += epsilon  # Back to original

        # 4. Compare
        error = np.abs(jacp_analytical - jacp_fd).max()

        if error > 1e-4:
            logger.error(f"Jacobian Mismatch for body '{body_name}': {error:.2e}")

        return float(error)
