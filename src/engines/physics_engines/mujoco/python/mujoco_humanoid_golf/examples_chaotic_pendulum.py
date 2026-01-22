"""Example demonstrations for the chaotic driven pendulum model.

This script demonstrates various control scenarios including:
1. Free oscillation with damping
2. Driven oscillation at resonance
3. PID control for stabilization
4. Energy-based swing-up control
5. Chaos exploration with phase portraits

Run this script to see interactive demonstrations of control principles
using the chaotic pendulum model.
"""

import abc
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from shared.python import constants

from .models import CHAOTIC_PENDULUM_XML


class ChaoticPendulumController(abc.ABC):
    """Base controller class for chaotic pendulum experiments."""

    def __init__(self, model: Any, data: Any) -> None:
        """Docstring for __init__."""
        self.model = model
        self.data = data
        self.g = constants.GRAVITY_M_S2
        self.L = 0.8  # pendulum length
        self.m = 1.0  # bob mass

    def get_state(self) -> tuple[float, float, float, float]:
        """Get current pendulum state."""
        x_base = self.data.qpos[0]  # base position
        theta = self.data.qpos[1]  # pendulum angle
        x_base_dot = self.data.qvel[0]  # base velocity
        theta_dot = self.data.qvel[1]  # angular velocity
        return x_base, theta, x_base_dot, theta_dot

    def compute_energy(self) -> float:
        """Compute total mechanical energy of pendulum.

        For a driven pendulum, the bob's velocity has contributions from both
        the base motion and the pendulum's angular motion.
        """
        _x_base, theta, x_base_dot, theta_dot = self.get_state()

        # Bob position in global frame
        # x_bob = x_base + L*sin(theta)
        # y_bob = -L*cos(theta)

        # Bob velocity in global frame
        v_x_bob = x_base_dot + self.L * theta_dot * np.cos(theta)
        v_y_bob = self.L * theta_dot * np.sin(theta)

        # Total kinetic energy (translational only, bob is point mass)
        ke = 0.5 * self.m * (v_x_bob**2 + v_y_bob**2)

        # Potential energy (reference at y=0, base height)
        pe = self.m * self.g * self.L * (1 - np.cos(theta))

        total_energy = ke + pe
        return float(total_energy)

    @abc.abstractmethod
    def control(self, time: float) -> tuple[float, float]:
        """Calculate control inputs. Should be overridden."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset to initial conditions. Should be overridden."""

    def apply_control(self, base_force: float, pendulum_torque: float) -> None:
        """Apply control inputs to the system."""
        self.data.ctrl[0] = base_force
        self.data.ctrl[1] = pendulum_torque


class FreeOscillationDemo(ChaoticPendulumController):
    """Demonstrate free oscillation with damping."""

    def __init__(self, model: Any, data: Any, initial_angle: float = np.pi / 6) -> None:
        """Docstring for __init__."""
        super().__init__(model, data)
        self.initial_angle = initial_angle

    def reset(self) -> None:
        """Reset to initial conditions."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[1] = self.initial_angle  # Set initial angle

    def control(self, time: float) -> tuple[float, float]:
        """No active control - free oscillation."""
        return 0.0, 0.0


class ResonanceDrivenDemo(ChaoticPendulumController):
    """Demonstrate resonance with sinusoidal base forcing."""

    def __init__(
        self,
        model: Any,
        data: Any,
        forcing_freq: float = 1.75,
        forcing_amp: float = 15.0,
    ) -> None:
        """Docstring for __init__."""
        super().__init__(model, data)
        self.forcing_freq = forcing_freq  # Hz
        self.forcing_amp = forcing_amp  # Newtons

    def reset(self) -> None:
        """Reset to initial conditions."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[1] = 0.1  # Small initial perturbation

    def control(self, time: float) -> tuple[float, float]:
        """Apply sinusoidal forcing at specified frequency."""
        base_force = self.forcing_amp * np.sin(2 * np.pi * self.forcing_freq * time)
        pendulum_torque = 0.0  # No direct pendulum control
        return base_force, pendulum_torque


class PIDStabilizationDemo(ChaoticPendulumController):
    """Stabilize pendulum at upright position using PID control."""

    def __init__(
        self,
        model: Any,
        data: Any,
        kp: float = 50.0,
        ki: float = 5.0,
        kd: float = 15.0,
    ) -> None:
        """Docstring for __init__."""
        super().__init__(model, data)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.dt = model.opt.timestep

    def reset(self) -> None:
        """Reset to near-upright position."""
        mujoco.mj_resetData(self.model, self.data)
        # Start near upright with small perturbation
        self.data.qpos[1] = np.pi + 0.1
        self.integral_error = 0.0
        self.prev_error = 0.0

    def control(self, time: float) -> tuple[float, float]:
        """PID control to stabilize at upright (θ = π)."""
        _, theta, _, _theta_dot = self.get_state()

        # Normalize angle to [-π, π]
        np.arctan2(np.sin(theta), np.cos(theta))

        # Target is upright (θ = π in normalized coords is 0 when measured from down)
        # We want θ near π, so error is π - θ
        error = np.pi - theta

        # Normalize error to [-π, π]
        error = np.arctan2(np.sin(error), np.cos(error))

        # PID computation
        self.integral_error += error * self.dt
        derivative_error = (error - self.prev_error) / self.dt
        self.prev_error = error

        # Control torque
        pendulum_torque = (
            self.kp * error + self.ki * self.integral_error + self.kd * derivative_error
        )

        # Clamp to actuator limits
        pendulum_torque = np.clip(pendulum_torque, -10.0, 10.0)

        base_force = 0.0  # No base forcing

        return base_force, pendulum_torque


class SwingUpControlDemo(ChaoticPendulumController):
    """Energy-based swing-up followed by stabilization."""

    def __init__(
        self,
        model: Any,
        data: Any,
        k_swingup: float = 3.0,
        k_stab: float = 50.0,
    ) -> None:
        """Docstring for __init__."""
        super().__init__(model, data)
        self.k_swingup = k_swingup
        self.k_stab = k_stab
        self.switch_threshold = 0.3  # rad from upright to switch control

    def reset(self) -> None:
        """Reset to downward position."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[1] = 0.0  # Start at bottom

    def control(self, time: float) -> tuple[float, float]:
        """Energy-based swing-up with stabilization."""
        _, theta, _, theta_dot = self.get_state()

        # Target energy (upright position)
        e_target = 2 * self.m * self.g * self.L
        e_current = self.compute_energy()

        # Distance from upright
        dist_from_upright = abs(
            np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi)),
        )

        if dist_from_upright > self.switch_threshold:
            # Swing-up phase: energy-based control
            energy_error = e_current - e_target
            pendulum_torque = -self.k_swingup * theta_dot * np.cos(theta) * energy_error
        else:
            # Stabilization phase: simple PD control
            error = np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi))
            pendulum_torque = -self.k_stab * error - 10.0 * theta_dot

        # Clamp to actuator limits
        pendulum_torque = np.clip(pendulum_torque, -10.0, 10.0)

        base_force = 0.0

        return base_force, pendulum_torque


class ChaosExplorationDemo(ChaoticPendulumController):
    """Explore chaotic dynamics with strong forcing."""

    def __init__(
        self,
        model: Any,
        data: Any,
        forcing_freq: float = 2.0,
        forcing_amp: float = 20.0,
        initial_angle: float = 0.5,
    ) -> None:
        """Docstring for __init__."""
        super().__init__(model, data)
        self.forcing_freq = forcing_freq
        self.forcing_amp = forcing_amp
        self.initial_angle = initial_angle

    def reset(self) -> None:
        """Reset with specified initial angle."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[1] = self.initial_angle

    def control(self, time: float) -> tuple[float, float]:
        """Apply strong forcing to induce chaos."""
        base_force = self.forcing_amp * np.sin(2 * np.pi * self.forcing_freq * time)
        pendulum_torque = 0.0
        return base_force, pendulum_torque


def run_simulation(
    controller: ChaoticPendulumController, duration: float = 20.0
) -> dict[str, np.ndarray]:
    """Run simulation with specified controller.

    Args:
        controller: Controller instance
        duration: Simulation duration in seconds

    Returns:
        Dictionary with time history and state variables

    Note:
        Uses the model's internal timestep (model.opt.timestep) for simulation.
    """
    controller.reset()

    # Use model's internal timestep to ensure synchronization
    dt = controller.model.opt.timestep
    n_steps = int(duration / dt)
    save_every = 10  # Save every 10 steps for efficiency

    # Storage
    time_history = []
    x_base_history = []
    theta_history = []
    x_base_dot_history = []
    theta_dot_history = []
    energy_history = []

    # Simulation loop
    for step in range(n_steps):
        t = step * dt

        # Get control action
        base_force, pendulum_torque = controller.control(t)

        # Apply control
        controller.apply_control(base_force, pendulum_torque)

        # Step simulation
        mujoco.mj_step(controller.model, controller.data)

        # Record data
        if step % save_every == 0:
            x_base, theta, x_base_dot, theta_dot = controller.get_state()
            energy = controller.compute_energy()

            time_history.append(t)
            x_base_history.append(x_base)
            theta_history.append(theta)
            x_base_dot_history.append(x_base_dot)
            theta_dot_history.append(theta_dot)
            energy_history.append(energy)

    return {
        "time": np.array(time_history),
        "x_base": np.array(x_base_history),
        "theta": np.array(theta_history),
        "x_base_dot": np.array(x_base_dot_history),
        "theta_dot": np.array(theta_dot_history),
        "energy": np.array(energy_history),
    }


def plot_results(
    results: dict[str, np.ndarray], title: str = "Chaotic Pendulum Simulation"
) -> None:
    """Plot simulation results."""
    _fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Time series: angle
    axes[0, 0].plot(results["time"], results["theta"])
    axes[0, 0].set_ylabel("Angle (rad)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].grid(True)
    axes[0, 0].set_title("Pendulum Angle vs Time")

    # Time series: angular velocity
    axes[1, 0].plot(results["time"], results["theta_dot"])
    axes[1, 0].set_ylabel("Angular Velocity (rad/s)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].grid(True)
    axes[1, 0].set_title("Angular Velocity vs Time")

    # Time series: base position
    axes[2, 0].plot(results["time"], results["x_base"])
    axes[2, 0].set_ylabel("Base Position (m)")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].grid(True)
    axes[2, 0].set_title("Base Position vs Time")

    # Phase portrait
    axes[0, 1].plot(results["theta"], results["theta_dot"], linewidth=0.5)
    axes[0, 1].set_xlabel("Angle (rad)")
    axes[0, 1].set_ylabel("Angular Velocity (rad/s)")
    axes[0, 1].grid(True)
    axes[0, 1].set_title("Phase Portrait")

    # Energy vs time
    axes[1, 1].plot(results["time"], results["energy"])
    axes[1, 1].set_ylabel("Energy (J)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].grid(True)
    axes[1, 1].set_title("Total Mechanical Energy vs Time")

    # Poincaré map (sample at forcing period for driven systems)
    # Sample every 0.5 seconds (forcing period)
    poincare_period = 0.5
    poincare_indices = [
        i
        for i, t in enumerate(results["time"])
        if abs(t - round(t / poincare_period) * poincare_period) < 0.01
    ]

    if len(poincare_indices) > 10:
        poincare_theta = results["theta"][poincare_indices]
        poincare_theta_dot = results["theta_dot"][poincare_indices]
        axes[2, 1].scatter(poincare_theta, poincare_theta_dot, s=1, c="blue", alpha=0.6)
        axes[2, 1].set_xlabel("Angle (rad)")
        axes[2, 1].set_ylabel("Angular Velocity (rad/s)")
        axes[2, 1].grid(True)
        axes[2, 1].set_title("Poincaré Map (sampled every 0.5s)")
    else:
        axes[2, 1].text(
            0.5,
            0.5,
            "Insufficient data for Poincaré map",
            ha="center",
            va="center",
            transform=axes[2, 1].transAxes,
        )

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXAMPLE DEMONSTRATIONS
# ============================================================================


def example_1_free_oscillation() -> None:
    """Example 1: Free oscillation with damping."""

    model = mujoco.MjModel.from_xml_string(CHAOTIC_PENDULUM_XML)
    data = mujoco.MjData(model)

    controller = FreeOscillationDemo(model, data, initial_angle=np.pi / 6)
    results = run_simulation(controller, duration=15.0)

    plot_results(results, title="Example 1: Free Oscillation with Damping")


def example_2_resonance() -> None:
    """Example 2: Driven oscillation at resonance frequency."""

    model = mujoco.MjModel.from_xml_string(CHAOTIC_PENDULUM_XML)
    data = mujoco.MjData(model)

    controller = ResonanceDrivenDemo(model, data, forcing_freq=1.75, forcing_amp=15.0)
    results = run_simulation(controller, duration=20.0)

    plot_results(results, title="Example 2: Resonance-Driven Oscillation")


def example_3_pid_stabilization() -> None:
    """Example 3: PID stabilization at upright position."""

    model = mujoco.MjModel.from_xml_string(CHAOTIC_PENDULUM_XML)
    data = mujoco.MjData(model)

    controller = PIDStabilizationDemo(model, data, kp=50.0, ki=5.0, kd=15.0)
    results = run_simulation(controller, duration=10.0)

    plot_results(results, title="Example 3: PID Stabilization at Upright")


def example_4_swing_up() -> None:
    """Example 4: Energy-based swing-up control."""

    model = mujoco.MjModel.from_xml_string(CHAOTIC_PENDULUM_XML)
    data = mujoco.MjData(model)

    controller = SwingUpControlDemo(model, data, k_swingup=3.0, k_stab=50.0)
    results = run_simulation(controller, duration=15.0)

    plot_results(results, title="Example 4: Energy-Based Swing-Up Control")


def example_5_chaos_exploration() -> None:
    """Example 5: Exploring chaotic dynamics."""

    model = mujoco.MjModel.from_xml_string(CHAOTIC_PENDULUM_XML)
    data = mujoco.MjData(model)

    controller = ChaosExplorationDemo(
        model,
        data,
        forcing_freq=2.0,
        forcing_amp=20.0,
        initial_angle=0.5,
    )
    results = run_simulation(controller, duration=30.0)

    plot_results(results, title="Example 5: Chaos Exploration")


def example_6_sensitivity_to_initial_conditions() -> None:
    """Example 6: Demonstrate sensitivity to initial conditions (hallmark of chaos)."""

    model1 = mujoco.MjModel.from_xml_string(CHAOTIC_PENDULUM_XML)
    data1 = mujoco.MjData(model1)
    model2 = mujoco.MjModel.from_xml_string(CHAOTIC_PENDULUM_XML)
    data2 = mujoco.MjData(model2)

    # Two controllers with slightly different initial angles
    controller1 = ChaosExplorationDemo(
        model1,
        data1,
        forcing_freq=2.0,
        forcing_amp=20.0,
        initial_angle=0.500,
    )
    controller2 = ChaosExplorationDemo(
        model2,
        data2,
        forcing_freq=2.0,
        forcing_amp=20.0,
        initial_angle=0.501,
    )

    results1 = run_simulation(controller1, duration=30.0)
    results2 = run_simulation(controller2, duration=30.0)

    # Plot comparison
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(
        results1["time"],
        results1["theta"],
        label="IC: θ₀ = 0.500 rad",
        linewidth=1.5,
    )
    axes[0, 0].plot(
        results2["time"],
        results2["theta"],
        label="IC: θ₀ = 0.501 rad",
        linewidth=1.5,
        alpha=0.8,
    )
    axes[0, 0].set_ylabel("Angle (rad)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title("Angle vs Time (Two Different Initial Conditions)")

    # Difference in angle over time
    angle_diff = np.abs(results1["theta"] - results2["theta"])
    axes[0, 1].semilogy(results1["time"], angle_diff)
    axes[0, 1].set_ylabel("Absolute Angle Difference (rad)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].grid(True)
    axes[0, 1].set_title("Divergence of Trajectories (Log Scale)")

    # Phase portraits
    axes[1, 0].plot(
        results1["theta"],
        results1["theta_dot"],
        linewidth=0.5,
        label="IC: θ₀ = 0.500 rad",
    )
    axes[1, 0].plot(
        results2["theta"],
        results2["theta_dot"],
        linewidth=0.5,
        label="IC: θ₀ = 0.501 rad",
        alpha=0.7,
    )
    axes[1, 0].set_xlabel("Angle (rad)")
    axes[1, 0].set_ylabel("Angular Velocity (rad/s)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_title("Phase Portraits Comparison")

    # Velocity comparison
    axes[1, 1].plot(
        results1["time"],
        results1["theta_dot"],
        label="IC: θ₀ = 0.500 rad",
        linewidth=1.5,
    )
    axes[1, 1].plot(
        results2["time"],
        results2["theta_dot"],
        label="IC: θ₀ = 0.501 rad",
        linewidth=1.5,
        alpha=0.8,
    )
    axes[1, 1].set_ylabel("Angular Velocity (rad/s)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_title("Angular Velocity vs Time")

    plt.suptitle(
        "Example 6: Sensitivity to Initial Conditions (Chaos)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def run_all_examples() -> None:
    """Run all example demonstrations."""

    example_1_free_oscillation()
    example_2_resonance()
    example_3_pid_stabilization()
    example_4_swing_up()
    example_5_chaos_exploration()
    example_6_sensitivity_to_initial_conditions()


if __name__ == "__main__":
    # Run all examples
    run_all_examples()

    # Or run individual examples:
    # example_1_free_oscillation()
    # example_2_resonance()
    # example_3_pid_stabilization()
    # example_4_swing_up()
    # example_5_chaos_exploration()
    # example_6_sensitivity_to_initial_conditions()
