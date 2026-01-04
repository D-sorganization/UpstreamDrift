"""
Core simulation logic for the Golf Swing Simulator.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from shared.python import constants

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Container for simulation results."""

    time: np.ndarray
    states: np.ndarray  # [N_steps, N_states]
    muscle_forces: np.ndarray  # [N_steps, N_muscles]
    control_signals: np.ndarray  # [N_steps, N_controls]
    joint_torques: np.ndarray  # [N_steps, N_joints]
    marker_positions: dict[str, np.ndarray]  # name -> [N_steps, 3]


class GolfSwingModel:
    """
    Represents the biomechanical model of a golf swing.

    This class abstracts the underlying physics engine (e.g., OpenSim).
    If OpenSim is not available, it falls back to an internal analytical model
    (Double Pendulum approximation) for demonstration purposes.
    """

    def __init__(self, model_path: str | None = None) -> None:
        """
        Initialize the model.

        Args:
            model_path: Path to an OpenSim .osim file. If None, uses
                internal demo model.
        """
        self.model_path = model_path
        self.use_opensim = False
        self._opensim_model: Any = None
        self._state: Any = None

        # Simulation parameters
        self.gravity = -constants.GRAVITY_M_S2
        self.dt = 0.001
        self.duration = 1.5  # seconds

        # Swing parameters (defaults for demo model)
        self.arm_length = 0.7  # meters
        self.club_length = 1.1  # meters
        self.arm_mass = 5.0  # kg
        self.club_mass = 0.4  # kg
        self.shoulder_torque = 50.0  # Nm (Peak)
        self.wrist_torque_passive = 1.0  # Nm

        self._try_load_opensim()

    def _try_load_opensim(self) -> None:
        """Attempt to load OpenSim libraries and model."""
        if self.model_path:
            try:
                import opensim  # type: ignore

                self._opensim_model = opensim.Model(self.model_path)
                self._state = self._opensim_model.initSystem()
                self.use_opensim = True
                logger.info(f"Loaded OpenSim model from {self.model_path}")
            except ImportError:
                logger.warning(
                    "OpenSim library not found. Falling back to internal demo model."
                )
            except Exception as e:
                logger.error(
                    f"Failed to load OpenSim model: {e}. "
                    f"Falling back to internal demo model."
                )

    def run_simulation(self) -> SimulationResult:
        """
        Run the forward dynamics simulation.

        Returns:
            SimulationResult object containing trajectories and forces.
        """
        if self.use_opensim:
            return self._run_opensim_simulation()
        else:
            return self._run_demo_simulation()

    def _run_opensim_simulation(self) -> SimulationResult:
        """Run simulation using OpenSim (Placeholder for actual implementation)."""
        # This would involve setting up the manager, controls, and integrating.
        # For now, we'll raise NotImplementedError or return dummy data if this
        # path were active. Since we likely don't have OpenSim in this env, we
        # focus on the demo model.
        # TODO: Implement full OpenSim integration once the environment includes valid OpenSim binaries and models.
        raise NotImplementedError("OpenSim integration pending environment setup.")

    def _run_demo_simulation(self) -> SimulationResult:
        """
        Run a simplified double pendulum simulation (Arm + Club).

        Equations of motion for double pendulum are solved using Euler integration
        for simplicity in this demo, or we can use analytical approx.
        """
        logger.info("Running demo simulation (Double Pendulum)...")

        steps = int(self.duration / self.dt)
        time = np.linspace(0, self.duration, steps)

        # State: [theta1, theta2, omega1, omega2]
        # theta1: Arm angle (0 = down, -pi/2 = backswing top)
        # theta2: Club angle relative to arm

        # Initial conditions: Backswing
        theta1 = -np.pi / 2.2  # Arm up/back
        theta2 = -np.pi / 1.5  # Wrist cocked
        omega1 = 0.0
        omega2 = 0.0

        states = np.zeros((steps, 4))
        muscle_forces = np.zeros((steps, 2))  # Dummy: Shoulder, Wrist
        control_signals = np.zeros((steps, 2))  # Activation
        joint_torques = np.zeros((steps, 2))  # Shoulder, Wrist

        # Marker positions for visualization
        # Shoulder (fixed), Hand, Clubhead
        pos_shoulder = np.zeros((steps, 3))
        pos_hand = np.zeros((steps, 3))
        pos_clubhead = np.zeros((steps, 3))

        for i, t in enumerate(time):
            # Simple control profile: Downswing initiation
            # Active torque starts after t=0.2
            activation = 0.0
            if t > 0.2:
                activation = min(1.0, float((t - 0.2) * 5))

            # Torques
            tau1 = self.shoulder_torque * activation * 2.0  # Shoulder acceleration

            # Wrist torque (passive + active release at impact approx)
            # Simplified: Spring-damper preventing hyper-extension + release
            tau2 = -0.1 * omega2  # Damping
            if t > 0.8:  # Release phase
                tau2 += float(5.0 * (t - 0.8))

            # Store controls/forces
            control_signals[i] = [activation, activation * 0.5]
            joint_torques[i] = [tau1, tau2]
            muscle_forces[i] = [tau1 / 0.05, tau2 / 0.02]  # Approx moment arms

            # Physics (Simplified decoupled + gravity)
            # Real golf swing physics is complex; this is a visual
            # approximation for the GUI.

            # Angular acceleration alpha = Torque / Inertia
            # Inertia approx
            I1 = self.arm_mass * (self.arm_length**2) / 3
            I2 = self.club_mass * (self.club_length**2) / 3

            # Gravity torques
            g_tau1 = (
                -self.arm_mass * self.gravity * (self.arm_length / 2) * np.sin(theta1)
            )
            # Club gravity effects complicated by compound pendulum, simplified here:
            g_tau2 = (
                -self.club_mass
                * self.gravity
                * (self.club_length / 2)
                * np.sin(theta1 + theta2)
            )

            alpha1 = (tau1 + g_tau1) / I1
            # Coupling effect simplified: Centrifugal force from arm pulls club out
            centrifugal_torque = (
                self.club_mass
                * (self.arm_length * omega1**2)
                * (self.club_length / 2)
                * np.sin(theta2)
            )
            alpha2 = (tau2 + g_tau2 - centrifugal_torque) / I2

            # Integrate
            omega1 += alpha1 * self.dt
            omega2 += alpha2 * self.dt
            theta1 += omega1 * self.dt
            theta2 += omega2 * self.dt

            # Store State
            states[i] = [theta1, theta2, omega1, omega2]

            # Kinematics
            # 2D in X-Z plane (Z is up)
            # Shoulder at (0, 1.8, 0)
            shoulder_pos = np.array([0.0, 1.8, 0.0])

            # Arm vector
            # Angle 0 is straight down.
            arm_vec = np.array([np.sin(theta1), -np.cos(theta1), 0.0]) * self.arm_length
            hand_p = shoulder_pos + arm_vec

            # Club vector
            # theta2 is relative to arm? Or global?
            # Let's say theta2 is relative to arm. 0 = straight line extension.
            club_angle = theta1 + theta2
            club_vec = (
                np.array(
                    [np.sin(club_angle), -np.cos(club_angle), 0.1 * np.sin(t * 10)]
                )
                * self.club_length
            )  # Add slight Y wobble for 3D effect

            clubhead_p = hand_p + club_vec

            pos_shoulder[i] = shoulder_pos
            pos_hand[i] = hand_p
            pos_clubhead[i] = clubhead_p

        return SimulationResult(
            time=time,
            states=states,
            muscle_forces=muscle_forces,
            control_signals=control_signals,
            joint_torques=joint_torques,
            marker_positions={
                "Shoulder": pos_shoulder,
                "Hand": pos_hand,
                "ClubHead": pos_clubhead,
            },
        )
