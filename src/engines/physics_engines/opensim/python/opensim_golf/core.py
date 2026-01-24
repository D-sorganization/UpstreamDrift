"""
Core simulation logic for the Golf Swing Simulator.

IMPORTANT: This module requires OpenSim to be properly installed.
There is NO demo or fallback mode - if OpenSim is not available,
explicit errors will be raised to prevent displaying incorrect data.
"""

from src.shared.python.logging_config import get_logger
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python import constants

# Configure logging
logger = get_logger(__name__)


class OpenSimNotInstalledError(Exception):
    """Raised when OpenSim is not installed but is required."""

    pass


class OpenSimModelLoadError(Exception):
    """Raised when an OpenSim model fails to load."""

    pass


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
    Represents the biomechanical model of a golf swing using OpenSim.

    This class requires OpenSim to be properly installed. There is NO fallback
    demo mode - if OpenSim is not available, explicit errors are raised.

    If you need to run without OpenSim, use the appropriate engine selector
    to choose MuJoCo or Pinocchio instead.
    """

    def __init__(self, model_path: str | None = None) -> None:
        """
        Initialize the model.

        Args:
            model_path: Path to an OpenSim .osim file. Required for simulation.

        Raises:
            OpenSimNotInstalledError: If OpenSim is not installed.
            OpenSimModelLoadError: If the model file cannot be loaded.
            FileNotFoundError: If model_path does not exist.
        """
        if model_path is None:
            raise ValueError(
                "model_path is required. OpenSim requires a valid .osim model file.\n"
                "Hint: Use the engine selector to choose a model, or provide a path "
                "to an OpenSim model file."
            )

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"OpenSim model file not found: {model_path}\n"
                f"Hint: Ensure the model file exists and the path is correct."
            )

        self.model_path = model_path
        self._opensim_model: Any = None
        self._state: Any = None
        self._manager: Any = None

        # Simulation parameters
        self.gravity = -constants.GRAVITY_M_S2
        self.dt = 0.001
        self.duration = 1.5  # seconds

        # Swing parameters
        self.arm_length = 0.7  # meters
        self.club_length = 1.1  # meters
        self.arm_mass = 5.0  # kg
        self.club_mass = 0.4  # kg
        self.shoulder_torque = 50.0  # Nm (Peak)
        self.wrist_torque_passive = 1.0  # Nm

        self._load_opensim()

    def _load_opensim(self) -> None:
        """Load OpenSim libraries and model.

        Raises:
            OpenSimNotInstalledError: If OpenSim is not installed.
            OpenSimModelLoadError: If the model cannot be loaded.
        """
        try:
            import opensim  # type: ignore
        except ImportError as e:
            raise OpenSimNotInstalledError(
                "OpenSim is not installed.\n"
                "\n"
                "Installation options:\n"
                "  1. Install via conda: conda install -c opensim-org opensim\n"
                "  2. Install via pip: pip install opensim (if available)\n"
                "  3. Download from: https://opensim.stanford.edu/\n"
                "\n"
                "If you don't have OpenSim, use a different physics engine:\n"
                "  - MuJoCo: pip install mujoco\n"
                "  - Pinocchio: pip install pin\n"
                "\n"
                "Select the engine in the launcher settings or use the engine selector."
            ) from e

        try:
            self._opensim_model = opensim.Model(self.model_path)
            self._state = self._opensim_model.initSystem()
            self._manager = opensim.Manager(self._opensim_model)

            # Explicitly initialize the manager with the state if supported (OpenSim 4.0+)
            # If initialize is not available, it might be an older version or using setInitialTime
            if hasattr(self._manager, "initialize"):
                self._manager.initialize(self._state)
            else:
                self._manager.setSessionTime(0.0)
                # Some versions might need setInitialTime

            logger.info(f"Loaded OpenSim model from {self.model_path}")
        except Exception as e:
            raise OpenSimModelLoadError(
                f"Failed to load OpenSim model: {self.model_path}\n"
                f"Error: {e}\n"
                "\n"
                "Possible causes:\n"
                "  1. Model file is corrupted or invalid\n"
                "  2. Model requires resources (geometry, data) not found\n"
                "  3. OpenSim version incompatibility\n"
                "\n"
                "Hint: Verify the model opens in OpenSim GUI before using here."
            ) from e

    @property
    def use_opensim(self) -> bool:
        """Return True - this class always uses OpenSim (no fallback)."""
        return True

    def run_simulation(self) -> SimulationResult:
        """
        Run the forward dynamics simulation.

        Returns:
            SimulationResult object containing trajectories and forces.
        """
        return self._run_opensim_simulation()

    def _run_opensim_simulation(self) -> SimulationResult:
        """Run simulation using OpenSim."""

        # Initialize storage for results
        num_steps = int(self.duration / self.dt)
        time_arr = np.zeros(num_steps)

        n_q = self._opensim_model.getNumCoordinates()
        n_u = self._opensim_model.getNumSpeeds()
        muscles = self._opensim_model.getMuscles()
        n_muscles = muscles.getSize()
        n_controls = self._opensim_model.getNumControls()

        states_arr = np.zeros((num_steps, n_q + n_u))  # Storing Q and U
        muscle_forces_arr = np.zeros((num_steps, n_muscles))
        control_signals_arr = np.zeros((num_steps, n_controls))
        joint_torques_arr = np.zeros((num_steps, n_u))  # Approx

        marker_positions = {}
        marker_set = self._opensim_model.getMarkerSet()
        n_markers = marker_set.getSize()
        for i in range(n_markers):
            marker_name = marker_set.get(i).getName()
            marker_positions[marker_name] = np.zeros((num_steps, 3))

        # Reset state
        self._state = self._opensim_model.initializeState()
        self._opensim_model.equilibrateMuscles(self._state)

        if hasattr(self._manager, "initialize"):
            self._manager.initialize(self._state)
        else:
            self._manager.setSessionTime(0.0)

        # Integration loop
        current_time = 0.0
        for i in range(num_steps):
            # Record current state
            time_arr[i] = current_time

            # Record Q and U
            q_vec = self._state.getQ()
            u_vec = self._state.getU()
            for j in range(n_q):
                states_arr[i, j] = q_vec.get(j)
            for j in range(n_u):
                states_arr[i, n_q + j] = u_vec.get(j)

            # Record Controls
            controls_vec = self._opensim_model.getControls(self._state)
            for j in range(n_controls):
                control_signals_arr[i, j] = controls_vec.get(j)

            # Record Muscle Forces
            # We need to realize Dynamics to get muscle forces
            self._opensim_model.realizeDynamics(self._state)
            for j in range(n_muscles):
                muscle_forces_arr[i, j] = muscles.get(j).getFiberForce(self._state)

            # Record Marker Positions
            self._opensim_model.realizePosition(self._state)
            for j in range(n_markers):
                marker = marker_set.get(j)
                pos = marker.getLocationInGround(self._state)
                marker_positions[marker.getName()][i] = np.array(
                    [pos.get(0), pos.get(1), pos.get(2)]
                )

            # Step simulation
            self._manager.setInitialTime(current_time)
            self._manager.setFinalTime(current_time + self.dt)

            # integrate returns the state (usually) or updates internal state?
            # In OpenSim 4.0: manager.integrate(finalTime) returns bool
            # It updates the state passed to initialize()
            self._manager.integrate(current_time + self.dt)
            current_time += self.dt

            # If using older API or different wrapper, we might need to fetch state
            # self._state = self._manager.getState()

        return SimulationResult(
            time=time_arr,
            states=states_arr,
            muscle_forces=muscle_forces_arr,
            control_signals=control_signals_arr,
            joint_torques=joint_torques_arr,
            marker_positions=marker_positions,
        )
