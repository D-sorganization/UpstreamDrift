"""
Core simulation logic for the Golf Swing Simulator.

IMPORTANT: This module requires OpenSim to be properly installed.
There is NO demo or fallback mode - if OpenSim is not available,
explicit errors will be raised to prevent displaying incorrect data.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shared.python import constants

# Configure logging
logger = logging.getLogger(__name__)


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

        Raises:
            NotImplementedError: Full OpenSim simulation integration pending.
        """
        return self._run_opensim_simulation()

    def _run_opensim_simulation(self) -> SimulationResult:
        """Run simulation using OpenSim.

        Raises:
            NotImplementedError: Full integration requires additional work.
        """
        # NOTE: Full OpenSim integration requires environment setup with valid
        # OpenSim binaries and models. This is a planned enhancement.
        raise NotImplementedError(
            "OpenSim simulation integration is not yet complete.\n"
            "\n"
            "Current status:\n"
            "  - OpenSim library: LOADED\n"
            "  - Model file: LOADED\n"
            "  - Simulation integration: PENDING\n"
            "\n"
            "For now, use MuJoCo or Pinocchio for simulation.\n"
            "OpenSim visualization and model inspection are available."
        )

    # NOTE: The previous _run_demo_simulation method has been removed.
    # There is NO fallback mode. If OpenSim cannot run, errors are raised.
    # Use MuJoCo or Pinocchio for simulation without OpenSim installed.
