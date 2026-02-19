import os
import sys

# Fix MuJoCo DLL loading issue on Windows with Python 3.13

# Must be set BEFORE any imports that might load mujoco

if "MUJOCO_PLUGIN_PATH" not in os.environ:
    os.environ["MUJOCO_PLUGIN_PATH"] = ""


import logging
from pathlib import Path
from typing import Any

import numpy as np
from humanoid_launcher_analysis import AnalysisMixin
from humanoid_launcher_sim import SimulationMixin
from humanoid_launcher_ui import UISetupMixin
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
)

from src.shared.python.config.configuration_manager import ConfigurationManager
from src.shared.python.data_io.path_utils import get_repo_root
from src.shared.python.engine_core.interfaces import RecorderInterface
from src.shared.python.logging_pkg.logging_config import (
    configure_gui_logging,
    get_logger,
)
from src.shared.python.ui.qt.process_worker import ProcessWorker

PROJECT_ROOT = get_repo_root()


# Configure logging using centralized module

configure_gui_logging()

logger = get_logger(__name__)


class RemoteRecorder(RecorderInterface):
    """Recorder that stores data received from remote process."""

    # Engine reference required by RecorderInterface protocol.

    # For remote recording, no direct engine access is available.

    engine: Any = None

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all recorded simulation data."""
        self.data: dict[str, Any] = {
            "times": [],
            "joint_positions": [],
            "joint_velocities": [],
            "joint_torques": [],
            "ztcf_accel": [],
            "zvcf_accel": [],
            "induced_accelerations": {},
        }

    def process_packet(self, packet: dict) -> None:
        """Ingest a simulation data packet into the recording store."""
        try:
            t = packet["time"]

            self.data["times"].append(t)

            self.data["joint_positions"].append(np.array(packet["qpos"]))

            self.data["joint_velocities"].append(np.array(packet["qvel"]))

            if "qfrc_actuator" in packet:
                self.data["joint_torques"].append(np.array(packet["qfrc_actuator"]))

            else:
                self.data["joint_torques"].append(np.zeros_like(packet["qvel"]))

            iaa = packet.get("iaa", {})

            for src, val in iaa.items():
                if src not in self.data["induced_accelerations"]:
                    self.data["induced_accelerations"][src] = []

                self.data["induced_accelerations"][src].append(np.array(val))

            cf = packet.get("cf", {})

            if "ztcf" in cf:
                self.data["ztcf_accel"].append(np.array(cf["ztcf"]))

            if "zvcf" in cf:
                self.data["zvcf_accel"].append(np.array(cf["zvcf"]))

        except (ValueError, TypeError, RuntimeError) as e:
            logging.error(f"Error processing packet: {e}")

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return time-aligned arrays for a named data field."""
        if not self.data["times"]:
            return np.array([]), np.array([])

        times = np.array(self.data["times"])

        if field_name == "joint_positions":
            return times, np.array(self.data["joint_positions"])

        elif field_name == "joint_velocities":
            return times, np.array(self.data["joint_velocities"])

        elif field_name == "joint_torques":
            return times, np.array(self.data["joint_torques"])

        elif field_name == "ztcf_accel":
            return times, np.array(self.data["ztcf_accel"])

        elif field_name == "zvcf_accel":
            return times, np.array(self.data["zvcf_accel"])

        return times, np.array([])

    def get_induced_acceleration_series(
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return induced acceleration time series for a source."""
        if not self.data["times"]:
            return np.array([]), np.array([])

        times = np.array(self.data["times"])

        key = str(source_name)

        if key in self.data["induced_accelerations"]:
            return times, np.array(self.data["induced_accelerations"][key])

        return times, np.array([])

    def set_analysis_config(self, config: dict) -> None:
        """Configure which advanced metrics to record/compute.

        For RemoteRecorder, this is a no-op since configuration is driven by
        the simulation process, not the GUI. The config is stored but not acted upon.
        """

        # Store config for potential future use, but don't act on it
        # since the simulation process controls what metrics are computed.

        self._analysis_config = config

    def export_to_dict(self) -> dict[str, Any]:
        """Export all recorded data as a dictionary."""
        return self.data


class ModernDarkPalette(QPalette):
    """Custom Dark Palette for a modern look."""

    def __init__(self) -> None:
        super().__init__()

        self.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
        self.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
        self.setColor(QPalette.ColorRole.AlternateBase, QColor(43, 43, 43))
        self.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        self.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        self.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        self.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        self.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)


class HumanoidLauncher(UISetupMixin, SimulationMixin, AnalysisMixin, QMainWindow):
    """Main launcher window for the Humanoid Golf Simulation Suite."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Humanoid Golf Simulation Suite")
        self.setMinimumSize(1000, 800)

        # Apply Theme
        QApplication.setStyle("Fusion")
        self.setPalette(ModernDarkPalette())

        # Paths
        self.current_dir = Path(__file__).parent.resolve()
        # engines/physics_engines/mujoco/python -> engines/physics_engines/mujoco
        self.repo_path = self.current_dir.parent
        # Save config in docker/src where the humanoid_golf simulation expects it
        self.config_path = self.repo_path / "docker" / "src" / "simulation_config.json"

        # State
        self.config_manager = ConfigurationManager(self.config_path)
        self.config = self.config_manager.load()
        self.simulation_thread: ProcessWorker | None = None
        self.recorder = RemoteRecorder()

        # UI Setup
        self.setup_ui()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Global Stylesheet for Rounded Buttons and Modern Look
    app.setStyleSheet("""
        QPushButton {
            border-radius: 5px;
            padding: 5px;
        }
        QPushButton:hover {
            border: 1px solid #4a90e2;
        }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            border-radius: 4px;
            padding: 4px;
            border: 1px solid #555;
            background-color: #333;
            color: white;
            selection-background-color: #4a90e2;
        }
        QComboBox QAbstractItemView {
            background-color: #333;
            color: white;
            selection-background-color: #4a90e2;
            border: 1px solid #555;
        }
        QLabel, QCheckBox, QRadioButton, QGroupBox {
            color: white;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #777;
            background: #2b2b2b;
        }
        QCheckBox::indicator:checked {
            background: #4a90e2;
            border: 1px solid #4a90e2;
        }
        QPushButton {
            color: white; /* Ensure button text is visible */
            background-color: #444; /* Default dark button bg */
        }
        /* Fix QMessageBox readability */
        QMessageBox {
            background-color: #333;
        }
        QMessageBox QLabel {
            color: white;
        }
    """)

    window = HumanoidLauncher()
    window.show()
    sys.exit(app.exec())
