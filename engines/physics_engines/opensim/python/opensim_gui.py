#!/usr/bin/env python3
"""
OpenSim Golf GUI
A PyQt6 interface for the OpenSim Golf Model.

IMPORTANT: This GUI requires OpenSim to be properly installed.
There is NO demo or fallback mode - if OpenSim is unavailable,
clear error dialogs will be shown.
"""
import sys
from typing import Any

import matplotlib
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

matplotlib.use("QtAgg")
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure

try:
    from engines.physics_engines.opensim.python.opensim_golf.core import (
        GolfSwingModel,
        OpenSimModelLoadError,
        OpenSimNotInstalledError,
    )
except ImportError:
    from .opensim_golf.core import (
        GolfSwingModel,
        OpenSimModelLoadError,
        OpenSimNotInstalledError,
    )


class OpenSimGolfGUI(QMainWindow):
    """OpenSim Golf Simulation GUI.

    This GUI requires a valid OpenSim model file. There is NO fallback
    or demo mode - errors are shown clearly when something fails.
    """

    def __init__(self, model_path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("OpenSim Golf Interface")
        self.resize(1000, 800)

        # Model state
        self.model: GolfSwingModel | None = None
        self.model_path = model_path
        self.result: Any = None
        self.initialization_error: str | None = None

        self.init_ui()
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt to load the OpenSim model if a path was provided."""
        if self.model_path is None:
            self._show_model_required_message()
            return

        try:
            self.model = GolfSwingModel(self.model_path)
            self._update_status("OpenSim Model Loaded", "green")
            self.btn_run.setEnabled(True)
        except OpenSimNotInstalledError as e:
            self.initialization_error = str(e)
            self._update_status("ERROR: OpenSim Not Installed", "red")
            self.btn_run.setEnabled(False)
            QMessageBox.critical(
                self,
                "OpenSim Not Installed",
                f"OpenSim is required but not installed.\n\n{e}",
            )
        except OpenSimModelLoadError as e:
            self.initialization_error = str(e)
            self._update_status("ERROR: Model Load Failed", "red")
            self.btn_run.setEnabled(False)
            QMessageBox.critical(
                self,
                "Model Load Failed",
                f"Failed to load OpenSim model.\n\n{e}",
            )
        except FileNotFoundError as e:
            self.initialization_error = str(e)
            self._update_status("ERROR: Model File Not Found", "red")
            self.btn_run.setEnabled(False)
            QMessageBox.critical(
                self,
                "Model File Not Found",
                f"The specified model file was not found.\n\n{e}",
            )
        except ValueError as e:
            self.initialization_error = str(e)
            self._update_status("No Model Loaded", "orange")
            self.btn_run.setEnabled(False)

    def _show_model_required_message(self) -> None:
        """Show message that a model is required."""
        self._update_status("No Model Loaded - Select a .osim File", "orange")
        self.btn_run.setEnabled(False)
        self.lbl_details.setText(
            "Click 'Load Model' to select an OpenSim .osim model file."
        )

    def _update_status(self, message: str, color: str) -> None:
        """Update the status label."""
        self.lbl_status.setText(f"Status: {message}")
        self.lbl_status.setStyleSheet(f"color: {color}; font-weight: bold;")

    def init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Header
        header = QLabel("OpenSim Golf Simulation")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Status (no more "Demo Fallback" - only real status)
        self.lbl_status = QLabel("Status: Initializing...")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_status)

        # Controls
        controls = QHBoxLayout()

        self.btn_load = QPushButton("Load Model")
        self.btn_load.clicked.connect(self._load_model_dialog)
        self.btn_load.setFixedWidth(150)
        self.btn_load.setStyleSheet(
            "background-color: #6c757d; color: white; padding: 10px; font-weight: bold;"
        )
        controls.addWidget(self.btn_load)

        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.clicked.connect(self.run_simulation)
        self.btn_run.setFixedWidth(200)
        self.btn_run.setEnabled(False)  # Disabled until model loaded
        self.btn_run.setStyleSheet(
            "background-color: #007acc; color: white; padding: 10px; font-weight: bold;"
        )
        controls.addWidget(self.btn_run)

        layout.addLayout(controls)

        # Visualization (Matplotlib)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Result Details
        self.lbl_details = QLabel("No simulation data.")
        self.lbl_details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_details)

    def _load_model_dialog(self) -> None:
        """Open file dialog to select an OpenSim model."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select OpenSim Model",
            "",
            "OpenSim Models (*.osim);;All Files (*)",
        )
        if file_path:
            self.model_path = file_path
            self._try_load_model()

    def run_simulation(self) -> None:
        if self.model is None:
            QMessageBox.warning(
                self,
                "No Model Loaded",
                "Please load an OpenSim model first.\n\n"
                "Click 'Load Model' to select a .osim file.",
            )
            return

        self.btn_run.setEnabled(False)
        self.lbl_details.setText("Running...")
        QApplication.processEvents()

        try:
            self.result = self.model.run_simulation()
            self.plot_results()
            if self.result:
                self.lbl_details.setText(
                    f"Simulation Complete. Duration: {self.result.time[-1]:.2f}s, "
                    f"Steps: {len(self.result.time)}"
                )
        except NotImplementedError as e:
            # Clear, informative error - not a silent fallback
            QMessageBox.warning(
                self,
                "Simulation Not Available",
                f"OpenSim simulation is not yet fully implemented.\n\n{e}",
            )
            self.lbl_details.setText("Simulation not available - see error message.")
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", str(e))
            self.lbl_details.setText("Error occurred - see error message.")
        finally:
            self.btn_run.setEnabled(True)

    def plot_results(self) -> None:
        if not self.result:
            return

        self.fig.clear()

        # Plot 1: Joint Angles
        ax1 = self.fig.add_subplot(221)
        ax1.plot(self.result.time, self.result.states[:, 0], label="Shoulder Angle")
        ax1.plot(self.result.time, self.result.states[:, 1], label="Wrist Angle")
        ax1.set_title("Joint Angles (rad)")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Joint Torques
        ax2 = self.fig.add_subplot(222)
        ax2.plot(
            self.result.time, self.result.joint_torques[:, 0], label="Shoulder Torque"
        )
        ax2.plot(
            self.result.time, self.result.joint_torques[:, 1], label="Wrist Torque"
        )
        ax2.set_title("Joint Torques (Nm)")
        ax2.legend()
        ax2.grid(True)

        # Plot 3: 2D Trajectory
        ax3 = self.fig.add_subplot(212)
        hand = self.result.marker_positions["Hand"]
        club = self.result.marker_positions["ClubHead"]
        ax3.plot(hand[:, 0], hand[:, 1], "b-", label="Hand Path")
        ax3.plot(club[:, 0], club[:, 1], "r-", label="Clubhead Path")
        ax3.set_title("Swing Trajectory (XZ Plane)")
        ax3.set_aspect("equal")
        ax3.legend()
        ax3.grid(True)

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Check for command line model path
    model_path = sys.argv[1] if len(sys.argv) > 1 else None

    window = OpenSimGolfGUI(model_path=model_path)
    window.show()
    sys.exit(app.exec())
