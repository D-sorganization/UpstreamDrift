#!/usr/bin/env python3
"""
OpenSim Golf GUI
A PyQt6 interface for the OpenSim Golf Model (or its demo fallback).
"""
import sys
from typing import Any

import matplotlib
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
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
    from engines.physics_engines.opensim.python.opensim_golf.core import GolfSwingModel
except ImportError:
    from .opensim_golf.core import GolfSwingModel


class OpenSimGolfGUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OpenSim Golf Interface")
        self.resize(1000, 800)

        # Model
        self.model = GolfSwingModel()
        self.result: Any = None

        self.init_ui()

    def init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Header
        header = QLabel("OpenSim Golf Simulation")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Status
        self.lbl_status = QLabel(
            "Engine Mode: "
            + ("OpenSim Core" if self.model.use_opensim else "Demo Fallback")
        )
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if not self.model.use_opensim:
            self.lbl_status.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
        layout.addWidget(self.lbl_status)

        # Controls
        controls = QHBoxLayout()

        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.clicked.connect(self.run_simulation)
        self.btn_run.setFixedWidth(200)
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

    def run_simulation(self) -> None:
        self.btn_run.setEnabled(False)
        self.lbl_details.setText("Running...")
        QApplication.processEvents()

        try:
            self.result = self.model.run_simulation()
            self.plot_results()
            if self.result:
                self.lbl_details.setText(
                    f"Simulation Complete. Duration: {self.result.time[-1]:.2f}s, Steps: {len(self.result.time)}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", str(e))
            self.lbl_details.setText("Error occurred.")
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
    window = OpenSimGolfGUI()
    window.show()
    sys.exit(app.exec())
