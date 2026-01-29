#!/usr/bin/env python3
"""
Matlab Unified Launcher
Consolidates Simscape Models and Analysis Tools into a single interface.
"""

import os
import subprocess
import sys
from pathlib import Path

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Constants for Paths
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
MATLAB_MODELS = [
    {
        "name": "Simscape 2D Model",
        "desc": "2D Simscape Multibody Golf Swing Model (.slx)",
        "path": "src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/GolfSwingZVCF.slx",
        "type": "model",
    },
    {
        "name": "Simscape 3D Model",
        "desc": "3D Simscape Multibody Golf Swing Model (.slx)",
        "path": "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/GolfSwing3D_Kinetic.slx",
        "type": "model",
    },
    {
        "name": "Dataset Generator",
        "desc": "Forward Dynamics Dataset Generator GUI (.m)",
        "path": "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/scripts/dataset_generator/Dataset_GUI.m",
        "type": "tool",
    },
    {
        "name": "Analysis GUI",
        "desc": "Golf Swing Analysis & Plotting Suite (.m)",
        "path": "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/2D GUI/main_scripts/golf_swing_analysis_gui.m",
        "type": "tool",
    },
]


class MatlabLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Matlab Models & Tools")
        self.resize(600, 500)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        title = QLabel("Matlab Golf Models")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel("Select a model or tool to launch in MATLAB.")
        desc.setStyleSheet("color: #666;")
        layout.addWidget(desc)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # Grid of buttons
        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)

        row = 0
        for item in MATLAB_MODELS:
            btn_frame = QFrame()
            btn_frame.setStyleSheet(
                """
                QFrame {
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                }
                QFrame:hover {
                    background-color: #e9ecef;
                    border-color: #adb5bd;
                }
            """
            )
            btn_layout = QVBoxLayout(btn_frame)

            lbl_name = QLabel(item["name"])
            lbl_name.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            lbl_name.setStyleSheet("border: none; background: transparent;")

            lbl_desc = QLabel(item["desc"])
            lbl_desc.setStyleSheet(
                "color: #666; font-size: 11px; border: none; background: transparent;"
            )
            lbl_desc.setWordWrap(True)

            btn_launch = QPushButton("Launch")
            btn_launch.setStyleSheet(
                """
                QPushButton {
                    background-color: #007bff;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 4px;
                }
                QPushButton:hover { background-color: #0056b3; }
            """
            )
            # Fix lambda closure early binding
            btn_launch.clicked.connect(
                lambda checked, _p=item["path"]: self.launch_file(_p)
            )

            btn_layout.addWidget(lbl_name)
            btn_layout.addWidget(lbl_desc)
            btn_layout.addWidget(btn_launch)

            grid_layout.addWidget(btn_frame, row // 2, row % 2)
            row += 1

        layout.addLayout(grid_layout)
        layout.addStretch()

    def launch_file(self, relative_path: str):
        full_path = REPO_ROOT / relative_path
        if not full_path.exists():
            QMessageBox.critical(self, "Error", f"File not found:\n{full_path}")
            return

        try:
            if os.name == "nt":
                os.startfile(full_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(full_path)], check=True)
            else:
                subprocess.run(["xdg-open", str(full_path)], check=True)
        except Exception as e:
            QMessageBox.critical(self, "Launch Failed", f"Could not launch file:\n{e}")


def main():
    app = QApplication(sys.argv)

    # Optional: Set stylesheet or theme
    app.setStyle("Fusion")

    window = MatlabLauncher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
