#!/usr/bin/env python3
"""
Unified MuJoCo Launcher
Hub for accessing MuJoCo Humanoid Simulation and Analysis Dashboard.
"""

import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()

MODES = [
    {
        "name": "Humanoid Simulation",
        "desc": "Full-body biomechanics simulation with muscle dynamics.",
        "path": "src/engines/physics_engines/mujoco/python/humanoid_launcher.py",
        "icon": "🏃",
    },
    {
        "name": "Analysis Dashboard",
        "desc": "Real-time plotting, video analysis, and data visualization.",
        "module": "mujoco_humanoid_golf",  # Launch as module for dashboard
        "cwd_suffix": "src/engines/physics_engines/mujoco/python",
        "icon": "📊",
    },
]


class MujocoUnifiedLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MuJoCo Unified Interface")
        self.resize(500, 350)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Header
        lbl_title = QLabel("MuJoCo Golf Engine")
        lbl_title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_title)

        lbl_sub = QLabel("Select simulation mode:")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sub.setStyleSheet("color: #666;")
        layout.addWidget(lbl_sub)

        # Buttons
        for mode in MODES:
            btn_frame = QFrame()
            btn_frame.setStyleSheet("""
                QFrame {
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                }
                QFrame:hover {
                    background-color: #e2e6ea;
                    border-color: #adb5bd;
                }
            """)
            frame_layout = QVBoxLayout(btn_frame)

            btn = QPushButton(f"{mode['icon']}  {mode['name']}")
            btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            btn.setStyleSheet(
                "text-align: left; border: none; background: transparent;"
            )
            # Pass mode dict to handler
            btn.clicked.connect(lambda checked, _m=mode: self.launch_mode(_m))

            desc = QLabel(mode["desc"])
            desc.setStyleSheet(
                "color: #555; font-size: 11px; margin-left: 28px; border: none; background: transparent;"
            )

            frame_layout.addWidget(btn)
            frame_layout.addWidget(desc)
            layout.addWidget(btn_frame)

        layout.addStretch()

    def launch_mode(self, mode):
        try:
            cmd = [sys.executable]
            cwd = REPO_ROOT

            if "path" in mode:
                script_path = REPO_ROOT / mode["path"]
                if not script_path.exists():
                    QMessageBox.critical(
                        self, "Error", f"Script not found:\n{script_path}"
                    )
                    return
                cmd.append(str(script_path))
            elif "module" in mode:
                cmd.extend(["-m", mode["module"]])
                if "cwd_suffix" in mode:
                    cwd = REPO_ROOT / mode["cwd_suffix"]

            # Launch
            subprocess.Popen(cmd, cwd=cwd)

        except Exception as e:
            QMessageBox.critical(self, "Launch Error", str(e))


def main():
    app = QApplication(sys.argv)
    window = MujocoUnifiedLauncher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
