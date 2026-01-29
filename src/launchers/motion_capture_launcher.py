#!/usr/bin/env python3
"""
Motion Capture & Analysis Launcher
Central hub for C3D visualization and Markerless Pose Estimation.
"""
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
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

APPS = [
    {
        "name": "C3D Motion Viewer",
        "desc": "Visualize and analyze optical motion capture data (.c3d).",
        "path": "src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/c3d_viewer.py",
        "icon": "📈",
    },
    {
        "name": "OpenPose Analysis",
        "desc": "High-accuracy body pose estimation (Academic License).",
        "path": "src/shared/python/pose_estimation/openpose_gui.py",
        "icon": "🎥",
    },
    {
        "name": "MediaPipe Analysis",
        "desc": "Fast, permissive license pose estimation (Apache 2.0).",
        "path": "src/shared/python/pose_estimation/mediapipe_gui.py",
        "icon": "⚡",
    },
]


class MoCapLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Capture")
        self.resize(500, 450)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Header
        lbl_title = QLabel("Motion Capture")
        lbl_title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_title)

        lbl_sub = QLabel("Select an analysis tool to launch:")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sub.setStyleSheet("color: #666;")
        layout.addWidget(lbl_sub)

        # Buttons
        for app in APPS:
            btn_frame = QFrame()
            btn_frame.setStyleSheet(
                """
                QFrame {
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                }
                QFrame:hover {
                    background-color: #e2e6ea;
                    border-color: #adb5bd;
                }
            """
            )
            frame_layout = QVBoxLayout(btn_frame)

            btn = QPushButton(f"{app['icon']}  {app['name']}")
            btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            btn.setStyleSheet(
                "text-align: left; border: none; background: transparent;"
            )
            btn.clicked.connect(lambda checked, _p=app["path"]: self.launch_app(_p))

            desc = QLabel(app["desc"])
            desc.setStyleSheet(
                "color: #555; font-size: 11px; margin-left: 28px; border: none; background: transparent;"
            )

            frame_layout.addWidget(btn)
            frame_layout.addWidget(desc)
            layout.addWidget(btn_frame)

        layout.addStretch()

    def launch_app(self, relative_path):
        script_path = REPO_ROOT / relative_path
        if not script_path.exists():
            QMessageBox.critical(self, "Error", f"Script not found:\n{script_path}")
            return

        try:
            # Launch in new process
            subprocess.Popen([sys.executable, str(script_path)], cwd=REPO_ROOT)
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", str(e))


def main():
    app = QApplication(sys.argv)
    window = MoCapLauncher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
