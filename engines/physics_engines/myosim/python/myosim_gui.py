#!/usr/bin/env python3
"""
MyoSim Suite GUI
Interface for multi-scale muscle simulation.
"""
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class MyoSimGUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MyoSim Suite")
        self.resize(600, 400)
        self.init_ui()

    def init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        lbl = QLabel("MyoSim Muscle Simulation")
        lbl.setFont(self.font())
        lbl.setStyleSheet("font-size: 20px; font-weight: bold;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

        # Placeholder controls
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Activation:"))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        ctrl_layout.addWidget(slider)
        layout.addLayout(ctrl_layout)

        btn = QPushButton("Simulate Sarcomere")
        layout.addWidget(btn)

        layout.addStretch()

        status = QLabel("Engine Status: Ready (Mock)")
        status.setStyleSheet("color: green")
        layout.addWidget(status)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyoSimGUI()
    window.show()
    sys.exit(app.exec())
