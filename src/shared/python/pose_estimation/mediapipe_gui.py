#!/usr/bin/env python3
"""
MediaPipe Analysis GUI
A user interface for configuring and running MediaPipe pose estimation.
"""

import sys
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MediaPipeGUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MediaPipe Analysis")
        self.resize(800, 600)
        self.init_ui()

    def init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Header
        lbl_title = QLabel("MediaPipe Video Analysis")
        lbl_title.setStyleSheet(
            "font-size: 18px; font-weight: bold; margin-bottom: 10px;"
        )
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_title)

        # Select Video
        self.btn_load = QPushButton("Load Video File...")
        self.btn_load.clicked.connect(self.load_video)
        layout.addWidget(self.btn_load)

        self.lbl_file = QLabel("No file selected.")
        self.lbl_file.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_file)

        # Configuration
        lbl_config = QLabel("Configuration:")
        layout.addWidget(lbl_config)

        self.config_box = QTextEdit()
        self.config_box.setPlainText(
            "# MediaPipe Configuration\n"
            "model_complexity=2\n"
            "min_detection_confidence=0.5\n"
            "min_tracking_confidence=0.5\n"
            "static_image_mode=False"
        )
        self.config_box.setMaximumHeight(100)
        layout.addWidget(self.config_box)

        # Run
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet(
            "background-color: #ff9900; color: white; padding: 10px; font-weight: bold;"
        )
        layout.addWidget(self.btn_run)

        # Progress
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # Log
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

    def load_video(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_name:
            self.lbl_file.setText(file_name)
            self.btn_run.setEnabled(True)
            self.log(f"Loaded video: {file_name}")

    def run_analysis(self) -> None:
        self.btn_run.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress.setValue(0)
        self.log("Starting MediaPipe analysis...")

        # Mocking the process for now - Integration with mediapipe_estimator.py comes here
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)

    def update_progress(self) -> None:
        val: int = self.progress.value()
        if val < 100:
            self.progress.setValue(val + 5) # Faster than OpenPose
            if val % 20 == 0:
                self.log(f"Processing frame {val}...")
        else:
            self.timer.stop()
            self.log("Analysis Complete!")
            self.btn_run.setEnabled(True)
            self.btn_load.setEnabled(True)
            QMessageBox.information(
                self,
                "Done",
                "Analysis finished successfully.\nLandmarks saved to output/",
            )

    def log(self, msg: str) -> None:
        self.log_area.append(msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MediaPipeGUI()
    window.show()
    sys.exit(app.exec())
