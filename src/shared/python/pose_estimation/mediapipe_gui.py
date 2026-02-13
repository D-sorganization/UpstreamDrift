#!/usr/bin/env python3
"""MediaPipe Analysis GUI — real estimator integration.

Provides a PyQt6 GUI for configuring and running MediaPipe pose estimation
on video files. Replaces the previous mock-only implementation.

Design by Contract:
    Preconditions:
        - Video file must exist and be readable
        - MediaPipe library must be installed (graceful fallback if missing)
    Postconditions:
        - Results are written to output/ directory as JSON
        - Progress is reported accurately based on frame count
    Invariants:
        - GUI remains responsive during analysis (QThread processing)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal
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

from src.shared.python.theme.style_constants import Styles

logger = logging.getLogger(__name__)


class _AnalysisWorker(QThread):
    """Background worker for running MediaPipe analysis.

    Emits signals for progress updates and completion, keeping
    the GUI thread responsive.
    """

    progress = pyqtSignal(int, int, str)  # current_frame, total_frames, message
    finished = pyqtSignal(list)  # results list
    error = pyqtSignal(str)  # error message

    def __init__(
        self,
        video_path: str,
        config: dict[str, Any],
        parent: QThread | None = None,
    ) -> None:
        super().__init__(parent)
        self._video_path = video_path
        self._config = config

    def run(self) -> None:
        """Execute the analysis in a background thread."""
        try:
            from src.shared.python.pose_estimation.mediapipe_estimator import (
                MediaPipeEstimator,
            )

            estimator = MediaPipeEstimator(
                min_detection_confidence=self._config.get(
                    "min_detection_confidence", 0.5
                ),
                min_tracking_confidence=self._config.get(
                    "min_tracking_confidence", 0.5
                ),
                enable_temporal_smoothing=True,
            )
            estimator.load_model()

            # Get frame count for progress
            try:
                import cv2

                cap = cv2.VideoCapture(self._video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except ImportError:
                total_frames = 0

            self.progress.emit(0, total_frames, "Starting analysis...")

            results = estimator.estimate_from_video(Path(self._video_path))

            self.finished.emit(results)

        except ImportError as e:
            self.error.emit(
                f"MediaPipe dependency not installed: {e}\n"
                "Install with: pip install mediapipe opencv-python"
            )
        except (RuntimeError, TypeError, AttributeError) as e:
            self.error.emit(f"Analysis failed: {e}")


def _parse_config(text: str) -> dict[str, Any]:
    """Parse configuration text into a dictionary.

    Args:
        text: Configuration text with key=value lines

    Returns:
        Dictionary of configuration values
    """
    config: dict[str, Any] = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Parse numeric values
            try:
                config[key] = float(value)
            except ValueError:
                if value.lower() in {"true", "false"}:
                    config[key] = value.lower() == "true"
                else:
                    config[key] = value
    return config


class MediaPipeGUI(QMainWindow):
    """MediaPipe Analysis GUI with real estimator integration."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MediaPipe Analysis")
        self.resize(800, 600)
        self._video_path: str = ""
        self._worker: _AnalysisWorker | None = None
        self.init_ui()

    def init_ui(self) -> None:
        """Initialize the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Header
        lbl_title = QLabel("MediaPipe Video Analysis")
        lbl_title.setStyleSheet(Styles.POSE_EST_TITLE)
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
            "min_detection_confidence=0.5\n"
            "min_tracking_confidence=0.5\n"
        )
        self.config_box.setMaximumHeight(100)
        layout.addWidget(self.config_box)

        # Run
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet(Styles.BTN_RUN_MEDIAPIPE)
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
        """Open file dialog to select a video file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_name:
            self._video_path = file_name
            self.lbl_file.setText(file_name)
            self.btn_run.setEnabled(True)
            self.log(f"Loaded video: {file_name}")

    def run_analysis(self) -> None:
        """Start the MediaPipe analysis in a background thread."""
        if not self._video_path:
            return

        self.btn_run.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress.setValue(0)

        config = _parse_config(self.config_box.toPlainText())
        self.log(f"Configuration: {config}")
        self.log("Starting MediaPipe analysis...")

        self._worker = _AnalysisWorker(self._video_path, config)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, current: int, total: int, message: str) -> None:
        """Handle progress updates from the worker thread."""
        if total > 0:
            pct = min(int((current / total) * 100), 100)
            self.progress.setValue(pct)
        self.log(message)

    def _on_finished(self, results: list[Any]) -> None:
        """Handle analysis completion."""
        self.progress.setValue(100)
        self.log(f"Analysis complete! Processed {len(results)} frames.")

        # Save results
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "mediapipe_results.json"

        serialized = []
        for result in results:
            entry: dict[str, Any] = {
                "timestamp": result.timestamp,
                "confidence": result.confidence,
                "joint_angles": result.joint_angles,
            }
            if result.raw_keypoints:
                entry["keypoint_count"] = len(result.raw_keypoints)
            serialized.append(entry)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

        self.log(f"Results saved to {output_file}")
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)

        QMessageBox.information(
            self,
            "Done",
            f"Analysis finished successfully.\n"
            f"Processed {len(results)} frames.\n"
            f"Results saved to {output_file}",
        )

    def _on_error(self, message: str) -> None:
        """Handle analysis errors."""
        self.log(f"ERROR: {message}")
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        QMessageBox.critical(self, "Error", message)

    def log(self, msg: str) -> None:
        """Append a message to the log area."""
        self.log_area.append(msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MediaPipeGUI()
    window.show()
    sys.exit(app.exec())
