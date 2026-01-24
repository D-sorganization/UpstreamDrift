from __future__ import annotations

import csv
import json
import typing

from PyQt6 import QtCore, QtWidgets

from src.shared.python.logging_config import get_logger

from ...sim_widget import MuJoCoSimWidget

if typing.TYPE_CHECKING:
    from ...advanced_gui import AdvancedGolfAnalysisWindow

logger = get_logger(__name__)


class AnalysisTab(QtWidgets.QWidget):
    """
    Tab for real-time biomechanical analysis and data export.
    Displays metrics like club head speed, energy, and recording stats.
    """

    def __init__(
        self,
        sim_widget: MuJoCoSimWidget,
        main_window: AdvancedGolfAnalysisWindow,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.sim_widget = sim_widget
        self.main_window = main_window

        self._setup_ui()

        # Update metrics timer
        self.metrics_timer = QtCore.QTimer(self)
        self.metrics_timer.timeout.connect(self.update_metrics)
        self.metrics_timer.start(100)  # Update every 100ms

    def _setup_ui(self) -> None:
        """Create the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Real-time metrics
        metrics_group = QtWidgets.QGroupBox("Real-Time Metrics")
        metrics_layout = QtWidgets.QFormLayout(metrics_group)

        self.club_speed_label = QtWidgets.QLabel("--")
        self.total_energy_label = QtWidgets.QLabel("--")
        self.recording_label = QtWidgets.QLabel(
            "Not recording"
        )  # Added for update_metrics logic
        self.recording_label.setStyleSheet("font-weight: bold; padding: 5px;")

        self.recording_time_label = QtWidgets.QLabel("--")
        self.num_frames_label = QtWidgets.QLabel("--")

        metrics_layout.addRow("Status:", self.recording_label)
        metrics_layout.addRow("Club Head Speed:", self.club_speed_label)
        metrics_layout.addRow("Total Energy:", self.total_energy_label)
        metrics_layout.addRow("Recording Time:", self.recording_time_label)
        metrics_layout.addRow("Frames Recorded:", self.num_frames_label)

        layout.addWidget(metrics_group)

        # Data export
        export_group = QtWidgets.QGroupBox("Data Export")
        export_layout = QtWidgets.QVBoxLayout(export_group)

        self.export_csv_btn = QtWidgets.QPushButton("Export to CSV")
        self.export_csv_btn.clicked.connect(self.on_export_csv)
        export_layout.addWidget(self.export_csv_btn)

        self.export_json_btn = QtWidgets.QPushButton("Export to JSON")
        self.export_json_btn.clicked.connect(self.on_export_json)
        export_layout.addWidget(self.export_json_btn)

        layout.addWidget(export_group)

        layout.addStretch(1)

    def update_metrics(self) -> None:
        """Update real-time metrics display."""
        recorder = self.sim_widget.get_recorder()
        analyzer = self.sim_widget.get_analyzer()

        # Update recording status
        if recorder.is_recording:
            duration = recorder.get_duration()
            num_frames = recorder.get_num_frames()
            self.recording_label.setText(
                f"Recording: {duration:.2f}s ({num_frames} frames)",
            )
            self.recording_label.setStyleSheet(
                "background-color: #d62728; color: white; font-weight: bold; "
                "padding: 5px;",
            )
        else:
            num_frames = recorder.get_num_frames()
            if num_frames > 0:
                duration = recorder.get_duration()
                self.recording_label.setText(
                    f"Stopped: {duration:.2f}s ({num_frames} frames)",
                )
                self.recording_label.setStyleSheet(
                    "background-color: #ff7f0e; color: white; font-weight: bold; "
                    "padding: 5px;",
                )
            else:
                self.recording_label.setText("Not recording")
                self.recording_label.setStyleSheet("font-weight: bold; padding: 5px;")

        # Update metrics
        if analyzer is not None:
            _, _, club_speed = analyzer.get_club_head_data()
            _, _, total_energy = analyzer.compute_energies()

            self.club_speed_label.setText(
                f"{club_speed * 2.23694:.1f} mph ({club_speed:.1f} m/s)",
            )
            self.total_energy_label.setText(f"{total_energy:.2f} J")

        self.recording_time_label.setText(f"{recorder.get_duration():.2f} s")
        self.num_frames_label.setText(str(recorder.get_num_frames()))

    def on_export_csv(self) -> None:
        """Export recorded data to CSV."""
        recorder = self.sim_widget.get_recorder()

        if recorder.get_num_frames() == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No recorded data available to export.",
            )
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            "",
            "CSV Files (*.csv)",
        )

        if filename:
            try:
                data_dict = recorder.export_to_dict()

                # Write to CSV
                with open(filename, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)

                    # Write header
                    writer.writerow(data_dict.keys())

                    # Write data rows
                    num_rows = len(next(iter(data_dict.values())))
                    for i in range(num_rows):
                        row = [
                            data_dict[key][i] if i < len(data_dict[key]) else ""
                            for key in data_dict
                        ]
                        writer.writerow(row)

                QtWidgets.QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Data exported to {filename}",
                )

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Error exporting data: {e!s}",
                )

    def on_export_json(self) -> None:
        """Export recorded data to JSON."""
        recorder = self.sim_widget.get_recorder()

        if recorder.get_num_frames() == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No recorded data available to export.",
            )
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export JSON",
            "",
            "JSON Files (*.json)",
        )

        if filename:
            try:
                data_dict = recorder.export_to_dict()

                with open(filename, "w") as jsonfile:
                    json.dump(data_dict, jsonfile, indent=2)

                QtWidgets.QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Data exported to {filename}",
                )

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Error exporting data: {e!s}",
                )
