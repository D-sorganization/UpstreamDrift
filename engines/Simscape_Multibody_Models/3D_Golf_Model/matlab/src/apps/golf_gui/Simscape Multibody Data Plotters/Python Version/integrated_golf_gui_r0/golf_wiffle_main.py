#!/usr/bin/env python3
"""
Golf Swing Visualizer - Wiffle_ProV1 Main Application
Enhanced main application with Excel data loading and advanced visualization
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from golf_gui_application import GolfVisualizerWidget
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from wiffle_data_loader import WiffleDataConfig, WiffleDataLoader

# ============================================================================
# DATA LOADING THREAD
# ============================================================================


class DataLoadingThread(QThread):
    """Background thread for loading Excel data"""

    dataLoaded = pyqtSignal(object, object, object)  # BASEQ, ZTCFQ, DELTAQ
    loadingProgress = pyqtSignal(str)
    loadingError = pyqtSignal(str)

    def __init__(self, excel_file_path: str, config: WiffleDataConfig):
        super().__init__()
        self.excel_file_path = excel_file_path
        self.config = config

    def run(self):
        """Load data in background thread"""
        try:
            self.loadingProgress.emit("Initializing Wiffle data loader...")

            # Create loader
            loader = WiffleDataLoader(self.config)

            self.loadingProgress.emit("Loading Excel data...")
            excel_data = loader.load_excel_data(self.excel_file_path)

            self.loadingProgress.emit("Converting to GUI format...")
            baseq, ztcfq, deltaq = loader.convert_to_gui_format(excel_data)

            self.loadingProgress.emit("Data loading completed!")
            self.dataLoaded.emit(baseq, ztcfq, deltaq)

        except Exception as e:
            self.loadingError.emit(f"Error loading data: {str(e)}")


# ============================================================================
# ENHANCED MAIN WINDOW
# ============================================================================


class WiffleGolfMainWindow(QMainWindow):
    """Enhanced main window with Wiffle_ProV1 data support"""

    def __init__(self):
        super().__init__()

        # Window setup
        self.setWindowTitle("Golf Swing Visualizer - Wiffle_ProV1 Edition")
        self.setGeometry(100, 100, 1600, 1000)

        # Data state
        self.baseq_data = None
        self.ztcfq_data = None
        self.deltaq_data = None
        self.data_loaded = False

        # Loading thread
        self.loading_thread = None

        # Create UI
        self._create_ui()
        self._create_menu_bar()
        self._create_toolbar()
        self._create_status_bar()
        self._apply_modern_style()

        # Auto-load default data if available
        self._try_auto_load_data()

        print("🎯 Wiffle Golf Main Window initialized")

    def _create_ui(self):
        """Create the main user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create visualizer widget
        self.visualizer_widget = GolfVisualizerWidget()
        main_layout.addWidget(self.visualizer_widget, 1)

        # Create control panels
        self._create_control_panels()

        # Connect signals
        self._connect_signals()

    def _create_control_panels(self):
        """Create dockable control panels"""
        # Data loading panel
        self.data_panel = self._create_data_panel()
        data_dock = QDockWidget("Data Loading", self)
        data_dock.setWidget(self.data_panel)
        data_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, data_dock)

        # Wiffle-specific controls
        self.wiffle_panel = self._create_wiffle_panel()
        wiffle_dock = QDockWidget("Wiffle Controls", self)
        wiffle_dock.setWidget(self.wiffle_panel)
        wiffle_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, wiffle_dock)

        # Analysis panel
        self.analysis_panel = self._create_analysis_panel()
        analysis_dock = QDockWidget("Analysis", self)
        analysis_dock.setWidget(self.analysis_panel)
        analysis_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, analysis_dock)

    def _create_data_panel(self) -> QWidget:
        """Create data loading control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Data Loading")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        # File selection
        file_group = QGroupBox("Excel File")
        file_layout = QVBoxLayout(file_group)

        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setWordWrap(True)
        file_layout.addWidget(self.file_path_label)

        self.load_file_btn = QPushButton("Load Excel File")
        self.load_file_btn.clicked.connect(self._load_excel_file)
        file_layout.addWidget(self.load_file_btn)

        layout.addWidget(file_group)

        # Loading progress
        progress_group = QGroupBox("Loading Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        self.progress_text = QTextEdit()
        self.progress_text.setMaximumHeight(100)
        self.progress_text.setVisible(False)
        progress_layout.addWidget(self.progress_text)

        layout.addWidget(progress_group)

        # Data info
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout(info_group)

        self.data_info_label = QLabel("No data loaded")
        self.data_info_label.setWordWrap(True)
        info_layout.addWidget(self.data_info_label)

        layout.addWidget(info_group)

        layout.addStretch()
        return panel

    def _create_wiffle_panel(self) -> QWidget:
        """Create Wiffle-specific control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Wiffle Controls")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        # Ball type selection
        ball_group = QGroupBox("Ball Type")
        ball_layout = QVBoxLayout(ball_group)

        self.ball_combo = QComboBox()
        self.ball_combo.addItems(["ProV1", "Wiffle", "Difference"])
        self.ball_combo.currentTextChanged.connect(self._on_ball_type_changed)
        ball_layout.addWidget(self.ball_combo)

        layout.addWidget(ball_group)

        # Data processing options
        processing_group = QGroupBox("Data Processing")
        processing_layout = QVBoxLayout(processing_group)

        self.normalize_time_cb = QCheckBox("Normalize Time")
        self.normalize_time_cb.setChecked(True)
        processing_layout.addWidget(self.normalize_time_cb)

        self.filter_noise_cb = QCheckBox("Filter Noise")
        self.filter_noise_cb.setChecked(True)
        processing_layout.addWidget(self.filter_noise_cb)

        self.interpolate_missing_cb = QCheckBox("Interpolate Missing")
        self.interpolate_missing_cb.setChecked(True)
        processing_layout.addWidget(self.interpolate_missing_cb)

        layout.addWidget(processing_group)

        # Reload button
        self.reload_btn = QPushButton("Reload with New Settings")
        self.reload_btn.clicked.connect(self._reload_data)
        self.reload_btn.setEnabled(False)
        layout.addWidget(self.reload_btn)

        layout.addStretch()
        return panel

    def _create_analysis_panel(self) -> QWidget:
        """Create analysis control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Analysis")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        # Comparison controls
        comparison_group = QGroupBox("Ball Comparison")
        comparison_layout = QVBoxLayout(comparison_group)

        self.show_prov1_cb = QCheckBox("Show ProV1")
        self.show_prov1_cb.setChecked(True)
        comparison_layout.addWidget(self.show_prov1_cb)

        self.show_wiffle_cb = QCheckBox("Show Wiffle")
        self.show_wiffle_cb.setChecked(True)
        comparison_layout.addWidget(self.show_wiffle_cb)

        self.show_difference_cb = QCheckBox("Show Difference")
        self.show_difference_cb.setChecked(False)
        comparison_layout.addWidget(self.show_difference_cb)

        layout.addWidget(comparison_group)

        # Metrics
        metrics_group = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        self.metrics_label = QLabel("No metrics available")
        self.metrics_label.setWordWrap(True)
        metrics_layout.addWidget(self.metrics_label)

        layout.addWidget(metrics_group)

        # Export options
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        self.export_comparison_btn = QPushButton("Export Comparison")
        self.export_comparison_btn.clicked.connect(self._export_comparison)
        self.export_comparison_btn.setEnabled(False)
        export_layout.addWidget(self.export_comparison_btn)

        layout.addWidget(export_group)

        layout.addStretch()
        return panel

    def _create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        load_action = QAction("Load Excel File", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._load_excel_file)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        reset_camera_action = QAction("Reset Camera", self)
        reset_camera_action.triggered.connect(self.visualizer_widget.reset_camera)
        view_menu.addAction(reset_camera_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_toolbar(self):
        """Create toolbar"""
        toolbar = self.addToolBar("Main Toolbar")

        # Load data action
        load_action = QAction("📁 Load Excel", self)
        load_action.triggered.connect(self._load_excel_file)
        toolbar.addAction(load_action)

        toolbar.addSeparator()

        # Playback controls
        play_action = QAction("▶️ Play", self)
        play_action.triggered.connect(self.visualizer_widget.play)
        toolbar.addAction(play_action)

        pause_action = QAction("⏸️ Pause", self)
        pause_action.triggered.connect(self.visualizer_widget.pause)
        toolbar.addAction(pause_action)

        toolbar.addSeparator()

        # Camera controls
        reset_camera_action = QAction("🎯 Reset Camera", self)
        reset_camera_action.triggered.connect(self.visualizer_widget.reset_camera)
        toolbar.addAction(reset_camera_action)

    def _create_status_bar(self):
        """Create status bar"""
        self.statusBar().showMessage("Ready to load Wiffle_ProV1 data")

    def _connect_signals(self):
        """Connect widget signals"""
        # Visualizer signals
        self.visualizer_widget.frameChanged.connect(self._on_frame_changed)
        self.visualizer_widget.statusMessage.connect(self.statusBar().showMessage)

        # Control panel signals
        self.show_prov1_cb.toggled.connect(self._on_visibility_changed)
        self.show_wiffle_cb.toggled.connect(self._on_visibility_changed)
        self.show_difference_cb.toggled.connect(self._on_visibility_changed)

    def _try_auto_load_data(self):
        """Try to automatically load the default Excel file"""
        default_file = Path("Matlab Inverse Dynamics/Wiffle_ProV1_club_3D_data.xlsx")
        if default_file.exists():
            self._load_excel_file(str(default_file))

    def _load_excel_file(self, file_path: str = None):
        """Load Excel file with Wiffle_ProV1 data"""
        if file_path is None:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Wiffle_ProV1 Excel File",
                str(Path("Matlab Inverse Dynamics")),
                "Excel Files (*.xlsx *.xls)",
            )

        if not file_path:
            return

        # Update UI
        self.file_path_label.setText(f"Loading: {Path(file_path).name}")
        self.progress_bar.setVisible(True)
        self.progress_text.setVisible(True)
        self.progress_text.clear()
        self.progress_text.append("Starting data load...")

        # Create configuration
        config = WiffleDataConfig(
            normalize_time=self.normalize_time_cb.isChecked(),
            filter_noise=self.filter_noise_cb.isChecked(),
            interpolate_missing=self.interpolate_missing_cb.isChecked(),
        )

        # Start loading thread
        self.loading_thread = DataLoadingThread(file_path, config)
        self.loading_thread.loadingProgress.connect(self._on_loading_progress)
        self.loading_thread.dataLoaded.connect(self._on_data_loaded)
        self.loading_thread.loadingError.connect(self._on_loading_error)
        self.loading_thread.start()

    def _on_loading_progress(self, message: str):
        """Handle loading progress updates"""
        self.progress_text.append(message)
        self.statusBar().showMessage(message)

    def _on_data_loaded(self, baseq, ztcfq, deltaq):
        """Handle successful data loading"""
        self.baseq_data = baseq
        self.ztcfq_data = ztcfq
        self.deltaq_data = deltaq
        self.data_loaded = True

        # Load data into visualizer
        self.visualizer_widget.load_data(baseq, ztcfq, deltaq)

        # Update UI
        self.file_path_label.setText(f"Loaded: {len(baseq)} frames")
        self.progress_bar.setVisible(False)
        self.progress_text.setVisible(False)
        self.reload_btn.setEnabled(True)
        self.export_comparison_btn.setEnabled(True)

        # Update data info
        info_text = (
            f"ProV1: {len(baseq)} frames\n"
            f"Wiffle: {len(ztcfq)} frames\n"
            f"Time: {baseq['Time'].min():.3f} - {baseq['Time'].max():.3f}s"
        )
        self.data_info_label.setText(info_text)

        # Calculate and display metrics
        self._calculate_metrics()

        self.statusBar().showMessage("Data loaded successfully!")

    def _on_loading_error(self, error_message: str):
        """Handle loading errors"""
        self.progress_bar.setVisible(False)
        self.progress_text.setVisible(False)
        QMessageBox.critical(self, "Loading Error", error_message)
        self.statusBar().showMessage("Data loading failed")

    def _on_ball_type_changed(self, ball_type: str):
        """Handle ball type selection change"""
        if not self.data_loaded:
            return

        # Update visualizer data based on selection
        if ball_type == "ProV1":
            self.visualizer_widget.load_data(
                self.baseq_data, self.ztcfq_data, self.deltaq_data
            )
        elif ball_type == "Wiffle":
            # Swap BASEQ and ZTCFQ to show Wiffle as primary
            self.visualizer_widget.load_data(
                self.ztcfq_data, self.baseq_data, self.deltaq_data
            )
        elif ball_type == "Difference":
            # Show difference data
            self.visualizer_widget.load_data(
                self.deltaq_data, self.baseq_data, self.ztcfq_data
            )

    def _on_visibility_changed(self):
        """Handle visibility checkbox changes"""
        if not self.data_loaded:
            return

        # Update render configuration based on checkboxes
        config = self.visualizer_widget.render_config

        # Update visibility based on current ball type
        ball_type = self.ball_combo.currentText()
        if ball_type == "ProV1":
            config.show_forces["BASEQ"] = self.show_prov1_cb.isChecked()
            config.show_forces["ZTCFQ"] = self.show_wiffle_cb.isChecked()
            config.show_forces["DELTAQ"] = self.show_difference_cb.isChecked()
        elif ball_type == "Wiffle":
            config.show_forces["ZTCFQ"] = self.show_prov1_cb.isChecked()
            config.show_forces["BASEQ"] = self.show_wiffle_cb.isChecked()
            config.show_forces["DELTAQ"] = self.show_difference_cb.isChecked()

        # Trigger visualizer update
        self.visualizer_widget.update()

    def _on_frame_changed(self, frame_idx: int):
        """Handle frame change events"""
        if self.data_loaded:
            # Update metrics for current frame
            self._update_frame_metrics(frame_idx)

    def _reload_data(self):
        """Reload data with current settings"""
        if hasattr(self, "loading_thread") and self.loading_thread:
            self.loading_thread.quit()
            self.loading_thread.wait()

        # Get current file path
        if hasattr(self, "file_path_label"):
            current_file = self.file_path_label.text()
            if current_file.startswith("Loaded:") or current_file.startswith(
                "Loading:"
            ):
                # Extract file path from previous load
                # This is a simplified approach - in practice you'd store the path
                self._load_excel_file()

    def _calculate_metrics(self):
        """Calculate comparison metrics between ProV1 and Wiffle"""
        if not self.data_loaded:
            return

        try:
            # Calculate basic metrics
            prov1_max_speed = self._calculate_max_speed(self.baseq_data)
            wiffle_max_speed = self._calculate_max_speed(self.ztcfq_data)

            # Calculate trajectory differences
            trajectory_diff = self._calculate_trajectory_difference()

            metrics_text = (
                f"Max Speed:\n"
                f"  ProV1: {prov1_max_speed:.2f} m/s\n"
                f"  Wiffle: {wiffle_max_speed:.2f} m/s\n"
                f"  Difference: {abs(prov1_max_speed - wiffle_max_speed):.2f} m/s\n\n"
                f"Trajectory RMS: {trajectory_diff:.3f} m"
            )

            self.metrics_label.setText(metrics_text)

        except Exception as e:
            self.metrics_label.setText(f"Error calculating metrics: {str(e)}")

    def _calculate_max_speed(self, data) -> float:
        """Calculate maximum clubhead speed"""
        try:
            # Calculate velocity from position data
            dt = data["Time"].diff().mean()
            vx = data["CHx"].diff() / dt
            vy = data["CHy"].diff() / dt
            vz = data["CHz"].diff() / dt

            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            return speed.max()
        except (KeyError, ValueError, IndexError):
            return 0.0

    def _calculate_trajectory_difference(self) -> float:
        """Calculate RMS difference between ProV1 and Wiffle trajectories"""
        try:
            # Calculate differences for clubhead position
            diff_x = self.baseq_data["CHx"] - self.ztcfq_data["CHx"]
            diff_y = self.baseq_data["CHy"] - self.ztcfq_data["CHy"]
            diff_z = self.baseq_data["CHz"] - self.ztcfq_data["CHz"]

            rms_diff = np.sqrt(np.mean(diff_x**2 + diff_y**2 + diff_z**2))
            return rms_diff
        except (KeyError, ValueError, IndexError):
            return 0.0

    def _update_frame_metrics(self, frame_idx: int):
        """Update metrics for current frame"""
        if not self.data_loaded or frame_idx >= len(self.baseq_data):
            return

        # Calculate frame-specific metrics
        try:
            prov1_pos = np.array(
                [
                    self.baseq_data.iloc[frame_idx]["CHx"],
                    self.baseq_data.iloc[frame_idx]["CHy"],
                    self.baseq_data.iloc[frame_idx]["CHz"],
                ]
            )

            wiffle_pos = np.array(
                [
                    self.ztcfq_data.iloc[frame_idx]["CHx"],
                    self.ztcfq_data.iloc[frame_idx]["CHy"],
                    self.ztcfq_data.iloc[frame_idx]["CHz"],
                ]
            )

            distance = np.linalg.norm(prov1_pos - wiffle_pos)

            # Update status bar with frame info
            time_val = self.baseq_data.iloc[frame_idx]["Time"]
            self.statusBar().showMessage(
                f"Frame {frame_idx + 1}, Time: {time_val:.3f}s, "
                f"Distance: {distance:.3f}m"
            )

        except Exception:
            pass

    def _export_comparison(self):
        """Export comparison data"""
        if not self.data_loaded:
            QMessageBox.warning(self, "No Data", "No data loaded to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Comparison Data",
            "wiffle_prov1_comparison.csv",
            "CSV Files (*.csv)",
        )

        if file_path:
            try:
                # Create comparison DataFrame
                comparison_data = pd.DataFrame()
                comparison_data["Time"] = self.baseq_data["Time"]

                # Add ProV1 positions
                comparison_data["ProV1_CHx"] = self.baseq_data["CHx"]
                comparison_data["ProV1_CHy"] = self.baseq_data["CHy"]
                comparison_data["ProV1_CHz"] = self.baseq_data["CHz"]

                # Add Wiffle positions
                comparison_data["Wiffle_CHx"] = self.ztcfq_data["CHx"]
                comparison_data["Wiffle_CHy"] = self.ztcfq_data["CHy"]
                comparison_data["Wiffle_CHz"] = self.ztcfq_data["CHz"]

                # Add differences
                comparison_data["Diff_CHx"] = self.deltaq_data["CHx"]
                comparison_data["Diff_CHy"] = self.deltaq_data["CHy"]
                comparison_data["Diff_CHz"] = self.deltaq_data["CHz"]

                comparison_data.to_csv(file_path, index=False)
                QMessageBox.information(
                    self, "Export Complete", f"Data exported to {file_path}"
                )

            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Error exporting data: {str(e)}"
                )

    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Golf Swing Visualizer",
            "Golf Swing Visualizer - Wiffle_ProV1 Edition\n\n"
            "Advanced 3D visualization tool for comparing golf ball "
            "motion capture data.\n\n"
            "Features:\n"
            "• Excel data loading and processing\n"
            "• Real-time 3D visualization\n"
            "• Ball comparison analysis\n"
            "• Performance metrics calculation\n"
            "• Data export capabilities\n\n"
            "Version: 1.0",
        )

    def _apply_modern_style(self):
        """Apply modern styling to the application"""
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """
        )


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Golf Swing Visualizer - Wiffle_ProV1")
    app.setApplicationVersion("1.0")

    # Create and show main window
    window = WiffleGolfMainWindow()
    window.show()

    # Start application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
