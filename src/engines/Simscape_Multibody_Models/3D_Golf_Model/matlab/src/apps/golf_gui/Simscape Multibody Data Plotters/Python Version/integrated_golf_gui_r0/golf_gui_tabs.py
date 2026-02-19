"""Tab widgets for the Golf Swing Visualizer application.

Contains MotionCaptureTab, SimulinkModelTab, and ComparisonTab.
Extracted from golf_gui_application.py for Single Responsibility Principle.
"""

from __future__ import annotations

import traceback

import numpy as np
from golf_data_core import FrameData, FrameProcessor, RenderConfig
from golf_playback_controller import SmoothPlaybackController
from golf_visualizer_widget import GolfVisualizerWidget
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from wiffle_data_loader import MotionDataLoader


class MotionCaptureTab(QWidget):
    """Tab for motion capture data visualization with smooth playback."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.parent = parent
        self.frame_processor = None

        # Use smooth playback controller instead of QTimer
        self.playback_controller = SmoothPlaybackController(self)
        self.playback_controller.frameUpdated.connect(self._on_smooth_frame_updated)
        self.playback_controller.positionChanged.connect(self._on_position_changed)

        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self) -> None:
        """Setup the motion capture tab UI."""
        layout = QVBoxLayout()

        # Control panel
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel)

        # 3D visualization area
        self.opengl_widget = GolfVisualizerWidget()
        layout.addWidget(self.opengl_widget)

        # Status bar
        self.status_label = QLabel("Ready - Load motion capture data to begin")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def _create_control_panel(self) -> QGroupBox:
        """Create the control panel for motion capture data."""
        panel = QGroupBox("Motion Capture Controls")
        layout = QGridLayout()

        # Data selection
        layout.addWidget(QLabel("Swing Type:"), 0, 0)
        self.swing_combo = QComboBox()
        self.swing_combo.addItems(["TW Wiffle", "TW ProV1", "GW Wiffle", "GW ProV1"])
        layout.addWidget(self.swing_combo, 0, 1)

        # Load button
        self.load_button = QPushButton("Load Data")
        self.load_button.setMaximumWidth(100)  # Make button smaller
        layout.addWidget(self.load_button, 0, 2)

        # Playback controls
        layout.addWidget(QLabel("Playback:"), 1, 0)

        self.play_button = QPushButton("Play")
        layout.addWidget(self.play_button, 1, 1)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        layout.addWidget(self.frame_slider, 1, 2)

        self.frame_label = QLabel("Frame: 0/0")
        layout.addWidget(self.frame_label, 1, 3)

        # Visualization options
        layout.addWidget(QLabel("Display:"), 2, 0)

        self.show_body_check = QCheckBox("Body Segments")
        self.show_body_check.setChecked(True)
        layout.addWidget(self.show_body_check, 2, 1)

        self.show_club_check = QCheckBox("Golf Club")
        self.show_club_check.setChecked(True)
        layout.addWidget(self.show_club_check, 2, 2)

        self.show_ground_check = QCheckBox("Ground")
        self.show_ground_check.setChecked(True)
        layout.addWidget(self.show_ground_check, 2, 3)

        panel.setLayout(layout)
        return panel

    def _setup_connections(self) -> None:
        """Setup signal connections."""
        self.load_button.clicked.connect(self._load_motion_capture_data)
        self.play_button.clicked.connect(self._toggle_playback)
        self.frame_slider.valueChanged.connect(self._on_slider_moved)
        self.swing_combo.currentTextChanged.connect(self._on_swing_changed)

    def _load_motion_capture_data(self) -> None:
        """Load motion capture data."""
        try:
            swing_type = self.swing_combo.currentText()
            self.status_label.setText(f"Loading {swing_type} data...")

            # Load data using the existing MotionDataLoader
            loader = MotionDataLoader()
            excel_data = loader.load_data()  # Load the Excel data first
            baseq_data, ztcfq_data, deltaq_data = loader.convert_to_gui_format(
                excel_data
            )

            # Create frame processor with config
            config = RenderConfig()
            self.frame_processor = FrameProcessor(
                (baseq_data, ztcfq_data, deltaq_data), config
            )

            # Load into smooth playback controller
            self.playback_controller.load_frame_processor(self.frame_processor)

            # Update UI
            total_frames = len(self.frame_processor.time_vector)
            self.frame_slider.setMaximum(total_frames - 1)
            self.frame_label.setText(f"Frame: 0/{total_frames}")

            # Initialize visualization
            self.opengl_widget.load_data_from_dataframes(
                (baseq_data, ztcfq_data, deltaq_data)
            )

            self.status_label.setText(
                f"Loaded {swing_type} data successfully - Smooth playback ready!"
            )

        except (RuntimeError, ValueError, OSError) as e:
            self.status_label.setText(f"Error loading data: {str(e)}")
            traceback.print_exc()

    def _on_swing_changed(self, swing_type: str) -> None:
        """Handle swing type change."""
        if self.frame_processor is not None:
            self._load_motion_capture_data()

    def _toggle_playback(self) -> None:
        """Toggle smooth playback."""
        if not self.frame_processor:
            return

        self.playback_controller.toggle_playback()

        if self.playback_controller.is_playing:
            self.play_button.setText("Pause")
        else:
            self.play_button.setText("Play")

    def _on_slider_moved(self, value: int) -> None:
        """Handle manual slider movement (scrubbing)."""
        # Seek to slider position for smooth scrubbing
        self.playback_controller.seek(float(value))

    def _on_position_changed(self, position: float) -> None:
        """Update UI when playback position changes."""
        total_frames = (
            len(self.frame_processor.time_vector) if self.frame_processor else 0
        )

        # Update frame label with fractional position for smooth display
        self.frame_label.setText(f"Frame: {position:.1f}/{total_frames}")

        # Update slider (without triggering valueChanged)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(int(position))
        self.frame_slider.blockSignals(False)

    def _on_smooth_frame_updated(self, frame_data: FrameData) -> None:
        """Called on every interpolated frame update (60+ FPS!)."""
        if not self.opengl_widget.renderer:
            return

        try:
            # Get current render config from UI checkboxes
            render_config = RenderConfig()
            render_config.show_body_segments = {
                "left_forearm": self.show_body_check.isChecked(),
                "left_upper_arm": self.show_body_check.isChecked(),
                "right_forearm": self.show_body_check.isChecked(),
                "right_upper_arm": self.show_body_check.isChecked(),
                "left_shoulder_neck": self.show_body_check.isChecked(),
                "right_shoulder_neck": self.show_body_check.isChecked(),
            }
            render_config.show_club = self.show_club_check.isChecked()
            render_config.show_ground = self.show_ground_check.isChecked()

            # Update 3D visualization with interpolated frame
            self.opengl_widget.update_frame(frame_data, render_config)

        except ImportError as e:
            self.status_label.setText(f"Visualization error: {str(e)}")


class SimulinkModelTab(QWidget):
    """Tab for Simulink model data visualization."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.parent = parent
        self.frame_processor = None

        # Use smooth playback controller
        self.playback_controller = SmoothPlaybackController(self)
        self.playback_controller.frameUpdated.connect(self._on_smooth_frame_updated)
        self.playback_controller.positionChanged.connect(self._on_position_changed)

        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self) -> None:
        """Setup the Simulink model tab UI."""
        layout = QVBoxLayout()

        # Control panel
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel)

        # 3D visualization area
        self.opengl_widget = GolfVisualizerWidget()
        layout.addWidget(self.opengl_widget)

        # Status bar
        self.status_label = QLabel("Ready - Load Simulink model data to begin")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def _create_control_panel(self) -> QGroupBox:
        """Create the control panel for Simulink model data."""
        panel = QGroupBox("Simulink Model Controls")
        layout = QGridLayout()

        # Data selection
        layout.addWidget(QLabel("Model Source:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(
            ["Simscape Multibody", "MuJoCo", "Drake", "Pinocchio"]
        )
        layout.addWidget(self.model_combo, 0, 1)

        # Load button
        self.load_button = QPushButton("Load Model Data")
        self.load_button.setMaximumWidth(150)
        layout.addWidget(self.load_button, 0, 2)

        # Playback controls
        layout.addWidget(QLabel("Playback:"), 1, 0)

        self.play_button = QPushButton("Play")
        layout.addWidget(self.play_button, 1, 1)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        layout.addWidget(self.frame_slider, 1, 2)

        self.frame_label = QLabel("Frame: 0/0")
        layout.addWidget(self.frame_label, 1, 3)

        # Visualization options
        layout.addWidget(QLabel("Display:"), 2, 0)

        self.show_body_check = QCheckBox("Robot Segments")
        self.show_body_check.setChecked(True)
        layout.addWidget(self.show_body_check, 2, 1)

        self.show_club_check = QCheckBox("Golf Club")
        self.show_club_check.setChecked(True)
        layout.addWidget(self.show_club_check, 2, 2)

        self.show_ground_check = QCheckBox("Ground")
        self.show_ground_check.setChecked(True)
        layout.addWidget(self.show_ground_check, 2, 3)

        panel.setLayout(layout)
        return panel

    def _setup_connections(self) -> None:
        """Setup signal connections."""
        self.load_button.clicked.connect(self._load_model_data)
        self.play_button.clicked.connect(self._toggle_playback)
        self.frame_slider.valueChanged.connect(self._on_slider_moved)

    def _load_model_data(self) -> None:
        """Load model data (currently using MotionDataLoader as proxy)."""
        try:
            model_source = self.model_combo.currentText()
            self.status_label.setText(f"Loading {model_source} data...")

            # Load data using the existing MotionDataLoader (proxy for now)
            loader = MotionDataLoader()
            excel_data = loader.load_data()
            baseq_data, ztcfq_data, deltaq_data = loader.convert_to_gui_format(
                excel_data
            )

            # Create frame processor with config
            config = RenderConfig()
            self.frame_processor = FrameProcessor(
                (baseq_data, ztcfq_data, deltaq_data), config
            )

            # Load into smooth playback controller
            self.playback_controller.load_frame_processor(self.frame_processor)

            # Update UI
            total_frames = len(self.frame_processor.time_vector)
            self.frame_slider.setMaximum(total_frames - 1)
            self.frame_label.setText(f"Frame: 0/{total_frames}")

            # Initialize visualization
            self.opengl_widget.load_data_from_dataframes(
                (baseq_data, ztcfq_data, deltaq_data)
            )

            self.status_label.setText(f"Loaded {model_source} data successfully")

        except (RuntimeError, ValueError, OSError) as e:
            self.status_label.setText(f"Error loading data: {str(e)}")
            traceback.print_exc()

    def _toggle_playback(self) -> None:
        """Toggle smooth playback."""
        if not self.frame_processor:
            return

        self.playback_controller.toggle_playback()

        if self.playback_controller.is_playing:
            self.play_button.setText("Pause")
        else:
            self.play_button.setText("Play")

    def _on_slider_moved(self, value: int) -> None:
        """Handle manual slider movement."""
        self.playback_controller.seek(float(value))

    def _on_position_changed(self, position: float) -> None:
        """Update UI when playback position changes."""
        total_frames = (
            len(self.frame_processor.time_vector) if self.frame_processor else 0
        )
        self.frame_label.setText(f"Frame: {position:.1f}/{total_frames}")
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(int(position))
        self.frame_slider.blockSignals(False)

    def _on_smooth_frame_updated(self, frame_data: FrameData) -> None:
        """Called on every interpolated frame update."""
        if not self.opengl_widget.renderer:
            return

        try:
            render_config = RenderConfig()
            # For robot model, we might want different segment mapping
            render_config.show_body_segments = {
                "left_forearm": self.show_body_check.isChecked(),
                "left_upper_arm": self.show_body_check.isChecked(),
                "right_forearm": self.show_body_check.isChecked(),
                "right_upper_arm": self.show_body_check.isChecked(),
                "left_shoulder_neck": self.show_body_check.isChecked(),
                "right_shoulder_neck": self.show_body_check.isChecked(),
            }
            render_config.show_club = self.show_club_check.isChecked()
            render_config.show_ground = self.show_ground_check.isChecked()

            self.opengl_widget.update_frame(frame_data, render_config)

        except ImportError as e:
            self.status_label.setText(f"Visualization error: {str(e)}")


class ComparisonTab(QWidget):
    """Tab for comparing motion capture vs Simulink model data."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.parent = parent
        self.frame_processor_mocap = None
        self.frame_processor_model = None

        # Shared playback controller driving both
        self.playback_controller = SmoothPlaybackController(self)
        self.playback_controller.frameUpdated.connect(self._on_smooth_frame_updated)
        self.playback_controller.positionChanged.connect(self._on_position_changed)

        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self) -> None:
        """Setup the comparison tab UI."""
        layout = QVBoxLayout()

        # Control panel
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel)

        # Split view visualization area
        split_layout = QGridLayout()

        # Left: Motion Capture
        self.mocap_widget = GolfVisualizerWidget()
        split_layout.addWidget(QLabel("Reference (Motion Capture)"), 0, 0)
        split_layout.addWidget(self.mocap_widget, 1, 0)

        # Right: Model
        self.model_widget = GolfVisualizerWidget()
        split_layout.addWidget(QLabel("Simulation (Model)"), 0, 1)
        split_layout.addWidget(self.model_widget, 1, 1)

        layout.addLayout(split_layout)

        # Metrics Panel
        self.metrics_label = QLabel("Comparison Metrics: Load data to begin analysis")
        self.metrics_label.setStyleSheet(
            "font-weight: bold; color: #333; padding: 5px;"
        )
        layout.addWidget(self.metrics_label)

        self.setLayout(layout)

    def _create_control_panel(self) -> QGroupBox:
        """Create comparison controls."""
        panel = QGroupBox("Comparison Controls")
        layout = QGridLayout()

        # Load buttons
        self.load_btn = QPushButton("Load Comparison Data")
        layout.addWidget(self.load_btn, 0, 0)

        # Playback controls
        layout.addWidget(QLabel("Playback:"), 0, 1)
        self.play_button = QPushButton("Play Sync")
        layout.addWidget(self.play_button, 0, 2)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        layout.addWidget(self.frame_slider, 0, 3)

        self.frame_label = QLabel("Frame: 0/0")
        layout.addWidget(self.frame_label, 0, 4)

        panel.setLayout(layout)
        return panel

    def _setup_connections(self) -> None:
        self.load_btn.clicked.connect(self._load_comparison_data)
        self.play_button.clicked.connect(self._toggle_playback)
        self.frame_slider.valueChanged.connect(self._on_slider_moved)

    def _load_comparison_data(self) -> None:
        """Load two datasets for comparison."""
        try:
            self.metrics_label.setText("Loading datasets...")

            # Load Data 1 (MoCap)
            loader1 = MotionDataLoader()
            excel_data1 = loader1.load_data()
            baseq1, ztcfq1, deltaq1 = loader1.convert_to_gui_format(excel_data1)
            config1 = RenderConfig()
            self.frame_processor_mocap = FrameProcessor(
                (baseq1, ztcfq1, deltaq1), config1
            )

            # Load Data 2 (Model) - Reusing same loader for prototype
            loader2 = MotionDataLoader()
            excel_data2 = loader2.load_data()
            baseq2, ztcfq2, deltaq2 = loader2.convert_to_gui_format(excel_data2)
            config2 = RenderConfig()
            self.frame_processor_model = FrameProcessor(
                (baseq2, ztcfq2, deltaq2), config2
            )

            # Initialize visualizers
            self.mocap_widget.load_data_from_dataframes((baseq1, ztcfq1, deltaq1))
            self.model_widget.load_data_from_dataframes((baseq2, ztcfq2, deltaq2))

            # Set controller to drive based on MoCap length
            self.playback_controller.load_frame_processor(self.frame_processor_mocap)

            # Update UI
            total_frames = len(self.frame_processor_mocap.time_vector)
            self.frame_slider.setMaximum(total_frames - 1)
            self.frame_label.setText(f"Frame: 0/{total_frames}")

            self.metrics_label.setText("Datasets Loaded. Ready to Compare.")

        except (RuntimeError, ValueError, OSError) as e:
            self.metrics_label.setText(f"Error loading data: {str(e)}")
            traceback.print_exc()

    def _toggle_playback(self) -> None:
        if not self.frame_processor_mocap:
            return
        self.playback_controller.toggle_playback()
        text = "Pause Sync" if self.playback_controller.is_playing else "Play Sync"
        self.play_button.setText(text)

    def _on_slider_moved(self, value: int) -> None:
        self.playback_controller.seek(float(value))

    def _on_position_changed(self, position: float) -> None:
        if self.frame_processor_mocap:
            total_frames = len(self.frame_processor_mocap.time_vector)
        else:
            total_frames = 0
        self.frame_label.setText(f"Frame: {position:.1f}/{total_frames}")
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(int(position))
        self.frame_slider.blockSignals(False)

    def _on_smooth_frame_updated(self, frame_data_mocap: FrameData) -> None:
        """Update both visualizers and metrics."""
        if not self.frame_processor_model:
            return

        # 1. Update MoCap View
        if self.mocap_widget.renderer:
            self.mocap_widget.update_frame(frame_data_mocap, RenderConfig())

        # 2. Get Interpolated Model Frame at same position
        pos = self.playback_controller.position
        total_frames_model = len(self.frame_processor_model.time_vector)
        pos_clamped = np.clip(pos, 0.0, total_frames_model - 1)
        low_idx = int(np.floor(pos_clamped))
        high_idx = min(low_idx + 1, total_frames_model - 1)
        t = pos_clamped - low_idx

        frame_low = self.frame_processor_model.get_frame_data(low_idx)
        frame_high = self.frame_processor_model.get_frame_data(high_idx)
        frame_data_model = SmoothPlaybackController._lerp_frame_data(
            frame_low, frame_high, t
        )

        # Update Model View
        if self.model_widget.renderer:
            self.model_widget.update_frame(frame_data_model, RenderConfig())

        # 3. Calculate Metrics (e.g. Midpoint Distance)
        mp1 = frame_data_mocap.midpoint
        mp2 = frame_data_model.midpoint
        if np.isfinite(mp1).all() and np.isfinite(mp2).all():
            dist = np.linalg.norm(mp1 - mp2)
            self.metrics_label.setText(
                f"Comparison Metrics | Midpoint Error: {dist:.4f} m"
            )
