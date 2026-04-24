# mypy: disable-error-code="attr-defined"
"""UI builders for the legacy motion-capture golf plotter."""

from __future__ import annotations

import logging

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

matplotlib.use("QtAgg")

logger = logging.getLogger(__name__)


class MotionCapturePlotterUIMixin:
    def setup_ui(self) -> None:
        """Setup the main UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Right plot panel
        plot_panel = self.create_plot_panel()
        main_layout.addWidget(plot_panel, stretch=1)

    def _create_data_loading_group(self) -> QGroupBox:
        """Create the data loading group box with source and file selectors."""
        file_group = QGroupBox("Data Loading")
        file_layout = QVBoxLayout(file_group)

        # Data source selection
        file_layout.addWidget(QLabel("Data Source:"))
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItem("Motion Capture (Excel)")
        self.data_source_combo.addItem("Simscape Multibody (CSV)")
        self.data_source_combo.addItem("Both (Simultaneous)")
        self.data_source_combo.currentTextChanged.connect(self.on_data_source_changed)
        file_layout.addWidget(self.data_source_combo)

        # File selection
        file_layout.addWidget(QLabel("Motion Capture File:"))
        self.motion_capture_file_combo = QComboBox()
        self.motion_capture_file_combo.addItem("Wiffle_ProV1_club_3D_data.xlsx")
        file_layout.addWidget(self.motion_capture_file_combo)

        file_layout.addWidget(QLabel("Simscape File:"))
        self.simscape_file_combo = QComboBox()
        self.simscape_file_combo.addItem("trial_001_20250802_204903.csv")
        file_layout.addWidget(self.simscape_file_combo)

        load_btn = QPushButton("Load File")
        load_btn.clicked.connect(self.load_file)
        file_layout.addWidget(load_btn)

        return file_group

    def _create_playback_controls_group(self) -> QGroupBox:
        """Create the playback controls group box with play, frame, and speed."""
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QVBoxLayout(playback_group)

        # Play/Pause button
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        playback_layout.addWidget(self.play_btn)

        # Frame slider
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        frame_layout.addWidget(self.frame_slider)
        self.frame_label = QLabel("0")
        frame_layout.addWidget(self.frame_label)
        playback_layout.addLayout(frame_layout)

        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 60)
        self.speed_slider.setValue(30)  # Default faster speed
        self.speed_slider.valueChanged.connect(self.on_speed_change)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("30")
        speed_layout.addWidget(self.speed_label)
        playback_layout.addLayout(speed_layout)

        return playback_group

    def _create_visualization_options_group(self) -> QGroupBox:
        """Create the visualization options group box with traces and sliders."""
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QVBoxLayout(viz_group)

        # Trajectory options
        self.trajectory_check = QCheckBox("Show Mid-Hands Path")
        self.trajectory_check.setChecked(True)
        self.trajectory_check.stateChanged.connect(self.update_visualization)
        viz_layout.addWidget(self.trajectory_check)

        self.club_path_check = QCheckBox("Show Club Head Path")
        self.club_path_check.setChecked(True)
        self.club_path_check.stateChanged.connect(self.update_visualization)
        viz_layout.addWidget(self.club_path_check)

        # Simscape segment trace options
        viz_layout.addWidget(QLabel("Simscape Segment Traces:"))
        self.segment_traces = {}
        segment_options = [
            ("club_head", "Club Head"),
            ("left_hand", "Left Hand"),
            ("right_hand", "Right Hand"),
            ("left_elbow", "Left Elbow"),
            ("right_elbow", "Right Elbow"),
            ("left_shoulder", "Left Shoulder"),
            ("right_shoulder", "Right Shoulder"),
            ("hub", "Hub"),
            ("spine", "Spine"),
            ("hip", "Hip"),
        ]

        for segment_key, segment_name in segment_options:
            checkbox = QCheckBox(f"Trace {segment_name}")
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(self.update_visualization)
            self.segment_traces[segment_key] = checkbox
            viz_layout.addWidget(checkbox)

        # Motion scaling
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Motion Scale:"))
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(1, 10)  # Reasonable range for scaling
        self.scale_slider.setValue(1)  # Default 1x scale
        self.scale_slider.valueChanged.connect(self.on_scale_change)
        scale_layout.addWidget(self.scale_slider)
        self.scale_label = QLabel("1x")
        scale_layout.addWidget(self.scale_label)
        viz_layout.addLayout(scale_layout)

        # Club length
        club_layout = QHBoxLayout()
        club_layout.addWidget(QLabel("Club Length:"))
        self.club_slider = QSlider(Qt.Orientation.Horizontal)
        self.club_slider.setRange(50, 150)
        self.club_slider.setValue(90)  # 0.9m default
        self.club_slider.valueChanged.connect(self.on_club_length_change)
        club_layout.addWidget(self.club_slider)
        self.club_label = QLabel("0.9m")
        club_layout.addWidget(self.club_label)
        viz_layout.addLayout(club_layout)

        return viz_group

    def _create_camera_controls_group(self) -> QGroupBox:
        """Create the camera views group box with preset view buttons."""
        camera_group = QGroupBox("Camera Views")
        camera_layout = QVBoxLayout(camera_group)

        camera_buttons = [
            ("Face-On", lambda: self.set_camera_view("face_on")),
            ("Down-the-Line", lambda: self.set_camera_view("down_line")),
            ("Top-Down", lambda: self.set_camera_view("top_down")),
            ("Isometric", lambda: self.set_camera_view("isometric")),
            ("Reset View", lambda: self.reset_view()),
        ]

        for text, command in camera_buttons:
            btn = QPushButton(text)
            btn.clicked.connect(command)
            camera_layout.addWidget(btn)

        return camera_group

    def _create_info_and_help_groups(self) -> tuple[QGroupBox, QGroupBox]:
        """Create the frame data info and 3D plot help group boxes.

        Returns (info_group, help_group) tuple.
        """
        # Analysis info
        info_group = QGroupBox("Current Frame Data")
        info_layout = QVBoxLayout(info_group)
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        info_layout.addWidget(self.info_text)

        # Interactive controls help
        help_group = QGroupBox("3D Plot Controls")
        help_layout = QVBoxLayout(help_group)
        help_text = """3D Plot Interaction:
• Left-click + drag: Rotate view
• Right-click + drag: Pan view
• Mouse wheel: Zoom in/out
• Use camera buttons for preset views"""
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_layout.addWidget(help_label)

        return info_group, help_group

    def create_control_panel(self) -> QScrollArea:
        """Create the left control panel with scroll area."""
        # Create a scroll area to contain all controls
        scroll_area = QScrollArea()
        scroll_area.setMaximumWidth(400)
        scroll_area.setMinimumWidth(350)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Create the actual content widget
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Motion Capture Plotter")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Add widget groups
        layout.addWidget(self._create_data_loading_group())

        # Swing selection
        swing_group = QGroupBox("Swing Selection")
        swing_layout = QVBoxLayout(swing_group)
        self.swing_combo = QComboBox()
        self.swing_combo.currentTextChanged.connect(self.on_swing_change)
        swing_layout.addWidget(self.swing_combo)
        layout.addWidget(swing_group)

        layout.addWidget(self._create_playback_controls_group())
        layout.addWidget(self._create_visualization_options_group())
        layout.addWidget(self._create_camera_controls_group())

        info_group, help_group = self._create_info_and_help_groups()
        layout.addWidget(info_group)
        layout.addWidget(help_group)

        layout.addStretch()

        # Set the content widget in the scroll area
        scroll_area.setWidget(panel)
        return scroll_area

    def create_plot_panel(self) -> QWidget:
        """Create the right plot panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Enable interactive features and 3D navigation
        self.ax.mouse_init()

        # Enable matplotlib's built-in 3D navigation
        self.ax.set_navigate(True)

        # Create canvas
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Connect mouse events for zoom/rotation with proper event handling
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        # Enable mouse tracking for better interaction
        self.canvas.setMouseTracking(True)

        # Initialize empty plot elements
        self.club_line = None
        self.club_head = None
        self.trajectory_line = None
        self.club_path_line = None

        # Show initial 3D scene
        self.setup_3d_scene()

        return panel
