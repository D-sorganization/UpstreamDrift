"""
Motion Capture Plotter - 3D visualization of golf swing data.

Decomposed via SRP into:
- mocap_data_loader.py: Data parsing/loading (Excel, CSV formats)
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("QtAgg")  # Use QtAgg backend for PyQt6 compatibility

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mocap_data_loader import (
    find_available_joints,
    get_simscape_joint_positions,
    parse_excel_row,
    process_excel_sheet,
    safe_float,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class MotionCapturePlotter(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Motion Capture Plotter - PyQt6")
        self.setGeometry(100, 100, 1400, 900)

        # Data storage - now supporting multiple data sources simultaneously
        self.swing_data = {}  # Motion capture data
        self.simscape_data = {}  # Simscape data
        self.current_swing = None
        self.current_frame = 0
        self.is_playing = False
        self.current_filter = "none"

        # Club parameters
        self.shaft_length = 0.9  # meters
        self.motion_scale = 1.0  # Use actual scale since we have real coordinates

        # Mouse interaction state
        self._last_pos = None

        # Setup UI
        self.setup_ui()

        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.next_frame)

        # Data source tracking
        self.current_data_source = "Motion Capture (Excel)"
        self.show_motion_capture = True
        self.show_simscape = False

        # Try to auto-load the Excel file if it exists
        self.auto_load_excel_file()

    def auto_load_excel_file(self) -> None:
        """Automatically load the Excel file if it exists in the current directory."""
        excel_files = [f for f in os.listdir(".") if f.endswith((".xlsx", ".xls"))]
        if excel_files:
            # Try to load the first Excel file found
            filename = excel_files[0]
            logger.info(f"Auto-loading Excel file: {filename}")
            self.load_excel_file(filename)

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

    def load_file(self) -> None:
        """Load data file based on current data source."""
        if self.current_data_source == "Motion Capture (Excel)":
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Excel File", "", "Excel Files (*.xlsx *.xls)"
            )
            if filename:
                self.load_excel_file(filename)
        else:  # Simscape Multibody (CSV)
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load CSV File", "", "CSV Files (*.csv)"
            )
            if filename:
                self.load_simscape_csv(filename)

    def on_data_source_changed(self, source) -> None:
        """Handle data source change."""
        self.current_data_source = source
        logger.info(f"Data source changed to: {source}")

        # Update visibility flags based on data source
        if source == "Motion Capture (Excel)":
            self.show_motion_capture = True
            self.show_simscape = False
            # Try to auto-load Excel file
            self.auto_load_excel_file()
        elif source == "Simscape Multibody (CSV)":
            self.show_motion_capture = False
            self.show_simscape = True
            # Try to auto-load CSV file
            self.auto_load_simscape_csv()
        else:  # Both (Simultaneous)
            self.show_motion_capture = True
            self.show_simscape = True
            # Try to auto-load both files
            self.auto_load_excel_file()
            self.auto_load_simscape_csv()

        # Update visualization
        self.update_visualization()

    def auto_load_simscape_csv(self) -> None:
        """Automatically load the Simscape CSV file if it exists."""
        csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]
        if csv_files:
            filename = csv_files[0]
            logger.info(f"Auto-loading Simscape CSV file: {filename}")
            self.load_simscape_csv(filename)

    # Delegates to mocap_data_loader module
    _safe_float = staticmethod(safe_float)
    _parse_excel_row = staticmethod(parse_excel_row)

    def _process_excel_sheet(self, filename, sheet_name) -> None:
        """Process a single Excel sheet and store parsed frames in swing_data."""
        result = process_excel_sheet(filename, sheet_name)
        if result is not None:
            self.swing_data[sheet_name] = result
            self.print_data_debug(sheet_name)

    def load_excel_file(self, filename) -> None:
        """Load and process Excel file."""
        try:
            logger.info(f"Loading file: {filename}")
            excel_file = pd.ExcelFile(filename)
            logger.info(f"Available sheets: {excel_file.sheet_names}")

            for sheet_name in ["TW_wiffle", "TW_ProV1", "GW_wiffle", "GW_ProV11"]:
                if sheet_name in excel_file.sheet_names:
                    self._process_excel_sheet(filename, sheet_name)

            # Update swing selection
            self.swing_combo.clear()
            self.swing_combo.addItems(list(self.swing_data.keys()))

            if self.swing_data:
                self.current_swing = list(self.swing_data.keys())[0]
                logger.info(f"Selected swing: {self.current_swing}")
                self.swing_combo.setCurrentText(self.current_swing)
                self.setup_frame_slider()
                self.update_visualization()
            else:
                logger.info("No valid swing data found in the file")

        except ImportError as e:
            logger.error(f"Error loading file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    # Delegates to mocap_data_loader module
    _simscape_joint_position_definitions = staticmethod(get_simscape_joint_positions)
    _find_available_joints = staticmethod(find_available_joints)

    def load_simscape_csv(self, filename) -> None:
        """Load and process Simscape CSV file."""
        try:
            logger.info(f"Loading Simscape CSV file: {filename}")

            df = pd.read_csv(filename)
            logger.debug(
                f"Successfully loaded CSV with {len(df)} rows "
                f"and {len(df.columns)} columns"
            )
            logger.info(
                f"Time range: {df['time'].min():.3f} to {df['time'].max():.3f} seconds"
            )

            joint_positions = self._simscape_joint_position_definitions()
            available_joints = self._find_available_joints(joint_positions, df.columns)

            if not available_joints:
                raise ValueError("No valid joint position data found in the CSV file")

            # Process the data into our standard format
            data = []
            for _i, row in df.iterrows():
                frame_data = {"time": row["time"]}
                for joint_name, columns in available_joints.items():
                    if all(col in df.columns for col in columns):
                        frame_data[f"{joint_name}_X"] = row[columns[0]]
                        frame_data[f"{joint_name}_Y"] = row[columns[1]]
                        frame_data[f"{joint_name}_Z"] = row[columns[2]]
                data.append(frame_data)

            # Store the data
            swing_name = "Simscape_Swing"
            self.simscape_data[swing_name] = pd.DataFrame(data)
            logger.debug(f"Successfully loaded {len(data)} frames for {swing_name}")

            # Update swing selection
            self.swing_combo.clear()
            self.swing_combo.addItems(list(self.swing_data.keys()))

            if self.swing_data:
                self.current_swing = swing_name
                logger.info(f"Selected swing: {self.current_swing}")
                self.swing_combo.setCurrentText(self.current_swing)
                self.setup_frame_slider()
                self.update_visualization()
            else:
                logger.info("No valid swing data found in the file")

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Error loading Simscape CSV file: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to load Simscape CSV file: {str(e)}"
            )

    def print_data_debug(self, sheet_name) -> None:
        """Print debug information about the loaded data."""
        if sheet_name in self.swing_data:
            data = self.swing_data[sheet_name]
            if not data.empty:
                logger.debug(f"\n=== Data Debug for {sheet_name} ===")
                logger.info(f"Number of frames: {len(data)}")
                logger.info(
                    f"Time range: {data['time'].min():.3f} to "
                    f"{data['time'].max():.3f} seconds"
                )
                logger.info("Mid-Hands Position ranges:")
                logger.info(
                    f"  X: {data['mid_X'].min():.3f} to {data['mid_X'].max():.3f}"
                )
                logger.info(
                    f"  Y: {data['mid_Y'].min():.3f} to {data['mid_Y'].max():.3f}"
                )
                logger.info(
                    f"  Z: {data['mid_Z'].min():.3f} to {data['mid_Z'].max():.3f}"
                )

                logger.info("Club Head Position ranges:")
                logger.info(
                    f"  X: {data['club_X'].min():.3f} to {data['club_X'].max():.3f}"
                )
                logger.info(
                    f"  Y: {data['club_Y'].min():.3f} to {data['club_Y'].max():.3f}"
                )
                logger.info(
                    f"  Z: {data['club_Z'].min():.3f} to {data['club_Z'].max():.3f}"
                )

                # Calculate total position ranges
                mid_range = np.max(
                    [
                        data["mid_X"].max() - data["mid_X"].min(),
                        data["mid_Y"].max() - data["mid_Y"].min(),
                        data["mid_Z"].max() - data["mid_Z"].min(),
                    ]
                )
                club_range = np.max(
                    [
                        data["club_X"].max() - data["club_X"].min(),
                        data["club_Y"].max() - data["club_Y"].min(),
                        data["club_Z"].max() - data["club_Z"].min(),
                    ]
                )

                logger.info(f"Mid-Hands motion range: {mid_range:.3f}")
                logger.info(f"Club Head motion range: {club_range:.3f}")

                logger.info("Data Analysis:")
                logger.info(
                    "  This data contains both mid-hands and club head positions"
                )
                logger.info(
                    "  Using actual measured positions instead of calculated ones"
                )
                logger.info(
                    "  Original data in inches, converted to meters for visualization"
                )
                logger.info(
                    "  Direction cosines (Xx, Xy, Xz, Yx, Yy, Yz, Zx, Zy, Zz) "
                    "are unitless"
                )
                logger.info("  Motion scaling applied to make visualization clearer")
                logger.info("=" * 40)

    def setup_frame_slider(self) -> None:
        """Setup the frame slider."""
        if self.current_swing in self.swing_data:
            data = self.swing_data[self.current_swing]
            max_frame = len(data) - 1
            self.frame_slider.setRange(0, max_frame)
            self.frame_slider.setValue(0)
            self.current_frame = 0

    def on_swing_change(self, swing_name) -> None:
        """Handle swing selection change."""
        if swing_name in self.swing_data:
            self.current_swing = swing_name
            self.setup_frame_slider()
            self.update_visualization()

    def on_frame_change(self, frame) -> None:
        """Handle frame slider change."""
        self.current_frame = frame
        self.frame_label.setText(str(frame))
        self.update_visualization()

    def on_speed_change(self, speed) -> None:
        """Handle speed slider change."""
        self.speed_label.setText(str(speed))
        if self.is_playing:
            self.animation_timer.setInterval(1000 // speed)

    def on_scale_change(self, scale) -> None:
        """Handle motion scale change."""
        self.motion_scale = scale
        self.scale_label.setText(f"{scale}x")
        self.update_visualization()

    def on_club_length_change(self, length_cm) -> None:
        """Handle club length change."""
        self.shaft_length = length_cm / 100.0  # Convert cm to meters
        self.club_label.setText(f"{self.shaft_length:.1f}m")
        self.update_visualization()

    def toggle_playback(self) -> None:
        """Toggle play/pause."""
        if self.is_playing:
            self.animation_timer.stop()
            self.play_btn.setText("Play")
            self.is_playing = False
        else:
            speed = self.speed_slider.value()
            self.animation_timer.start(1000 // speed)
            self.play_btn.setText("Pause")
            self.is_playing = True

    def next_frame(self) -> None:
        """Advance to next frame."""
        if self.current_swing in self.swing_data:
            data = self.swing_data[self.current_swing]
            if self.current_frame < len(data) - 1:
                self.current_frame += 1
                self.frame_slider.setValue(self.current_frame)
            else:
                # Loop back to start
                self.current_frame = 0
                self.frame_slider.setValue(0)

    def setup_3d_scene(self) -> None:
        """Setup the 3D scene with ground plane and ball."""
        self.ax.clear()

        # Ground plane - positioned at calculated ground level
        ground_level = self.calculate_ground_level()
        x_ground = np.linspace(-3, 3, 10)
        y_ground = np.linspace(-3, 3, 10)
        X_ground, Y_ground = np.meshgrid(x_ground, y_ground)
        Z_ground = np.full_like(X_ground, ground_level)  # Set to actual ground level
        self.ax.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.3, color="green")

        # Golf ball - positioned at origin for golf swing analysis
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        ball_radius = 0.021
        x_ball = ball_radius * np.outer(np.cos(u), np.sin(v))
        y_ball = ball_radius * np.outer(np.sin(u), np.sin(v))
        z_ball = ball_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax.plot_surface(x_ball, y_ball, z_ball, color="white", alpha=0.8)

        # Set axis labels and limits
        self.ax.set_xlabel("X (Target Line)")
        self.ax.set_ylabel("Y (Ball Direction)")
        self.ax.set_zlabel("Z (Vertical)")
        self.ax.set_xlim([-2.0, 2.0])
        self.ax.set_ylim([-1.0, 3.0])

        # Set initial Z limits - will be adjusted based on data
        self.ax.set_zlim([-0.5, 2.5])

        # Set initial view
        self.ax.view_init(elev=15, azim=-45)

    def calculate_ground_level(self) -> float:
        """Set ground level to fixed -2.5 meters."""
        return -2.5

    def adjust_plot_limits_to_ground(self) -> None:
        """Adjust plot limits so ground level is at the bottom."""
        ground_level = self.calculate_ground_level()

        # Set Z limits with ground at bottom and reasonable height above
        z_range = 3.0  # 3 meters total height
        self.ax.set_zlim([ground_level, ground_level + z_range])

        logger.info(f"Ground level set to: {ground_level:.3f}m")

    def update_visualization(self) -> None:
        """Update the 3D visualization with proper coordinate system."""
        # Store current view angles before clearing
        current_elev = self.ax.elev
        current_azim = self.ax.azim

        # Clear and setup scene
        self.setup_3d_scene()

        # Adjust ground level based on data
        self.adjust_plot_limits_to_ground()

        # Restore the view angles to maintain user's camera position
        self.ax.view_init(elev=current_elev, azim=current_azim)

        # Visualize motion capture data if enabled
        if self.show_motion_capture and self.swing_data:
            # Find the first available swing data
            available_swings = list(self.swing_data.keys())
            if available_swings and self.current_frame < len(
                self.swing_data[available_swings[0]]
            ):
                motion_data = self.swing_data[available_swings[0]]
                frame_data = motion_data.iloc[self.current_frame]
                self.visualize_motion_capture_data(frame_data, motion_data)

        # Visualize Simscape data if enabled
        if self.show_simscape and self.simscape_data:
            # Find the first available Simscape data
            available_simscape = list(self.simscape_data.keys())
            if available_simscape and self.current_frame < len(
                self.simscape_data[available_simscape[0]]
            ):
                simscape_data = self.simscape_data[available_simscape[0]]
                frame_data = simscape_data.iloc[self.current_frame]
                self.visualize_simscape_data(frame_data, simscape_data)

        # Update info text with combined data
        self.update_info_text(
            None
        )  # Pass None since we're handling multiple data sources

        # Redraw canvas
        self.canvas.draw()

    def _draw_motion_capture_trajectory_paths(self, data) -> None:
        """Draw mid-hands and club head trajectory paths for motion capture data.

        Parameters:
            data: full DataFrame of all frames
        """
        if self.trajectory_check.isChecked() and len(data) > 1:
            # Mid-hands path (blue dashed) - flip X for right-handed swing
            trajectory = np.array(
                [
                    [
                        -row["mid_X"] * self.motion_scale,
                        row["mid_Y"] * self.motion_scale,
                        row["mid_Z"] * self.motion_scale,
                    ]
                    for _, row in data.iterrows()
                ]
            )
            self.ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                "b--",
                alpha=0.6,
                linewidth=2,
                label="Mid-Hands Path",
            )

        if self.club_path_check.isChecked() and len(data) > 1:
            # Club head path (red dashed) - flip X for right-handed swing
            club_path = np.array(
                [
                    [
                        -row["club_X"] * self.motion_scale,
                        row["club_Y"] * self.motion_scale,
                        row["club_Z"] * self.motion_scale,
                    ]
                    for _, row in data.iterrows()
                ]
            )
            self.ax.plot(
                club_path[:, 0],
                club_path[:, 1],
                club_path[:, 2],
                "r--",
                alpha=0.6,
                linewidth=2,
                label="Club Head Path",
            )

    def visualize_motion_capture_data(self, frame_data, data) -> None:
        """Visualize motion capture data (Excel format)."""
        # Use actual mid-hands and club head positions from the data
        # For right-handed golfers: X should be flipped to show proper swing direction
        mid_hands = np.array(
            [
                -frame_data["mid_X"]
                * self.motion_scale,  # Flip X for right-handed swing
                frame_data["mid_Y"] * self.motion_scale,
                frame_data["mid_Z"] * self.motion_scale,
            ]
        )

        club_head = np.array(
            [
                -frame_data["club_X"]
                * self.motion_scale,  # Flip X for right-handed swing
                frame_data["club_Y"] * self.motion_scale,
                frame_data["club_Z"] * self.motion_scale,
            ]
        )

        # Draw the club shaft, head, face normal, and golf ball
        self._draw_club_with_face_normal(club_head, mid_hands)

        # Draw trajectory paths
        self._draw_motion_capture_trajectory_paths(data)

    def _extract_joint_positions(self, frame_data) -> dict[str, np.ndarray]:
        """Extract scaled joint positions from a Simscape frame.

        Returns a dict mapping joint names to numpy position arrays.
        """
        joints = {}
        joint_names = [
            "club_head",
            "left_hand",
            "right_hand",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "hub",
            "spine",
            "hip",
        ]
        for joint_name in joint_names:
            if f"{joint_name}_X" in frame_data:
                joints[joint_name] = np.array(
                    [
                        -frame_data[f"{joint_name}_X"]
                        * self.motion_scale,  # Flip X for right-handed swing
                        frame_data[f"{joint_name}_Y"] * self.motion_scale,
                        frame_data[f"{joint_name}_Z"] * self.motion_scale,
                    ]
                )
        return joints

    def _draw_club_with_face_normal(self, club_head_pos, grip_pos) -> None:
        """Draw the club shaft, head sphere, face normal vector, and golf ball.

        Parameters:
            club_head_pos: numpy array of club head position [x, y, z]
            grip_pos: numpy array of grip (left hand) position [x, y, z]
        """
        # Draw club shaft from grip to club head
        club_points = np.array([grip_pos, club_head_pos])
        self.ax.plot(
            club_points[:, 0],
            club_points[:, 1],
            club_points[:, 2],
            color="gray",
            linewidth=6,
            alpha=0.9,
            label="Club Shaft",
        )

        # Draw club head as sphere with better appearance
        u = np.linspace(0, 2 * np.pi, 8)
        v = np.linspace(0, np.pi, 8)
        head_size = 0.03  # Realistic club head size
        x_head = head_size * np.outer(np.cos(u), np.sin(v)) + club_head_pos[0]
        y_head = head_size * np.outer(np.sin(u), np.sin(v)) + club_head_pos[1]
        z_head = head_size * np.outer(np.ones(np.size(u)), np.cos(v)) + club_head_pos[2]
        self.ax.plot_surface(
            x_head, y_head, z_head, color="darkgray", alpha=0.9, label="Club Head"
        )

        # Calculate and draw club face normal vector
        shaft_direction = club_head_pos - grip_pos
        shaft_length = np.linalg.norm(shaft_direction)

        if shaft_length > 0:
            shaft_direction = shaft_direction / shaft_length

            # Calculate face normal (perpendicular to shaft)
            up_vector = np.array([0, 0, 1])  # Vertical up
            face_normal = np.cross(shaft_direction, up_vector)
            face_normal_length = np.linalg.norm(face_normal)

            if face_normal_length > 0:
                face_normal = face_normal / face_normal_length
                self._draw_face_normal_and_ball(club_head_pos, face_normal)

    def _draw_face_normal_and_ball(self, club_head_pos, face_normal) -> None:
        """Draw the face normal vector arrow and a golf ball in front of the club.

        Parameters:
            club_head_pos: numpy array of club head position [x, y, z]
            face_normal: unit numpy array of face normal direction
        """
        # Draw face normal vector (red arrow) - longer and more visible
        normal_length = 0.25  # 25cm normal vector (longer)
        normal_end = club_head_pos + face_normal * normal_length

        # Draw the normal vector as a thick line
        normal_points = np.array([club_head_pos, normal_end])
        self.ax.plot(
            normal_points[:, 0],
            normal_points[:, 1],
            normal_points[:, 2],
            "red",
            linewidth=6,
            alpha=1.0,
            label="Face Normal",
        )

        # Add a larger arrowhead at the end
        self.ax.scatter(
            normal_end[0],
            normal_end[1],
            normal_end[2],
            c="red",
            s=200,
            marker=">",
            alpha=1.0,
        )

        # Add a small sphere at the start of the normal for better visibility
        self.ax.scatter(
            club_head_pos[0],
            club_head_pos[1],
            club_head_pos[2],
            c="red",
            s=50,
            marker="o",
            alpha=0.8,
        )

        # Draw golf ball positioned for center strike
        ball_offset_distance = 0.08  # 8cm in front of club face
        ball_position = club_head_pos + face_normal * ball_offset_distance

        # Draw golf ball as a white sphere
        ball_radius = 0.021  # Standard golf ball radius
        u_ball = np.linspace(0, 2 * np.pi, 12)
        v_ball = np.linspace(0, np.pi, 12)
        x_ball = (
            ball_radius * np.outer(np.cos(u_ball), np.sin(v_ball)) + ball_position[0]
        )
        y_ball = (
            ball_radius * np.outer(np.sin(u_ball), np.sin(v_ball)) + ball_position[1]
        )
        z_ball = (
            ball_radius * np.outer(np.ones(np.size(u_ball)), np.cos(v_ball))
            + ball_position[2]
        )
        self.ax.plot_surface(
            x_ball,
            y_ball,
            z_ball,
            color="white",
            alpha=0.95,
            edgecolor="lightgray",
            linewidth=0.5,
            label="Golf Ball",
        )

    def _draw_body_segments_and_markers(self, joints, segment_definitions) -> None:
        """Draw body segment lines and joint marker dots.

        Parameters:
            joints: dict mapping joint names to numpy position arrays
            segment_definitions: list of (start_joint, end_joint, color) tuples
        """
        # Draw body segments
        for start_joint, end_joint, color in segment_definitions:
            if start_joint in joints and end_joint in joints:
                segment_points = np.array([joints[start_joint], joints[end_joint]])
                self.ax.plot(
                    segment_points[:, 0],
                    segment_points[:, 1],
                    segment_points[:, 2],
                    color=color,
                    linewidth=3,
                    alpha=0.7,
                )

        # Draw joint markers
        for _, position in joints.items():
            self.ax.scatter(
                position[0], position[1], position[2], color="black", s=50, alpha=0.8
            )

    def _draw_simscape_trajectory_paths(self, joints, data) -> None:
        """Draw club head and hands trajectory paths for Simscape data.

        Parameters:
            joints: dict mapping joint names to numpy position arrays
            data: full DataFrame of all frames
        """
        # Club head trajectory
        if (
            self.trajectory_check.isChecked()
            and len(data) > 1
            and "club_head" in joints
        ):
            club_trajectory = np.array(
                [
                    [
                        -row["club_head_X"]
                        * self.motion_scale,  # Flip X for right-handed swing
                        row["club_head_Y"] * self.motion_scale,
                        row["club_head_Z"] * self.motion_scale,
                    ]
                    for _, row in data.iterrows()
                    if "club_head_X" in row
                ]
            )
            if len(club_trajectory) > 1:
                self.ax.plot(
                    club_trajectory[:, 0],
                    club_trajectory[:, 1],
                    club_trajectory[:, 2],
                    "r--",
                    alpha=0.6,
                    linewidth=2,
                    label="Club Head Path",
                )

            # Hands trajectory
        if self.club_path_check.isChecked() and len(data) > 1 and "left_hand" in joints:
            hands_trajectory = np.array(
                [
                    [
                        -row["left_hand_X"]
                        * self.motion_scale,  # Flip X for right-handed swing
                        row["left_hand_Y"] * self.motion_scale,
                        row["left_hand_Z"] * self.motion_scale,
                    ]
                    for _, row in data.iterrows()
                    if "left_hand_X" in row
                ]
            )
            if len(hands_trajectory) > 1:
                self.ax.plot(
                    hands_trajectory[:, 0],
                    hands_trajectory[:, 1],
                    hands_trajectory[:, 2],
                    "b--",
                    alpha=0.6,
                    linewidth=2,
                    label="Hands Path",
                )

    def _draw_segment_traces(self, frame_data, data) -> None:
        """Draw optional per-segment trace paths for Simscape data.

        Parameters:
            frame_data: current frame's data row
            data: full DataFrame of all frames
        """
        trace_colors = {
            "club_head": "red",
            "left_hand": "blue",
            "right_hand": "cyan",
            "left_elbow": "green",
            "right_elbow": "lime",
            "left_shoulder": "orange",
            "right_shoulder": "yellow",
            "hub": "purple",
            "spine": "magenta",
            "hip": "brown",
        }

        for segment_key, checkbox in self.segment_traces.items():
            if (
                checkbox.isChecked()
                and f"{segment_key}_X" in frame_data
                and len(data) > 1
            ):
                # Create trajectory for this segment
                segment_trajectory = np.array(
                    [
                        [
                            -row[f"{segment_key}_X"]
                            * self.motion_scale,  # Flip X for right-handed swing
                            row[f"{segment_key}_Y"] * self.motion_scale,
                            row[f"{segment_key}_Z"] * self.motion_scale,
                        ]
                        for _, row in data.iterrows()
                        if f"{segment_key}_X" in row
                    ]
                )
                if len(segment_trajectory) > 1:
                    color = trace_colors.get(segment_key, "gray")
                    self.ax.plot(
                        segment_trajectory[:, 0],
                        segment_trajectory[:, 1],
                        segment_trajectory[:, 2],
                        color=color,
                        linestyle="--",
                        alpha=0.6,
                        linewidth=2,
                        label=f"{segment_key.replace('_', ' ').title()} Path",
                    )

    def visualize_simscape_data(self, frame_data, data) -> None:
        """Visualize Simscape multibody data (CSV format)."""
        # Define colors for different body segments
        colors = {
            "club": "red",
            "hands": "blue",
            "arms": "green",
            "shoulders": "orange",
            "torso": "purple",
            "hips": "brown",
        }

        joints = self._extract_joint_positions(frame_data)

        # Define segments connecting joints
        segment_definitions = [
            ("left_hand", "right_hand", colors["hands"]),  # Midpoint
            ("left_hand", "left_elbow", colors["arms"]),
            ("right_hand", "right_elbow", colors["arms"]),
            ("left_elbow", "left_shoulder", colors["arms"]),
            ("right_elbow", "right_shoulder", colors["arms"]),
            ("left_shoulder", "right_shoulder", colors["shoulders"]),
            ("left_shoulder", "hub", colors["torso"]),
            ("right_shoulder", "hub", colors["torso"]),
            ("hub", "spine", colors["torso"]),
            ("spine", "hip", colors["torso"]),
        ]

        # Draw club if available
        if "club_head" in joints and "left_hand" in joints:
            self._draw_club_with_face_normal(joints["club_head"], joints["left_hand"])

        self._draw_body_segments_and_markers(joints, segment_definitions)
        self._draw_simscape_trajectory_paths(joints, data)
        self._draw_segment_traces(frame_data, data)

    def update_info_text(self, frame_data) -> None:
        """Update the information text display."""
        info = f"Frame: {self.current_frame}\n"
        info += f"Data Source: {self.current_data_source}\n"
        info += f"Motion Scale: {self.motion_scale}x\n\n"

        # Show motion capture data if available
        if self.show_motion_capture and self.swing_data:
            available_swings = list(self.swing_data.keys())
            if available_swings and self.current_frame < len(
                self.swing_data[available_swings[0]]
            ):
                motion_data = self.swing_data[available_swings[0]]
                motion_frame = motion_data.iloc[self.current_frame]
                info += "Motion Capture Data:\n"
                info += f"  Time: {motion_frame['time']:.3f}s\n"
                info += (
                    f"  Mid-Hands: ({motion_frame['mid_X']:.3f}, "
                    f"{motion_frame['mid_Y']:.3f}, "
                    f"{motion_frame['mid_Z']:.3f})\n"
                )
                info += (
                    f"  Club Head: ({motion_frame['club_X']:.3f}, "
                    f"{motion_frame['club_Y']:.3f}, "
                    f"{motion_frame['club_Z']:.3f})\n\n"
                )

        # Show Simscape data if available
        if self.show_simscape and self.simscape_data:
            available_simscape = list(self.simscape_data.keys())
            if available_simscape and self.current_frame < len(
                self.simscape_data[available_simscape[0]]
            ):
                simscape_data = self.simscape_data[available_simscape[0]]
                simscape_frame = simscape_data.iloc[self.current_frame]
                info += "Simscape Data:\n"
                info += f"  Time: {simscape_frame['time']:.3f}s\n"
                info += "  Available Joints:\n"
                joint_count = 0
                for joint_name in [
                    "club_head",
                    "left_hand",
                    "right_hand",
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "hub",
                    "spine",
                    "hip",
                ]:
                    if f"{joint_name}_X" in simscape_frame:
                        info += f"    {joint_name}: ✓\n"
                        joint_count += 1
                    else:
                        info += f"    {joint_name}: ✗\n"
                info += f"\nTotal Joints: {joint_count}"

        self.info_text.setText(info)

    def set_camera_view(self, view) -> None:
        """Set predefined camera views."""
        if view == "face_on":
            # Face-on view: looking at golfer from front (toward +X target line)
            self.ax.view_init(elev=15, azim=90)
        elif view == "down_line":
            # Down-the-line view: looking from side (90° from face-on)
            self.ax.view_init(elev=15, azim=180)
        elif view == "top_down":
            # Top-down view: looking down from above
            self.ax.view_init(elev=90, azim=0)
        elif view == "isometric":
            # Isometric view: 3D perspective
            self.ax.view_init(elev=15, azim=-45)

        # Force redraw
        self.canvas.draw_idle()
        self.canvas.flush_events()

    def reset_view(self) -> None:
        """Reset the 3D view to the default isometric view and limits."""
        # Reset view angles
        self.ax.view_init(elev=15, azim=-45)

        # Reset plot limits to default
        self.ax.set_xlim([-2.0, 2.0])
        self.ax.set_ylim([-1.0, 3.0])
        self.ax.set_zlim([-0.5, 2.5])

        # Force redraw
        self.canvas.draw_idle()
        self.canvas.flush_events()

    def on_scroll(self, event) -> None:
        """Handle mouse scroll for zooming."""
        if event.inaxes != self.ax:
            return

        logger.info(f"Scroll event: button={event.button}, step={event.step}")

        # Get current view limits
        x_lim = self.ax.get_xlim()
        y_lim = self.ax.get_ylim()
        z_lim = self.ax.get_zlim()

        # Determine zoom factor based on scroll direction
        zoom_factor = 0.9 if event.button == "up" or event.step > 0 else 1.1  # Zoom out

        # Calculate centers
        x_center = (x_lim[0] + x_lim[1]) / 2
        y_center = (y_lim[0] + y_lim[1]) / 2
        z_center = (z_lim[0] + z_lim[1]) / 2

        # Calculate new ranges
        x_range = (x_lim[1] - x_lim[0]) * zoom_factor
        y_range = (y_lim[1] - y_lim[0]) * zoom_factor
        z_range = (z_lim[1] - z_lim[0]) * zoom_factor

        # Set new limits
        self.ax.set_xlim([x_center - x_range / 2, x_center + x_range / 2])
        self.ax.set_ylim([y_center - y_range / 2, y_center + y_range / 2])
        self.ax.set_zlim([z_center - z_range / 2, z_center + z_range / 2])

        self.canvas.draw()
        logger.info(f"Zooming: factor={zoom_factor}")

    def on_mouse_press(self, event) -> None:
        """Handle mouse button press for rotation/panning."""
        if event.inaxes != self.ax:
            return
        # Store initial position for rotation/panning (use screen coordinates)
        self._last_pos = (event.x, event.y)
        logger.info(f"Mouse press: button={event.button}, pos=({event.x}, {event.y})")

    def on_mouse_release(self, event) -> None:
        """Handle mouse button release."""
        self._last_pos = None

    def on_mouse_move(self, event) -> None:
        """Handle mouse movement for rotation/panning."""
        if event.inaxes != self.ax or self._last_pos is None:
            return

        if hasattr(event, "button") and event.button == 1:  # Left click - rotate
            # Get current view angles
            elev = self.ax.elev
            azim = self.ax.azim

            # Calculate change in position (use screen coordinates for better control)
            dx = event.x - self._last_pos[0]
            dy = event.y - self._last_pos[1]

            # Update view angles (scale the movement)
            self.ax.view_init(elev=elev + dy * 0.5, azim=azim + dx * 0.5)
            self.canvas.draw()
            logger.info(
                f"Rotating: dx={dx}, dy={dy}, "
                f"new_elev={elev + dy * 0.5}, new_azim={azim + dx * 0.5}"
            )

        elif hasattr(event, "button") and event.button == 3:  # Right click - pan
            # Get current limits
            x_lim = self.ax.get_xlim()
            y_lim = self.ax.get_ylim()
            z_lim = self.ax.get_zlim()

            # Calculate change in position (use screen coordinates)
            dx = event.x - self._last_pos[0]
            dy = event.y - self._last_pos[1]

            # Update limits (scale the movement)
            x_range = x_lim[1] - x_lim[0]
            y_range = y_lim[1] - y_lim[0]
            z_lim[1] - z_lim[0]

            pan_scale = 0.01  # Adjust this for panning sensitivity

            self.ax.set_xlim(
                [
                    x_lim[0] - dx * x_range * pan_scale,
                    x_lim[1] - dx * x_range * pan_scale,
                ]
            )
            self.ax.set_ylim(
                [
                    y_lim[0] + dy * y_range * pan_scale,
                    y_lim[1] + dy * y_range * pan_scale,
                ]
            )
            self.canvas.draw()
            logger.info(f"Panning: dx={dx}, dy={dy}")

        self._last_pos = (event.x, event.y)


def main() -> None:
    """Launch the Motion Capture Plotter GUI application."""
    app = QApplication(sys.argv)
    window = MotionCapturePlotter()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
