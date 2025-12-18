import os
import sys

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
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
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class GolfSwingAnalyzerPyQt6(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Golf Swing Motion Capture Analyzer - PyQt6")
        self.setGeometry(100, 100, 1600, 1000)

        # Data storage
        self.swing_data = {}
        self.key_frames = {}
        self.current_swing = "TW_ProV1"
        self.current_frame = 0
        self.current_filter = "None"
        self.eval_offset = 0.0  # inches
        self.is_playing = False
        self.show_trajectory = True
        self.show_force_vectors = True

        # Club parameters
        self.club_mass = 0.2  # kg
        self.shaft_length = 1.2  # meters

        # Scaling parameters
        self.motion_scale = 1.0
        self.club_length_override = 0.9

        self.setup_ui()
        self.load_default_data()

    def setup_ui(self):
        """Setup the main UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)

        # Right 3D plot panel
        plot_panel = self.create_plot_panel()
        main_layout.addWidget(plot_panel, 3)

    def create_control_panel(self):
        """Create the left control panel"""
        panel = QWidget()
        panel.setMaximumWidth(400)
        panel.setMinimumWidth(350)

        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Golf Swing Analyzer")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # File loading
        file_group = QGroupBox("Data Loading")
        file_layout = QVBoxLayout(file_group)
        load_btn = QPushButton("Load Excel File")
        load_btn.clicked.connect(self.load_file)
        file_layout.addWidget(load_btn)
        layout.addWidget(file_group)

        # Swing selection
        swing_group = QGroupBox("Swing Selection")
        swing_layout = QVBoxLayout(swing_group)
        self.swing_combo = QComboBox()
        self.swing_combo.addItems(["TW_ProV1", "TW_wiffle", "GW_ProV11", "GW_wiffle"])
        self.swing_combo.currentTextChanged.connect(self.on_swing_change)
        swing_layout.addWidget(self.swing_combo)
        layout.addWidget(swing_group)

        # Playback controls
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QVBoxLayout(playback_group)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        playback_layout.addWidget(self.play_button)

        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        playback_layout.addWidget(QLabel("Frame:"))
        playback_layout.addWidget(self.frame_slider)

        self.frame_info = QLabel("Frame: 0 / 0")
        playback_layout.addWidget(self.frame_info)

        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(30)
        self.speed_slider.setValue(10)
        speed_layout.addWidget(self.speed_slider)
        playback_layout.addLayout(speed_layout)

        layout.addWidget(playback_group)

        # Camera controls
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

        layout.addWidget(camera_group)

        # Data scaling
        scaling_group = QGroupBox("Data Scaling")
        scaling_layout = QVBoxLayout(scaling_group)

        # Motion scale
        scaling_layout.addWidget(QLabel("Motion Scale Factor:"))
        self.motion_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.motion_scale_slider.setMinimum(1)
        self.motion_scale_slider.setMaximum(100)
        self.motion_scale_slider.setValue(10)
        self.motion_scale_slider.valueChanged.connect(self.on_motion_scale_change)
        scaling_layout.addWidget(self.motion_scale_slider)

        # Club length
        scaling_layout.addWidget(QLabel("Club Length (m):"))
        self.club_length_slider = QSlider(Qt.Orientation.Horizontal)
        self.club_length_slider.setMinimum(50)
        self.club_length_slider.setMaximum(150)
        self.club_length_slider.setValue(90)
        self.club_length_slider.valueChanged.connect(self.on_club_length_change)
        scaling_layout.addWidget(self.club_length_slider)

        layout.addWidget(scaling_group)

        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)

        self.trajectory_check = QCheckBox("Show Trajectory")
        self.trajectory_check.setChecked(True)
        self.trajectory_check.toggled.connect(self.update_display_options)
        display_layout.addWidget(self.trajectory_check)

        self.force_check = QCheckBox("Show Force Vectors")
        self.force_check.setChecked(True)
        self.force_check.toggled.connect(self.update_display_options)
        display_layout.addWidget(self.force_check)

        layout.addWidget(display_group)

        # Info display
        info_group = QGroupBox("Current Frame Data")
        info_layout = QVBoxLayout(info_group)
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(200)
        info_layout.addWidget(self.info_text)
        layout.addWidget(info_group)

        # Help
        help_group = QGroupBox("3D Plot Controls")
        help_layout = QVBoxLayout(help_group)
        help_text = QLabel(
            """3D Plot Interaction:
• Left-click + drag: Rotate view
• Right-click + drag: Pan view
• Mouse wheel: Zoom in/out
• Use camera buttons for preset views"""
        )
        help_layout.addWidget(help_text)
        layout.addWidget(help_group)

        layout.addStretch()
        return panel

    def create_plot_panel(self):
        """Create the right 3D plot panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 10), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Create canvas
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Setup 3D scene
        self.setup_3d_scene()

        # Connect mouse events
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        # Interaction state
        self.is_rotating = False
        self.is_panning = False
        self.last_mouse_pos = None

        return panel

    def setup_3d_scene(self):
        """Setup the 3D scene with proper coordinate system"""
        self.ax.clear()

        # Ground plane
        x_ground = np.linspace(-3, 3, 10)
        y_ground = np.linspace(-3, 3, 10)
        X_ground, Y_ground = np.meshgrid(x_ground, y_ground)
        Z_ground = np.zeros_like(X_ground) - 0.1
        self.ax.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.3, color="green")

        # Golf ball - positioned at origin (0,0,0) for proper club alignment
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        ball_radius = 0.021
        x_ball = ball_radius * np.outer(np.cos(u), np.sin(v))
        y_ball = ball_radius * np.outer(np.sin(u), np.sin(v))
        z_ball = ball_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax.plot_surface(x_ball, y_ball, z_ball, color="white")

        # Set labels and limits
        self.ax.set_xlabel("X (m) - Target Line")
        self.ax.set_ylabel("Y (m) - Toward Ball")
        self.ax.set_zlabel("Z (m) - Vertical")

        # Adjust limits for better visualization
        self.ax.set_xlim([-2.0, 2.0])
        self.ax.set_ylim([-1.0, 3.0])
        self.ax.set_zlim([-0.5, 2.5])

        # Set initial view
        self.ax.view_init(elev=15, azim=-45)
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw()

    def load_default_data(self):
        """Try to load the default Excel file"""
        default_file = "Wiffle_ProV1_club_3D_data.xlsx"
        if os.path.exists(default_file):
            self.load_excel_file(default_file)

    def load_file(self):
        """Load Excel file dialog"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Golf Swing Data File",
            "",
            "Excel files (*.xlsx);;All files (*.*)",
        )
        if filename:
            self.load_excel_file(filename)

    def load_excel_file(self, filename):
        """Load and process Excel file with proper coordinate interpretation"""
        try:
            excel_file = pd.ExcelFile(filename)

            for sheet_name in ["TW_wiffle", "TW_ProV1", "GW_wiffle", "GW_ProV11"]:
                if sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(filename, sheet_name=sheet_name, header=None)

                    # Extract key frames
                    key_frame_data = {}
                    for col in range(2, min(10, len(df.columns))):
                        if pd.notna(df.iloc[0, col]) and str(df.iloc[0, col]) in [
                            "A",
                            "T",
                            "I",
                            "F",
                        ]:
                            if col + 1 < len(df.columns) and pd.notna(
                                df.iloc[0, col + 1]
                            ):
                                key_frame_data[str(df.iloc[0, col])] = int(
                                    df.iloc[0, col + 1]
                                )

                    self.key_frames[sheet_name] = key_frame_data

                    # Extract motion data with proper coordinate interpretation
                    motion_data = []
                    for row in range(3, len(df)):
                        if pd.notna(df.iloc[row, 0]):
                            row_data = {
                                "sample": (
                                    int(df.iloc[row, 0])
                                    if pd.notna(df.iloc[row, 0])
                                    else 0
                                ),
                                "time": (
                                    float(df.iloc[row, 1])
                                    if pd.notna(df.iloc[row, 1])
                                    else 0
                                ),
                                "X": (
                                    float(df.iloc[row, 2]) / 1000
                                    if pd.notna(df.iloc[row, 2])
                                    else 0
                                ),
                                "Y": (
                                    float(df.iloc[row, 3]) / 1000
                                    if pd.notna(df.iloc[row, 3])
                                    else 0
                                ),
                                "Z": (
                                    float(df.iloc[row, 4]) / 1000
                                    if pd.notna(df.iloc[row, 4])
                                    else 0
                                ),
                                "Xx": (
                                    float(df.iloc[row, 5])
                                    if pd.notna(df.iloc[row, 5])
                                    else 0
                                ),
                                "Xy": (
                                    float(df.iloc[row, 6])
                                    if pd.notna(df.iloc[row, 6])
                                    else 0
                                ),
                                "Xz": (
                                    float(df.iloc[row, 7])
                                    if pd.notna(df.iloc[row, 7])
                                    else 0
                                ),
                                "Yx": (
                                    float(df.iloc[row, 8])
                                    if pd.notna(df.iloc[row, 8])
                                    else 0
                                ),
                                "Yy": (
                                    float(df.iloc[row, 9])
                                    if pd.notna(df.iloc[row, 9])
                                    else 0
                                ),
                                "Yz": (
                                    float(df.iloc[row, 10])
                                    if pd.notna(df.iloc[row, 10])
                                    else 0
                                ),
                            }
                            motion_data.append(row_data)

                    self.swing_data[sheet_name] = pd.DataFrame(motion_data)
                    self.print_data_debug(sheet_name)

            # Update frame slider
            if self.current_swing in self.swing_data:
                max_frame = len(self.swing_data[self.current_swing]) - 1
                self.frame_slider.setMaximum(max_frame)

            self.update_visualization()
            QMessageBox.information(self, "Success", f"Loaded data from {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def print_data_debug(self, sheet_name):
        """Print debug information about the loaded data"""
        if sheet_name in self.swing_data:
            data = self.swing_data[sheet_name]
            if not data.empty:
                print(f"\n=== Data Debug for {sheet_name} ===")
                print(f"Number of frames: {len(data)}")
                print(
                    f"Time range: {data['time'].min():.3f} to "
                    f"{data['time'].max():.3f} seconds"
                )
                print("Position ranges (meters):")
                print(f"  X: {data['X'].min():.3f} to {data['X'].max():.3f}")
                print(f"  Y: {data['Y'].min():.3f} to {data['Y'].max():.3f}")
                print(f"  Z: {data['Z'].min():.3f} to {data['Z'].max():.3f}")

                # Calculate motion range
                positions = np.array(
                    [[row["X"], row["Y"], row["Z"]] for _, row in data.iterrows()]
                )
                position_range = np.max(positions, axis=0) - np.min(positions, axis=0)
                total_range = np.linalg.norm(position_range)
                print(f"Total position range: {total_range:.3f} meters")

                print("\nData Analysis:")
                print("  Motion range is very small - this suggests:")
                print("    - Data might be tracking a single point on the club")
                print("    - Possibly near the grip or sensor attachment point")
                print("    - Not the full club head motion")

                if total_range < 0.5:
                    print(f"  WARNING: Very small motion range ({total_range:.3f}m)")
                    print(
                        "    - For comparison, a golf swing typically has "
                        "2-4m of club head motion"
                    )
                    print(
                        "    - This data appears to track a fixed point, "
                        "not the full swing arc"
                    )
                    print(
                        "    - Using standard club length and motion scaling "
                        "for visualization"
                    )

                print("=" * 40)

    def update_visualization(self):
        """Update the 3D visualization with proper coordinate system"""
        if self.current_swing not in self.swing_data:
            return

        data = self.swing_data[self.current_swing]
        if data.empty or self.current_frame >= len(data):
            return

        # Get current frame data
        frame_data = data.iloc[self.current_frame]

        # Clear and setup scene
        self.setup_3d_scene()

        # FIXED: Proper coordinate system interpretation
        # The data appears to track a point on the club (likely near the grip)
        # We need to position the club so that the clubhead points toward the ball

        # Get the tracked point position (likely near the grip)
        tracked_point = np.array([frame_data["X"], frame_data["Y"], frame_data["Z"]])

        # Apply motion scaling
        motion_scale = self.motion_scale / 10.0  # Convert slider value
        if motion_scale != 1.0:
            tracked_point = tracked_point * motion_scale

        # Get club orientation vectors
        x_axis = np.array([frame_data["Xx"], frame_data["Xy"], frame_data["Xz"]])
        y_axis = np.array([frame_data["Yx"], frame_data["Yy"], frame_data["Yz"]])
        z_axis = np.cross(x_axis, y_axis)

        # Normalize vectors
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # FIXED: Position club so clubhead points toward ball
        # The z_axis should point from grip toward clubhead
        # The ball is at origin (0,0,0), so we need to position the club accordingly

        club_length = self.club_length_override / 100.0  # Convert slider value

        # Calculate club positions
        # Grip position is the tracked point
        grip_pos = tracked_point

        # Clubhead position is grip + club_length * z_axis
        clubhead_pos = grip_pos + z_axis * club_length

        # Draw club shaft
        shaft_points = np.array([grip_pos, clubhead_pos])
        self.ax.plot(
            shaft_points[:, 0],
            shaft_points[:, 1],
            shaft_points[:, 2],
            "k-",
            linewidth=4,
            label="Shaft",
        )

        # Draw clubhead (scaled relative to club length)
        head_size = club_length * 0.08
        head_depth = club_length * 0.04

        head_vertices = np.array(
            [
                # Face of the club
                clubhead_pos + x_axis * head_size + y_axis * head_size / 2,
                clubhead_pos - x_axis * head_size + y_axis * head_size / 2,
                clubhead_pos - x_axis * head_size - y_axis * head_size / 2,
                clubhead_pos + x_axis * head_size - y_axis * head_size / 2,
                # Back of the club
                clubhead_pos
                + z_axis * head_depth
                + x_axis * head_size
                + y_axis * head_size / 2,
                clubhead_pos
                + z_axis * head_depth
                - x_axis * head_size
                + y_axis * head_size / 2,
                clubhead_pos
                + z_axis * head_depth
                - x_axis * head_size
                - y_axis * head_size / 2,
                clubhead_pos
                + z_axis * head_depth
                + x_axis * head_size
                - y_axis * head_size / 2,
            ]
        )

        # Draw clubhead outline
        for i in range(4):
            start = head_vertices[i]
            end = head_vertices[(i + 1) % 4]
            self.ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                "gray",
                linewidth=3,
            )

        for i in range(4):
            start = head_vertices[i + 4]
            end = head_vertices[((i + 1) % 4) + 4]
            self.ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                "gray",
                linewidth=3,
            )

        for i in range(4):
            start = head_vertices[i]
            end = head_vertices[i + 4]
            self.ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                "gray",
                linewidth=2,
            )

        # Draw grip
        grip_end = grip_pos + z_axis * (club_length * 0.15)
        grip_points = np.array([grip_pos, grip_end])
        self.ax.plot(
            grip_points[:, 0],
            grip_points[:, 1],
            grip_points[:, 2],
            "brown",
            linewidth=6,
            label="Grip",
        )

        # Draw trajectory if enabled
        if self.trajectory_check.isChecked() and len(data) > 1:
            trajectory = np.array(
                [[row["X"], row["Y"], row["Z"]] for _, row in data.iterrows()]
            )
            if motion_scale != 1.0:
                trajectory = trajectory * motion_scale
            self.ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                "r--",
                alpha=0.6,
                linewidth=1,
                label="Trajectory",
            )

        # Update info display
        self.update_info_display(frame_data)

        self.canvas.draw()

    def update_info_display(self, frame_data):
        """Update the information display"""
        info = f"Frame: {self.current_frame + 1}\n"
        info += f"Time: {frame_data['time']:.3f} s\n\n"

        info += "Position (m):\n"
        info += f"  X: {frame_data['X']:.4f}\n"
        info += f"  Y: {frame_data['Y']:.4f}\n"
        info += f"  Z: {frame_data['Z']:.4f}\n\n"

        info += f"Motion Scale: {self.motion_scale / 10.0:.1f}x\n"
        info += f"Club Length: {self.club_length_override / 100.0:.2f}m\n"

        self.info_text.setText(info)

    # Event handlers
    def on_swing_change(self, swing_name):
        """Handle swing selection change"""
        self.current_swing = swing_name
        self.current_frame = 0
        self.frame_slider.setValue(0)

        if self.current_swing in self.swing_data:
            max_frame = len(self.swing_data[self.current_swing]) - 1
            self.frame_slider.setMaximum(max_frame)

        self.update_visualization()
        self.update_frame_info()

    def on_frame_change(self, value):
        """Handle frame slider change"""
        self.current_frame = value
        self.update_visualization()
        self.update_frame_info()

    def on_motion_scale_change(self, value):
        """Handle motion scale change"""
        self.motion_scale = value
        self.update_visualization()

    def on_club_length_change(self, value):
        """Handle club length change"""
        self.club_length_override = value
        self.update_visualization()

    def update_display_options(self):
        """Handle display option changes"""
        self.update_visualization()

    def update_frame_info(self):
        """Update frame information display"""
        if self.current_swing in self.swing_data:
            max_frame = len(self.swing_data[self.current_swing]) - 1
            self.frame_info.setText(f"Frame: {self.current_frame} / {max_frame}")

    def toggle_playback(self):
        """Toggle play/pause"""
        self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_button.setText("Pause")
            self.animate()
        else:
            self.play_button.setText("Play")

    def animate(self):
        """Animation loop for playback"""
        if not self.is_playing:
            return

        if self.current_swing in self.swing_data:
            max_frame = len(self.swing_data[self.current_swing]) - 1

            if self.current_frame >= max_frame:
                self.current_frame = 0
            else:
                self.current_frame += 1

            self.frame_slider.setValue(self.current_frame)
            self.update_visualization()
            self.update_frame_info()

            # Schedule next frame
            delay = int(
                1000 / (self.speed_slider.value() * 2)
            )  # Convert speed to delay
            QTimer.singleShot(delay, self.animate)
        else:
            self.is_playing = False
            self.play_button.setText("Play")

    def set_camera_view(self, view):
        """Set predefined camera views"""
        if view == "face_on":
            self.ax.view_init(elev=0, azim=0)
        elif view == "down_line":
            self.ax.view_init(elev=0, azim=180)
        elif view == "top_down":
            self.ax.view_init(elev=90, azim=0)
        elif view == "isometric":
            self.ax.view_init(elev=15, azim=-45)

        self.canvas.draw()

    def reset_view(self):
        """Reset the 3D view"""
        self.ax.view_init(elev=15, azim=-45)
        self.ax.set_xlim([-2.0, 2.0])
        self.ax.set_ylim([-1.0, 3.0])
        self.ax.set_zlim([-0.5, 2.5])
        self.canvas.draw()

    def on_mouse_press(self, event):
        """Handle mouse button press"""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left mouse button
            self.is_rotating = True
        elif event.button == 3:  # Right mouse button
            self.is_panning = True

        self.last_mouse_pos = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        """Handle mouse button release"""
        self.is_rotating = False
        self.is_panning = False
        self.last_mouse_pos = None

    def on_mouse_move(self, event):
        """Handle mouse movement"""
        if event.inaxes != self.ax or self.last_mouse_pos is None:
            return

        if not (self.is_rotating or self.is_panning):
            return

        dx = event.xdata - self.last_mouse_pos[0]
        dy = event.ydata - self.last_mouse_pos[1]

        if self.is_rotating:
            self.ax.azim += dx * 2
            self.ax.elev += dy * 2
            self.canvas.draw()
        elif self.is_panning:
            x_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
            y_range = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
            pan_factor = 0.1
            self.ax.set_xlim(self.ax.get_xlim() - dx * x_range * pan_factor)
            self.ax.set_ylim(self.ax.get_ylim() - dy * y_range * pan_factor)
            self.canvas.draw()

        self.last_mouse_pos = (event.xdata, event.ydata)

    def on_scroll(self, event):
        """Handle mouse scroll for zooming"""
        if event.inaxes != self.ax:
            return

        zoom_factor = 1.1 if event.button == "up" else 0.9

        x_lim = self.ax.get_xlim()
        y_lim = self.ax.get_ylim()
        z_lim = self.ax.get_zlim()

        x_center = (x_lim[0] + x_lim[1]) / 2
        y_center = (y_lim[0] + y_lim[1]) / 2
        z_center = (z_lim[0] + z_lim[1]) / 2

        x_range = (x_lim[1] - x_lim[0]) / 2
        y_range = (y_lim[1] - y_lim[0]) / 2
        z_range = (z_lim[1] - z_lim[0]) / 2

        self.ax.set_xlim(
            x_center - x_range * zoom_factor, x_center + x_range * zoom_factor
        )
        self.ax.set_ylim(
            y_center - y_range * zoom_factor, y_center + y_range * zoom_factor
        )
        self.ax.set_zlim(
            z_center - z_range * zoom_factor, z_center + z_range * zoom_factor
        )

        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    window = GolfSwingAnalyzerPyQt6()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
