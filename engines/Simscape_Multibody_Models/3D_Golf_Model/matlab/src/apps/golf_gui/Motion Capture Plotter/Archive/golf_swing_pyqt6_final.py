import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Qt5Agg")  # Force Qt5 backend for compatibility

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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
        self.setWindowTitle("Golf Swing Analyzer - PyQt6")
        self.setGeometry(100, 100, 1400, 900)

        # Data storage
        self.swing_data = {}
        self.current_swing = None
        self.current_frame = 0
        self.is_playing = False
        self.current_filter = "none"

        # Club parameters
        self.shaft_length = 0.9  # meters
        self.motion_scale = 1.0

        # Setup UI
        self.setup_ui()

        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.next_frame)

    def setup_ui(self):
        """Setup the main UI"""
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
        self.swing_combo.currentTextChanged.connect(self.on_swing_change)
        swing_layout.addWidget(self.swing_combo)
        layout.addWidget(swing_group)

        # Playback controls
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

        layout.addWidget(playback_group)

        # Visualization options
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

        # Motion scaling
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Motion Scale:"))
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(1, 10)
        self.scale_slider.setValue(3)  # Default 3x scale
        self.scale_slider.valueChanged.connect(self.on_scale_change)
        scale_layout.addWidget(self.scale_slider)
        self.scale_label = QLabel("3x")
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

        layout.addWidget(viz_group)

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

        # Analysis info
        info_group = QGroupBox("Current Frame Data")
        info_layout = QVBoxLayout(info_group)
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        info_layout.addWidget(self.info_text)
        layout.addWidget(info_group)

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
        layout.addWidget(help_group)

        layout.addStretch()
        return panel

    def create_plot_panel(self):
        """Create the right plot panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Enable interactive features
        self.ax.mouse_init()

        # Create canvas
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Connect mouse events for zoom/rotation
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        # Initialize empty plot elements
        self.club_line = None
        self.club_head = None
        self.trajectory_line = None
        self.club_path_line = None

        return panel

    def load_file(self):
        """Load Excel file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Excel File", "", "Excel Files (*.xlsx *.xls)"
        )

        if filename:
            self.load_excel_file(filename)

    def load_excel_file(self, filename):
        """Load and process Excel file"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(filename)

            for sheet_name in ["TW_wiffle", "TW_ProV1", "GW_wiffle", "GW_ProV11"]:
                if sheet_name in excel_file.sheet_names:
                    # Read the sheet
                    df = pd.read_excel(filename, sheet_name=sheet_name, header=None)

                    # Extract key frames from first row
                    key_frame_data = {}
                    for col in range(2, min(10, len(df.columns))):
                        if pd.notna(df.iloc[0, col]) and str(df.iloc[0, col]) in [
                            "A",
                            "T",
                            "I",
                            "F",
                        ]:
                            key_frame_data[str(df.iloc[0, col])] = col

                    # Process data starting from row 2
                    if len(df) > 1:
                        # Extract position and orientation data
                        data = []
                        for i in range(1, len(df)):
                            row = df.iloc[i]
                            if len(row) >= 15:  # Ensure we have enough columns
                                frame_data = {
                                    "time": row[0] if pd.notna(row[0]) else i - 1,
                                    "X": (
                                        row[1] / 1000.0 if pd.notna(row[1]) else 0
                                    ),  # Convert mm to m
                                    "Y": row[2] / 1000.0 if pd.notna(row[2]) else 0,
                                    "Z": row[3] / 1000.0 if pd.notna(row[3]) else 0,
                                    "Xx": row[4] if pd.notna(row[4]) else 0,
                                    "Xy": row[5] if pd.notna(row[5]) else 0,
                                    "Xz": row[6] if pd.notna(row[6]) else 0,
                                    "Yx": row[7] if pd.notna(row[7]) else 0,
                                    "Yy": row[8] if pd.notna(row[8]) else 0,
                                    "Yz": row[9] if pd.notna(row[9]) else 0,
                                    "Zx": row[10] if pd.notna(row[10]) else 0,
                                    "Zy": row[11] if pd.notna(row[11]) else 0,
                                    "Zz": row[12] if pd.notna(row[12]) else 0,
                                }
                                data.append(frame_data)

                        if data:
                            self.swing_data[sheet_name] = pd.DataFrame(data)
                            self.print_data_debug(sheet_name)

            # Update swing selection
            self.swing_combo.clear()
            self.swing_combo.addItems(list(self.swing_data.keys()))

            if self.swing_data:
                self.current_swing = list(self.swing_data.keys())[0]
                self.swing_combo.setCurrentText(self.current_swing)
                self.setup_frame_slider()
                self.update_visualization()

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

                # Calculate total position range
                pos_range = np.max(
                    [
                        data["X"].max() - data["X"].min(),
                        data["Y"].max() - data["Y"].min(),
                        data["Z"].max() - data["Z"].min(),
                    ]
                )
                print(f"Total position range: {pos_range:.3f} meters")

                print("Data Analysis:")
                print("  Motion range is very small - this suggests:")
                print("    - Data might be tracking a single point on the club")
                print("    - Possibly near the grip or sensor attachment point")
                print("    - Not the full club head motion")
                print(f"  WARNING: Very small motion range ({pos_range:.3f}m)")
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

                print("Orientation vector ranges:")
                print(f"  Xx: {data['Xx'].min():.3f} to {data['Xx'].max():.3f}")
                print(f"  Xy: {data['Xy'].min():.3f} to {data['Xy'].max():.3f}")
                print(f"  Xz: {data['Xz'].min():.3f} to {data['Xz'].max():.3f}")
                print(f"  Yx: {data['Yx'].min():.3f} to {data['Yx'].max():.3f}")
                print(f"  Yy: {data['Yy'].min():.3f} to {data['Yy'].max():.3f}")
                print(f"  Yz: {data['Yz'].min():.3f} to {data['Yz'].max():.3f}")

                # Validate orientation vectors
                x_norms = np.sqrt(data["Xx"] ** 2 + data["Xy"] ** 2 + data["Xz"] ** 2)
                y_norms = np.sqrt(data["Yx"] ** 2 + data["Yy"] ** 2 + data["Yz"] ** 2)
                print("Orientation vector validation:")
                print(
                    f"  X-axis norm range: {x_norms.min():.3f} to {x_norms.max():.3f} (should be ~1.0)"
                )
                print(
                    f"  Y-axis norm range: {y_norms.min():.3f} to {y_norms.max():.3f} (should be ~1.0)"
                )
                print("=" * 40)

    def setup_frame_slider(self):
        """Setup the frame slider"""
        if self.current_swing in self.swing_data:
            data = self.swing_data[self.current_swing]
            max_frame = len(data) - 1
            self.frame_slider.setRange(0, max_frame)
            self.frame_slider.setValue(0)
            self.current_frame = 0

    def on_swing_change(self, swing_name):
        """Handle swing selection change"""
        if swing_name in self.swing_data:
            self.current_swing = swing_name
            self.setup_frame_slider()
            self.update_visualization()

    def on_frame_change(self, frame):
        """Handle frame slider change"""
        self.current_frame = frame
        self.frame_label.setText(str(frame))
        self.update_visualization()

    def on_speed_change(self, speed):
        """Handle speed slider change"""
        self.speed_label.setText(str(speed))
        if self.is_playing:
            self.animation_timer.setInterval(1000 // speed)

    def on_scale_change(self, scale):
        """Handle motion scale change"""
        self.motion_scale = scale
        self.scale_label.setText(f"{scale}x")
        self.update_visualization()

    def on_club_length_change(self, length_cm):
        """Handle club length change"""
        self.shaft_length = length_cm / 100.0  # Convert cm to meters
        self.club_label.setText(f"{self.shaft_length:.1f}m")
        self.update_visualization()

    def toggle_playback(self):
        """Toggle play/pause"""
        if self.is_playing:
            self.animation_timer.stop()
            self.play_btn.setText("Play")
            self.is_playing = False
        else:
            speed = self.speed_slider.value()
            self.animation_timer.start(1000 // speed)
            self.play_btn.setText("Pause")
            self.is_playing = True

    def next_frame(self):
        """Advance to next frame"""
        if self.current_swing in self.swing_data:
            data = self.swing_data[self.current_swing]
            if self.current_frame < len(data) - 1:
                self.current_frame += 1
                self.frame_slider.setValue(self.current_frame)
            else:
                # Loop back to start
                self.current_frame = 0
                self.frame_slider.setValue(0)

    def setup_3d_scene(self):
        """Setup the 3D scene with ground plane and ball"""
        self.ax.clear()

        # Ground plane - adjusted for golf swing coordinate system
        x_ground = np.linspace(-3, 3, 10)
        y_ground = np.linspace(-3, 3, 10)
        X_ground, Y_ground = np.meshgrid(x_ground, y_ground)
        Z_ground = np.zeros_like(X_ground) - 0.1
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
        self.ax.set_zlim([-0.5, 2.5])

        # Set initial view
        self.ax.view_init(elev=15, azim=-45)

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

        # Apply motion scaling to the tracked position
        mid_hands = np.array(
            [
                frame_data["X"] * self.motion_scale,
                frame_data["Y"] * self.motion_scale,
                frame_data["Z"] * self.motion_scale,
            ]
        )

        # Get orientation vectors (these define the club's orientation)
        x_axis = np.array([frame_data["Xx"], frame_data["Xy"], frame_data["Xz"]])
        y_axis = np.array([frame_data["Yx"], frame_data["Yy"], frame_data["Yz"]])
        z_axis = np.array([frame_data["Zx"], frame_data["Zy"], frame_data["Zz"]])

        # Normalize vectors
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # The z-axis should point from grip to clubhead
        # Calculate club head position
        club_head = mid_hands + z_axis * self.shaft_length

        # Draw the club shaft
        shaft_points = np.array([mid_hands, club_head])
        self.ax.plot(
            shaft_points[:, 0],
            shaft_points[:, 1],
            shaft_points[:, 2],
            "b-",
            linewidth=8,
            alpha=0.8,
            label="Club Shaft",
        )

        # Draw club head (larger and more visible)
        head_size = self.shaft_length * 0.15  # 15% of club length
        head_points = []
        for i in range(8):
            angle = i * np.pi / 4
            offset = head_size * np.array(
                [
                    np.cos(angle) * x_axis[0] + np.sin(angle) * y_axis[0],
                    np.cos(angle) * x_axis[1] + np.sin(angle) * y_axis[1],
                    np.cos(angle) * x_axis[2] + np.sin(angle) * y_axis[2],
                ]
            )
            head_points.append(club_head + offset)

        head_points = np.array(head_points)
        self.ax.plot(
            head_points[:, 0],
            head_points[:, 1],
            head_points[:, 2],
            "r-",
            linewidth=6,
            alpha=0.9,
            label="Club Head",
        )

        # Draw trajectory paths
        if self.trajectory_check.isChecked() and len(data) > 1:
            # Mid-hands path (blue dashed)
            trajectory = np.array(
                [
                    [
                        row["X"] * self.motion_scale,
                        row["Y"] * self.motion_scale,
                        row["Z"] * self.motion_scale,
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
            # Club head path (red dashed)
            club_path = []
            for _, row in data.iterrows():
                pos = np.array(
                    [
                        row["X"] * self.motion_scale,
                        row["Y"] * self.motion_scale,
                        row["Z"] * self.motion_scale,
                    ]
                )
                z_vec = np.array([row["Zx"], row["Zy"], row["Zz"]])
                z_vec = z_vec / np.linalg.norm(z_vec)
                club_head_pos = pos + z_vec * self.shaft_length
                club_path.append(club_head_pos)

            club_path = np.array(club_path)
            self.ax.plot(
                club_path[:, 0],
                club_path[:, 1],
                club_path[:, 2],
                "r--",
                alpha=0.6,
                linewidth=2,
                label="Club Head Path",
            )

        # Update info text
        self.update_info_text(frame_data)

        # Redraw canvas
        self.canvas.draw()

    def update_info_text(self, frame_data):
        """Update the information text display"""
        info = f"Frame: {self.current_frame}\n"
        info += f"Time: {frame_data['time']:.3f}s\n"
        info += "Position (m):\n"
        info += f"  X: {frame_data['X']:.3f}\n"
        info += f"  Y: {frame_data['Y']:.3f}\n"
        info += f"  Z: {frame_data['Z']:.3f}\n"
        info += f"Motion Scale: {self.motion_scale}x\n"
        info += f"Club Length: {self.shaft_length:.2f}m"

        self.info_text.setText(info)

    def set_camera_view(self, view):
        """Set predefined camera views"""
        if view == "face_on":
            # Face-on view: looking at golfer from front
            self.ax.view_init(elev=0, azim=0)
        elif view == "down_line":
            # Down-the-line view: looking from behind golfer toward target
            self.ax.view_init(elev=0, azim=180)
        elif view == "top_down":
            # Top-down view: looking down from above
            self.ax.view_init(elev=90, azim=0)
        elif view == "isometric":
            # Isometric view: 3D perspective
            self.ax.view_init(elev=15, azim=-45)

        self.canvas.draw()

    def reset_view(self):
        """Reset the 3D view to the default isometric view and limits"""
        # Reset view angles
        self.ax.view_init(elev=15, azim=-45)

        # Reset plot limits to default
        self.ax.set_xlim([-2.0, 2.0])
        self.ax.set_ylim([-1.0, 3.0])
        self.ax.set_zlim([-0.5, 2.5])

        self.canvas.draw()

    def on_scroll(self, event):
        """Handle mouse scroll for zooming"""
        if event.inaxes != self.ax:
            return

        # Get current view limits
        x_lim = self.ax.get_xlim()
        y_lim = self.ax.get_ylim()
        z_lim = self.ax.get_zlim()

        # Determine zoom factor based on scroll direction
        if event.button == "up":
            zoom_factor = 0.9  # Zoom in
        else:
            zoom_factor = 1.1  # Zoom out

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

    def on_mouse_press(self, event):
        """Handle mouse button press for rotation/panning"""
        if event.inaxes != self.ax:
            return
        # Store initial position for rotation/panning
        self._last_pos = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        """Handle mouse button release"""
        self._last_pos = None

    def on_mouse_move(self, event):
        """Handle mouse movement for rotation/panning"""
        if event.inaxes != self.ax or self._last_pos is None:
            return

        if event.button == 1:  # Left click - rotate
            # Get current view angles
            elev = self.ax.elev
            azim = self.ax.azim

            # Calculate change in position
            dx = event.xdata - self._last_pos[0]
            dy = event.ydata - self._last_pos[1]

            # Update view angles
            self.ax.view_init(elev=elev + dy, azim=azim + dx)
            self.canvas.draw()

        elif event.button == 3:  # Right click - pan
            # Get current limits
            x_lim = self.ax.get_xlim()
            y_lim = self.ax.get_ylim()

            # Calculate change in position
            dx = event.xdata - self._last_pos[0]
            dy = event.ydata - self._last_pos[1]

            # Update limits
            x_range = x_lim[1] - x_lim[0]
            y_range = y_lim[1] - y_lim[0]

            self.ax.set_xlim([x_lim[0] - dx * x_range, x_lim[1] - dx * x_range])
            self.ax.set_ylim([y_lim[0] - dy * y_range, y_lim[1] - dy * y_range])
            self.canvas.draw()

        self._last_pos = (event.xdata, event.ydata)


def main():
    app = QApplication(sys.argv)
    window = GolfSwingAnalyzerPyQt6()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
