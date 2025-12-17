import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import signal


class GolfSwingAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Golf Swing Motion Capture Analyzer")
        self.root.geometry("1400x900")

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

        self.setup_gui()
        self.load_default_data()

    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for controls
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        control_frame.pack_propagate(False)

        # Right panel for 3D plot
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_controls(control_frame)
        self.setup_plot(plot_frame)

    def setup_controls(self, parent):
        """Setup control panel"""
        # Title
        title_label = ttk.Label(
            parent, text="Golf Swing Analyzer", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 10))

        # File loading
        file_frame = ttk.LabelFrame(parent, text="Data Loading", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="Load Excel File", command=self.load_file).pack(
            fill=tk.X
        )

        # Swing selection
        swing_frame = ttk.LabelFrame(parent, text="Swing Selection", padding=10)
        swing_frame.pack(fill=tk.X, pady=(0, 10))

        self.swing_var = tk.StringVar(value=self.current_swing)
        swing_options = ["TW_ProV1", "TW_wiffle", "GW_ProV11", "GW_wiffle"]
        self.swing_combo = ttk.Combobox(
            swing_frame,
            textvariable=self.swing_var,
            values=swing_options,
            state="readonly",
        )
        self.swing_combo.pack(fill=tk.X)
        self.swing_combo.bind("<<ComboboxSelected>>", self.on_swing_change)

        # Playback controls
        playback_frame = ttk.LabelFrame(parent, text="Playback Controls", padding=10)
        playback_frame.pack(fill=tk.X, pady=(0, 10))

        # Play/Pause button
        self.play_button = ttk.Button(
            playback_frame, text="Play", command=self.toggle_playback
        )
        self.play_button.pack(fill=tk.X, pady=(0, 5))

        # Frame slider
        self.frame_var = tk.IntVar()
        self.frame_scale = ttk.Scale(
            playback_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.frame_var,
            command=self.on_frame_change,
        )
        self.frame_scale.pack(fill=tk.X, pady=(0, 5))

        # Frame info
        self.frame_info = ttk.Label(playback_frame, text="Frame: 0 / 0")
        self.frame_info.pack()

        # Speed control
        speed_frame = ttk.Frame(playback_frame)
        speed_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(
            speed_frame,
            from_=0.1,
            to=3.0,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
        )
        speed_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Camera controls
        camera_frame = ttk.LabelFrame(parent, text="Camera Views", padding=10)
        camera_frame.pack(fill=tk.X, pady=(0, 10))

        camera_buttons = [
            ("Face-On", lambda: self.set_camera_view("face_on")),
            ("Down-the-Line", lambda: self.set_camera_view("down_line")),
            ("Top-Down", lambda: self.set_camera_view("top_down")),
            ("Isometric", lambda: self.set_camera_view("isometric")),
            ("Reset View", lambda: self.reset_view()),
        ]

        for i, (text, command) in enumerate(camera_buttons):
            row, col = i // 2, i % 2
            btn = ttk.Button(camera_frame, text=text, command=command)
            btn.grid(row=row, column=col, sticky="ew", padx=2, pady=2)

        camera_frame.grid_columnconfigure(0, weight=1)
        camera_frame.grid_columnconfigure(1, weight=1)

        # Filtering options
        filter_frame = ttk.LabelFrame(parent, text="Data Filtering", padding=10)
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        self.filter_var = tk.StringVar(value="None")
        filter_options = [
            "None",
            "Moving Average",
            "Savitzky-Golay",
            "Butterworth 6Hz",
            "Butterworth 8Hz",
            "Butterworth 10Hz",
        ]
        filter_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_var,
            values=filter_options,
            state="readonly",
        )
        filter_combo.pack(fill=tk.X)
        filter_combo.bind("<<ComboboxSelected>>", self.on_filter_change)

        # Evaluation point offset
        offset_frame = ttk.LabelFrame(parent, text="Evaluation Point", padding=10)
        offset_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(offset_frame, text="Offset (inches):").pack()
        self.offset_var = tk.DoubleVar()
        offset_scale = ttk.Scale(
            offset_frame,
            from_=-2,
            to=2,
            orient=tk.HORIZONTAL,
            variable=self.offset_var,
            command=self.on_offset_change,
        )
        offset_scale.pack(fill=tk.X)

        # Data scaling options
        scaling_frame = ttk.LabelFrame(parent, text="Data Scaling", padding=10)
        scaling_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(scaling_frame, text="Motion Scale Factor:").pack()
        self.scale_var = tk.DoubleVar(value=1.0)
        scale_scale = ttk.Scale(
            scaling_frame,
            from_=0.1,
            to=10.0,
            orient=tk.HORIZONTAL,
            variable=self.scale_var,
            command=self.on_scale_change,
        )
        scale_scale.pack(fill=tk.X)

        # Club length override
        ttk.Label(scaling_frame, text="Club Length (m):").pack()
        self.club_length_var = tk.DoubleVar(value=0.9)
        club_length_scale = ttk.Scale(
            scaling_frame,
            from_=0.5,
            to=1.5,
            orient=tk.HORIZONTAL,
            variable=self.club_length_var,
            command=self.on_club_length_change,
        )
        club_length_scale.pack(fill=tk.X)

        # Display options
        display_frame = ttk.LabelFrame(parent, text="Display Options", padding=10)
        display_frame.pack(fill=tk.X, pady=(0, 10))

        self.trajectory_var = tk.BooleanVar(value=True)
        trajectory_check = ttk.Checkbutton(
            display_frame,
            text="Show Trajectory",
            variable=self.trajectory_var,
            command=self.update_display_options,
        )
        trajectory_check.pack(anchor=tk.W)

        self.force_var = tk.BooleanVar(value=True)
        force_check = ttk.Checkbutton(
            display_frame,
            text="Show Force Vectors",
            variable=self.force_var,
            command=self.update_display_options,
        )
        force_check.pack(anchor=tk.W)

        # Analysis info
        info_frame = ttk.LabelFrame(parent, text="Current Frame Data", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_text = tk.Text(info_frame, height=8, width=30)
        info_scroll = ttk.Scrollbar(
            info_frame, orient=tk.VERTICAL, command=self.info_text.yview
        )
        self.info_text.configure(yscrollcommand=info_scroll.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Interactive controls help
        help_frame = ttk.LabelFrame(parent, text="3D Plot Controls", padding=10)
        help_frame.pack(fill=tk.X, pady=(0, 10))

        help_text = """3D Plot Interaction:
• Left-click + drag: Rotate view
• Right-click + drag: Pan view
• Mouse wheel: Zoom in/out
• Use camera buttons for preset views"""

        help_label = ttk.Label(help_frame, text=help_text, justify=tk.LEFT)
        help_label.pack(anchor=tk.W)

    def setup_plot(self, parent):
        """Setup 3D matplotlib plot"""
        # Enable interactive mode for zoom and rotation
        plt.ion()

        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Enable interactive features
        self.ax.mouse_init()

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize empty plot elements
        self.club_line = None
        self.club_head = None
        self.trajectory_line = None
        self.force_arrow = None
        self.torque_arrow = None
        self.key_points = []

        # Add mouse event handlers for better interaction
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        # Interaction state
        self.is_rotating = False
        self.is_panning = False
        self.last_mouse_pos = None

        self.setup_3d_scene()

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
        z_ball = ball_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + ball_radius
        self.ax.plot_surface(x_ball, y_ball, z_ball, color="white")

        # Set labels and limits - adjusted for golf swing coordinate system
        self.ax.set_xlabel("X (m) - Target Line")
        self.ax.set_ylabel("Y (m) - Toward Ball")
        self.ax.set_zlabel("Z (m) - Vertical")

        # Adjust limits based on typical golf swing range
        self.ax.set_xlim([-2.0, 2.0])  # Target line range
        self.ax.set_ylim([-1.0, 3.0])  # Ball direction range
        self.ax.set_zlim([-0.5, 2.5])  # Vertical range

        # Set initial view - better for golf swing analysis
        self.ax.view_init(elev=15, azim=-45)

        # Add grid for better spatial reference
        self.ax.grid(True, alpha=0.3)

    def load_default_data(self):
        """Try to load the default Excel file"""
        default_file = "Wiffle_ProV1_club_3D_data.xlsx"
        if os.path.exists(default_file):
            self.load_excel_file(default_file)

    def load_file(self):
        """Load Excel file dialog"""
        filename = filedialog.askopenfilename(
            title="Select Golf Swing Data File",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
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
                            if col + 1 < len(df.columns) and pd.notna(
                                df.iloc[0, col + 1]
                            ):
                                key_frame_data[str(df.iloc[0, col])] = int(
                                    df.iloc[0, col + 1]
                                )

                    self.key_frames[sheet_name] = key_frame_data

                    # Extract motion data starting from row 3 (0-indexed)
                    motion_data = []
                    for row in range(3, len(df)):
                        if pd.notna(df.iloc[row, 0]):  # Sample number exists
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
                                ),  # mm to m
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

                    # Debug: Print data statistics
                    self.print_data_debug(sheet_name)

            # Update frame slider maximum
            if self.current_swing in self.swing_data:
                max_frame = len(self.swing_data[self.current_swing]) - 1
                self.frame_scale.configure(to=max_frame)

            self.update_visualization()
            messagebox.showinfo("Success", f"Loaded data from {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def print_data_debug(self, sheet_name):
        """Print debug information about the loaded data"""
        if sheet_name in self.swing_data:
            data = self.swing_data[sheet_name]
            if not data.empty:
                print(f"\n=== Data Debug for {sheet_name} ===")
                print(f"Number of frames: {len(data)}")
                print(
                    f"Time range: {data['time'].min():.3f} to {data['time'].max():.3f} seconds"
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

                # Analyze the data characteristics
                print("\nData Analysis:")
                print("  Motion range is very small - this suggests:")
                print("    - Data might be tracking a single point on the club")
                print("    - Possibly near the grip or sensor attachment point")
                print("    - Not the full club head motion")

                # Check if data might be in different units
                if total_range < 0.5:
                    print(f"  WARNING: Very small motion range ({total_range:.3f}m)")
                    print(
                        "    - For comparison, a golf swing typically has 2-4m of club head motion"
                    )
                    print(
                        "    - This data appears to track a fixed point, not the full swing arc"
                    )
                    print("    - Using standard club length (0.9m) for visualization")
                else:
                    print("  Motion range seems reasonable for a golf swing")

                # Check orientation vectors
                print("\nOrientation vector ranges:")
                print(f"  Xx: {data['Xx'].min():.3f} to {data['Xx'].max():.3f}")
                print(f"  Xy: {data['Xy'].min():.3f} to {data['Xy'].max():.3f}")
                print(f"  Xz: {data['Xz'].min():.3f} to {data['Xz'].max():.3f}")
                print(f"  Yx: {data['Yx'].min():.3f} to {data['Yx'].max():.3f}")
                print(f"  Yy: {data['Yy'].min():.3f} to {data['Yy'].max():.3f}")
                print(f"  Yz: {data['Yz'].min():.3f} to {data['Yz'].max():.3f}")

                # Check if orientation vectors are unit vectors
                x_norms = np.sqrt(data["Xx"] ** 2 + data["Xy"] ** 2 + data["Xz"] ** 2)
                y_norms = np.sqrt(data["Yx"] ** 2 + data["Yy"] ** 2 + data["Yz"] ** 2)
                print("\nOrientation vector validation:")
                print(
                    f"  X-axis norm range: {x_norms.min():.3f} to {x_norms.max():.3f} (should be ~1.0)"
                )
                print(
                    f"  Y-axis norm range: {y_norms.min():.3f} to {y_norms.max():.3f} (should be ~1.0)"
                )

                print("=" * 40)

    def apply_filter(self, data, method):
        """Apply selected filter to the data"""
        if method == "None" or data.empty:
            return data

        filtered_data = data.copy()
        position_cols = ["X", "Y", "Z"]
        orientation_cols = ["Xx", "Xy", "Xz", "Yx", "Yy", "Yz"]

        for col in position_cols + orientation_cols:
            if col in filtered_data.columns:
                if method == "Moving Average":
                    filtered_data[col] = (
                        filtered_data[col].rolling(window=5, center=True).mean()
                    )
                elif method == "Savitzky-Golay":
                    if len(filtered_data) > 9:
                        filtered_data[col] = signal.savgol_filter(
                            filtered_data[col], 9, 3
                        )
                elif "Butterworth" in method:
                    cutoff = int(method.split()[-1].replace("Hz", ""))
                    fs = 240  # Assumed sampling frequency
                    nyquist = fs / 2
                    normalized_cutoff = cutoff / nyquist
                    b, a = signal.butter(4, normalized_cutoff, btype="low")
                    filtered_data[col] = signal.filtfilt(b, a, filtered_data[col])

        # Forward fill any NaN values created by filtering
        filtered_data = filtered_data.fillna(method="ffill").fillna(method="bfill")
        return filtered_data

    def calculate_kinematics(self, data):
        """Calculate velocity and acceleration from position data"""
        if len(data) < 3:
            return None

        kinematics = []
        dt_mean = np.mean(np.diff(data["time"]))

        for i in range(len(data)):
            # Central difference for velocity
            if i == 0:
                vel_x = (data.iloc[i + 1]["X"] - data.iloc[i]["X"]) / dt_mean
                vel_y = (data.iloc[i + 1]["Y"] - data.iloc[i]["Y"]) / dt_mean
                vel_z = (data.iloc[i + 1]["Z"] - data.iloc[i]["Z"]) / dt_mean
            elif i == len(data) - 1:
                vel_x = (data.iloc[i]["X"] - data.iloc[i - 1]["X"]) / dt_mean
                vel_y = (data.iloc[i]["Y"] - data.iloc[i - 1]["Y"]) / dt_mean
                vel_z = (data.iloc[i]["Z"] - data.iloc[i - 1]["Z"]) / dt_mean
            else:
                vel_x = (data.iloc[i + 1]["X"] - data.iloc[i - 1]["X"]) / (2 * dt_mean)
                vel_y = (data.iloc[i + 1]["Y"] - data.iloc[i - 1]["Y"]) / (2 * dt_mean)
                vel_z = (data.iloc[i + 1]["Z"] - data.iloc[i - 1]["Z"]) / (2 * dt_mean)

            # Central difference for acceleration
            if i <= 1 or i >= len(data) - 2:
                acc_x = acc_y = acc_z = 0
            else:
                acc_x = (
                    data.iloc[i + 1]["X"]
                    - 2 * data.iloc[i]["X"]
                    + data.iloc[i - 1]["X"]
                ) / (dt_mean**2)
                acc_y = (
                    data.iloc[i + 1]["Y"]
                    - 2 * data.iloc[i]["Y"]
                    + data.iloc[i - 1]["Y"]
                ) / (dt_mean**2)
                acc_z = (
                    data.iloc[i + 1]["Z"]
                    - 2 * data.iloc[i]["Z"]
                    + data.iloc[i - 1]["Z"]
                ) / (dt_mean**2)

            kinematics.append(
                {
                    "velocity": np.array([vel_x, vel_y, vel_z]),
                    "acceleration": np.array([acc_x, acc_y, acc_z]),
                }
            )

        return kinematics

    def calculate_dynamics(self, frame_data, kinematics, frame_idx):
        """Calculate force and torque vectors"""
        if not kinematics or frame_idx >= len(kinematics):
            return {"force": np.zeros(3), "torque": np.zeros(3)}

        # Force = mass * acceleration
        acceleration = kinematics[frame_idx]["acceleration"]
        force = self.club_mass * acceleration

        # Simplified torque calculation
        # In reality, this would require more complex rigid body dynamics
        torque = np.array(
            [
                force[1] * 0.5,  # Approximation based on lever arm
                -force[0] * 0.5,
                force[2] * 0.2,
            ]
        )

        return {"force": force, "torque": torque}

    def update_visualization(self):
        """Update the 3D visualization"""
        if self.current_swing not in self.swing_data:
            return

        data = self.swing_data[self.current_swing]
        if data.empty or self.current_frame >= len(data):
            return

        # Apply filtering
        filtered_data = self.apply_filter(data, self.current_filter)

        # Calculate kinematics
        kinematics = self.calculate_kinematics(filtered_data)

        # Get current frame data
        frame_data = filtered_data.iloc[self.current_frame]

        # Clear previous club visualization
        self.setup_3d_scene()

        # FIXED: Proper coordinate system conversion
        # Original data: X (target line), Y (toward ball), Z (vertical)
        # Visualization: Keep X as X, Y as Y, Z as Z for proper orientation
        mid_hands = np.array([frame_data["X"], frame_data["Y"], frame_data["Z"]])

        # FIXED: Proper orientation vector handling
        # The orientation vectors should represent the club's coordinate system
        x_axis = np.array([frame_data["Xx"], frame_data["Xy"], frame_data["Xz"]])
        y_axis = np.array([frame_data["Yx"], frame_data["Yy"], frame_data["Yz"]])
        z_axis = np.cross(x_axis, y_axis)

        # Normalize vectors to ensure they're unit vectors
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # FIXED: Better club length estimation based on motion analysis
        # The data appears to be tracking a single point, so we need to estimate
        # the full club length based on typical golf club dimensions
        if len(filtered_data) > 10:
            # Calculate the range of motion to understand the scale
            positions = np.array(
                [[row["X"], row["Y"], row["Z"]] for _, row in filtered_data.iterrows()]
            )
            position_range = np.max(positions, axis=0) - np.min(positions, axis=0)

            # The position data seems to be tracking a point on the club (likely near the grip)
            # For a typical golf swing, the club head should move much more than the grip
            # Let's estimate the club length based on typical proportions

            # Use the user-defined club length if available
            club_length = self.club_length_var.get()

            # Apply motion scaling to make the swing arc more realistic
            motion_scale = self.scale_var.get()
            if motion_scale != 1.0:
                # Scale the position data to make the swing arc larger
                mid_hands = mid_hands * motion_scale
                # Also scale the trajectory if it's being shown
                if self.trajectory_var.get() and len(filtered_data) > 1:
                    trajectory = np.array(
                        [
                            [row["X"], row["Y"], row["Z"]]
                            for _, row in filtered_data.iterrows()
                        ]
                    )
                    trajectory = trajectory * motion_scale
        else:
            club_length = self.club_length_var.get()

        # Calculate club positions
        eval_offset_m = self.eval_offset * 0.0254  # inches to meters
        eval_point = mid_hands + z_axis * eval_offset_m
        club_face = mid_hands + z_axis * (-club_length)

        # Draw club shaft
        shaft_points = np.array([mid_hands, club_face])
        self.ax.plot(
            shaft_points[:, 0],
            shaft_points[:, 1],
            shaft_points[:, 2],
            "k-",
            linewidth=4,
            label="Shaft",
        )

        # FIXED: Scale clubhead size relative to club length
        head_size = club_length * 0.08  # Proportional to club length
        head_depth = club_length * 0.04  # Proportional to club length

        # Create a more realistic clubhead shape (driver-like)
        head_vertices = np.array(
            [
                # Face of the club (front)
                club_face + x_axis * head_size + y_axis * head_size / 2,
                club_face - x_axis * head_size + y_axis * head_size / 2,
                club_face - x_axis * head_size - y_axis * head_size / 2,
                club_face + x_axis * head_size - y_axis * head_size / 2,
                # Back of the club
                club_face
                + z_axis * head_depth
                + x_axis * head_size
                + y_axis * head_size / 2,
                club_face
                + z_axis * head_depth
                - x_axis * head_size
                + y_axis * head_size / 2,
                club_face
                + z_axis * head_depth
                - x_axis * head_size
                - y_axis * head_size / 2,
                club_face
                + z_axis * head_depth
                + x_axis * head_size
                - y_axis * head_size / 2,
            ]
        )

        # Draw clubhead outline (face)
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

        # Draw clubhead outline (back)
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

        # Connect face to back
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
        grip_end = mid_hands + z_axis * (club_length * 0.15)  # Proportional grip length
        grip_points = np.array([mid_hands, grip_end])
        self.ax.plot(
            grip_points[:, 0],
            grip_points[:, 1],
            grip_points[:, 2],
            "brown",
            linewidth=6,
            label="Grip",
        )

        # FIXED: Proper trajectory plotting with scaling
        if self.trajectory_var.get() and len(filtered_data) > 1:
            trajectory = np.array(
                [[row["X"], row["Y"], row["Z"]] for _, row in filtered_data.iterrows()]
            )
            # Apply motion scaling if enabled
            motion_scale = self.scale_var.get()
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

        # Show force vectors if enabled
        if self.force_var.get() and kinematics:
            dynamics = self.calculate_dynamics(
                frame_data, kinematics, self.current_frame
            )

            force_scale = 0.01  # Scale for visualization
            force_end = eval_point + dynamics["force"] * force_scale
            self.ax.quiver(
                eval_point[0],
                eval_point[1],
                eval_point[2],
                dynamics["force"][0] * force_scale,
                dynamics["force"][1] * force_scale,
                dynamics["force"][2] * force_scale,
                color="red",
                arrow_length_ratio=0.1,
                linewidth=2,
                label="Force",
            )

            torque_scale = 0.05
            torque_end = eval_point + dynamics["torque"] * torque_scale
            self.ax.quiver(
                eval_point[0],
                eval_point[1],
                eval_point[2],
                dynamics["torque"][0] * torque_scale,
                dynamics["torque"][1] * torque_scale,
                dynamics["torque"][2] * torque_scale,
                color="blue",
                arrow_length_ratio=0.1,
                linewidth=2,
                label="Torque",
            )

        # Mark key frames
        if self.current_swing in self.key_frames:
            key_colors = {"A": "green", "T": "yellow", "I": "red", "F": "blue"}
            key_names = {"A": "Address", "T": "Top", "I": "Impact", "F": "Finish"}

            for key, frame_num in self.key_frames[self.current_swing].items():
                if abs(self.current_frame - frame_num) < 3:
                    marker_pos = mid_hands + np.array([0, 0, 0.1])
                    self.ax.scatter(
                        marker_pos[0],
                        marker_pos[1],
                        marker_pos[2],
                        c=key_colors.get(key, "white"),
                        s=100,
                        alpha=0.8,
                        label=f"{key_names.get(key, key)}",
                    )

        # Update info display
        self.update_info_display(frame_data, kinematics)

        self.canvas.draw()

    def update_info_display(self, frame_data, kinematics):
        """Update the information display"""
        self.info_text.delete(1.0, tk.END)

        info = f"Frame: {self.current_frame + 1}\n"
        info += f"Time: {frame_data['time']:.3f} s\n\n"

        info += "Position (m):\n"
        info += f"  X: {frame_data['X']:.4f}\n"
        info += f"  Y: {frame_data['Y']:.4f}\n"
        info += f"  Z: {frame_data['Z']:.4f}\n\n"

        if kinematics and self.current_frame < len(kinematics):
            vel = kinematics[self.current_frame]["velocity"]
            acc = kinematics[self.current_frame]["acceleration"]

            info += "Velocity (m/s):\n"
            info += f"  X: {vel[0]:.3f}\n"
            info += f"  Y: {vel[1]:.3f}\n"
            info += f"  Z: {vel[2]:.3f}\n\n"

            info += "Acceleration (m/s²):\n"
            info += f"  X: {acc[0]:.2f}\n"
            info += f"  Y: {acc[1]:.2f}\n"
            info += f"  Z: {acc[2]:.2f}\n\n"

            dynamics = self.calculate_dynamics(
                frame_data, kinematics, self.current_frame
            )

            info += "Force (N):\n"
            info += f"  X: {dynamics['force'][0]:.2f}\n"
            info += f"  Y: {dynamics['force'][1]:.2f}\n"
            info += f"  Z: {dynamics['force'][2]:.2f}\n\n"

            info += "Torque (Nm):\n"
            info += f"  X: {dynamics['torque'][0]:.3f}\n"
            info += f"  Y: {dynamics['torque'][1]:.3f}\n"
            info += f"  Z: {dynamics['torque'][2]:.3f}\n"

        self.info_text.insert(1.0, info)

    def update_frame_info(self):
        """Update frame information display"""
        if self.current_swing in self.swing_data:
            max_frame = len(self.swing_data[self.current_swing]) - 1
            self.frame_info.config(text=f"Frame: {self.current_frame} / {max_frame}")

    # Event handlers
    def on_swing_change(self, event=None):
        """Handle swing selection change"""
        self.current_swing = self.swing_var.get()
        self.current_frame = 0
        self.frame_var.set(0)

        if self.current_swing in self.swing_data:
            max_frame = len(self.swing_data[self.current_swing]) - 1
            self.frame_scale.configure(to=max_frame)

        self.update_visualization()
        self.update_frame_info()

    def on_frame_change(self, value):
        """Handle frame slider change"""
        self.current_frame = int(float(value))
        self.update_visualization()
        self.update_frame_info()

    def on_filter_change(self, event=None):
        """Handle filter selection change"""
        self.current_filter = self.filter_var.get()
        self.update_visualization()

    def on_offset_change(self, value):
        """Handle evaluation point offset change"""
        self.eval_offset = float(value)
        self.update_visualization()

    def on_scale_change(self, value):
        """Handle motion scale factor change"""
        self.scale_var.set(float(value))
        self.update_visualization()

    def on_club_length_change(self, value):
        """Handle club length override change"""
        self.club_length_var.set(float(value))
        self.update_visualization()

    def update_display_options(self):
        """Handle display option changes"""
        self.show_trajectory = self.trajectory_var.get()
        self.show_force_vectors = self.force_var.get()
        self.update_visualization()

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
        self.ax.set_xlim([-2.0, 2.0])  # Target line range
        self.ax.set_ylim([-1.0, 3.0])  # Ball direction range
        self.ax.set_zlim([-0.5, 2.5])  # Vertical range

        self.canvas.draw()

    def on_mouse_press(self, event):
        """Handle mouse button press for rotation/panning"""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left mouse button - rotation
            self.is_rotating = True
        elif event.button == 3:  # Right mouse button - panning
            self.is_panning = True

        self.last_mouse_pos = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        """Handle mouse button release"""
        self.is_rotating = False
        self.is_panning = False
        self.last_mouse_pos = None

    def on_mouse_move(self, event):
        """Handle mouse movement for rotation/panning"""
        if event.inaxes != self.ax or self.last_mouse_pos is None:
            return

        if not (self.is_rotating or self.is_panning):
            return

        dx = event.xdata - self.last_mouse_pos[0]
        dy = event.ydata - self.last_mouse_pos[1]

        if self.is_rotating:
            # Rotate the view
            self.ax.azim += dx * 2
            self.ax.elev += dy * 2
            self.canvas.draw()
        elif self.is_panning:
            # Pan the view (adjust limits)
            x_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
            y_range = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
            z_range = self.ax.get_zlim()[1] - self.ax.get_zlim()[0]

            pan_factor = 0.1
            self.ax.set_xlim(self.ax.get_xlim() - dx * x_range * pan_factor)
            self.ax.set_ylim(self.ax.get_ylim() - dy * y_range * pan_factor)
            self.canvas.draw()

        self.last_mouse_pos = (event.xdata, event.ydata)

    def on_scroll(self, event):
        """Handle mouse scroll for zooming"""
        if event.inaxes != self.ax:
            return

        # Zoom factor
        zoom_factor = 1.1 if event.button == "up" else 0.9

        # Get current limits
        x_lim = self.ax.get_xlim()
        y_lim = self.ax.get_ylim()
        z_lim = self.ax.get_zlim()

        # Calculate new limits
        x_center = (x_lim[0] + x_lim[1]) / 2
        y_center = (y_lim[0] + y_lim[1]) / 2
        z_center = (z_lim[0] + z_lim[1]) / 2

        x_range = (x_lim[1] - x_lim[0]) / 2
        y_range = (y_lim[1] - y_lim[0]) / 2
        z_range = (z_lim[1] - z_lim[0]) / 2

        # Apply zoom
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

    def toggle_playback(self):
        """Toggle play/pause"""
        self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_button.config(text="Pause")
            self.animate()
        else:
            self.play_button.config(text="Play")

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

            self.frame_var.set(self.current_frame)
            self.update_visualization()
            self.update_frame_info()

            # Schedule next frame
            delay = int(50 / self.speed_var.get())  # Base 20fps, adjusted by speed
            self.root.after(delay, self.animate)
        else:
            self.is_playing = False
            self.play_button.config(text="Play")


def main():
    root = tk.Tk()
    app = GolfSwingAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
