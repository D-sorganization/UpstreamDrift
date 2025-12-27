"""
Interactive driven double pendulum GUI with 3D visualization and advanced features.

Features:
- 3D rotation and zoom
- Gravity toggle
- Inclined plane constraint toggle
- Out-of-plane angle for 3D motion
- Immediate position updates on parameter changes
- Data output with configurable granularity
- Organized input categories
- Visual angle reference indicators
"""

from __future__ import annotations

import contextlib
import csv
import math
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from double_pendulum_model.physics.double_pendulum import (
    DEFAULT_PLANE_INCLINATION_DEG,
    DoublePendulumDynamics,
    DoublePendulumParameters,
    DoublePendulumState,
    compile_forcing_functions,
)

TIME_STEP = 0.01


@dataclass
class UserInputs:
    shoulder_angle_deg: float
    wrist_angle_deg: float
    out_of_plane_angle_deg: float
    shoulder_expression: str
    wrist_expression: str
    upper_length_m: float
    upper_mass_kg: float
    upper_com_ratio: float
    lower_length_m: float
    shaft_mass_kg: float
    clubhead_mass_kg: float
    shaft_com_ratio: float
    plane_inclination_deg: float
    damping_shoulder: float
    damping_wrist: float
    gravity_enabled: bool
    constrained_to_plane: bool


class DoublePendulumApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Driven Double Pendulum — 3D Control Affine Model")
        self.root.geometry("1400x900")

        # State variables
        self.state: DoublePendulumState | None = None
        self.dynamics: DoublePendulumDynamics | None = None
        self.time = 0.0
        self.running = False

        # Data logging
        self.data_logging_enabled = False
        self.data_granularity = 1  # Log every N steps
        self.data_step_counter = 0
        self.data_file: csv.Writer | None = None
        self.data_file_handle: object | None = None
        self.data_file_stack: contextlib.ExitStack | None = None

        # Build UI
        self._build_ui()

        # Initialize with default state after UI is built
        # Use after() to ensure UI is fully rendered
        self.root.after(100, self._update_pendulum_immediately)

    def _build_ui(self) -> None:
        # Create main frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create 3D visualization
        self.fig = Figure(figsize=(9, 9), dpi=100, facecolor="white")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X (m)", fontsize=10)
        self.ax.set_ylabel("Y (m)", fontsize=10)
        self.ax.set_zlabel("Z (m)", fontsize=10)
        self.ax.set_title("Double Pendulum 3D View", fontsize=12, fontweight="bold")

        # Set initial view limits
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-1, 1])

        # Enable interactive rotation and zoom
        self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
        self.canvas.get_tk_widget().pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5)
        )

        # Draw initial empty plot
        self.canvas.draw()

        # Create control panel with better styling
        panel_frame = tk.Frame(main_frame, bg="#f0f0f0", width=350)
        panel_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        panel_frame.pack_propagate(False)

        # Scrollable frame for controls
        canvas_scroll = tk.Canvas(panel_frame, bg="white", highlightthickness=0)
        scrollbar = tk.Scrollbar(
            panel_frame, orient="vertical", command=canvas_scroll.yview
        )
        scrollable_frame = tk.Frame(canvas_scroll, bg="white")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")),
        )

        canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        self.entries = {}
        row = 0

        # Title
        title_label = tk.Label(
            scrollable_frame,
            text="Double Pendulum Controls",
            font=("Arial", 14, "bold"),
            bg="white",
        )
        title_label.grid(row=row, column=0, columnspan=2, pady=(10, 15), sticky="ew")
        row += 1

        # === INITIAL CONDITIONS ===
        self._create_section_header(scrollable_frame, "Initial Conditions", row)
        row += 1

        def labeled_row(
            label: str, default: str, row: int, tooltip: str = ""
        ) -> tk.Entry:
            frame = tk.Frame(scrollable_frame, bg="white")
            frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)

            label_widget = tk.Label(frame, text=label, bg="white", width=20, anchor="w")
            label_widget.pack(side=tk.LEFT, padx=(0, 5))

            entry = tk.Entry(frame, width=12)
            entry.insert(0, default)
            entry.pack(side=tk.LEFT)
            entry.bind("<KeyRelease>", lambda e: self._on_parameter_change())

            if tooltip:
                self._create_tooltip(label_widget, tooltip)

            self.entries[label] = entry
            return entry

        labeled_row(
            "Shoulder angle (deg)",
            "-45",
            row,
            "Angle of upper segment from vertical (0° = straight down)",
        )
        row += 1
        labeled_row(
            "Wrist angle (deg)",
            "-90",
            row,
            "Relative angle of lower segment from upper segment (0° = aligned)",
        )
        row += 1
        labeled_row(
            "Out-of-plane angle (deg)",
            "0",
            row,
            "Angle above/below plane (only used when not constrained)",
        )
        row += 1

        # === PHYSICAL PARAMETERS ===
        self._create_section_header(scrollable_frame, "Physical Parameters", row)
        row += 1

        labeled_row("Upper length (m)", "0.75", row)
        row += 1
        labeled_row("Upper mass (kg)", "7.5", row)
        row += 1
        labeled_row(
            "Upper COM ratio",
            "0.45",
            row,
            "Center of mass position as fraction of length",
        )
        row += 1
        labeled_row("Lower length (m)", "1.0", row)
        row += 1
        labeled_row("Shaft mass (kg)", "0.35", row)
        row += 1
        labeled_row("Clubhead mass (kg)", "0.20", row)
        row += 1
        labeled_row("Shaft COM ratio", "0.43", row)
        row += 1
        labeled_row(
            "Plane incline (deg)",
            str(DEFAULT_PLANE_INCLINATION_DEG),
            row,
            "Inclination of swing plane from vertical",
        )
        row += 1

        # === DAMPING PARAMETERS ===
        self._create_section_header(scrollable_frame, "Damping Parameters", row)
        row += 1

        labeled_row(
            "Shoulder damping",
            "0.4",
            row,
            "Damping coefficient for upper segment (N·m·s/rad)",
        )
        row += 1
        labeled_row(
            "Wrist damping",
            "0.25",
            row,
            "Damping coefficient for lower segment (N·m·s/rad)",
        )
        row += 1

        # === CONTROL INPUTS ===
        self._create_section_header(scrollable_frame, "Control Inputs", row)
        row += 1

        labeled_row(
            "Shoulder torque f(t)",
            "0.0",
            row,
            "Torque expression using t, theta1, theta2, omega1, omega2",
        )
        row += 1
        labeled_row("Wrist torque f(t)", "0.0", row)
        row += 1

        # === SIMULATION OPTIONS ===
        self._create_section_header(scrollable_frame, "Simulation Options", row)
        row += 1

        # Gravity toggle
        gravity_frame = tk.Frame(scrollable_frame, bg="white")
        gravity_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=5)
        self.gravity_var = tk.BooleanVar(value=True)
        gravity_check = tk.Checkbutton(
            gravity_frame,
            text="Gravity Enabled",
            variable=self.gravity_var,
            command=self._on_parameter_change,
            bg="white",
            font=("Arial", 10),
        )
        gravity_check.pack(side=tk.LEFT)
        self._create_tooltip(gravity_check, "Enable/disable gravitational force")
        row += 1

        # Constraint toggle
        constraint_frame = tk.Frame(scrollable_frame, bg="white")
        constraint_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=5)
        self.constraint_var = tk.BooleanVar(value=True)
        constraint_check = tk.Checkbutton(
            constraint_frame,
            text="Constrained to Plane",
            variable=self.constraint_var,
            command=self._on_parameter_change,
            bg="white",
            font=("Arial", 10),
        )
        constraint_check.pack(side=tk.LEFT)
        self._create_tooltip(constraint_check, "Constrain motion to inclined plane")
        row += 1

        # Control buttons
        button_frame = tk.Frame(scrollable_frame, bg="white")
        button_frame.grid(row=row, column=0, columnspan=2, pady=15)

        start_btn = tk.Button(
            button_frame,
            text="Start",
            command=self.start,
            width=8,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
        )
        start_btn.pack(side=tk.LEFT, padx=2)

        pause_btn = tk.Button(
            button_frame,
            text="Pause",
            command=self.pause,
            width=8,
            bg="#FF9800",
            fg="white",
            font=("Arial", 10, "bold"),
        )
        pause_btn.pack(side=tk.LEFT, padx=2)

        reset_btn = tk.Button(
            button_frame,
            text="Reset",
            command=self.reset,
            width=8,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10, "bold"),
        )
        reset_btn.pack(side=tk.LEFT, padx=2)
        row += 1

        # === DATA LOGGING ===
        self._create_section_header(scrollable_frame, "Data Logging", row)
        row += 1

        data_frame = tk.Frame(scrollable_frame, bg="white")
        data_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=5)

        self.data_logging_var = tk.BooleanVar(value=False)
        data_check = tk.Checkbutton(
            data_frame,
            text="Enable Data Output",
            variable=self.data_logging_var,
            command=self._on_data_logging_change,
            bg="white",
            font=("Arial", 10),
        )
        data_check.pack(side=tk.LEFT)
        row += 1

        granularity_frame = tk.Frame(scrollable_frame, bg="white")
        granularity_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        tk.Label(
            granularity_frame, text="Granularity (every N steps):", bg="white"
        ).pack(side=tk.LEFT, padx=(20, 5))
        self.granularity_var = tk.StringVar(value="1")
        granularity_entry = tk.Entry(
            granularity_frame, textvariable=self.granularity_var, width=8
        )
        granularity_entry.pack(side=tk.LEFT)
        granularity_entry.bind("<KeyRelease>", lambda e: self._on_granularity_change())
        row += 1

        # === STATUS DISPLAY ===
        self._create_section_header(scrollable_frame, "Status", row)
        row += 1

        self.torque_label = tk.Label(
            scrollable_frame,
            text="Torques: --\nTime: 0.00s",
            justify=tk.LEFT,
            wraplength=300,
            bg="white",
            font=("Courier", 9),
            anchor="w",
            padx=5,
            pady=5,
            relief=tk.SUNKEN,
            borderwidth=1,
        )
        self.torque_label.grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)

        # Configure column weights
        scrollable_frame.columnconfigure(0, weight=1)
        scrollable_frame.columnconfigure(1, weight=1)

        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_section_header(self, parent, text: str, row: int) -> None:
        """Create a styled section header."""
        header_frame = tk.Frame(parent, bg="#e0e0e0", height=30)
        header_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 5))
        header_frame.grid_propagate(False)

        label = tk.Label(
            header_frame,
            text=text,
            font=("Arial", 11, "bold"),
            bg="#e0e0e0",
            fg="#333333",
        )
        label.pack(side=tk.LEFT, padx=10, pady=5)

    def _create_tooltip(self, widget, text: str) -> None:
        """Create a simple tooltip."""

        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(
                tooltip,
                text=text,
                bg="#ffffe0",
                relief=tk.SOLID,
                borderwidth=1,
                font=("Arial", 8),
                wraplength=200,
            )
            label.pack()
            widget.tooltip = tooltip

        def on_leave(event):
            if hasattr(widget, "tooltip"):
                widget.tooltip.destroy()
                del widget.tooltip

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _read_inputs(self) -> UserInputs:
        def get_float(label: str) -> float:
            try:
                if label not in self.entries:
                    print(f"Warning: Entry '{label}' not found in entries dict")
                    return 0.0
                value = self.entries[label].get()
                if not value:
                    return 0.0
                return float(value)
            except ValueError as e:
                print(f"Error converting '{label}' to float: {e}")
                return 0.0
            except Exception as e:
                print(f"Unexpected error reading '{label}': {e}")
                return 0.0

        def get_str(label: str) -> str:
            try:
                if label not in self.entries:
                    return "0.0"
                return self.entries[label].get() or "0.0"
            except Exception as e:
                print(f"Error reading '{label}': {e}")
                return "0.0"

        return UserInputs(
            shoulder_angle_deg=get_float("Shoulder angle (deg)"),
            wrist_angle_deg=get_float("Wrist angle (deg)"),
            out_of_plane_angle_deg=get_float("Out-of-plane angle (deg)"),
            shoulder_expression=get_str("Shoulder torque f(t)"),
            wrist_expression=get_str("Wrist torque f(t)"),
            upper_length_m=get_float("Upper length (m)"),
            upper_mass_kg=get_float("Upper mass (kg)"),
            upper_com_ratio=get_float("Upper COM ratio"),
            lower_length_m=get_float("Lower length (m)"),
            shaft_mass_kg=get_float("Shaft mass (kg)"),
            clubhead_mass_kg=get_float("Clubhead mass (kg)"),
            shaft_com_ratio=get_float("Shaft COM ratio"),
            plane_inclination_deg=get_float("Plane incline (deg)"),
            damping_shoulder=get_float("Shoulder damping"),
            damping_wrist=get_float("Wrist damping"),
            gravity_enabled=self.gravity_var.get(),
            constrained_to_plane=self.constraint_var.get(),
        )

    def _on_parameter_change(self) -> None:
        """Called when any parameter changes - updates pendulum immediately."""
        self._update_pendulum_immediately()

    def _on_data_logging_change(self) -> None:
        """Handle data logging checkbox change."""
        self.data_logging_enabled = self.data_logging_var.get()
        if self.data_logging_enabled:
            self._start_data_logging()
        else:
            self._stop_data_logging()

    def _on_granularity_change(self) -> None:
        """Handle granularity change."""
        try:
            self.data_granularity = max(1, int(self.granularity_var.get()))
        except ValueError:
            self.data_granularity = 1

    def _start_data_logging(self) -> None:
        """Start logging data to CSV file."""
        if self.data_file_handle is not None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pendulum_data_{timestamp}.csv"
        self.data_file_stack = contextlib.ExitStack()
        self.data_file_handle = self.data_file_stack.enter_context(
            open(filename, "w", newline="")  # noqa: SIM115
        )
        self.data_file = csv.writer(self.data_file_handle)
        self.data_file.writerow(
            [
                "time",
                "theta1",
                "theta2",
                "phi",
                "omega1",
                "omega2",
                "omega_phi",
                "tau1",
                "tau2",
                "grav1",
                "grav2",
                "damp1",
                "damp2",
                "coriolis1",
                "coriolis2",
            ]
        )
        self.data_step_counter = 0

    def _stop_data_logging(self) -> None:
        """Stop logging data and close file."""
        if self.data_file_stack is not None:
            self.data_file_stack.close()
            self.data_file_stack = None
            self.data_file_handle = None
            self.data_file = None

    def _log_data(self) -> None:
        """Log current state to file if logging is enabled."""
        if not self.data_logging_enabled or self.data_file is None:
            return

        self.data_step_counter += 1
        if self.data_step_counter >= self.data_granularity:
            self.data_step_counter = 0

            if self.state is not None and self.dynamics is not None:
                torques = self.dynamics.applied_torques(self.time, self.state)
                breakdown = self.dynamics.joint_torque_breakdown(self.state, torques)

                self.data_file.writerow(
                    [
                        f"{self.time:.6f}",
                        f"{self.state.theta1:.6f}",
                        f"{self.state.theta2:.6f}",
                        f"{self.state.phi:.6f}",
                        f"{self.state.omega1:.6f}",
                        f"{self.state.omega2:.6f}",
                        f"{self.state.omega_phi:.6f}",
                        f"{torques[0]:.6f}",
                        f"{torques[1]:.6f}",
                        f"{breakdown.gravitational[0]:.6f}",
                        f"{breakdown.gravitational[1]:.6f}",
                        f"{breakdown.damping[0]:.6f}",
                        f"{breakdown.damping[1]:.6f}",
                        f"{breakdown.coriolis_centripetal[0]:.6f}",
                        f"{breakdown.coriolis_centripetal[1]:.6f}",
                    ]
                )
                self.data_file_handle.flush()

    def _update_pendulum_immediately(self) -> None:
        """Update pendulum position immediately when parameters change."""
        try:
            user_inputs = self._read_inputs()
            # Calculate upper segment inertia about COM
            # For a uniform rod: I = (1/12) * m * L^2 (COM at L/2)
            # When COM ratio != 0.5, we use an approximation that scales with
            # the COM position. This assumes a mass distribution consistent
            # with the specified COM ratio
            com_ratio = user_inputs.upper_com_ratio
            if abs(com_ratio - 0.5) < 0.01:
                # Close to uniform rod - use standard formula
                upper_inertia = (
                    (1.0 / 12.0)
                    * user_inputs.upper_mass_kg
                    * user_inputs.upper_length_m**2
                )
            else:
                # For non-uniform distribution, use an approximation
                # Scale the uniform rod inertia based on how far COM is from center
                # This is an approximation - exact value depends on actual
                # mass distribution
                uniform_inertia = (
                    (1.0 / 12.0)
                    * user_inputs.upper_mass_kg
                    * user_inputs.upper_length_m**2
                )
                # Adjust based on COM position (empirical scaling factor)
                com_offset_factor = 1.0 + 3.0 * (com_ratio - 0.5) ** 2
                upper_inertia = uniform_inertia * com_offset_factor

            parameters = DoublePendulumParameters(
                upper_segment=DoublePendulumParameters.default().upper_segment.__class__(
                    length_m=user_inputs.upper_length_m,
                    mass_kg=user_inputs.upper_mass_kg,
                    center_of_mass_ratio=user_inputs.upper_com_ratio,
                    inertia_about_com=upper_inertia,
                ),
                lower_segment=DoublePendulumParameters.default().lower_segment.__class__(
                    length_m=user_inputs.lower_length_m,
                    shaft_mass_kg=user_inputs.shaft_mass_kg,
                    clubhead_mass_kg=user_inputs.clubhead_mass_kg,
                    shaft_com_ratio=user_inputs.shaft_com_ratio,
                ),
                plane_inclination_deg=user_inputs.plane_inclination_deg,
                damping_shoulder=user_inputs.damping_shoulder,
                damping_wrist=user_inputs.damping_wrist,
                gravity_enabled=user_inputs.gravity_enabled,
                constrained_to_plane=user_inputs.constrained_to_plane,
            )

            forcing = compile_forcing_functions(
                user_inputs.shoulder_expression, user_inputs.wrist_expression
            )
            self.dynamics = DoublePendulumDynamics(
                parameters=parameters, forcing_functions=forcing
            )

            # Update state with new parameters
            # If simulation is running and angles change, pause to avoid physically
            # inconsistent states. Changing angles while preserving velocities would
            # create discontinuities. Non-angle parameters can change without resetting
            # motion.
            was_running = self.running
            angles_changed = False

            if was_running and self.state is not None:
                # Check if angles actually changed
                old_theta1 = math.degrees(self.state.theta1)
                old_theta2 = math.degrees(self.state.theta2)
                old_phi = (
                    math.degrees(self.state.phi) if hasattr(self.state, "phi") else 0.0
                )

                angles_changed = (
                    abs(old_theta1 - user_inputs.shoulder_angle_deg) > 0.1
                    or abs(old_theta2 - user_inputs.wrist_angle_deg) > 0.1
                    or abs(old_phi - user_inputs.out_of_plane_angle_deg) > 0.1
                )

                if angles_changed:
                    # Pause simulation when angles change to maintain
                    # physical consistency
                    self.running = False

            # Only reset velocities if angles changed or if this is initial setup
            if angles_changed or self.state is None or not was_running:
                # Reset state when angles change (or on initial setup)
                self.state = DoublePendulumState(
                    theta1=math.radians(user_inputs.shoulder_angle_deg),
                    theta2=math.radians(user_inputs.wrist_angle_deg),
                    omega1=0.0,
                    omega2=0.0,
                    phi=math.radians(user_inputs.out_of_plane_angle_deg),
                    omega_phi=0.0,
                )
            else:
                # Preserve state when only non-angle parameters changed
                # Just update the angles if they changed slightly (within tolerance)
                self.state = DoublePendulumState(
                    theta1=math.radians(user_inputs.shoulder_angle_deg),
                    theta2=math.radians(user_inputs.wrist_angle_deg),
                    omega1=self.state.omega1,
                    omega2=self.state.omega2,
                    phi=math.radians(user_inputs.out_of_plane_angle_deg),
                    omega_phi=(
                        self.state.omega_phi
                        if hasattr(self.state, "omega_phi")
                        else 0.0
                    ),
                )

            self._draw_pendulum_3d()
        except Exception as error:  # noqa: BLE001
            # Still try to draw something even if there's an error
            if self.state is None or self.dynamics is None:
                # Initialize with defaults if not set
                self.dynamics = DoublePendulumDynamics()
                self.state = DoublePendulumState(
                    theta1=math.radians(-45),
                    theta2=math.radians(-90),
                    omega1=0.0,
                    omega2=0.0,
                    phi=0.0,
                    omega_phi=0.0,
                )
                self._draw_pendulum_3d()
            raise RuntimeError(f"Error updating pendulum: {error}") from error

    def start(self) -> None:
        """Start or resume simulation."""
        self._update_pendulum_immediately()
        self.running = True
        self._update()

    def pause(self) -> None:
        """Pause simulation."""
        self.running = False

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.running = False
        self.time = 0.0
        user_inputs = self._read_inputs()
        self.state = DoublePendulumState(
            theta1=math.radians(user_inputs.shoulder_angle_deg),
            theta2=math.radians(user_inputs.wrist_angle_deg),
            omega1=0.0,
            omega2=0.0,
            phi=math.radians(user_inputs.out_of_plane_angle_deg),
            omega_phi=0.0,
        )
        self._update_pendulum_immediately()

    def _update(self) -> None:
        """Update simulation step."""
        if not self.running or self.state is None or self.dynamics is None:
            return

        # For now, phi doesn't affect dynamics (2D model), but we preserve it
        # In a full 3D model, phi would have its own dynamics
        self.state = self.dynamics.step(self.time, self.state, TIME_STEP)
        self.time += TIME_STEP

        self._draw_pendulum_3d()
        self._log_data()

        torques = self.dynamics.applied_torques(self.time, self.state)
        breakdown = self.dynamics.joint_torque_breakdown(self.state, torques)
        self.torque_label.config(
            text=(
                f"Applied (Nm): shoulder={torques[0]:.2f}, wrist={torques[1]:.2f}\n"
                f"Gravity: {breakdown.gravitational[0]:.2f}, "
                f"{breakdown.gravitational[1]:.2f}\n"
                f"Time: {self.time:.2f}s"
            )
        )
        self.root.after(int(TIME_STEP * 1000), self._update)

    def _draw_pendulum_3d(self) -> None:
        """Draw pendulum in 3D space with correct pivot point and angle references."""
        if self.state is None or self.dynamics is None:
            print(f"DEBUG: state={self.state}, dynamics={self.dynamics}")
            return

        try:
            self.ax.clear()
        except Exception as e:
            print(f"Error clearing axes: {e}")
            return

        upper = self.dynamics.parameters.upper_segment
        lower = self.dynamics.parameters.lower_segment

        # PIVOT POINT: Base of upper segment (hub/pivot)
        # This is the fixed point from which the pendulum swings
        pivot = np.array([0.0, 0.0, 0.0])

        # Calculate positions in the plane
        # Coordinate system: X=horizontal, Y=depth, Z=vertical (positive Z is up)
        # theta1: angle of upper segment from vertical
        # (0° = pointing straight down = negative Z)
        # theta2: relative angle of lower segment from upper segment
        theta1 = self.state.theta1
        theta2 = self.state.theta2

        # Upper segment: extends from pivot
        # In standard pendulum coordinates: theta=0 means vertical down (negative Z)
        # So we use: x = l * sin(theta), z = -l * cos(theta) where z is vertical
        upper_vec = np.array(
            [
                math.sin(theta1) * upper.length_m,  # Horizontal component (X)
                0.0,  # Depth component (Y, initially zero)
                -math.cos(theta1)
                * upper.length_m,  # Vertical component (Z, negative = down)
            ]
        )
        elbow = pivot + upper_vec

        # Lower segment: extends from elbow
        # Absolute angle is theta1 + theta2
        lower_abs_angle = theta1 + theta2
        lower_vec = np.array(
            [
                math.sin(lower_abs_angle) * lower.length_m,  # Horizontal (X)
                0.0,  # Depth (Y)
                -math.cos(lower_abs_angle)
                * lower.length_m,  # Vertical (Z, negative = down)
            ]
        )
        wrist = elbow + lower_vec

        # Apply out-of-plane rotation if not constrained
        # This rotates around the vertical (Z) axis to create motion
        # in the Y (depth) direction
        if not self.dynamics.parameters.constrained_to_plane:
            phi = self.state.phi if hasattr(self.state, "phi") else 0.0
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)

            # Rotate around vertical (Z) axis to move in Y (depth) direction
            def rotate_out_of_plane(point):
                x, y, z = point[0], point[1], point[2]
                # Rotate around vertical (Z) axis
                new_x = x * cos_phi - y * sin_phi
                new_y = x * sin_phi + y * cos_phi
                return np.array([new_x, new_y, z])

            elbow = rotate_out_of_plane(elbow)
            wrist = rotate_out_of_plane(wrist)

        # Apply plane rotation if constrained
        if self.dynamics.parameters.constrained_to_plane:
            plane_angle = self.dynamics.parameters.plane_inclination_rad
            cos_plane = math.cos(plane_angle)
            sin_plane = math.sin(plane_angle)

            def rotate_plane(point):
                y, z = point[1], point[2]
                new_y = y * cos_plane - z * sin_plane
                new_z = y * sin_plane + z * cos_plane
                return np.array([point[0], new_y, new_z])

            pivot = rotate_plane(pivot)
            elbow = rotate_plane(elbow)
            wrist = rotate_plane(wrist)

        # Draw reference axes and angle indicators
        max_range = (upper.length_m + lower.length_m) * 1.3

        # Draw vertical reference line (shows theta1=0 reference in plane)
        # Z is vertical, so theta1=0 means pointing in negative Z direction
        self.ax.plot(
            [pivot[0], pivot[0]],
            [pivot[1], pivot[1]],
            [pivot[2], pivot[2] - max_range * 0.3],
            "k--",
            linewidth=1,
            alpha=0.3,
            label="Plane Vertical (θ₁=0°)",
        )

        # Draw horizontal reference line (in X direction)
        self.ax.plot(
            [pivot[0] - max_range * 0.3, pivot[0] + max_range * 0.3],
            [pivot[1], pivot[1]],
            [pivot[2], pivot[2]],
            "k--",
            linewidth=1,
            alpha=0.3,
            label="Plane Horizontal",
        )

        # Draw angle arc for theta1 (in X-Z plane)
        arc_points = 20
        arc_theta = np.linspace(0, theta1, arc_points)
        arc_radius = max_range * 0.2
        arc_x = [pivot[0] + arc_radius * math.sin(t) for t in arc_theta]
        arc_y = [pivot[1]] * arc_points  # Y stays constant
        arc_z = [
            pivot[2] - arc_radius * math.cos(t) for t in arc_theta
        ]  # Z is vertical
        if len(arc_x) > 1:
            self.ax.plot(arc_x, arc_y, arc_z, "b-", linewidth=2, alpha=0.5)

        # Draw gravity vector - ALWAYS points straight down in WORLD coordinates
        # Gravity is independent of plane rotation - it always points in
        # negative Z direction. This is the true direction of gravity in the
        # real world (Z is vertical, positive Z is up)
        gravity_length = max_range * 0.35

        # Position gravity arrow at a visible location
        gravity_start = pivot + np.array(
            [max_range * 0.6, max_range * 0.2, max_range * 0.3]
        )

        # Gravity vector ALWAYS points straight down in world coordinates (negative Z)
        # This is independent of any plane rotation
        gravity_vec_world = np.array([0, 0, -gravity_length])

        # Draw gravity arrow with quiver - this will show the true gravity direction
        self.ax.quiver(
            gravity_start[0],
            gravity_start[1],
            gravity_start[2],
            gravity_vec_world[0],
            gravity_vec_world[1],
            gravity_vec_world[2],
            color="#00AA00",
            arrow_length_ratio=0.3,
            linewidth=5,
            label="Gravity (g) - Always Vertical Down",
            alpha=0.95,
        )

        # Add prominent gravity label with arrow symbol
        gravity_label_pos = gravity_start + gravity_vec_world * 0.5
        self.ax.text(
            gravity_label_pos[0] + max_range * 0.1,
            gravity_label_pos[1],
            gravity_label_pos[2],
            "g↓",
            fontsize=16,
            color="#00AA00",
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#E8F5E9",
                "alpha": 0.95,
                "edgecolor": "#00AA00",
                "linewidth": 3,
            },
        )

        # Draw a reference line showing true vertical (world vertical) from pivot
        # This helps visualize that gravity is always straight down,
        # even when plane is inclined
        world_vertical_start = pivot + np.array([-max_range * 0.5, -max_range * 0.5, 0])
        world_vertical_end = world_vertical_start + np.array([0, 0, -max_range * 0.3])
        self.ax.plot(
            [world_vertical_start[0], world_vertical_end[0]],
            [world_vertical_start[1], world_vertical_end[1]],
            [world_vertical_start[2], world_vertical_end[2]],
            "g--",
            linewidth=2.5,
            alpha=0.7,
            label="True Vertical (World Gravity Direction)",
        )

        # Draw pendulum segments with clear color coding
        # UPPER SEGMENT: Blue (shoulder to elbow)
        self.ax.plot(
            [pivot[0], elbow[0]],
            [pivot[1], elbow[1]],
            [pivot[2], elbow[2]],
            color="#2E86AB",
            linewidth=7,
            label="Upper Segment (Shoulder)",
            alpha=0.9,
            zorder=5,
        )

        # LOWER SEGMENT: Red (elbow to wrist/clubhead)
        self.ax.plot(
            [elbow[0], wrist[0]],
            [elbow[1], wrist[1]],
            [elbow[2], wrist[2]],
            color="#A23B72",
            linewidth=8,
            label="Lower Segment (Wrist)",
            alpha=0.9,
            zorder=5,
        )

        # Draw pivot point (hub) - make it prominent
        self.ax.scatter(
            *pivot,
            color="black",
            s=250,
            marker="o",
            label="Pivot (Hub)",
            edgecolors="white",
            linewidths=3,
            zorder=10,
        )

        # Draw elbow joint - blue to match upper segment
        self.ax.scatter(
            *elbow,
            color="#2E86AB",
            s=100,
            marker="o",
            edgecolors="white",
            linewidths=2,
            zorder=9,
        )

        # Draw wrist/end point (clubhead) - red to match lower segment
        self.ax.scatter(
            *wrist,
            color="#A23B72",
            s=180,
            marker="o",
            label="End Point (Clubhead)",
            edgecolors="white",
            linewidths=3,
            zorder=9,
        )

        # Add text labels for segments
        # Upper segment label (midpoint)
        upper_mid = (pivot + elbow) / 2
        self.ax.text(
            upper_mid[0],
            upper_mid[1],
            upper_mid[2],
            "UPPER",
            fontsize=9,
            color="#2E86AB",
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "#2E86AB",
            },
            ha="center",
        )

        # Lower segment label (midpoint)
        lower_mid = (elbow + wrist) / 2
        self.ax.text(
            lower_mid[0],
            lower_mid[1],
            lower_mid[2],
            "LOWER",
            fontsize=9,
            color="#A23B72",
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "#A23B72",
            },
            ha="center",
        )

        # Draw plane if constrained
        # The plane is in the X-Y plane, rotated around X axis by plane_angle
        if self.dynamics.parameters.constrained_to_plane:
            plane_size = (upper.length_m + lower.length_m) * 1.2
            x_plane = np.linspace(-plane_size, plane_size, 15)
            y_plane = np.linspace(-plane_size, plane_size, 15)
            x_plane_grid, y_plane_grid = np.meshgrid(x_plane, y_plane)

            plane_angle = self.dynamics.parameters.plane_inclination_rad
            # Rotate around X axis: Y becomes Y*cos - Z*sin, Z becomes Y*sin + Z*cos
            # For the plane surface, we start with Z=0, so:
            z_plane = y_plane_grid * math.sin(plane_angle)
            y_plane_rotated = y_plane_grid * math.cos(plane_angle)

            self.ax.plot_surface(
                x_plane_grid,
                y_plane_rotated,
                z_plane,
                alpha=0.15,
                color="gray",
                edgecolor="none",
            )

        # Set equal aspect ratio and labels
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([-max_range * 0.5, max_range * 0.5])
        self.ax.set_xlabel("X (m)", fontsize=10)
        self.ax.set_ylabel("Y (m)", fontsize=10)
        self.ax.set_zlabel("Z (m)", fontsize=10)
        self.ax.set_title(
            "Double Pendulum 3D View\nPivot at origin, θ₁=0° is vertical down",
            fontsize=11,
            fontweight="bold",
        )

        # Add legend
        self.ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

        self.canvas.draw()

    def __del__(self) -> None:
        """Cleanup on destruction."""
        self._stop_data_logging()


def run_app() -> None:
    root = tk.Tk()
    DoublePendulumApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_app()
