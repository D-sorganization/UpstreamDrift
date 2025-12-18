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
import logging
import math
import tkinter as tk
import typing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

from double_pendulum_model.physics.double_pendulum import (
    DEFAULT_ARM_CENTER_OF_MASS_RATIO,
    DEFAULT_ARM_LENGTH_M,
    DEFAULT_ARM_MASS_KG,
    DEFAULT_CLUBHEAD_MASS_KG,
    DEFAULT_DAMPING_SHOULDER,
    DEFAULT_DAMPING_WRIST,
    DEFAULT_PLANE_INCLINATION_DEG,
    DEFAULT_SHAFT_COM_RATIO,
    DEFAULT_SHAFT_LENGTH_M,
    DEFAULT_SHAFT_MASS_KG,
    DoublePendulumDynamics,
    DoublePendulumParameters,
    DoublePendulumState,
    compile_forcing_functions,
)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)

TIME_STEP = 0.01

# Tolerance for angle comparisons in the GUI [degrees] (UI precision only)
ANGLE_TOLERANCE_DEG = 0.1

# Tolerance for center-of-mass ratio comparisons [unitless, ratio]
# Used to determine when two COM positions are effectively equal.
COM_TOLERANCE = 0.01


@dataclass
class UIEntryConfig:
    label: str
    default: str
    tooltip: str = ""


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
        self.data_file: typing.Any | None = None
        self.data_file_handle: typing.TextIO | None = None
        self.data_file_stack: contextlib.ExitStack | None = None

        # Build UI
        self._build_ui()

        # Initialize with default state after UI is built
        # Use after() to ensure UI is fully rendered
        self.root.after(100, self._update_pendulum_immediately)

    def _build_ui(self) -> None:
        """Build the user interface."""
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._setup_visualization(main_frame)
        self._setup_controls(main_frame)

    def _setup_visualization(self, parent: tk.Widget) -> None:
        """Setup 3D visualization area."""
        self.fig = Figure(figsize=(9, 9), dpi=100, facecolor="white")
        self.ax: Axes3D = typing.cast(
            Axes3D, self.fig.add_subplot(111, projection="3d")
        )
        self.ax.set_xlabel("X (m)", fontsize=10)
        self.ax.set_ylabel("Y (m)", fontsize=10)
        self.ax.set_zlabel("Z (m)", fontsize=10)
        self.ax.set_title("Double Pendulum 3D View", fontsize=12, fontweight="bold")

        self.ax.set_xlim((-2, 2))
        self.ax.set_ylim((-2, 2))
        self.ax.set_zlim((-1, 1))

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5)
        )
        self.canvas.draw()

    def _setup_controls(self, parent: tk.Widget) -> None:
        """Setup control panel."""
        panel_frame = tk.Frame(parent, bg="#f0f0f0", width=350)
        panel_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        panel_frame.pack_propagate(False)  # noqa: FBT003

        canvas_scroll = tk.Canvas(panel_frame, bg="white", highlightthickness=0)
        scrollbar = tk.Scrollbar(
            panel_frame, orient="vertical", command=canvas_scroll.yview
        )
        scrollable_frame = tk.Frame(canvas_scroll, bg="white")

        scrollable_frame.bind(
            "<Configure>",
            lambda _e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")),
        )

        canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        self.entries: dict[str, tk.Entry] = {}
        row = 0

        title_label = tk.Label(
            scrollable_frame,
            text="Double Pendulum Controls",
            font=("Arial", 14, "bold"),
            bg="white",
        )
        title_label.grid(row=row, column=0, columnspan=2, pady=(10, 15), sticky="ew")
        row += 1

        row = self._setup_initial_conditions(scrollable_frame, row)
        row = self._setup_physical_parameters(scrollable_frame, row)
        row = self._setup_damping_parameters(scrollable_frame, row)
        row = self._setup_control_inputs(scrollable_frame, row)
        row = self._setup_simulation_options(scrollable_frame, row)
        row = self._setup_data_logging(scrollable_frame, row)
        self._setup_status(scrollable_frame, row)

        scrollable_frame.columnconfigure(0, weight=1)
        scrollable_frame.columnconfigure(1, weight=1)

        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _add_labeled_row(
        self, parent: tk.Widget, row: int, config: UIEntryConfig
    ) -> tk.Entry:
        """Add a labeled entry row to the control panel."""
        frame = tk.Frame(parent, bg="white")
        frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)

        label_widget = tk.Label(
            frame, text=config.label, bg="white", width=20, anchor="w"
        )
        label_widget.pack(side=tk.LEFT, padx=(0, 5))

        entry = tk.Entry(frame, width=12)
        entry.insert(0, config.default)
        entry.pack(side=tk.LEFT)
        entry.bind("<KeyRelease>", lambda _e: self._on_parameter_change())

        if config.tooltip:
            self._create_tooltip(label_widget, config.tooltip)

        self.entries[config.label] = entry
        return entry

    def _setup_initial_conditions(self, parent: tk.Widget, row: int) -> int:
        self._create_section_header(parent, "Initial Conditions", row)
        row += 1
        self._add_labeled_row(
            parent,
            row,
            UIEntryConfig(
                "Shoulder angle (deg)",
                "-45",
                "Angle of upper segment from vertical (0° = straight down)",
            ),
        )
        row += 1
        self._add_labeled_row(
            parent,
            row,
            UIEntryConfig(
                "Wrist angle (deg)",
                "-90",
                "Relative angle of lower segment from upper segment (0° = aligned)",
            ),
        )
        row += 1
        self._add_labeled_row(
            parent,
            row,
            UIEntryConfig(
                "Out-of-plane angle (deg)",
                "0",
                "Angle above/below plane (only used when not constrained)",
            ),
        )
        row += 1
        return row

    def _setup_physical_parameters(self, parent: tk.Widget, row: int) -> int:
        self._create_section_header(parent, "Physical Parameters", row)
        row += 1
        self._add_labeled_row(
            parent,
            row,
            UIEntryConfig("Upper length (m)", str(DEFAULT_ARM_LENGTH_M)),
        )
        row += 1
        self._add_labeled_row(
            parent, row, UIEntryConfig("Upper mass (kg)", str(DEFAULT_ARM_MASS_KG))
        )
        row += 1
        self._add_labeled_row(
            parent,
            row,
            UIEntryConfig(
                "Upper COM ratio",
                str(DEFAULT_ARM_CENTER_OF_MASS_RATIO),
                "Center of mass position as fraction of length",
            ),
        )
        row += 1
        self._add_labeled_row(
            parent,
            row,
            UIEntryConfig("Lower length (m)", str(DEFAULT_SHAFT_LENGTH_M)),
        )
        row += 1
        self._add_labeled_row(
            parent, row, UIEntryConfig("Shaft mass (kg)", str(DEFAULT_SHAFT_MASS_KG))
        )
        row += 1
        self._add_labeled_row(
            parent,
            row,
            UIEntryConfig("Clubhead mass (kg)", str(DEFAULT_CLUBHEAD_MASS_KG)),
        )
        row += 1
        self._add_labeled_row(
            parent, row, UIEntryConfig("Shaft COM ratio", str(DEFAULT_SHAFT_COM_RATIO))
        )
        row += 1
        self._add_labeled_row(
            parent,
            row,
            UIEntryConfig(
                "Plane incline (deg)",
                str(DEFAULT_PLANE_INCLINATION_DEG),
                "Inclination of swing plane from vertical",
            ),
        )
        row += 1
        return row

    def _setup_damping_parameters(self, parent: tk.Widget, row: int) -> int:
        self._create_section_header(parent, "Damping Parameters", row)
        row += 1
        self._add_labeled_row(
            parent,
            row,
            UIEntryConfig(
                "Shoulder damping",
                str(DEFAULT_DAMPING_SHOULDER),
                "Damping coefficient for upper segment (N·m·s/rad)",
            ),
        )
        row += 1
        self._add_labeled_row(
            parent,
            row,
            UIEntryConfig(
                "Wrist damping",
                str(DEFAULT_DAMPING_WRIST),
                "Damping coefficient for lower segment (N·m·s/rad)",
            ),
        )
        row += 1
        return row

    def _setup_control_inputs(self, parent: tk.Widget, row: int) -> int:
        self._create_section_header(parent, "Control Inputs", row)
        row += 1
        self._add_labeled_row(
            parent,
            row,
            UIEntryConfig(
                "Shoulder torque f(t)",
                "0.0",
                "Torque expression using t, theta1, theta2, omega1, omega2",
            ),
        )
        row += 1
        self._add_labeled_row(parent, row, UIEntryConfig("Wrist torque f(t)", "0.0"))
        row += 1
        return row

    def _setup_simulation_options(self, parent: tk.Widget, row: int) -> int:
        self._create_section_header(parent, "Simulation Options", row)
        row += 1

        gravity_frame = tk.Frame(parent, bg="white")
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

        constraint_frame = tk.Frame(parent, bg="white")
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

        button_frame = tk.Frame(parent, bg="white")
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
        return row

    def _setup_data_logging(self, parent: tk.Widget, row: int) -> int:
        self._create_section_header(parent, "Data Logging", row)
        row += 1

        data_frame = tk.Frame(parent, bg="white")
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

        granularity_frame = tk.Frame(parent, bg="white")
        granularity_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        tk.Label(
            granularity_frame, text="Granularity (every N steps):", bg="white"
        ).pack(side=tk.LEFT, padx=(20, 5))
        self.granularity_var = tk.StringVar(value="1")
        granularity_entry = tk.Entry(
            granularity_frame, textvariable=self.granularity_var, width=8
        )
        granularity_entry.pack(side=tk.LEFT)
        granularity_entry.bind("<KeyRelease>", lambda _e: self._on_granularity_change())
        row += 1
        return row

    def _setup_status(self, parent: tk.Widget, row: int) -> int:
        self._create_section_header(parent, "Status", row)
        row += 1

        self.torque_label = tk.Label(
            parent,
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
        return row

    def _create_section_header(self, parent: tk.Widget, text: str, row: int) -> None:
        """Create a styled section header."""
        header_frame = tk.Frame(parent, bg="#e0e0e0", height=30)
        header_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 5))
        header_frame.grid_propagate(False)  # noqa: FBT003

        label = tk.Label(
            header_frame,
            text=text,
            font=("Arial", 11, "bold"),
            bg="#e0e0e0",
            fg="#333333",
        )
        label.pack(side=tk.LEFT, padx=10, pady=5)

    def _create_tooltip(self, widget: tk.Widget, text: str) -> None:
        """Create a simple tooltip."""

        def on_enter(event: tk.Event) -> None:
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)  # noqa: FBT003
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
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
            widget.tooltip = tooltip  # type: ignore[attr-defined]

        def on_leave(_event: tk.Event) -> None:
            if hasattr(widget, "tooltip"):
                widget.tooltip.destroy()  # type: ignore[attr-defined]
                del widget.tooltip  # type: ignore[attr-defined]

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _read_inputs(self) -> UserInputs:
        def get_float(label: str) -> float:
            result = 0.0
            if label not in self.entries:
                logger.warning("Entry '%s' not found", label)
                return result

            val = self.entries[label].get()
            if val:
                try:
                    result = float(val)
                except ValueError:
                    logger.exception("Error converting '%s' to float", label)
                except Exception:
                    logger.exception("Unexpected error reading '%s'", label)
            return result

        def get_str(label: str) -> str:
            if label not in self.entries:
                return "0.0"
            return self.entries[label].get() or "0.0"

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

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
        filename = f"pendulum_data_{timestamp}.csv"
        self.data_file_stack = contextlib.ExitStack()
        self.data_file_handle = self.data_file_stack.enter_context(
            Path(filename).open("w", newline="")  # noqa: SIM115
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
                if self.data_file_handle:
                    self.data_file_handle.flush()

    def _calculate_upper_inertia(self, user_inputs: UserInputs) -> float:
        """Calculate inertia of upper segment."""
        com_ratio = user_inputs.upper_com_ratio
        if abs(com_ratio - 0.5) < COM_TOLERANCE:
            return (
                (1.0 / 12.0) * user_inputs.upper_mass_kg * user_inputs.upper_length_m**2
            )

        uniform_inertia = (
            (1.0 / 12.0) * user_inputs.upper_mass_kg * user_inputs.upper_length_m**2
        )
        com_offset_factor = 1.0 + 3.0 * (com_ratio - 0.5) ** 2
        return uniform_inertia * com_offset_factor

    def _create_parameters(
        self, user_inputs: UserInputs, upper_inertia: float
    ) -> DoublePendulumParameters:
        """Create DoublePendulumParameters from inputs."""
        return DoublePendulumParameters(
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

    def _update_pendulum_immediately(self) -> None:
        """Update pendulum position immediately when parameters change."""
        try:
            user_inputs = self._read_inputs()
            upper_inertia = self._calculate_upper_inertia(user_inputs)
            parameters = self._create_parameters(user_inputs, upper_inertia)

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
                    abs(old_theta1 - user_inputs.shoulder_angle_deg)
                    > ANGLE_TOLERANCE_DEG
                    or abs(old_theta2 - user_inputs.wrist_angle_deg)
                    > ANGLE_TOLERANCE_DEG
                    or abs(old_phi - user_inputs.out_of_plane_angle_deg)
                    > ANGLE_TOLERANCE_DEG
                )

                if angles_changed:
                    # Pause simulation when angles change to maintain consistency
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
        except (ValueError, TypeError, RuntimeError, ArithmeticError) as error:
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
            msg = f"Error updating pendulum: {error}"
            raise RuntimeError(msg) from error

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
        """Draw pendulum in 3D space using helper methods."""
        import numpy as np

        if self.state is None or self.dynamics is None:
            logger.debug("DEBUG: state=%s, dynamics=%s", self.state, self.dynamics)
            return

        try:
            self.ax.clear()
        except Exception:
            logger.exception("Error clearing axes")
            return

        # Prepare
        pivot = np.array([0.0, 0.0, 0.0])
        upper = self.dynamics.parameters.upper_segment
        lower = self.dynamics.parameters.lower_segment
        max_range = (upper.length_m + lower.length_m) * 1.3

        # Calculate Positions
        pivot, elbow, wrist = self._calculate_3d_positions(pivot)

        # Draw Elements
        self._draw_reference_lines(pivot, max_range, self.state.theta1)
        self._draw_segments(pivot, elbow, wrist)
        self._draw_plane(upper.length_m + lower.length_m)

        # Finalize Plot
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
        self.ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        self.canvas.draw()

    def _calculate_3d_positions(
        self, pivot: npt.NDArray[np.float64]
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Calculate the 3D positions of pendulum joints."""
        import numpy as np

        if self.state is None or self.dynamics is None:
            return pivot, pivot, pivot

        upper = self.dynamics.parameters.upper_segment
        lower = self.dynamics.parameters.lower_segment
        theta1 = self.state.theta1
        theta2 = self.state.theta2

        # Upper segment vector
        upper_vec = np.array(
            [
                math.sin(theta1) * upper.length_m,
                0.0,
                -math.cos(theta1) * upper.length_m,
            ]
        )
        elbow = pivot + upper_vec

        # Lower segment vector
        lower_abs_angle = theta1 + theta2
        lower_vec = np.array(
            [
                math.sin(lower_abs_angle) * lower.length_m,
                0.0,
                -math.cos(lower_abs_angle) * lower.length_m,
            ]
        )
        wrist = elbow + lower_vec

        # Apply rotations
        if not self.dynamics.parameters.constrained_to_plane:
            # Out of plane rotation
            phi = self.state.phi if hasattr(self.state, "phi") else 0.0
            elbow = self._rotate_out_of_plane(elbow, phi)
            wrist = self._rotate_out_of_plane(wrist, phi)

        if self.dynamics.parameters.constrained_to_plane:
            # Plane inclination rotation
            plane_angle = self.dynamics.parameters.plane_inclination_rad
            pivot = self._rotate_plane(pivot, plane_angle)
            elbow = self._rotate_plane(elbow, plane_angle)
            wrist = self._rotate_plane(wrist, plane_angle)

        return pivot, elbow, wrist

    def _rotate_out_of_plane(
        self, point: npt.NDArray[np.float64], phi: float
    ) -> npt.NDArray[np.float64]:
        """Rotate point around Z axis by phi."""
        import numpy as np

        x, y, z = point[0], point[1], point[2]
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        new_x = x * cos_phi - y * sin_phi
        new_y = x * sin_phi + y * cos_phi
        return np.array([new_x, new_y, z])

    def _rotate_plane(
        self, point: npt.NDArray[np.float64], angle: float
    ) -> npt.NDArray[np.float64]:
        """Rotate point around X axis by angle."""
        import numpy as np

        y, z = point[1], point[2]
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        new_y = y * cos_a - z * sin_a
        new_z = y * sin_a + z * cos_a
        return np.array([point[0], new_y, new_z])

    def _draw_reference_lines(
        self, pivot: npt.NDArray[np.float64], max_range: float, theta1: float
    ) -> None:
        """Draw reference lines and gravity."""
        import numpy as np

        # Vertical reference
        self.ax.plot(
            [pivot[0], pivot[0]],
            [pivot[1], pivot[1]],
            [pivot[2], pivot[2] - max_range * 0.3],
            "k--",
            linewidth=1,
            alpha=0.3,
            label="Plane Vertical (θ₁=0°)",
        )
        # Horizontal reference
        self.ax.plot(
            [pivot[0] - max_range * 0.3, pivot[0] + max_range * 0.3],
            [pivot[1], pivot[1]],
            [pivot[2], pivot[2]],
            "k--",
            linewidth=1,
            alpha=0.3,
            label="Plane Horizontal",
        )
        # Theta1 arc
        arc_theta = np.linspace(0, theta1, 20)
        arc_radius = max_range * 0.2
        arc_x = [pivot[0] + arc_radius * math.sin(t) for t in arc_theta]
        arc_z = [pivot[2] - arc_radius * math.cos(t) for t in arc_theta]
        if len(arc_x) > 1:
            self.ax.plot(arc_x, [pivot[1]] * 20, arc_z, "b-", linewidth=2, alpha=0.5)

        # Gravity
        gravity_len = max_range * 0.35
        g_start = pivot + np.array([max_range * 0.6, max_range * 0.2, max_range * 0.3])
        g_vec = np.array([0, 0, -gravity_len])
        self.ax.quiver(
            g_start[0],
            g_start[1],
            g_start[2],
            g_vec[0],
            g_vec[1],
            g_vec[2],
            color="#00AA00",
            arrow_length_ratio=0.3,
            linewidth=5,
            label="Gravity (g) - Always Vertical Down",
            alpha=0.95,
        )
        # Gravity label
        g_label_pos = g_start + g_vec * 0.5
        self.ax.text(
            g_label_pos[0] + max_range * 0.1,
            g_label_pos[1],
            g_label_pos[2],
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

    def _draw_segments(
        self,
        pivot: npt.NDArray[np.float64],
        elbow: npt.NDArray[np.float64],
        wrist: npt.NDArray[np.float64],
    ) -> None:
        """Draw the pendulum segments and joints."""
        # Upper Segment
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
        # Lower Segment
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
        # Joints
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
        self.ax.scatter(
            *elbow,
            color="#2E86AB",
            s=100,
            marker="o",
            edgecolors="white",
            linewidths=2,
            zorder=9,
        )
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

        # Labels
        upper_mid = (pivot + elbow) / 2
        self.ax.text(
            upper_mid[0],
            upper_mid[1],
            upper_mid[2],
            "UPPER",
            fontsize=9,
            color="#2E86AB",
            weight="bold",
            ha="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "#2E86AB",
            },
        )

        lower_mid = (elbow + wrist) / 2
        self.ax.text(
            lower_mid[0],
            lower_mid[1],
            lower_mid[2],
            "LOWER",
            fontsize=9,
            color="#A23B72",
            weight="bold",
            ha="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "#A23B72",
            },
        )

    def _draw_plane(self, size: float) -> None:
        """Draw the inclined plane surface."""
        import numpy as np

        if not self.dynamics or not self.dynamics.parameters.constrained_to_plane:
            return

        plane_size = size * 1.2
        x_plane = np.linspace(-plane_size, plane_size, 15)
        y_plane = np.linspace(-plane_size, plane_size, 15)
        x_grid, y_grid = np.meshgrid(x_plane, y_plane)

        angle = self.dynamics.parameters.plane_inclination_rad
        z_plane = y_grid * math.sin(angle)
        y_rot = y_grid * math.cos(angle)

        self.ax.plot_surface(
            x_grid, y_rot, z_plane, alpha=0.15, color="gray", edgecolor="none"
        )

    def __del__(self) -> None:
        """Cleanup on destruction."""
        self._stop_data_logging()


def run_app() -> None:
    root = tk.Tk()
    DoublePendulumApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_app()
