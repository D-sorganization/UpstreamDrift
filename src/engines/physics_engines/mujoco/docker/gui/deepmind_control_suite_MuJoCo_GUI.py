import contextlib
import datetime
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import threading
from functools import partial

logger = logging.getLogger(__name__)

try:
    import queue
    import tkinter as tk
    from tkinter import colorchooser, filedialog, messagebox, ttk
except ImportError:
    logger.error("Error: 'tkinter' module not found.")
    if platform.system() == "Linux":
        logger.info("Please install it by running: sudo apt-get install python3-tk")
    sys.exit(1)

# Default Config
DEFAULT_COLORS = {
    "shirt": [0.6, 0.6, 0.6, 1.0],
    "pants": [0.4, 0.2, 0.0, 1.0],
    "shoes": [0.1, 0.1, 0.1, 1.0],
    "skin": [0.8, 0.6, 0.4, 1.0],
    "eyes": [1.0, 1.0, 1.0, 1.0],
    "club": [0.8, 0.8, 0.8, 1.0],
}


class GolfSimulationGUI:
    def __init__(self, root) -> None:
        """Initialize the GUI."""
        self.root = root
        self.root.title("MuJoCo Golf Simulation Suite")
        self.root.geometry("900x750")
        self.root.configure(bg="#2b2b2b")

        # Set minimum window size
        self.root.minsize(800, 600)

        # Configure window icon and styling
        with contextlib.suppress(RuntimeError, ValueError, AttributeError):
            # Try to set a modern window style
            self.root.tk.call("tk", "scaling", 1.2)

        # Detect Environment
        self.is_windows = platform.system() == "Windows"

        if self.is_windows:
            # Dynamic path resolution
            current_dir = os.path.dirname(os.path.abspath(__file__))  # .../docker/gui
            docker_dir = os.path.dirname(current_dir)  # .../docker
            self.repo_path = os.path.dirname(docker_dir)  # .../ (Repo Root)

            # Convert to WSL path (e.g. C:\Users -> /mnt/c/Users)
            drive = self.repo_path[0].lower()
            rel_path = self.repo_path[2:].replace("\\", "/")
            self.wsl_path = f"/mnt/{drive}{rel_path}"
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            docker_dir = os.path.dirname(current_dir)
            self.repo_path = os.path.dirname(docker_dir)
            self.wsl_path = "/workspace"

        self.config_path = os.path.join(self.repo_path, "simulation_config.json")

        # Variables
        self.colors = DEFAULT_COLORS.copy()
        self.height_var = tk.DoubleVar(value=1.8)
        self.weight_var = tk.DoubleVar(value=100.0)

        self.control_mode_var = tk.StringVar(value="pd")
        self.live_view_var = tk.BooleanVar(value=False)
        self.save_path_var = tk.StringVar(value="")
        self.load_path_var = tk.StringVar(value="")

        self.club_length_var = tk.DoubleVar(value=1.0)
        self.club_mass_var = tk.DoubleVar(value=0.5)
        self.two_handed_var = tk.BooleanVar(value=False)
        self.enhance_face_var = tk.BooleanVar(value=False)
        self.articulated_fingers_var = tk.BooleanVar(value=False)

        self.stop_event = threading.Event()  # Initialize to avoid AttributeError
        self.load_config()

        # Configure modern styling
        self.setup_styles()

        # MAIN TABS
        self.notebook = ttk.Notebook(root, style="Modern.TNotebook")
        self.notebook.pack(expand=True, fill="both", padx=15, pady=15)

        # Bind tab selection to maintain consistent styling
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Tab 1: Simulation
        self.tab_sim = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_sim, text="Simulation")
        self.setup_sim_tab()

        # Tab 2: Appearance
        self.tab_appearance = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_appearance, text="Appearance")
        self.setup_appearance_tab()

        # Tab 3: Equipment & Model
        self.tab_equip = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_equip, text="Equipment & Model")
        self.setup_equip_tab()

    def load_config(self) -> None:
        """Load configuration from JSON."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    data = json.load(f)
                    if "colors" in data:
                        for k, v in data["colors"].items():
                            self.colors[k] = v

                    self.height_var.set(data.get("height_m", 1.8))
                    self.weight_var.set(data.get("weight_percent", 100.0))

                    self.control_mode_var.set(data.get("control_mode", "pd"))
                    self.live_view_var.set(data.get("live_view", False))
                    self.save_path_var.set(data.get("save_state_path", ""))
                    self.load_path_var.set(data.get("load_state_path", ""))

                    self.club_length_var.set(data.get("club_length", 1.0))
                    self.club_mass_var.set(data.get("club_mass", 0.5))
                    self.two_handed_var.set(data.get("two_handed", False))
                    self.enhance_face_var.set(data.get("enhance_face", False))
                    self.articulated_fingers_var.set(
                        data.get("articulated_fingers", False)
                    )

            except (FileNotFoundError, PermissionError, OSError) as e:
                logger.error("Error loading config: %s", e)

    def save_config(self) -> None:
        """Save configuration to JSON."""
        data = {
            "colors": self.colors,
            "height_m": self.height_var.get(),
            "weight_percent": self.weight_var.get(),
            "control_mode": self.control_mode_var.get(),
            "live_view": self.live_view_var.get(),
            "save_state_path": self.save_path_var.get(),
            "load_state_path": self.load_path_var.get(),
            "club_length": self.club_length_var.get(),
            "club_mass": self.club_mass_var.get(),
            "two_handed": self.two_handed_var.get(),
            "enhance_face": self.enhance_face_var.get(),
            "articulated_fingers": self.articulated_fingers_var.get(),
        }
        try:
            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=4)
            logger.info("Saved config to %s", self.config_path)
        except (FileNotFoundError, PermissionError, OSError) as e:
            messagebox.showerror("Error", f"Could not save config: {e}")

    def setup_styles(self) -> None:
        """Configure modern styling for the application."""
        style = ttk.Style()
        style.theme_use("clam")

        colors = self._get_color_scheme()
        self._configure_notebook_styles(style, colors)
        self._configure_frame_styles(style, colors)
        self._configure_label_styles(style, colors)
        self._configure_button_styles(style)
        self._configure_widget_styles(style, colors)

    @staticmethod
    def _get_color_scheme() -> dict[str, str]:
        return {
            "bg": "#2b2b2b",
            "fg": "#ffffff",
            "select_bg": "#404040",
            "select_fg": "#ffffff",
            "accent": "#0078d4",
            "success": "#107c10",
            "warning": "#ff8c00",
            "error": "#d13438",
            "purple": "#8b5cf6",
        }

    @staticmethod
    def _configure_notebook_styles(style: ttk.Style, colors: dict[str, str]) -> None:
        style.configure("Modern.TNotebook", background=colors["bg"], borderwidth=0)
        style.configure(
            "Modern.TNotebook.Tab",
            background=colors["select_bg"],
            foreground=colors["fg"],
            padding=[20, 12],
            font=("Segoe UI", 10, "bold"),
            focuscolor="none",
        )
        style.map(
            "Modern.TNotebook.Tab",
            background=[
                ("selected", colors["accent"]),
                ("active", colors["select_bg"]),
                ("!active", colors["select_bg"]),
            ],
            foreground=[
                ("selected", "#ffffff"),
                ("active", colors["fg"]),
                ("!active", colors["fg"]),
            ],
            padding=[
                ("selected", [20, 12]),
                ("active", [20, 12]),
                ("!active", [20, 12]),
            ],
        )

    @staticmethod
    def _configure_frame_styles(style: ttk.Style, colors: dict[str, str]) -> None:
        style.configure("Modern.TFrame", background=colors["bg"])
        style.configure(
            "Card.TFrame", background=colors["select_bg"], relief="flat", borderwidth=1
        )

    @staticmethod
    def _configure_label_styles(style: ttk.Style, colors: dict[str, str]) -> None:
        style.configure(
            "Modern.TLabel",
            background=colors["bg"],
            foreground=colors["fg"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Title.TLabel",
            background=colors["bg"],
            foreground=colors["fg"],
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "Heading.TLabel",
            background=colors["bg"],
            foreground=colors["accent"],
            font=("Segoe UI", 12, "bold"),
        )

    @staticmethod
    def _configure_button_styles(style: ttk.Style) -> None:
        style.configure("Modern.TButton", font=("Segoe UI", 10), padding=[15, 8])
        style.configure(
            "Primary.TButton", font=("Segoe UI", 11, "bold"), padding=[20, 10]
        )
        style.configure(
            "Success.TButton", font=("Segoe UI", 11, "bold"), padding=[20, 10]
        )
        style.configure("Warning.TButton", font=("Segoe UI", 10), padding=[15, 8])
        style.configure(
            "Danger.TButton", font=("Segoe UI", 11, "bold"), padding=[15, 8]
        )

    @staticmethod
    def _configure_widget_styles(style: ttk.Style, colors: dict[str, str]) -> None:
        style.configure(
            "Modern.TCombobox",
            fieldbackground=colors["select_bg"],
            background=colors["select_bg"],
            foreground=colors["fg"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Modern.TCheckbutton",
            background=colors["bg"],
            foreground=colors["fg"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Modern.TLabelframe",
            background=colors["bg"],
            foreground=colors["accent"],
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "Modern.TLabelframe.Label",
            background=colors["bg"],
            foreground=colors["accent"],
            font=("Segoe UI", 11, "bold"),
        )

    def setup_sim_tab(self) -> None:
        """Setup the simulation tab."""
        # Main container with padding
        main_container = ttk.Frame(self.tab_sim, style="Modern.TFrame")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        self._setup_sim_title(main_container)
        self._setup_sim_settings_card(main_container)
        self._setup_sim_state_card(main_container)
        self._setup_sim_action_buttons(main_container)
        self._setup_sim_log_section(main_container)

    def _setup_sim_title(self, parent) -> None:
        """Create the simulation tab title section."""
        title_frame = ttk.Frame(parent, style="Modern.TFrame")
        title_frame.pack(fill="x", pady=(0, 20))

        title = ttk.Label(
            title_frame, text="🏌️ Humanoid Golf Simulation", style="Title.TLabel"
        )
        title.pack(anchor="center")

        subtitle = ttk.Label(
            title_frame,
            text="Advanced biomechanical golf swing analysis",
            style="Modern.TLabel",
        )
        subtitle.pack(anchor="center", pady=(5, 0))

    def _setup_sim_settings_card(self, parent) -> None:
        """Create the simulation settings card with control mode and live view."""
        settings_card = ttk.LabelFrame(
            parent, text="⚙️ Simulation Settings", style="Modern.TLabelframe"
        )
        settings_card.pack(fill="x", pady=(0, 15))

        settings_inner = ttk.Frame(settings_card, style="Modern.TFrame")
        settings_inner.pack(fill="x", padx=20, pady=15)

        # Control mode section
        control_frame = ttk.Frame(settings_inner, style="Modern.TFrame")
        control_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(control_frame, text="Control Mode:", style="Modern.TLabel").pack(
            side="left"
        )
        control_combo = ttk.Combobox(
            control_frame,
            textvariable=self.control_mode_var,
            values=["pd", "lqr", "poly"],
            state="readonly",
            style="Modern.TCombobox",
            width=15,
        )
        control_combo.pack(side="left", padx=(10, 0))

        # Live view checkbox
        live_view_frame = ttk.Frame(settings_inner, style="Modern.TFrame")
        live_view_frame.pack(fill="x")

        ttk.Checkbutton(
            live_view_frame,
            text="🖥️ Live Interactive View (requires X11/VcXsrv)",
            variable=self.live_view_var,
            style="Modern.TCheckbutton",
        ).pack(side="left")

    def _setup_sim_state_card(self, parent) -> None:
        """Create the state management card with load/save path entries."""
        state_card = ttk.LabelFrame(
            parent, text="💾 State Management", style="Modern.TLabelframe"
        )
        state_card.pack(fill="x", pady=(0, 15))

        state_inner = ttk.Frame(state_card, style="Modern.TFrame")
        state_inner.pack(fill="x", padx=20, pady=15)

        # Load state section
        load_frame = ttk.Frame(state_inner, style="Modern.TFrame")
        load_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(load_frame, text="Load Initial State:", style="Modern.TLabel").pack(
            anchor="w"
        )
        load_entry_frame = ttk.Frame(load_frame, style="Modern.TFrame")
        load_entry_frame.pack(fill="x", pady=(5, 0))

        load_entry = tk.Entry(
            load_entry_frame,
            textvariable=self.load_path_var,
            font=("Segoe UI", 10),
            bg="#404040",
            fg="white",
            insertbackground="white",
            relief="flat",
            bd=5,
        )
        load_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        ttk.Button(
            load_entry_frame,
            text="📁 Browse",
            command=lambda: self.browse_file(self.load_path_var),
            style="Modern.TButton",
        ).pack(side="right")

        # Save state section
        save_frame = ttk.Frame(state_inner, style="Modern.TFrame")
        save_frame.pack(fill="x")

        ttk.Label(save_frame, text="Save Final State:", style="Modern.TLabel").pack(
            anchor="w"
        )
        save_entry_frame = ttk.Frame(save_frame, style="Modern.TFrame")
        save_entry_frame.pack(fill="x", pady=(5, 0))

        save_entry = tk.Entry(
            save_entry_frame,
            textvariable=self.save_path_var,
            font=("Segoe UI", 10),
            bg="#404040",
            fg="white",
            insertbackground="white",
            relief="flat",
            bd=5,
        )
        save_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        ttk.Button(
            save_entry_frame,
            text="📁 Browse",
            command=lambda: self.browse_file(self.save_path_var, save=True),
            style="Modern.TButton",
        ).pack(side="right")

    def _setup_sim_action_buttons(self, parent) -> None:
        """Create the simulation control and results action buttons."""
        action_card = ttk.LabelFrame(
            parent, text="🎮 Simulation Controls", style="Modern.TLabelframe"
        )
        action_card.pack(fill="x", pady=(0, 15))

        action_inner = ttk.Frame(action_card, style="Modern.TFrame")
        action_inner.pack(fill="x", padx=20, pady=15)

        self._create_primary_action_buttons(action_inner)
        self._create_secondary_action_buttons(action_inner)

    def _create_primary_action_buttons(self, parent) -> None:
        primary_frame = ttk.Frame(parent, style="Modern.TFrame")
        primary_frame.pack(fill="x", pady=(0, 10))

        self.btn_run = tk.Button(
            primary_frame,
            text="🚀 RUN SIMULATION",
            command=self.start_simulation,
            bg="#107c10",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            bd=0,
            padx=30,
            pady=12,
            cursor="hand2",
        )
        self.btn_run.pack(side="left", padx=(0, 15))

        self.btn_stop = tk.Button(
            primary_frame,
            text="⏹️ STOP",
            command=self.stop_simulation,
            bg="#d13438",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            bd=0,
            padx=20,
            pady=12,
            state=tk.DISABLED,
            cursor="hand2",
        )
        self.btn_stop.pack(side="left", padx=(0, 15))

        self.btn_rebuild = tk.Button(
            primary_frame,
            text="🔧 UPDATE ENV",
            command=self.rebuild_docker,
            bg="#8b5cf6",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            bd=0,
            padx=20,
            pady=12,
            cursor="hand2",
        )
        self.btn_rebuild.pack(side="right")

    def _create_secondary_action_buttons(self, parent) -> None:
        secondary_frame = ttk.Frame(parent, style="Modern.TFrame")
        secondary_frame.pack(fill="x")

        results_label = ttk.Label(
            secondary_frame, text="📊 Results:", style="Modern.TLabel"
        )
        results_label.pack(side="left", padx=(0, 10))

        self.btn_open_video = tk.Button(
            secondary_frame,
            text="🎥 Open Video",
            command=self.open_video,
            bg="#404040",
            fg="white",
            font=("Segoe UI", 10),
            relief="flat",
            bd=0,
            padx=15,
            pady=8,
            state=tk.DISABLED,
            cursor="hand2",
        )
        self.btn_open_video.pack(side="left", padx=(0, 10))

        self.btn_open_data = tk.Button(
            secondary_frame,
            text="📈 Open Data (CSV)",
            command=self.open_data,
            bg="#404040",
            fg="white",
            font=("Segoe UI", 10),
            relief="flat",
            bd=0,
            padx=15,
            pady=8,
            state=tk.DISABLED,
            cursor="hand2",
        )
        self.btn_open_data.pack(side="left")

    def _setup_sim_log_section(self, parent) -> None:
        """Create the simulation log section with text area and scrollbar."""
        log_card = ttk.LabelFrame(
            parent, text="📋 Simulation Log", style="Modern.TLabelframe"
        )
        log_card.pack(fill="both", expand=True)

        log_inner = ttk.Frame(log_card, style="Modern.TFrame")
        log_inner.pack(fill="both", expand=True, padx=20, pady=15)

        # Log header with clear button
        log_header = ttk.Frame(log_inner, style="Modern.TFrame")
        log_header.pack(fill="x", pady=(0, 10))

        log_status = ttk.Label(
            log_header, text="Real-time simulation output:", style="Modern.TLabel"
        )
        log_status.pack(side="left")

        self.btn_clear_log = tk.Button(
            log_header,
            text="🗑️ Clear Log",
            command=self.clear_log,
            bg="#ff8c00",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
        )
        self.btn_clear_log.pack(side="right")

        # Log text area with modern styling
        log_frame = ttk.Frame(log_inner, style="Modern.TFrame")
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(
            log_frame,
            height=12,
            bg="#1e1e1e",
            fg="#ffffff",
            font=("Consolas", 10),
            relief="flat",
            bd=0,
            padx=10,
            pady=10,
            wrap="word",
            insertbackground="#ffffff",
        )
        self.log_text.pack(side="left", fill="both", expand=True)

        # Modern scrollbar
        scrollbar = tk.Scrollbar(
            log_frame,
            command=self.log_text.yview,
            bg="#404040",
            troughcolor="#2b2b2b",
            activebackground="#606060",
        )
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

    def setup_appearance_tab(self) -> None:
        """Setup the appearance tab."""
        # Main container
        main_container = ttk.Frame(self.tab_appearance, style="Modern.TFrame")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        title = ttk.Label(
            main_container,
            text="🎨 Humanoid Appearance Customization",
            style="Title.TLabel",
        )
        title.pack(pady=(0, 20))

        self._setup_dimensions_card(main_container)
        self._setup_colors_card(main_container)
        self._setup_appearance_save_button(main_container)

    def _setup_dimensions_card(self, parent: ttk.Frame) -> None:
        """Create the physical dimensions card with height and weight controls."""
        dimensions_card = ttk.LabelFrame(
            parent, text="📏 Physical Dimensions", style="Modern.TLabelframe"
        )
        dimensions_card.pack(fill="x", pady=(0, 20))

        dim_inner = ttk.Frame(dimensions_card, style="Modern.TFrame")
        dim_inner.pack(fill="x", padx=20, pady=15)

        self._setup_height_control(dim_inner)
        self._setup_weight_control(dim_inner)

    def _setup_height_control(self, parent: ttk.Frame) -> None:
        """Create the height spinbox control."""
        height_frame = ttk.Frame(parent, style="Modern.TFrame")
        height_frame.pack(fill="x", pady=(0, 15))

        ttk.Label(height_frame, text="Height (meters):", style="Modern.TLabel").pack(
            side="left"
        )
        height_spinbox = tk.Spinbox(
            height_frame,
            from_=0.5,
            to=3.0,
            increment=0.05,
            textvariable=self.height_var,
            width=8,
            font=("Segoe UI", 10),
            bg="#404040",
            fg="white",
            buttonbackground="#606060",
            relief="flat",
            bd=5,
        )
        height_spinbox.pack(side="right")

    def _setup_weight_control(self, parent: ttk.Frame) -> None:
        """Create the weight scale control with label."""
        weight_frame = ttk.Frame(parent, style="Modern.TFrame")
        weight_frame.pack(fill="x")

        weight_label_frame = ttk.Frame(weight_frame, style="Modern.TFrame")
        weight_label_frame.pack(fill="x", pady=(0, 5))

        ttk.Label(
            weight_label_frame, text="Weight (% of standard):", style="Modern.TLabel"
        ).pack(side="left")
        weight_value = ttk.Label(
            weight_label_frame, textvariable=self.weight_var, style="Modern.TLabel"
        )
        weight_value.pack(side="right")

        weight_scale = tk.Scale(
            weight_frame,
            from_=50,
            to=200,
            orient=tk.HORIZONTAL,
            variable=self.weight_var,
            length=300,
            bg="#2b2b2b",
            fg="white",
            troughcolor="#404040",
            activebackground="#0078d4",
            highlightthickness=0,
            relief="flat",
        )
        weight_scale.pack(fill="x")

    def _setup_colors_card(self, parent: ttk.Frame) -> None:
        """Create the body colors card with color pickers for each body part."""
        colors_card = ttk.LabelFrame(
            parent, text="🎨 Body Colors", style="Modern.TLabelframe"
        )
        colors_card.pack(fill="x", pady=(0, 20))

        colors_inner = ttk.Frame(colors_card, style="Modern.TFrame")
        colors_inner.pack(fill="x", padx=20, pady=15)

        self.color_widgets: dict[str, tk.Canvas] = {}

        color_parts = [
            ("👕 Shirt", "shirt"),
            ("👖 Pants", "pants"),
            ("👟 Shoes", "shoes"),
            ("👤 Skin", "skin"),
            ("🏌️ Club", "club"),
        ]

        for _, (display_name, part_key) in enumerate(color_parts):
            self._create_color_picker_row(colors_inner, display_name, part_key)

    def _create_color_picker_row(
        self, parent: ttk.Frame, display_name: str, part_key: str
    ) -> None:
        """Create a single color picker row with label, swatch, and pick button."""
        color_row = ttk.Frame(parent, style="Modern.TFrame")
        color_row.pack(fill="x", pady=5)

        # Label
        ttk.Label(color_row, text=display_name, style="Modern.TLabel", width=12).pack(
            side="left"
        )

        # Color swatch
        swatch_frame = ttk.Frame(color_row, style="Modern.TFrame")
        swatch_frame.pack(side="left", padx=(10, 15))

        canvas = tk.Canvas(
            swatch_frame,
            width=50,
            height=30,
            bg="white",
            relief="solid",
            borderwidth=2,
            highlightthickness=0,
        )
        canvas.pack()
        self.color_widgets[part_key] = canvas
        self.update_swatch(part_key)

        # Pick color button
        color_btn = tk.Button(
            color_row,
            text="🎨 Pick Color",
            command=partial(self.pick_color, part_key),
            bg="#0078d4",
            fg="white",
            font=("Segoe UI", 9),
            relief="flat",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
        )
        color_btn.pack(side="right")

    def _setup_appearance_save_button(self, parent: ttk.Frame) -> None:
        """Create the save appearance settings button."""
        save_frame = ttk.Frame(parent, style="Modern.TFrame")
        save_frame.pack(fill="x", pady=20)

        save_btn = tk.Button(
            save_frame,
            text="💾 Save Appearance Settings",
            command=self.save_config,
            bg="#107c10",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            bd=0,
            padx=25,
            pady=10,
            cursor="hand2",
        )
        save_btn.pack(anchor="center")

    def setup_equip_tab(self) -> None:
        """Setup the equipment tab."""
        main_container = ttk.Frame(self.tab_equip, style="Modern.TFrame")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        title = ttk.Label(
            main_container,
            text="⚙️ Equipment & Model Configuration",
            style="Title.TLabel",
        )
        title.pack(pady=(0, 20))

        self._create_club_parameters_card(main_container)
        self._create_advanced_features_card(main_container)
        self._create_equip_save_button(main_container)

    def _create_club_parameters_card(self, parent: ttk.Frame) -> None:
        club_card = ttk.LabelFrame(
            parent, text="🏌️ Golf Club Parameters", style="Modern.TLabelframe"
        )
        club_card.pack(fill="x", pady=(0, 20))

        club_inner = ttk.Frame(club_card, style="Modern.TFrame")
        club_inner.pack(fill="x", padx=20, pady=15)

        self._create_scale_control(
            club_inner,
            label="Club Length (meters):",
            variable=self.club_length_var,
            from_=0.5,
            to=1.5,
            resolution=0.05,
            bottom_pad=15,
        )
        self._create_scale_control(
            club_inner,
            label="Club Mass (kg):",
            variable=self.club_mass_var,
            from_=0.1,
            to=2.0,
            resolution=0.1,
            bottom_pad=0,
        )

    def _create_scale_control(
        self,
        parent: ttk.Frame,
        *,
        label: str,
        variable: tk.DoubleVar,
        from_: float,
        to: float,
        resolution: float,
        bottom_pad: int,
    ) -> None:
        frame = ttk.Frame(parent, style="Modern.TFrame")
        frame.pack(fill="x", pady=(0, bottom_pad))

        label_frame = ttk.Frame(frame, style="Modern.TFrame")
        label_frame.pack(fill="x", pady=(0, 5))

        ttk.Label(label_frame, text=label, style="Modern.TLabel").pack(side="left")
        ttk.Label(label_frame, textvariable=variable, style="Modern.TLabel").pack(
            side="right"
        )

        tk.Scale(
            frame,
            from_=from_,
            to=to,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            variable=variable,
            length=300,
            bg="#2b2b2b",
            fg="white",
            troughcolor="#404040",
            activebackground="#0078d4",
            highlightthickness=0,
            relief="flat",
        ).pack(fill="x")

    def _create_advanced_features_card(self, parent: ttk.Frame) -> None:
        features_card = ttk.LabelFrame(
            parent,
            text="🔬 Advanced Model Features",
            style="Modern.TLabelframe",
        )
        features_card.pack(fill="x", pady=(0, 20))

        features_inner = ttk.Frame(features_card, style="Modern.TFrame")
        features_inner.pack(fill="x", padx=20, pady=15)

        features = [
            ("🤝 Two-Handed Grip (Constrained)", self.two_handed_var),
            ("😊 Enhanced Face (Nose, Mouth)", self.enhance_face_var),
            ("🖐️ Articulated Fingers (Segments)", self.articulated_fingers_var),
        ]

        for feature_text, feature_var in features:
            feature_frame = ttk.Frame(features_inner, style="Modern.TFrame")
            feature_frame.pack(fill="x", pady=5)

            checkbox = tk.Checkbutton(
                feature_frame,
                text=feature_text,
                variable=feature_var,
                bg="#2b2b2b",
                fg="white",
                selectcolor="#404040",
                activebackground="#2b2b2b",
                activeforeground="white",
                font=("Segoe UI", 10),
                relief="flat",
                bd=0,
                highlightthickness=0,
            )
            checkbox.pack(side="left")

    def _create_equip_save_button(self, parent: ttk.Frame) -> None:
        save_frame = ttk.Frame(parent, style="Modern.TFrame")
        save_frame.pack(fill="x", pady=20)

        save_btn = tk.Button(
            save_frame,
            text="💾 Save Equipment Settings",
            command=self.save_config,
            bg="#107c10",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            bd=0,
            padx=25,
            pady=10,
            cursor="hand2",
        )
        save_btn.pack(anchor="center")

    def browse_file(self, var, save=False) -> None:
        """Open file dialog to browse for file."""
        if save:
            path = filedialog.asksaveasfilename(
                defaultextension=".json", filetypes=[("JSON State", "*.json")]
            )
        else:
            path = filedialog.askopenfilename(filetypes=[("JSON State", "*.json")])
        if path:
            var.set(path)

    def update_swatch(self, part) -> None:
        """Update color swatch."""
        rgba = self.colors[part]
        r, g, b = (int(c * 255) for c in rgba[:3])
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        self.color_widgets[part].config(bg=hex_color)

    def pick_color(self, part) -> None:
        """Open color picker dialog."""
        current_rgba = self.colors[part]
        current_rgb_int = tuple(int(c * 255) for c in current_rgba[:3])

        color = colorchooser.askcolor(
            initialcolor=f"#{current_rgb_int[0]:02x}{current_rgb_int[1]:02x}{current_rgb_int[2]:02x}",
            title=f"Choose {part.title()} Color",
        )

        if color[0]:
            rgb = color[0]
            new_rgba = [x / 255.0 for x in rgb] + [1.0]
            self.colors[part] = new_rgba
            self.update_swatch(part)
            self.save_config()

    def log(self, message) -> None:
        """Log message to GUI console."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def clear_log(self) -> None:
        """Clear the simulation log."""
        self.log_text.delete("1.0", tk.END)

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log(f"[{timestamp}] Log cleared.")

    def _on_tab_changed(self, event) -> None:
        """Handle tab selection changes to maintain consistent styling."""
        # Force update of tab styling to prevent height changes
        self.notebook.update_idletasks()

    def start_simulation(self) -> None:
        """Start the simulation process."""
        self.save_config()
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.log("Starting simulation...")
        self.stop_event = threading.Event()
        threading.Thread(target=self._run_docker_process, daemon=True).start()

    def stop_simulation(self) -> None:
        """Stop the simulation process."""
        if hasattr(self, "process") and self.process:
            self.log("Stopping simulation...")
            self.process.terminate()
            # Force kill if needed
            # self.process.kill()
        self.stop_event.set()

    @staticmethod
    def _generate_update_dockerfile() -> str:
        """Generate a minimal Dockerfile to add missing dependencies."""
        return (
            "# Add missing dependencies to existing robotics_env\n"
            "FROM robotics_env:latest\n\n"
            "# Install missing dependencies in the existing virtual "
            "environment\n"
            'RUN /opt/mujoco-env/bin/pip install "defusedxml>=0.7.1" '
            '"PyQt6>=6.6.0"\n\n'
            "# Update PATH to use robotics_env by default\n"
            'ENV PATH="/opt/mujoco-env/bin:$PATH"\n'
            'ENV VIRTUAL_ENV="/opt/mujoco-env"\n'
        )

    def _run_docker_build(self, temp_dir: str, cmd: list[str]) -> int:
        """Execute the docker build command and return the exit code."""
        if self.is_windows:
            create_new_console = 0x00000010
            result = subprocess.run(
                ["cmd", "/k", *cmd],
                cwd=temp_dir,
                creationflags=create_new_console,  # type: ignore[call-arg]
            )
        else:
            result = subprocess.run(cmd, cwd=temp_dir, check=True)
        return result.returncode

    def _verify_docker_update(self) -> None:
        """Run a quick container test to verify defusedxml is available."""
        test_cmd = [
            "docker",
            "run",
            "--rm",
            "robotics_env",
            "python",
            "-c",
            "import defusedxml; print('✅ defusedxml confirmed working')",
        ]
        test_result = subprocess.run(test_cmd, capture_output=True, text=True)
        if test_result.returncode == 0:
            self.root.after(0, self.log, test_result.stdout.strip())
        else:
            self.root.after(0, self.log, "⚠️ Update completed but test failed")

    def rebuild_docker(self) -> None:
        """Add missing dependencies to the existing robotics_env Docker image."""
        from tkinter import messagebox

        msg = (
            "This will add missing dependencies (like defusedxml) to the existing "
            "robotics_env.\n"
            "This should be quick since we're just adding packages. Continue?"
        )
        result = messagebox.askyesno(
            "Update Robotics Environment",
            msg,
        )

        if not result:
            return

        self.log("Updating robotics_env with missing dependencies...")
        self.btn_rebuild.config(state=tk.DISABLED)

        def run_update() -> None:
            """Rebuild the Docker image with missing dependencies."""
            try:
                dockerfile_content = self._generate_update_dockerfile()

                with tempfile.TemporaryDirectory() as temp_dir:
                    dockerfile_path = os.path.join(temp_dir, "Dockerfile")
                    with open(dockerfile_path, "w") as f:
                        f.write(dockerfile_content)

                    cmd = ["docker", "build", "-t", "robotics_env", "."]
                    self.root.after(0, self.log, f"Running: {' '.join(cmd)}")
                    self.root.after(0, self.log, "Adding defusedxml to robotics_env...")

                    returncode = self._run_docker_build(temp_dir, cmd)

                    if returncode == 0:
                        self.root.after(
                            0, self.log, "✅ robotics_env updated successfully!"
                        )
                        self.root.after(
                            0,
                            self.log,
                            "defusedxml and other dependencies are now available.",
                        )
                        self._verify_docker_update()
                    else:
                        self.root.after(
                            0,
                            self.log,
                            f"❌ Update failed with code {returncode}",
                        )

            except ImportError as e:
                self.root.after(0, self.log, f"❌ Update failed: {e}")
            finally:
                self.root.after(0, lambda: self.btn_rebuild.config(state=tk.NORMAL))

        threading.Thread(target=run_update, daemon=True).start()

    def _build_docker_command(self) -> list[str]:
        """Build the docker run command for the simulation subprocess."""
        if self.is_windows:
            cmd = [
                "wsl",
                "docker",
                "run",
                "--rm",
                "-v",
                f"{self.wsl_path}:/workspace",
                "-w",
                "/workspace/python",
            ]

            if self.live_view_var.get():
                # Allow GUI to display on host Windows X Server (VcXsrv)
                cmd.extend(["-e", "DISPLAY=host.docker.internal:0"])
                cmd.extend(["-e", "MUJOCO_GL=glfw"])
                cmd.extend(["-e", "PYOPENGL_PLATFORM=glx"])
            else:
                cmd.extend(["-e", "MUJOCO_GL=osmesa"])

            cmd.extend(
                [
                    "robotics_env",
                    "/opt/mujoco-env/bin/python",
                    "-u",
                    "-m",
                    "mujoco_humanoid_golf",
                ]
            )
        else:
            cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{self.repo_path}:/workspace",
                "-w",
                "/workspace/python",
            ]

            if self.live_view_var.get():
                cmd.extend(["-e", f"DISPLAY={os.environ.get('DISPLAY', ':0')}"])
                cmd.extend(["-e", "MUJOCO_GL=glfw"])
                cmd.extend(["-e", "PYOPENGL_PLATFORM=glx"])
                cmd.extend(["-v", "/tmp/.X11-unix:/tmp/.X11-unix"])  # nosec B108
            else:
                cmd.extend(["-e", "MUJOCO_GL=osmesa"])

            cmd.extend(
                [
                    "robotics_env",
                    "/opt/mujoco-env/bin/python",
                    "-m",
                    "mujoco_humanoid_golf",
                ]
            )

        return cmd

    def _stream_process_output(self) -> None:
        """Read subprocess stdout via a queue and log lines to the GUI."""
        q: queue.Queue[str | None] = queue.Queue()

        def enqueue_output(out, output_queue) -> None:
            """Enqueue output from subprocess."""
            try:
                for line in iter(out.readline, ""):
                    output_queue.put(line)
                out.close()
            except (RuntimeError, ValueError, OSError) as e:
                with contextlib.suppress(RuntimeError, ValueError, AttributeError):
                    self.root.after(0, self.log, f"Exception in enqueue_output: {e}")
            output_queue.put(None)  # Sentinel

        t = threading.Thread(
            target=enqueue_output, args=(self.process.stdout, q), daemon=True
        )
        t.start()

        while True:
            # Check user stop
            if self.stop_event.is_set() and self.process.poll() is None:
                self.process.terminate()

            try:
                output = q.get(timeout=0.1)
            except queue.Empty:
                if self.process.poll() is not None and not t.is_alive():
                    break
                continue

            if output is None:  # Sentinel
                break

            self.root.after(0, self.log, output.strip())

    def _handle_process_failure(self, rc) -> None:
        """Log error details and suggest solutions for common failures."""
        self.root.after(0, self.log, f"Process exited with code {rc}")
        if self.process.stderr:
            err = self.process.stderr.read()
            if err:
                self.root.after(0, self.log, f"ERROR: {err}")
            # Check for specific common errors and provide solutions
            if "defusedxml" in err:
                self.root.after(
                    0,
                    self.log,
                    "SOLUTION: Missing defusedxml dependency. "
                    "Please rebuild Docker image.",
                )
                self.root.after(0, self.log, "Run: docker build -t robotics_env .")
            elif "ModuleNotFoundError" in err:
                self.root.after(
                    0,
                    self.log,
                    "SOLUTION: Missing Python dependency. "
                    "Check Dockerfile and rebuild.",
                )
            elif "DISPLAY" in err or "X11" in err:
                self.root.after(
                    0,
                    self.log,
                    "SOLUTION: X11/Display issue. "
                    "Try disabling 'Live Interactive View'.",
                )

    def _reset_buttons_state(self) -> None:
        """Reset run/stop buttons to their default enabled states."""
        self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))

    def _run_docker_process(self) -> None:
        """Run the simulation in a subprocess."""
        cmd = self._build_docker_command()

        try:
            self.log(f"Running command: {' '.join(cmd)}")

            # Race condition fix: Check stop event before starting
            if self.stop_event.is_set():
                self.log("Simulation cancelled.")
                self._reset_buttons_state()
                return

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            self._stream_process_output()

            rc = self.process.poll()

            # Check if stopped by user
            if self.stop_event.is_set():
                self.root.after(0, self.log, "Simulation stopped by user.")
                self._reset_buttons_state()
            elif rc == 0:
                self.root.after(0, self.on_sim_success)
            else:
                self._handle_process_failure(rc)
                self._reset_buttons_state()

        except ImportError as e:
            self.root.after(0, self.log, f"Failed to run subprocess: {e}")
            self._reset_buttons_state()

    def on_sim_success(self) -> None:
        """Handle successful simulation completion."""
        self.log("Simulation Completed Successfully!")
        self.btn_run.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_open_video.config(state=tk.NORMAL)
        self.btn_open_data.config(state=tk.NORMAL)

    def open_file(self, filepath) -> None:
        """Open a file with the default application."""
        if self.is_windows:
            if os.path.exists(filepath):
                os.startfile(filepath)  # type: ignore[attr-defined]
            else:
                messagebox.showerror("Error", f"File not found: {filepath}")
        elif os.path.exists(filepath):
            try:
                subprocess.call(["xdg-open", filepath])
            except FileNotFoundError:
                messagebox.showerror("Error", "xdg-open not found. Cannot open file.")
        else:
            messagebox.showerror("Error", f"File not found: {filepath}")

    def open_video(self) -> None:
        """Open the generated video file."""
        vid_path = os.path.join(self.repo_path, "humanoid_golf.mp4")
        self.open_file(vid_path)

    def open_data(self) -> None:
        """Open the generated data file."""
        csv_path = os.path.join(self.repo_path, "golf_data.csv")
        self.open_file(csv_path)


if __name__ == "__main__":
    root = tk.Tk()
    app = GolfSimulationGUI(root)
    root.mainloop()
