import contextlib
import logging
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk

# Config - UPDATED FOR GOLF_MODELING_SUITE
REPOS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "launchersassets"))
DOCKER_IMAGE_NAME = "robotics_env"
logger = logging.getLogger(__name__)

# Map display names to folder names - UPDATED PATHS
MODELS = {
    "MuJoCo Golf Model": "engines/physics_engines/mujoco",
    "Drake Golf Model": "engines/physics_engines/drake",
    "Pinocchio Golf Model": "engines/physics_engines/pinocchio",
}

# Map display names to image filenames
MODEL_IMAGES = {
    "MuJoCo Golf Model": "mujoco.png",
    "Drake Golf Model": "drake.png",
    "Pinocchio Golf Model": "pinocchio.png",
    "Playground (Solar System)": "playground.png",
}

# Theme Colors
COLOR_BG = "#1E1E1E"
COLOR_FG = "#FFFFFF"
COLOR_ACCENT = "#007ACC"
COLOR_SECONDARY_BG = "#252526"
COLOR_HOVER = "#3E3E42"


class ToolTip:
    """
    Creates a tooltip for a given widget as the mouse hovers over it.
    """

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip_window: tk.Toplevel | None = None
        self.id: str | None = None
        self.x = self.y = 0
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def configure(self, text: str) -> None:
        self.text = text

    def enter(self, _event=None) -> None:
        self.schedule()

    def leave(self, _event=None) -> None:
        self.unschedule()
        self.hidetip()

    def schedule(self) -> None:
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)

    def unschedule(self) -> None:
        id_ = self.id
        self.id = None
        if id_:
            self.widget.after_cancel(id_)

    def showtip(self, _event=None) -> None:
        x = y = 0
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)  # noqa: FBT003
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#2D2D30",
            foreground="#E0E0E0",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9),
        )
        label.pack(ipadx=5, ipady=3)

    def hidetip(self) -> None:
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


class UnifiedLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        self.title("Golf Modeling Suite - Unified Launcher")
        self.geometry("900x650")
        self.configure(bg=COLOR_BG)

        self.docker_available = False  # Track state

        self.style = ttk.Style()
        self.style.theme_use("clam")  # Use clam for better customizability
        self.configure_styles()

        self.images = {}  # Keep references to avoid GC
        self.setup_ui()
        self.start_check_docker()

    def configure_styles(self):
        style = self.style

        # General Frame
        style.configure("TFrame", background=COLOR_BG)
        style.configure(
            "TLabelframe", background=COLOR_BG, foreground=COLOR_FG, relief="flat"
        )
        style.configure(
            "TLabelframe.Label",
            background=COLOR_BG,
            foreground=COLOR_ACCENT,
            font=("Segoe UI", 12, "bold"),
        )

        # Label
        style.configure(
            "TLabel", background=COLOR_BG, foreground=COLOR_FG, font=("Segoe UI", 11)
        )

        # Radiobutton
        style.configure(
            "TRadiobutton",
            background=COLOR_BG,
            foreground=COLOR_FG,
            font=("Segoe UI", 11),
            indicatorbackground=COLOR_BG,
            indicatorforeground=COLOR_FG,
            selectcolor=COLOR_SECONDARY_BG,
        )
        style.map(
            "TRadiobutton",
            background=[("active", COLOR_HOVER)],
            indicatorcolor=[("selected", COLOR_ACCENT)],
        )

        # Checkbutton
        style.configure(
            "TCheckbutton",
            background=COLOR_BG,
            foreground=COLOR_FG,
            font=("Segoe UI", 11),
            focuscolor=COLOR_BG,
        )
        style.map(
            "TCheckbutton",
            background=[("active", COLOR_HOVER)],
            indicatorcolor=[("selected", COLOR_ACCENT)],
        )

        # Button
        style.configure(
            "TButton",
            font=("Segoe UI", 11, "bold"),
            background=COLOR_ACCENT,
            foreground="#FFFFFF",
            borderwidth=0,
            padding=10,
        )
        style.map(
            "TButton",
            background=[("active", "#005F9E"), ("disabled", "#555555")],
            foreground=[("disabled", "#AAAAAA")],
        )

        # Custom styles
        style.configure(
            "Title.TLabel", font=("Segoe UI", 24, "bold"), foreground=COLOR_FG
        )
        style.configure(
            "Status.TLabel",
            background=COLOR_SECONDARY_BG,
            foreground="#AAAAAA",
            font=("Segoe UI", 9),
        )

    def setup_ui(self):
        # Main Layout
        main_container = ttk.Frame(self, style="TFrame")
        main_container.pack(fill="both", expand=True, padx=30, pady=30)

        # Title
        lbl_title = ttk.Label(
            main_container, text="Golf Modeling Suite", style="Title.TLabel"
        )
        lbl_title.pack(pady=(0, 20), anchor="center")

        # Model Selection Area
        frame_select = ttk.LabelFrame(main_container, text="Select Physics Engine")
        frame_select.pack(fill="both", expand=True, pady=10)

        self.var_model = tk.StringVar(value="MuJoCo Golf Model")

        # Create a grid for models
        grid_frame = ttk.Frame(frame_select)
        grid_frame.pack(fill="both", expand=True, padx=10, pady=10)

        row = 0
        col = 0
        for name in MODELS:
            # Load Image
            img_path = os.path.join(ASSETS_DIR, MODEL_IMAGES.get(name, ""))
            photo = None
            if os.path.exists(img_path):
                try:
                    photo = tk.PhotoImage(file=img_path)
                    self.images[name] = photo
                except Exception as e:
                    logger.warning("Failed to load image for %s: %s", name, e)

            # Card-like container for each option
            card = tk.Frame(
                grid_frame,
                bg=COLOR_SECONDARY_BG,
                bd=0,
                highlightthickness=1,
                highlightbackground="#333",
            )
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            grid_frame.columnconfigure(col, weight=1)

            # Content inside card
            # Image
            if name in self.images:
                lbl_img = tk.Label(card, image=self.images[name], bg=COLOR_SECONDARY_BG)
                lbl_img.pack(side="top", pady=10)

            # Radio selection
            rb = ttk.Radiobutton(
                card,
                text=name,
                variable=self.var_model,
                value=name,
                style="TRadiobutton",
            )
            rb.pack(side="top", pady=(0, 10))

            col += 1
            if col > 1:  # 2 columns
                col = 0
                row += 1

        # Options
        frame_opts = ttk.LabelFrame(main_container, text="Configuration")
        frame_opts.pack(fill="x", pady=20)

        self.var_live = tk.BooleanVar(value=True)
        self.var_gpu = tk.BooleanVar(value=False)

        opts_inner = ttk.Frame(frame_opts)
        opts_inner.pack(padx=10, pady=10)

        cb_live = ttk.Checkbutton(
            opts_inner,
            text="Live Visualization (VcXsrv)",
            variable=self.var_live,
            style="TCheckbutton",
        )
        cb_live.pack(side="left", padx=20)
        ToolTip(
            cb_live,
            "Requires VcXsrv on Windows (automatically handled) or X11 on Linux.",
        )

        cb_gpu = ttk.Checkbutton(
            opts_inner,
            text="GPU Acceleration (NVIDIA)",
            variable=self.var_gpu,
            style="TCheckbutton",
        )
        cb_gpu.pack(side="left", padx=20)
        ToolTip(cb_gpu, "Requires NVIDIA Container Toolkit and compatible GPU.")

        # Actions
        frame_actions = ttk.Frame(main_container)
        frame_actions.pack(pady=20)

        self.btn_launch = ttk.Button(
            frame_actions,
            text="LAUNCH SIMULATION",
            command=self.launch,
            style="TButton",
        )
        self.btn_launch.pack(side="left", padx=10, ipadx=20)
        self.tooltip_launch = ToolTip(
            self.btn_launch,
            "Start the simulation environment with selected options. (Enter)",
        )

        self.btn_build = ttk.Button(
            frame_actions,
            text="BUILD DOCKER",
            command=self.build_docker,
            style="TButton",
        )
        self.btn_build.pack(side="left", padx=10)
        ToolTip(
            self.btn_build,
            "Builds the 'robotics_env' Docker image from the MuJoCo engine directory.",
        )

        btn_quit = ttk.Button(
            frame_actions, text="QUIT", command=self.destroy, style="TButton"
        )
        btn_quit.pack(side="left", padx=10)

        # Keyboard shortcuts
        self.bind("<Return>", lambda _e: self.launch())
        self.bind("<Control-q>", lambda _e: self.destroy())

        # Log Area
        log_frame = ttk.LabelFrame(main_container, text="Simulation Log")
        log_frame.pack(fill="both", expand=True, pady=(10, 0))

        # Create scrollable text area for logs
        log_inner = ttk.Frame(log_frame)
        log_inner.pack(fill="both", expand=True, padx=10, pady=10)

        self.log_text = tk.Text(
            log_inner,
            height=8,
            bg=COLOR_SECONDARY_BG,
            fg=COLOR_FG,
            font=("Consolas", 9),
            wrap="word",
            state="disabled",
        )

        scrollbar = ttk.Scrollbar(
            log_inner, orient="vertical", command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.pack(fill="x", padx=10, pady=(0, 10))

        btn_clear_log = ttk.Button(
            log_controls, text="Clear Log", command=self.clear_log, style="TButton"
        )
        btn_clear_log.pack(side="right")

        # Status Bar
        self.lbl_status = ttk.Label(
            self, text=" Initializing...", style="Status.TLabel", anchor="w"
        )
        self.lbl_status.pack(side="bottom", fill="x", ipady=5)

    def log(self, msg):
        self.lbl_status.config(text=f" {msg}")
        logger.info(msg)

        # Also add to log text area
        self.log_text.config(state="normal")
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {msg}\n")
        self.log_text.see("end")  # Auto-scroll to bottom
        self.log_text.config(state="disabled")

    def clear_log(self):
        """Clear the simulation log text area."""
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        self.log("Log cleared.")

    def start_check_docker(self):
        # Start in thread to avoid blocking UI startup
        self.log("Checking Docker...")
        # Temporarily disable launch until check completes
        self.btn_launch.config(state="disabled")
        threading.Thread(target=self._check_docker_thread, daemon=True).start()

    def _check_docker_thread(self):
        try:
            subprocess.run(
                ["docker", "--version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Schedule success on main thread
            self.after(0, self._on_docker_found)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Schedule failure on main thread
            self.after(0, self._on_docker_missing)

    def _on_docker_found(self):
        self.log("Ready. Docker detected.")
        self.docker_available = True
        self.btn_launch.config(state="normal")
        self.tooltip_launch.configure(
            "Start the simulation environment with selected options. (Enter)"
        )

    def _on_docker_missing(self):
        self.docker_available = False
        self.log("Error: Docker not found. Please install Docker Desktop.")
        self.btn_launch.config(state="disabled")
        self.tooltip_launch.configure(
            "Docker not found! Please install Docker Desktop to launch."
        )

    def get_repo_path(self, repo_name):
        # UPDATED: Paths are now relative to Golf_Modeling_Suite root
        return os.path.join(REPOS_ROOT, repo_name)

    def build_docker(self):
        # Build using the Dockerfile in engines/physics_engines/mujoco/docker
        mujoco_repo = self.get_repo_path(MODELS["MuJoCo Golf Model"])
        docker_dir = os.path.join(mujoco_repo, "docker")

        if not os.path.exists(docker_dir):
            messagebox.showerror("Error", f"Docker context not found at {docker_dir}")
            return

        cmd = ["docker", "build", "-t", DOCKER_IMAGE_NAME, "."]

        # UI Guards
        self.btn_launch.config(state="disabled")
        self.btn_build.config(state="disabled")
        self.config(cursor="watch")
        self.log("Building Docker image... (check terminal for details)")

        def run_build():
            try:
                # Open a new console for build output if on Windows, so user can see
                # progress
                creation_flags = 0
                if os.name == "nt":
                    creation_flags = subprocess.CREATE_NEW_CONSOLE

                subprocess.run(
                    cmd, cwd=docker_dir, check=True, creationflags=creation_flags
                )

                self.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Success", "Docker image built successfully!"
                    ),
                )
                self.after(0, lambda: self.log("Build complete."))
            except subprocess.CalledProcessError as err:
                err_msg = str(err)
                self.after(
                    0, lambda: messagebox.showerror("Error", f"Build failed: {err_msg}")
                )
                self.after(0, lambda: self.log("Build failed."))
            finally:
                self.after(0, self._restore_ui_state)

        threading.Thread(target=run_build, daemon=True).start()

    def _restore_ui_state(self):
        self.config(cursor="")
        self.btn_build.config(state="normal")
        if self.docker_available:
            self.btn_launch.config(state="normal")

    def launch(self):
        if not self.docker_available:
            messagebox.showwarning(
                "Docker Missing",
                (
                    "Docker is not detected or the check is still in progress. "
                    "Please wait or install Docker."
                ),
            )
            return

        model_name = self.var_model.get()
        repo_dir = MODELS[model_name]
        abs_repo_path = self.get_repo_path(repo_dir)

        if not os.path.exists(abs_repo_path):
            messagebox.showerror("Error", f"Engine not found: {abs_repo_path}")
            return

        if model_name == "MuJoCo Golf Model":
            self._launch_mujoco_gui(abs_repo_path)
            return

        self._launch_docker_container(model_name, abs_repo_path)

    def _launch_mujoco_gui(self, abs_repo_path):
        # UPDATED: MuJoCo GUI path for new structure
        gui_script = os.path.join(
            abs_repo_path, "docker", "gui", "deepmind_control_suite_MuJoCo_GUI.py"
        )

        self.log("Starting MuJoCo GUI simulation...")
        self.log(f"GUI script path: {gui_script}")

        if not os.path.exists(gui_script):
            error_msg = f"GUI script not found: {gui_script}"
            self.log(f"ERROR: {error_msg}")
            messagebox.showerror("Error", error_msg)
            return

        try:
            # Launch the separate GUI process
            process = subprocess.Popen(
                [sys.executable, gui_script], cwd=os.path.dirname(gui_script)
            )
            self.log(f"MuJoCo GUI launched successfully (PID: {process.pid})")
        except Exception as e:
            error_msg = f"Failed to launch MuJoCo GUI: {e}"
            self.log(f"ERROR: {error_msg}")
            messagebox.showerror("Error", error_msg)

    def _configure_drake(self, cmd):
        """Configure Drake specific Docker arguments."""
        cmd.extend(["-p", "7000-7010:7000-7010"])
        cmd.extend(["-e", "MESHCAT_HOST=0.0.0.0"])
        return 7000

    def _schedule_browser_open(self, host_port):
        """Schedule a browser open for the Meshcat server."""

        def open_browser():
            time.sleep(3)
            with contextlib.suppress(Exception):
                webbrowser.open(f"http://localhost:{host_port}")

        threading.Thread(target=open_browser, daemon=True).start()

    def _launch_docker_container(self, model_name, abs_repo_path):
        # Prepare Docker Command for physics engines
        cmd = ["docker", "run", "--rm", "-it"]

        # Volumes - mount the engine directory to /workspace
        mount_path = abs_repo_path.replace("\\", "/")
        cmd.extend(["-v", f"{mount_path}:/workspace"])
        cmd.extend(["-w", "/workspace"])

        # X11 / Display
        if self.var_live.get():
            if os.name == "nt":
                cmd.extend(["-e", "DISPLAY=host.docker.internal:0"])
                cmd.extend(["-e", "MUJOCO_GL=glfw"])
            else:
                cmd.extend(["-e", f"DISPLAY={os.environ.get('DISPLAY', ':0')}"])
                cmd.extend(["-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw"])  # noqa: S108

        # GPU
        if self.var_gpu.get():
            cmd.append("--gpus=all")

        host_port = None
        if model_name == "Drake Golf Model":
            host_port = self._configure_drake(cmd)

        # Image
        cmd.append(DOCKER_IMAGE_NAME)

        # UPDATED: Engine-specific launch commands for new structure
        if model_name == "Drake Golf Model":
            cmd.extend(["python", "python/src/golf_gui.py"])
            if host_port:
                self._schedule_browser_open(host_port)
        elif model_name == "Pinocchio Golf Model":
            cmd.extend(["python", "python/pinocchio_golf/gui.py"])

        # Log the command being executed
        self.log(f"Starting {model_name} simulation...")
        self.log(f"Command: {' '.join(cmd)}")

        # Launch in terminal
        if os.name == "nt":
            try:
                create_new_console = 0x00000010
                process = subprocess.Popen(
                    ["cmd", "/k", *cmd], creationflags=create_new_console
                )
                self.log(f"Simulation launched in new terminal (PID: {process.pid})")
            except Exception as e:
                self.log(f"ERROR: Failed to launch simulation: {e}")
                messagebox.showerror(
                    "Launch Error", f"Failed to start simulation:\n{e}"
                )
        else:
            try:
                process = subprocess.Popen(cmd)
                self.log(f"Simulation launched (PID: {process.pid})")
            except Exception as e:
                self.log(f"ERROR: Failed to launch simulation: {e}")
                messagebox.showerror(
                    "Launch Error", f"Failed to start simulation:\n{e}"
                )


if __name__ == "__main__":
    app = UnifiedLauncher()
    app.mainloop()
