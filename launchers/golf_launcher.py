#!/usr/bin/env python3
"""
Unified Golf Modeling Suite Launcher (PyQt6)
Features:
- Modern UI with rounded corners.
- Modular Docker Environment Management.
- Integrated Help and Documentation.
"""

import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
REPOS_ROOT = Path(__file__).parent.parent.resolve()
ASSETS_DIR = Path(__file__).parent / "assets"
DOCKER_IMAGE_NAME = "robotics_env"
GRID_COLUMNS = 4

MODELS_DICT = {
    "MuJoCo Humanoid": "engines/physics_engines/mujoco",
    "MuJoCo Dashboard": "engines/physics_engines/mujoco",
    "Drake Golf Model": "engines/physics_engines/drake",
    "Pinocchio Golf Model": "engines/physics_engines/pinocchio",
}

MODEL_IMAGES = {
    "MuJoCo Humanoid": "mujoco_humanoid.png",
    "MuJoCo Dashboard": "mujoco_hand.png",
    "Drake Golf Model": "drake.png",
    "Pinocchio Golf Model": "pinocchio.png",
}

MODEL_DESCRIPTIONS = {
    "MuJoCo Humanoid": "High-fidelity whole-body biomechanics simulation. Features a "
    "23-DOF humanoid model with active muscle sites, ground reaction force (GRF) "
    "visualization, and detailed contact dynamics. Ideal for analyzing kinetic "
    "chains and joint torque generation during the swing.",
    "MuJoCo Dashboard": "Interactive research workbench for comparative analysis. "
    "Switch instantly between double-pendulum, wrist-cocking, and full-body models. "
    "Includes real-time plots for phase space trajectories, energy conservation "
    "verification, and parameter tuning sliders.",
    "Drake Golf Model": "Control-theoretic golf robot focusing on trajectory "
    "optimization."
    "Utilizes Drake's rigorous multibody dynamics and constraint solvers to generate "
    "physically consistent swing paths. Features stabilizing controllers and inverse "
    "dynamics solvers.",
    "Pinocchio Golf Model": "Ultra-fast rigid body dynamics engine based on "
    "Featherstone's spatial algebra."
    " Specialized for rapid iteration and derivative computation. "
    "Validates kinematic chains and provides baseline capabilities for trajectory "
    "optimization algorithms.",
}

DOCKER_STAGES = ["all", "mujoco", "pinocchio", "drake", "base"]


class DockerCheckThread(QThread):
    result = pyqtSignal(bool)

    def run(self):
        try:
            subprocess.run(
                ["docker", "--version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self.result.emit(True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.result.emit(False)


class DockerBuildThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, target_stage="all"):
        super().__init__()
        self.target_stage = target_stage

    def run(self):
        mujoco_path = REPOS_ROOT / MODELS_DICT["MuJoCo Humanoid"]
        # Dockerfile is at engines/physics_engines/mujoco/Dockerfile
        docker_context = mujoco_path

        if not docker_context.exists():
            self.finished_signal.emit(False, f"Path not found: {docker_context}")
            return

        cmd = [
            "docker",
            "build",
            "-t",
            DOCKER_IMAGE_NAME,
            "--target",
            self.target_stage,
            "--progress=plain",  # Force unbuffered output for real-time logs
            ".",
        ]

        self.log_signal.emit(f"Starting build for target: {self.target_stage}")
        self.log_signal.emit(f"Context: {docker_context}")
        self.log_signal.emit(f"Command: {' '.join(cmd)}")

        try:
            # Set environment to disable output buffering
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            process = subprocess.Popen(
                cmd,
                cwd=str(docker_context),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered to ensure real-time output
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            # Read output real-time
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    self.log_signal.emit(line.strip())

            process.wait()

            if process.returncode == 0:
                self.finished_signal.emit(True, "Build successful.")
            else:
                self.finished_signal.emit(
                    False, f"Build failed with code {process.returncode}"
                )

        except Exception as e:
            self.finished_signal.emit(False, str(e))


class EnvironmentDialog(QDialog):
    """Dialog to manage Docker environment and view dependencies."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Environment")
        self.resize(700, 500)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        # --- Build Tab ---
        tab_build = QWidget()
        build_layout = QVBoxLayout(tab_build)

        lbl = QLabel(
            "Rebuild the Docker environment to ensure all dependencies are correct.\n"
            "You can choose a specific target to speed up build time."
        )
        lbl.setWordWrap(True)
        build_layout.addWidget(lbl)

        form = QHBoxLayout()
        form.addWidget(QLabel("Target Stage:"))
        self.combo_stage = QComboBox()
        self.combo_stage.addItems(DOCKER_STAGES)
        form.addWidget(self.combo_stage)
        build_layout.addLayout(form)

        self.btn_build = QPushButton("Build Environment")
        self.btn_build.clicked.connect(self.start_build)
        build_layout.addWidget(self.btn_build)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet(
            "background-color: #1e1e1e; color: #00ff00; font-family: Consolas;"
        )
        build_layout.addWidget(self.console)

        tabs.addTab(tab_build, "Build Docker")

        # --- Dependencies Tab ---
        tab_deps = QWidget()
        dep_layout = QVBoxLayout(tab_deps)

        # Determine dependencies text (mocked logic or reading Dockerfile)
        self.txt_deps = QTextEdit()
        self.txt_deps.setReadOnly(True)
        self.txt_deps.setHtml(
            """
        <h3>Core Dependencies</h3>
        <ul>
            <li><b>MuJoCo:</b> Physics Engine (Latest)</li>
            <li><b>dm_control:</b> DeepMind Control Suite</li>
            <li><b>NumPy / SciPy:</b> Scientific Computing</li>
            <li><b>PyQt6:</b> GUI Framework</li>
        </ul>
        <h3>Robotics Extensions</h3>
        <ul>
            <li><b>Pinocchio:</b> Rigid Body Dynamics</li>
            <li><b>Pink:</b> Inverse Kinematics</li>
            <li><b>Drake:</b> Model-based design (Optional)</li>
            <li><b>Meshcat:</b> Web visualization</li>
        </ul>
        <h3>System</h3>
        <ul>
            <li><b>OpenGL/EGL:</b> Hardware Accelerated Rendering</li>
            <li><b>X11:</b> Display Server Support</li>
        </ul>
        """
        )
        dep_layout.addWidget(self.txt_deps)
        tabs.addTab(tab_deps, "Dependencies")

        # Buttons
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def start_build(self):
        target = self.combo_stage.currentText()
        self.btn_build.setEnabled(False)
        self.console.clear()

        self.build_thread = DockerBuildThread(target)
        self.build_thread.log_signal.connect(self.append_log)
        self.build_thread.finished_signal.connect(self.build_finished)
        self.build_thread.start()

    def append_log(self, text):
        self.console.append(text)
        self.console.moveCursor(self.console.textCursor().MoveOperation.End)

    def build_finished(self, success, msg):
        self.btn_build.setEnabled(True)
        if success:
            QMessageBox.information(self, "Success", msg)
        else:
            QMessageBox.critical(self, "Build Failed", msg)


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Golf Suite - Help")
        self.resize(800, 600)

        layout = QVBoxLayout(self)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

        # Load help content
        help_path = ASSETS_DIR / "help.md"
        if help_path.exists():
            self.text_area.setMarkdown(help_path.read_text(encoding="utf-8"))
        else:
            self.text_area.setText("Help file not found.")

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


class GolfLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Golf Modeling Suite")
        self.resize(1400, 900)

        # Set Icon
        icon_path = ASSETS_DIR / "golf_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # State
        self.docker_available = False
        self.selected_model = None
        self.model_cards = {}

        self.init_ui()
        self.check_docker()

    def init_ui(self):
        # Main Widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # --- Top Bar ---
        top_bar = QHBoxLayout()

        # Status Indicator
        self.lbl_status = QLabel("Checking Docker...")
        self.lbl_status.setStyleSheet("color: #aaaaaa; font-weight: bold;")
        top_bar.addWidget(self.lbl_status)
        top_bar.addStretch()

        btn_env = QPushButton("Manage Environment")
        btn_env.clicked.connect(self.open_environment_manager)
        top_bar.addWidget(btn_env)

        btn_help = QPushButton("Help")
        btn_help.clicked.connect(self.open_help)
        top_bar.addWidget(btn_help)

        main_layout.addLayout(top_bar)

        # --- Model Grid ---
        grid_area = QScrollArea()
        grid_area.setWidgetResizable(True)
        grid_area.setFrameShape(QFrame.Shape.NoFrame)
        grid_area.setStyleSheet("background: transparent;")

        grid_widget = QWidget()
        self.grid_layout = QGridLayout(grid_widget)
        self.grid_layout.setSpacing(20)

        # Populate Grid
        row, col = 0, 0
        for name, _ in MODELS_DICT.items():
            card = self.create_model_card(name)
            self.model_cards[name] = card
            self.grid_layout.addWidget(card, row, col)
            col += 1
            if col >= GRID_COLUMNS:
                col = 0
                row += 1

        grid_area.setWidget(grid_widget)
        main_layout.addWidget(grid_area)

        # --- Configuration & Launch ---
        bottom_bar = QFrame()
        bottom_bar.setObjectName("BottomBar")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(20, 20, 20, 20)

        # Options
        opts_layout = QVBoxLayout()
        self.chk_live = QCheckBox("Live Visualization")
        self.chk_live.setChecked(True)
        self.chk_live.setToolTip("Enable VcXsrv/X11 display")

        self.chk_gpu = QCheckBox("GPU Acceleration")
        self.chk_gpu.setChecked(False)
        self.chk_gpu.setToolTip("Requires NVIDIA Container Toolkit")

        opts_layout.addWidget(self.chk_live)
        opts_layout.addWidget(self.chk_gpu)
        bottom_layout.addLayout(opts_layout)

        bottom_layout.addStretch()

        # Launch Button
        self.btn_launch = QPushButton("LAUNCH SIMULATION")
        self.btn_launch.setObjectName("LaunchButton")
        self.btn_launch.setFixedHeight(50)
        self.btn_launch.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.btn_launch.clicked.connect(self.launch_simulation)
        self.btn_launch.setEnabled(False)  # Wait for selection and docker
        bottom_layout.addWidget(self.btn_launch)

        main_layout.addWidget(bottom_bar)

        # --- Styling ---
        self.apply_styles()

        # Select first model by default
        self.select_model("MuJoCo Humanoid")

    def create_model_card(self, name):
        """Creates a clickable card widget."""
        card = QFrame()
        card.setObjectName("ModelCard")
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.mousePressEvent = lambda e: self.select_model(name)
        card.mouseDoubleClickEvent = lambda e: self.launch_model_direct(name)

        layout = QVBoxLayout(card)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Image
        img_name = MODEL_IMAGES.get(name)
        img_path = ASSETS_DIR / img_name if img_name else None

        lbl_img = QLabel()
        # Fixed card thumbnail area: 200x200 matches the model-card layout spec
        # and provides padding around the 180x180 scaled image.
        lbl_img.setFixedSize(200, 200)
        lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if img_path and img_path.exists():
            pixmap = QPixmap(str(img_path))
            pixmap = pixmap.scaled(
                180,
                180,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            lbl_img.setPixmap(pixmap)
        else:
            lbl_img.setText("No Image")
            lbl_img.setStyleSheet("color: #666; font-style: italic;")

        layout.addWidget(lbl_img)

        # Label
        lbl_name = QLabel(name)
        lbl_name.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        lbl_name.setWordWrap(True)
        lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_name)

        # Description
        desc_text = MODEL_DESCRIPTIONS.get(name, "")
        lbl_desc = QLabel(desc_text)
        lbl_desc.setFont(QFont("Segoe UI", 9))
        lbl_desc.setStyleSheet("color: #cccccc;")
        lbl_desc.setWordWrap(True)
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_desc)

        return card

    def launch_model_direct(self, name):
        """Selects and immediately launches the model (for double-click)."""
        self.select_model(name)
        if self.btn_launch.isEnabled():
            self.launch_simulation()

    def select_model(self, name):
        self.selected_model = name

        # Update Styles
        for model_name, card in self.model_cards.items():
            if model_name == name:
                card.setStyleSheet(
                    """
                    QFrame#ModelCard {
                        background-color: #333333;
                        border: 2px solid #007acc;
                        border-radius: 10px;
                    }
                """
                )
            else:
                card.setStyleSheet(
                    """
                    QFrame#ModelCard {
                        background-color: #252526;
                        border: 1px solid #3e3e42;
                        border-radius: 10px;
                    }
                    QFrame#ModelCard:hover {
                        border: 1px solid #555555;
                        background-color: #2d2d30;
                    }
                """
                )

        self.update_launch_button()

    def update_launch_button(self):
        if self.docker_available and self.selected_model:
            self.btn_launch.setEnabled(True)
            self.btn_launch.setText(f"LAUNCH {self.selected_model.upper()}")
        elif not self.docker_available:
            self.btn_launch.setEnabled(False)
            self.btn_launch.setText("DOCKER NOT FOUND")
        else:
            self.btn_launch.setEnabled(False)
            self.btn_launch.setText("SELECT A MODEL")

    def apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: "Segoe UI";
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0098ff;
            }
            QPushButton:pressed {
                background-color: #005c99;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #888888;
            }
            QFrame#BottomBar {
                background-color: #252526;
                border-top: 1px solid #3e3e42;
                border-radius: 0px;
            }
            QPushButton#LaunchButton {
                background-color: #28a745;
                font-size: 14px;
            }
            QPushButton#LaunchButton:hover {
                background-color: #34ce57;
            }
            QCheckBox {
                color: #dddddd;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 1px solid #666;
            }
            QCheckBox::indicator:checked {
                background-color: #007acc;
                border-color: #007acc;
            }
            QTabWidget::pane {
                border: 1px solid #3e3e42;
                border-radius: 5px;
            }
            QTabBar::tab {
                background: #252526;
                color: #ccc;
                padding: 10px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #333;
                color: white;
                border-bottom: 2px solid #007acc;
            }
            QComboBox {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 5px;
            }
        """
        )

    def check_docker(self):
        self.check_thread = DockerCheckThread()
        self.check_thread.result.connect(self.on_docker_check_complete)
        self.check_thread.start()

    def on_docker_check_complete(self, available):
        self.docker_available = available
        if available:
            self.lbl_status.setText("● System Ready")
            self.lbl_status.setStyleSheet("color: #28a745; font-weight: bold;")
        else:
            self.lbl_status.setText("● Docker Not Found")
            self.lbl_status.setStyleSheet("color: #dc3545; font-weight: bold;")
            QMessageBox.warning(
                self,
                "Docker Missing",
                "Docker is required to run simulations.\n"
                "Please install Docker Desktop.",
            )
        self.update_launch_button()

    def open_help(self):
        dlg = HelpDialog(self)
        dlg.exec()

    def open_environment_manager(self):
        dlg = EnvironmentDialog(self)
        dlg.exec()

    def launch_simulation(self):
        if not self.selected_model:
            return

        model_name = self.selected_model
        repo_rel_path = MODELS_DICT[model_name]
        abs_repo_path = REPOS_ROOT / repo_rel_path

        if not abs_repo_path.exists():
            QMessageBox.critical(self, "Error", f"Path not found: {abs_repo_path}")
            return

        # Engine-specific launch logic
        try:
            custom_launchers = {
                "MuJoCo Humanoid": self._custom_launch_humanoid,
                "MuJoCo Dashboard": self._custom_launch_comprehensive,
            }

            launcher = custom_launchers.get(model_name)
            if launcher:
                launcher(abs_repo_path)
            else:
                self._launch_docker_container(model_name, abs_repo_path)
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", str(e))

    def _custom_launch_humanoid(self, abs_repo_path):
        script = abs_repo_path / "python/humanoid_launcher.py"
        if not script.exists():
            raise FileNotFoundError(
                f"Script not found: {script}. "
                "Ensure the MuJoCo engine and selected model repository "
                "are properly installed."
            )

        logger.info(f"Launching Humanoid GUI: {script}")
        subprocess.Popen([sys.executable, str(script)], cwd=script.parent)

    def _custom_launch_comprehensive(self, abs_repo_path):
        python_dir = abs_repo_path / "python"
        logger.info(f"Launching Comprehensive GUI from {python_dir}")
        subprocess.Popen([sys.executable, "-m", "mujoco_humanoid_golf"], cwd=python_dir)

    def _launch_docker_container(self, model_name, abs_repo_path):
        cmd = ["docker", "run", "--rm", "-it"]

        # Volumes
        mount_path = str(abs_repo_path).replace("\\", "/")
        cmd.extend(["-v", f"{mount_path}:/workspace"])
        cmd.extend(["-w", "/workspace/python"])

        # Display/X11
        if self.chk_live.isChecked():
            if os.name == "nt":
                cmd.extend(["-e", "DISPLAY=host.docker.internal:0"])
                cmd.extend(["-e", "MUJOCO_GL=glfw"])
                cmd.extend(["-e", "LIBGL_ALWAYS_INDIRECT=1"])
            else:
                cmd.extend(["-e", f"DISPLAY={os.environ.get('DISPLAY', ':0')}"])
                cmd.extend(["-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw"])

        # GPU
        if self.chk_gpu.isChecked():
            cmd.append("--gpus=all")

        # Network for Meshcat (Drake/Pinocchio)
        host_port = None
        if "Drake" in model_name or "Pinocchio" in model_name:
            cmd.extend(["-p", "7000-7010:7000-7010"])
            cmd.extend(["-e", "MESHCAT_HOST=0.0.0.0"])
            host_port = 7000

        cmd.append(DOCKER_IMAGE_NAME)

        # Entry Command
        if "Drake" in model_name:
            # Run as module for relative imports (workdir is now /workspace/python)
            # FIX: Use drake_gui_app instead of the empty golf_gui
            cmd.extend(["/opt/mujoco-env/bin/python", "-m", "src.drake_gui_app"])

            if host_port:
                logger.info(f"Drake Meshcat will be available on host port {host_port}")
                self._start_meshcat_browser(host_port)

        elif "Pinocchio" in model_name:
            # Run from python dir
            cmd.extend(["/opt/mujoco-env/bin/python", "pinocchio_golf/gui.py"])

            if host_port:
                logger.info(
                    f"Pinocchio Meshcat will be available on host port {host_port}"
                )
                self._start_meshcat_browser(host_port)

        logger.info(f"Final Docker Command: {' '.join(cmd)}")

        # Launch in Terminal
        if os.name == "nt":
            # Add a diagnostic echo and pause to the terminal so users can see errors
            diagnostic_cmd = [
                "cmd",
                "/k",
                "echo Launching simulation container... && echo Command: "
                + " ".join(cmd)
                + " && "
                + " ".join(cmd),
            ]
            logger.info("Starting new console for simulation...")
            subprocess.Popen(
                diagnostic_cmd, creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            subprocess.Popen(cmd)

    def _start_meshcat_browser(self, port):
        def open_url():
            time.sleep(3)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=open_url, daemon=True).start()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Global Font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = GolfLauncher()
    window.show()
    sys.exit(app.exec())
