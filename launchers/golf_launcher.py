#!/usr/bin/env python3
"""
Unified Golf Modeling Suite Launcher (PyQt6)
Features:
- Modern UI with rounded corners.
- Modular Docker Environment Management.
- Integrated Help and Documentation.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QMimeData, QPoint, Qt, QThread, pyqtSignal
from PyQt6.QtGui import (
    QCloseEvent,
    QDrag,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QMouseEvent,
    QPixmap,
)
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

from shared.python.engine_manager import EngineManager, EngineType
from shared.python.model_registry import ModelRegistry

# Windows-specific subprocess constants
CREATE_NO_WINDOW: int
CREATE_NEW_CONSOLE: int

if os.name == "nt":
    try:
        CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        CREATE_NEW_CONSOLE = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    except AttributeError:
        CREATE_NO_WINDOW = 0x08000000
        CREATE_NEW_CONSOLE = 0x00000010
else:
    CREATE_NO_WINDOW = 0
    CREATE_NEW_CONSOLE = 0

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
REPOS_ROOT = Path(__file__).parent.parent.resolve()
ASSETS_DIR = Path(__file__).parent / "assets"
CONFIG_DIR = REPOS_ROOT / ".kiro" / "launcher"
LAYOUT_CONFIG_FILE = CONFIG_DIR / "layout.json"
DOCKER_IMAGE_NAME = "robotics_env"
GRID_COLUMNS = 4  # Changed to 3x4 grid (12 tiles total)

MODEL_IMAGES = {
    "MuJoCo Humanoid": "mujoco_humanoid.png",
    "MuJoCo Dashboard": "mujoco_hand.png",
    "Drake Golf Model": "drake.png",
    "Pinocchio Golf Model": "pinocchio.png",
    "OpenSim Golf": "openpose.jpg",
    "MyoSim Suite": "myosim.png",
    "OpenPose Analysis": "opensim.png",
    "Matlab Simscape": "simscape_multibody.png",
    "URDF Generator": "urdf_icon.png",
    "C3D Motion Viewer": "c3d_icon.png",  # Add C3D viewer icon
}

DOCKER_STAGES = ["all", "mujoco", "pinocchio", "drake", "base"]


class DraggableModelCard(QFrame):
    """Draggable model card widget with reordering support."""

    def __init__(self, model: Any, parent: Any = None):
        # Only pass parent to QFrame if it's a proper QWidget, otherwise use None
        # Check if parent is actually a QWidget by checking its type
        from unittest.mock import Mock

        from PyQt6.QtWidgets import QWidget

        if parent is None or isinstance(parent, Mock):
            qt_parent = None
        elif isinstance(parent, QWidget):
            qt_parent = parent
        else:
            qt_parent = None

        super().__init__(qt_parent)
        self.model = model
        self.parent_launcher = parent
        self.setAcceptDrops(True)
        self.setObjectName("ModelCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.drag_start_position = QPoint()
        self.setup_ui()

    def setup_ui(self) -> None:
        """Set up the card UI."""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Image
        img_name = MODEL_IMAGES.get(self.model.name)
        if not img_name:
            # Try to guess based on ID
            if "mujoco" in self.model.id:
                img_name = "mujoco_humanoid.png"
            elif "drake" in self.model.id:
                img_name = "drake.png"
            elif "pinocchio" in self.model.id:
                img_name = "pinocchio.png"
            elif "urdf" in self.model.id:
                img_name = "urdf_icon.png"
            elif "c3d" in self.model.id:
                img_name = "c3d_icon.png"

        img_path = ASSETS_DIR / img_name if img_name else None

        lbl_img = QLabel()
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
        lbl_name = QLabel(self.model.name)
        lbl_name.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        lbl_name.setWordWrap(True)
        lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_name)

        # Description
        lbl_desc = QLabel(self.model.description)
        lbl_desc.setFont(QFont("Segoe UI", 9))
        lbl_desc.setStyleSheet("color: #cccccc;")
        lbl_desc.setWordWrap(True)
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_desc)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        """Handle mouse press for selection and drag initiation."""
        if event and (
            event.button() == Qt.MouseButton.LeftButton or event.button() == 1
        ):
            self.drag_start_position = event.position().toPoint()
            if self.parent_launcher:
                self.parent_launcher.select_model(self.model.id)
        # Don't call super() as QLabel doesn't have mousePressEvent

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        """Handle mouse move for drag operation."""
        if not event or not (event.buttons() & Qt.MouseButton.LeftButton):
            return

        if (
            not hasattr(self, "drag_start_position")
            or self.drag_start_position.isNull()
        ):
            return

        if (
            event.position().toPoint() - self.drag_start_position
        ).manhattanLength() < QApplication.startDragDistance():
            return

        try:
            # Start drag operation
            drag = QDrag(self)
            mimeData = QMimeData()
            mimeData.setText(f"model_card:{self.model.id}")
            drag.setMimeData(mimeData)

            # Create drag pixmap (simplified to avoid painter issues)
            pixmap = self.grab()
            drag.setPixmap(pixmap)
            drag.setHotSpot(self.drag_start_position)

            # Execute drag
            drag.exec(Qt.DropAction.MoveAction)
        except Exception as e:
            logger.error(f"Drag operation failed: {e}")

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:
        """Handle double-click to launch model."""
        if self.parent_launcher:
            self.parent_launcher.launch_model_direct(self.model.id)
        # Don't call super() as QLabel doesn't have mouseDoubleClickEvent

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        """Handle drag enter event."""
        if event:
            mime_data = event.mimeData()
            if (
                mime_data
                and mime_data.hasText()
                and mime_data.text().startswith("model_card:")
            ):
                event.acceptProposedAction()
            else:
                event.ignore()

    def dropEvent(self, event: QDropEvent | None) -> None:
        """Handle drop event."""
        if event:
            mime_data = event.mimeData()
            if (
                mime_data
                and mime_data.hasText()
                and mime_data.text().startswith("model_card:")
            ):
                source_model_id = mime_data.text().split(":", 1)[1]
                if self.parent_launcher and source_model_id != self.model.id:
                    self.parent_launcher._swap_models(source_model_id, self.model.id)
                event.acceptProposedAction()
            else:
                event.ignore()


class DockerCheckThread(QThread):
    result = pyqtSignal(bool)

    def run(self) -> None:
        """Run docker check."""
        try:
            subprocess.run(
                ["docker", "--version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
                creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self.result.emit(True)
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            self.result.emit(False)


class DockerBuildThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, target_stage: str = "all") -> None:
        """Initialize the build thread."""
        super().__init__()
        self.target_stage = target_stage

    def run(self) -> None:
        """Run the docker build command."""
        # Assume MuJoCo path for context as it's the primary engine root
        mujoco_path = REPOS_ROOT / "engines/physics_engines/mujoco"
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
                creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            # Read output real-time
            while True:
                if process.stdout is not None:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        self.log_signal.emit(line.strip())
                else:
                    if process.poll() is not None:
                        break

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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage Environment")
        self.resize(700, 500)
        self.setup_ui()

    def setup_ui(self) -> None:
        """Setup the UI components."""
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

    def start_build(self) -> None:
        """Start the docker build process."""
        target = self.combo_stage.currentText()
        self.btn_build.setEnabled(False)
        self.console.clear()

        self.build_thread = DockerBuildThread(target)
        self.build_thread.log_signal.connect(self.append_log)
        self.build_thread.finished_signal.connect(self.build_finished)
        self.build_thread.start()

    def append_log(self, text: str) -> None:
        """Append text to the log console."""
        self.console.append(text)
        self.console.moveCursor(self.console.textCursor().MoveOperation.End)

    def build_finished(self, success: bool, msg: str) -> None:
        """Handle build completion."""
        self.btn_build.setEnabled(True)
        if success:
            QMessageBox.information(self, "Success", msg)
        else:
            QMessageBox.critical(self, "Build Failed", msg)


class HelpDialog(QDialog):
    """Dialog to display help documentation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the help dialog."""
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
    """Main application window for the launcher."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Golf Modeling Suite - GolfingRobot")
        self.resize(1400, 900)

        # Set Icon - Use Windows-optimized icon for maximum clarity on Windows
        icon_candidates = [
            ASSETS_DIR
            / "golf_robot_windows_optimized.png",  # Windows-optimized (best for Windows)
            ASSETS_DIR / "golf_robot_ultra_sharp.png",  # Ultra-sharp version
            ASSETS_DIR / "golf_robot_cropped_icon.png",  # Cropped version
            ASSETS_DIR / "golf_robot_icon.png",  # High-quality standard
            ASSETS_DIR / "golf_icon.png",  # Original fallback
        ]

        icon_loaded = False
        for icon_path in icon_candidates:
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
                logger.info("Loaded icon: %s", icon_path.name)
                icon_loaded = True
                break

        if not icon_loaded:
            logger.warning("No icon files found")

        # State
        self.docker_available = False
        self.selected_model: str | None = None
        self.model_cards: dict[str, Any] = {}
        self.model_order: list[str] = []  # Track model order for drag-and-drop

        # Load Registry
        try:
            self.registry: ModelRegistry | None = ModelRegistry(
                REPOS_ROOT / "config/models.yaml"
            )
        except ImportError:
            logger.error("Failed to import ModelRegistry. Registry unavailable.")
            self.registry = None

        # Engine Manager for local discovery
        self.engine_manager: Any = None
        try:
            self.engine_manager = EngineManager(REPOS_ROOT)
        except Exception as e:
            logger.warning(f"Failed to initialize EngineManager: {e}")
            self.engine_manager = None

        self.init_ui()
        self.check_docker()

        # Load saved layout
        self._load_layout()

    def _save_layout(self) -> None:
        """Save the current model layout to configuration file."""
        try:
            # Ensure config directory exists
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)

            layout_data = {
                "model_order": self.model_order,
                "selected_model": self.selected_model,
                "window_geometry": {
                    "x": self.x(),
                    "y": self.y(),
                    "width": self.width(),
                    "height": self.height(),
                },
                "options": {
                    "live_visualization": (
                        self.chk_live.isChecked() if hasattr(self, "chk_live") else True
                    ),
                    "gpu_acceleration": (
                        self.chk_gpu.isChecked() if hasattr(self, "chk_gpu") else False
                    ),
                },
            }

            with open(LAYOUT_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(layout_data, f, indent=2)

            logger.info(f"Layout saved to {LAYOUT_CONFIG_FILE}")

        except Exception as e:
            logger.error(f"Failed to save layout: {e}")

    def _load_layout(self) -> None:
        """Load the saved model layout from configuration file."""
        try:
            if not LAYOUT_CONFIG_FILE.exists():
                logger.info("No saved layout found, using default")
                return

            with open(LAYOUT_CONFIG_FILE, encoding="utf-8") as f:
                layout_data = json.load(f)

            # Restore model order if valid
            saved_order = layout_data.get("model_order", [])
            if saved_order and len(saved_order) == len(self.model_order):
                # Verify all models still exist
                if all(model_id in self.model_cards for model_id in saved_order):
                    self.model_order = saved_order
                    self._rebuild_grid() # Use _rebuild_grid as it exists
                    logger.info("Model layout restored from saved configuration")

            # Restore window geometry
            geo = layout_data.get("window_geometry", {})
            if geo:
                # Ensure window title bar is visible (y >= 30)
                # And center if it looks weird
                x = geo.get("x", 100)
                y = geo.get("y", 100)
                w = geo.get("width", 1280)
                h = geo.get("height", 800)
                
                # Clamp Y to avoid being off-screen top
                if y < 30:
                    y = 50
                
                self.setGeometry(x, y, w, h)
            else:
                self._center_window()

            # Restore options
            options = layout_data.get("options", {})
            if hasattr(self, "chk_live"):
                self.chk_live.setChecked(options.get("live_visualization", True))
            if hasattr(self, "chk_gpu"):
                self.chk_gpu.setChecked(options.get("gpu_acceleration", False))

            # Restore selected model
            saved_selection = layout_data.get("selected_model")
            if saved_selection and saved_selection in self.model_cards:
                self.select_model(saved_selection)

            self._rebuild_grid() # Use _rebuild_grid as it exists
            logger.info("Layout loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load layout: {e}")
            self._center_window()

    def _center_window(self) -> None:
        """Center the window on the primary screen."""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            w = self.width() if self.width() > 100 else 1280
            h = self.height() if self.height() > 100 else 800
            
            x = screen_geo.x() + (screen_geo.width() - w) // 2
            y = screen_geo.y() + (screen_geo.height() - h) // 2
            
            # Ensure not too high
            y = max(y, 50)
            
            self.setGeometry(x, y, w, h)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Handle window close event to save layout."""
        self._save_layout()
        super().closeEvent(event)

    def init_ui(self) -> None:
        """Initialize the user interface."""
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

        # Populate Grid - 3x4 layout with URDF generator and C3D viewer
        row, col = 0, 0
        if self.registry:
            # Get first 10 models plus add URDF generator and C3D viewer as 11th and 12th
            models = self.registry.get_all_models()[:10]

            # Add URDF generator as a special model
            urdf_model = type(
                "URDFModel",
                (),
                {
                    "id": "urdf_generator",
                    "name": "URDF Generator",
                    "description": "Interactive URDF model builder",
                    "type": "urdf_generator",
                },
            )()
            models.append(urdf_model)

            # Add C3D viewer as a special model
            c3d_model = type(
                "C3DModel",
                (),
                {
                    "id": "c3d_viewer",
                    "name": "C3D Motion Viewer",
                    "description": "C3D motion capture file viewer and analyzer",
                    "type": "c3d_viewer",
                },
            )()
            models.append(c3d_model)

            for model in models:
                card = DraggableModelCard(model, self)
                self.model_cards[model.id] = card  # Use ID as key
                self.model_order.append(model.id)  # Track order
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

        # Select first model by default if available
        if self.registry:
            models = self.registry.get_all_models()
            if models:
                # Prefer MuJoCo Humanoid if available
                humanoid = next(
                    (m for m in models if m.name == "MuJoCo Humanoid"), None
                )
                if humanoid:
                    self.select_model(humanoid.id)
                else:
                    self.select_model(models[0].id)

    def _swap_models(self, source_id: str, target_id: str) -> None:
        """Swap two models in the grid layout."""
        if source_id not in self.model_cards or target_id not in self.model_cards:
            return

        # Find positions in the order list
        source_idx = self.model_order.index(source_id)
        target_idx = self.model_order.index(target_id)

        # Swap in order list
        self.model_order[source_idx], self.model_order[target_idx] = (
            self.model_order[target_idx],
            self.model_order[source_idx],
        )

        # Rebuild the grid layout
        self._rebuild_grid()

        # Save layout after swap
        self._save_layout()

        logger.info(f"Swapped models: {source_id} <-> {target_id}")

    def _rebuild_grid(self) -> None:
        """Rebuild the grid layout based on current model order."""
        # Clear the layout
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.itemAt(i)
            if item:
                child = item.widget()
                if child:
                    self.grid_layout.removeWidget(child)

        # Re-add widgets in new order
        row, col = 0, 0
        for model_id in self.model_order:
            if model_id in self.model_cards:
                card = self.model_cards[model_id]
                self.grid_layout.addWidget(card, row, col)
                col += 1
                if col >= GRID_COLUMNS:
                    col = 0
                    row += 1

    def create_model_card(self, model: Any) -> QFrame:
        """Creates a clickable card widget."""
        name = model.name
        model_id = model.id
        card = QFrame()
        card.setObjectName("ModelCard")
        card.setCursor(Qt.CursorShape.PointingHandCursor)

        # Create proper event handlers instead of assigning to methods
        def handle_mouse_press(e: Any) -> None:
            self.select_model(model_id)

        def handle_mouse_double_click(e: Any) -> None:
            self.launch_model_direct(model_id)

        card.mousePressEvent = handle_mouse_press  # type: ignore[assignment]
        card.mouseDoubleClickEvent = handle_mouse_double_click  # type: ignore[assignment]

        layout = QVBoxLayout(card)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Image
        # Temporary fallback mapping until images are in registry
        img_name = MODEL_IMAGES.get(name)
        if not img_name:
            # Try to guess based on ID
            if "mujoco" in model.id:
                img_name = "mujoco_humanoid.png"
            elif "drake" in model.id:
                img_name = "drake.png"
            elif "pinocchio" in model.id:
                img_name = "pinocchio.png"

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
        desc_text = model.description
        lbl_desc = QLabel(desc_text)
        lbl_desc.setFont(QFont("Segoe UI", 9))
        lbl_desc.setStyleSheet("color: #cccccc;")
        lbl_desc.setWordWrap(True)
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_desc)

        return card

    def launch_model_direct(self, model_id: str) -> None:
        """Selects and immediately launches the model (for double-click)."""
        self.select_model(model_id)
        if model_id == "urdf_generator":
            self._launch_urdf_generator()
        elif model_id == "c3d_viewer":
            self._launch_c3d_viewer()
        elif self.btn_launch.isEnabled():
            self.launch_simulation()

    def _launch_urdf_generator(self) -> None:
        """Launch the URDF generator application."""
        try:
            import subprocess
            import sys
            from pathlib import Path

            # Path to URDF generator
            suite_root = Path(__file__).parent.parent
            urdf_script = (
                suite_root / "tools" / "urdf_generator" / "launch_urdf_generator.py"
            )

            if not urdf_script.exists():
                QMessageBox.warning(
                    self,
                    "URDF Generator Not Found",
                    f"URDF Generator script not found at:\n{urdf_script}\n\n"
                    "Please ensure the URDF generator is properly installed.",
                )
                return

            logger.info("Launching URDF Generator...")

            # Launch in new process
            if os.name == "nt":
                # Windows: Launch in new console
                subprocess.Popen(
                    [sys.executable, str(urdf_script)],
                    creationflags=CREATE_NEW_CONSOLE,
                    cwd=str(suite_root),
                )
            else:
                # Unix: Launch in background
                subprocess.Popen(
                    [sys.executable, str(urdf_script)], cwd=str(suite_root)
                )

        except Exception as e:
            logger.error(f"Failed to launch URDF Generator: {e}")
            QMessageBox.critical(
                self, "Launch Error", f"Failed to launch URDF Generator:\n{e}"
            )

    def _launch_c3d_viewer(self) -> None:
        """Launch the C3D motion viewer application."""
        try:
            import subprocess
            import sys
            from pathlib import Path

            # Path to C3D viewer
            suite_root = Path(__file__).parent.parent
            c3d_script = (
                suite_root
                / "engines"
                / "Simscape_Multibody_Models"
                / "3D_Golf_Model"
                / "python"
                / "src"
                / "apps"
                / "c3d_viewer.py"
            )

            if not c3d_script.exists():
                QMessageBox.warning(
                    self,
                    "C3D Viewer Not Found",
                    f"C3D Viewer script not found at:\n{c3d_script}\n\n"
                    "Please ensure the C3D viewer is properly installed.",
                )
                return

            logger.info("Launching C3D Motion Viewer...")

            # Launch in new process
            if os.name == "nt":
                # Windows: Launch in new console
                subprocess.Popen(
                    [sys.executable, str(c3d_script)],
                    creationflags=CREATE_NEW_CONSOLE,
                    cwd=str(c3d_script.parent),
                )
            else:
                # Unix: Launch in background
                subprocess.Popen(
                    [sys.executable, str(c3d_script)], cwd=str(c3d_script.parent)
                )

        except Exception as e:
            logger.error(f"Failed to launch C3D Viewer: {e}")
            QMessageBox.critical(
                self, "Launch Error", f"Failed to launch C3D Viewer:\n{e}"
            )

    def select_model(self, model_id: str) -> None:
        """Select a model and update UI."""
        self.selected_model = model_id

        # Get model name for display
        model_name = model_id
        if self.registry:
            model = self.registry.get_model(model_id)
            if model:
                model_name = model.name

        # Update Styles for draggable cards
        for m_id, card in self.model_cards.items():
            if m_id == model_id:
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

        self.update_launch_button(model_name)

    def update_launch_button(self, model_name: str | None = None) -> None:
        """Update the launch button state."""
        model_type = None
        if self.selected_model and self.registry:
            model = self.registry.get_model(self.selected_model)
            if model:
                model_name = model.name
                model_type = model.type

        if not self.selected_model:
            self.btn_launch.setEnabled(False)
            self.btn_launch.setText("SELECT A MODEL")
            return

        # Check local availability first
        is_local = False
        if self.engine_manager and model_type:
            engine_type = self._get_engine_type(model_type)
            if engine_type:
                probe = self.engine_manager.probes.get(engine_type)
                if probe and probe.is_available():
                    is_local = True

        if is_local:
            self.btn_launch.setEnabled(True)
            self.btn_launch.setText(f"LAUNCH {str(model_name).upper()} (LOCAL)")
            self.btn_launch.setStyleSheet("background-color: #28a745; color: white;")
        elif self.docker_available:
            self.btn_launch.setEnabled(True)
            self.btn_launch.setText(f"LAUNCH {str(model_name).upper()} (DOCKER)")
            self.btn_launch.setStyleSheet("background-color: #007acc; color: white;")
        else:
            self.btn_launch.setEnabled(False)
            self.btn_launch.setText("ENGINE NOT FOUND (LOCAL OR DOCKER)")
            self.btn_launch.setStyleSheet("background-color: #444444; color: #888888;")

    def _get_engine_type(self, model_type: str) -> EngineType | None:
        """Map model type to EngineType."""
        if "humanoid" in model_type or "mujoco" in model_type:
            return EngineType.MUJOCO
        elif "drake" in model_type:
            return EngineType.DRAKE
        elif "pinocchio" in model_type:
            return EngineType.PINOCCHIO
        elif "opensim" in model_type:
            return EngineType.OPENSIM
        return None

    def apply_styles(self) -> None:
        """Apply custom stylesheets."""
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

    def check_docker(self) -> None:
        """Start the docker check thread."""
        self.check_thread = DockerCheckThread()
        self.check_thread.result.connect(self.on_docker_check_complete)
        self.check_thread.start()

    def on_docker_check_complete(self, available: bool) -> None:
        """Handle docker check result."""
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

    def open_help(self) -> None:
        """Open the help dialog."""
        dlg = HelpDialog(self)
        dlg.exec()

    def open_environment_manager(self) -> None:
        """Open the environment manager dialog."""
        dlg = EnvironmentDialog(self)
        dlg.exec()

    def launch_simulation(self) -> None:
        """Launch the selected simulation."""
        if not self.selected_model:
            return

        model_id = self.selected_model

        # Handle special applications
        if model_id == "urdf_generator":
            self._launch_urdf_generator()
            return
        elif model_id == "c3d_viewer":
            self._launch_c3d_viewer()
            return

        if not self.registry:
            QMessageBox.critical(self, "Error", "Model registry is unavailable.")
            return

        model = self.registry.get_model(model_id)
        if not model:
            QMessageBox.critical(
                self, "Error", f"Model not found in registry: {model_id}"
            )
            return

        path = REPOS_ROOT / model.path

        if not path or not path.exists():
            QMessageBox.critical(self, "Error", f"Path not found: {path}")
            return

        # Determine execution mode (Local vs Docker)
        is_local_fit = False
        if self.engine_manager:
            engine_type = self._get_engine_type(model.type)
            # If we don't know the engine type, we assume it's a generic file launch which effectively relies on local util
            if not engine_type:
                # Generic XML/MJCF usually implies local viewer
                is_local_fit = True
            else:
                probe = self.engine_manager.probes.get(engine_type)
                if probe and probe.is_available():
                    is_local_fit = True

        # Override: If User manually selected Docker? (For now, we prioritize Local if available)
        # However, checking 'is_local_fit' allows us to Fallback to Docker if Local is broken.

        launch_locally = is_local_fit

        try:
            if launch_locally:
                if model.type == "custom_humanoid":
                    self._custom_launch_humanoid(path)
                elif model.type == "custom_dashboard":
                    self._custom_launch_comprehensive(path)
                elif model.type == "mjcf" or str(path).endswith(".xml"):
                    self._launch_generic_mjcf(path)
                else:
                    self._launch_docker_container(model, path)
            else:
                # Force Docker Launch
                self._launch_docker_container(model, path)
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", str(e))

    def _launch_generic_mjcf(self, path: Path) -> None:
        """Launch generic MJCF file in passive viewer."""
        logger.info(f"Launching generic MJCF: {path}")
        try:
            # Use the python executable to run a simple viewer script or module
            # Creating a temporary script or using -c
            cmd = [
                sys.executable,
                "-c",
                f"import mujoco; import mujoco.viewer; m=mujoco.MjModel.from_xml_path(r'{str(path)}'); mujoco.viewer.launch(m)",
            ]
            subprocess.Popen(cmd)
        except Exception as e:
            QMessageBox.critical(self, "Viewer Error", str(e))

    def _custom_launch_humanoid(self, abs_repo_path: Path) -> None:
        """Launch the humanoid GUI directly."""
        script = abs_repo_path / "python/humanoid_launcher.py"
        if not script.exists():
            raise FileNotFoundError(
                f"Script not found: {script}. "
                "Ensure the MuJoCo engine and selected model repository "
                "are properly installed."
            )

        logger.info(f"Launching Humanoid GUI: {script}")
        subprocess.Popen([sys.executable, str(script)], cwd=script.parent)

    def _custom_launch_comprehensive(self, abs_repo_path: Path) -> None:
        """Launch the comprehensive dashboard directly."""
        python_dir = abs_repo_path / "python"
        logger.info(f"Launching Comprehensive GUI from {python_dir}")
        subprocess.Popen([sys.executable, "-m", "mujoco_humanoid_golf"], cwd=python_dir)

    def _launch_docker_container(self, model: Any, abs_repo_path: Path) -> None:
        """Launch the simulation in a docker container."""
        # Parts of the command
        docker_base = ["docker", "run", "--rm", "-it"]
        run_flags = []
        exec_cmd = []

        # Volumes - mount entire suite root to /workspace
        suite_root = Path(__file__).parent.parent
        mount_path = str(suite_root).replace("\\", "/")
        run_flags.extend(["-v", f"{mount_path}:/workspace"])

        # Environment variables for Python path and shared modules
        run_flags.extend(
            ["-e", "PYTHONPATH=/workspace:/workspace/shared/python:/workspace/engines"]
        )

        # Mount 'shared' directory so that scripts can import shared modules
        shared_host_path = suite_root / "shared"
        if shared_host_path.exists():
            mount_shared_str = str(shared_host_path).replace("\\", "/")
            run_flags.extend(["-v", f"{mount_shared_str}:/shared"])

        # Display/X11
        if self.chk_live.isChecked():
            if os.name == "nt":
                run_flags.extend(["-e", "DISPLAY=host.docker.internal:0"])
                run_flags.extend(["-e", "MUJOCO_GL=glfw"])
            else:
                run_flags.extend(["-e", f"DISPLAY={os.environ.get('DISPLAY', ':0')}"])
                run_flags.extend(["-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw"])
        else:
            # Force headless OSMesa rendering if Live View is disabled
            run_flags.extend(["-e", "MUJOCO_GL=osmesa"])

        # GPU
        if self.chk_gpu.isChecked():
            run_flags.append("--gpus=all")

        # Network for Meshcat (Drake/Pinocchio)
        host_port = None
        if "drake" in model.type or "pinocchio" in model.type:
            run_flags.extend(["-p", "7000-7010:7000-7010"])
            run_flags.extend(["-e", "MESHCAT_HOST=0.0.0.0"])
            host_port = 7000

        # Define Working Dictionary and Command based on model type
        if model.type == "drake":
            run_flags.extend(["-w", "/workspace/engines/physics_engines/drake/python"])
            exec_cmd = ["python", "-m", "src.drake_gui_app"]
            if host_port:
                logger.info(f"Drake Meshcat will be available on host port {host_port}")
                self._start_meshcat_browser(host_port)

        elif model.type == "custom_humanoid":
            run_flags.extend(["-w", "/workspace/engines/physics_engines/mujoco/python"])
            exec_cmd = ["python", "humanoid_launcher.py"]

        elif model.type == "custom_dashboard":
            run_flags.extend(["-w", "/workspace/engines/physics_engines/mujoco/python"])
            exec_cmd = ["python", "-m", "mujoco_humanoid_golf"]

        elif model.type == "pinocchio":
            run_flags.extend(
                ["-w", "/workspace/engines/physics_engines/pinocchio/python"]
            )
            exec_cmd = ["python", "pinocchio_golf/gui.py"]
            if host_port:
                logger.info(
                    f"Pinocchio Meshcat will be available on host port {host_port}"
                )
                self._start_meshcat_browser(host_port)

        # Construct Final Command:
        # docker run [FLAGS] IMAGE [COMMAND]
        cmd = docker_base + run_flags + [DOCKER_IMAGE_NAME] + exec_cmd

        logger.info(f"Final Docker Command: {subprocess.list2cmdline(cmd)}")

        # Launch in Terminal
        if os.name == "nt":
            # Add a diagnostic echo and pause to the terminal so users can see errors
            diagnostic_cmd = [
                "cmd",
                "/k",
                "echo Launching simulation container... && "
                + subprocess.list2cmdline(cmd),
            ]
            logger.info("Starting new console for simulation...")
            subprocess.Popen(diagnostic_cmd, creationflags=CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(cmd)

    def _start_meshcat_browser(self, port: int) -> None:
        """Start the meshcat browser."""

        def open_url() -> None:
            """Open the browser URL."""
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
