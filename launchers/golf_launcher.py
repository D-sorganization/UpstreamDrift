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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QMimeData, QPoint, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QCloseEvent,
    QColor,
    QDrag,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplashScreen,
    QStyle,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shared.python.engine_manager import EngineManager, EngineType
from shared.python.model_registry import ModelRegistry
from shared.python.secure_subprocess import (
    SecureSubprocessError,
    secure_popen,
    secure_run,
)

# Optional AI Assistant import (graceful degradation if not available)
try:
    from shared.python.ai.gui import AIAssistantPanel, AISettings, AISettingsDialog

    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

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
    "Dataset Generator GUI": "simscape_multibody.png",
    "Golf Swing Analysis GUI": "opensim.png",
    "MATLAB Code Analyzer": "urdf_icon.png",
}

DOCKER_STAGES = ["all", "mujoco", "pinocchio", "drake", "base"]


@dataclass
class SpecialApp:
    id: str
    name: str
    description: str
    type: str
    path: str


SPECIAL_APPS = [
    SpecialApp(
        id="urdf_generator",
        name="URDF Generator",
        description="Interactive URDF model builder",
        type="utility",
        path="tools/urdf_generator/launch_urdf_generator.py",
    ),
    SpecialApp(
        id="c3d_viewer",
        name="C3D Motion Viewer",
        description="C3D motion capture file viewer and analyzer",
        type="utility",
        path=(
            "engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/c3d_viewer.py"
        ),
    ),
    SpecialApp(
        id="matlab_dataset_gui",
        name="Dataset Generator GUI",
        description="MATLAB forward dynamics dataset generator",
        type="matlab_app",
        path=(
            "engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/scripts/"
            "dataset_generator/Dataset_GUI.m"
        ),
    ),
    SpecialApp(
        id="matlab_golf_gui",
        name="Golf Swing Analysis GUI",
        description="MATLAB plotting suite with skeleton visualization",
        type="matlab_app",
        path=(
            "engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/"
            "golf_gui/2D GUI/main_scripts/golf_swing_analysis_gui.m"
        ),
    ),
    SpecialApp(
        id="matlab_code_analyzer",
        name="MATLAB Code Analyzer",
        description="Static analysis and code quality dashboard",
        type="matlab_app",
        path=(
            "engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/"
            "code_analysis_gui/launchCodeAnalyzer.m"
        ),
    ),
]


class GolfSplashScreen(QSplashScreen):
    """Custom splash screen for Golf Modeling Suite."""

    def __init__(self) -> None:
        """Initialize the splash screen."""
        # Create a splash pixmap
        splash_pix = QPixmap(600, 400)
        splash_pix.fill(QColor("#1e1e1e"))

        super().__init__(splash_pix)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        # Add loading message
        self.loading_message = "Initializing Golf Modeling Suite..."
        self.progress = 0

    def drawContents(self, painter: QPainter | None) -> None:
        """Draw custom content on splash screen."""
        if painter is None:
            return

        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))

        # Draw title
        painter.drawText(
            self.rect().adjusted(20, 100, -20, -200),
            Qt.AlignmentFlag.AlignCenter,
            "⛳ Golf Modeling Suite",
        )

        # Draw subtitle
        painter.setFont(QFont("Segoe UI", 10))
        painter.setPen(QColor("#cccccc"))
        painter.drawText(
            self.rect().adjusted(20, 150, -20, -150),
            Qt.AlignmentFlag.AlignCenter,
            "Professional Biomechanics & Robotics Platform",
        )

        # Draw loading message
        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor("#007acc"))
        painter.drawText(
            self.rect().adjusted(20, 200, -20, -100),
            Qt.AlignmentFlag.AlignCenter,
            self.loading_message,
        )

        # Draw progress bar
        bar_width = 400
        bar_height = 6
        bar_x = (self.width() - bar_width) // 2
        bar_y = 280

        # Background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#333333"))
        painter.drawRoundedRect(bar_x, bar_y, bar_width, bar_height, 3, 3)

        # Progress
        painter.setBrush(QColor("#007acc"))
        progress_width = int(bar_width * (self.progress / 100))
        painter.drawRoundedRect(bar_x, bar_y, progress_width, bar_height, 3, 3)

        # Version info
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#666666"))
        painter.drawText(
            self.rect().adjusted(20, 0, -20, -20),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
            "v1.0.0-beta",
        )

    def show_message(self, message: str, progress: int) -> None:
        """Show a message with progress."""
        self.loading_message = message
        self.progress = progress
        self.showMessage(
            message, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter
        )
        self.repaint()
        QApplication.processEvents()


class DraggableModelCard(QFrame):
    """Draggable model card widget with reordering support."""

    def __init__(self, model: Any, parent: Any = None):
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
        self.setAcceptDrops(False)  # Initially disabled, will be enabled by toggle
        # Match initial drag-and-drop state to the parent's layout_edit_mode if available.
        initial_accept_drops = bool(getattr(parent, "layout_edit_mode", False))
        self.setAcceptDrops(initial_accept_drops)
        self.setObjectName("ModelCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
        # Ensure the image container has proper styling for centering
        lbl_img.setStyleSheet(
            """
            QLabel {
                border: none;
                background: transparent;
                text-align: center;
            }
        """
        )

        if img_path and img_path.exists():
            pixmap = QPixmap(str(img_path))
            # Scale image to fit within container while maintaining aspect ratio
            pixmap = pixmap.scaled(
                180,
                180,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            lbl_img.setPixmap(pixmap)
        else:
            lbl_img.setText("No Image")
            lbl_img.setStyleSheet(
                """
                QLabel {
                    color: #666;
                    font-style: italic;
                    border: none;
                    background: transparent;
                    text-align: center;
                }
            """
            )

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

        # Status Chip
        status_text, status_color = self._get_status_info()
        lbl_status = QLabel(status_text)
        lbl_status.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl_status.setStyleSheet(
            f"background-color: {status_color}; color: white; padding: 2px 6px; border-radius: 4px;"
        )
        lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_status.setFixedWidth(80)  # Fixed width for consistency

        # Center the chip
        chip_layout = QHBoxLayout()
        chip_layout.addStretch()
        chip_layout.addWidget(lbl_status)
        chip_layout.addStretch()
        layout.addLayout(chip_layout)

        # Accessibility
        self.setAccessibleName(self.model.name)
        self.setAccessibleDescription(
            f"{self.model.description}. Status: {status_text}"
        )

    def _get_status_info(self) -> tuple[str, str]:
        """Get status text and color based on model type and availability."""
        t = getattr(self.model, "type", "").lower()

        if t in [
            "custom_humanoid",
            "custom_dashboard",
            "drake",
            "pinocchio",
            "openpose",
        ]:
            return "GUI Ready", "#28a745"  # Green

        path_str = str(getattr(self.model, "path", ""))
        if t == "mjcf" or path_str.endswith(".xml"):
            return "Viewer", "#17a2b8"  # Info Blue
        elif t in ["opensim", "myosim"]:
            # Check actual availability instead of showing misleading "Demo" status
            return self._check_engine_availability(t)
        elif t in ["matlab", "matlab_app"]:
            return "External", "#6f42c1"  # Purple
        elif t in ["urdf_generator", "c3d_viewer"]:
            return "Utility", "#6c757d"  # Gray

        return "Unknown", "#6c757d"

    def _check_engine_availability(self, engine_type: str) -> tuple[str, str]:
        """Check if a specific engine is installed and return appropriate status.

        Args:
            engine_type: The engine type to check ('opensim' or 'myosim').

        Returns:
            Tuple of (status_text, color_hex).
        """
        try:
            if engine_type == "opensim":
                import opensim  # noqa: F401

                return "Engine Ready", "#28a745"  # Green
            elif engine_type == "myosim":
                import myosuite  # noqa: F401

                return "Engine Ready", "#28a745"  # Green
        except ImportError:
            pass

        # Engine not installed - show accurate status
        return "Needs Setup", "#fd7e14"  # Orange

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        """Handle keyboard interaction."""
        if not event or not self.parent_launcher:
            super().keyPressEvent(event)
            return

        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.parent_launcher.select_model(self.model.id)
            event.accept()
            # If Enter/Return, also considering launching or just selecting?
            # Consistent behavior: Space/Enter selects. Double-click launches.
            # Tab to 'Launch' button and Enter to launch is the standard accessible flow.
        else:
            super().keyPressEvent(event)

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

        # Check if layout editing is enabled
        if self.parent_launcher and not getattr(
            self.parent_launcher, "layout_edit_mode", False
        ):
            return

        if (
            not hasattr(self, "drag_start_position")
            or self.drag_start_position.isNull()
        ):
            return

        # Calculate drag distance
        drag_vector = event.position().toPoint() - self.drag_start_position
        drag_dist = drag_vector.manhattanLength()
        start_drag_dist = QApplication.startDragDistance()

        # Check distance (handle potential Mock comparison errors in tests)
        try:
            if drag_dist < start_drag_dist:
                return
        except TypeError:
            # Assume threshold met if types incompatible (e.g. Mock vs int)
            pass

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
            secure_run(
                ["docker", "--version"],
                timeout=5.0,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.result.emit(True)
        except (
            SecureSubprocessError,
            FileNotFoundError,
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
                encoding="utf-8",
                errors="replace",
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

    DEFAULT_BTN_STYLE = ""
    SUCCESS_BTN_STYLE = "background-color: #28a745; color: white;"

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

        # Action Buttons Layout
        actions_layout = QHBoxLayout()

        self.btn_build = QPushButton("Build Environment")
        self.btn_build.clicked.connect(self.start_build)
        actions_layout.addWidget(self.btn_build)

        self.btn_copy_log = QPushButton("Copy Log")
        self.btn_copy_log.setToolTip("Copy the build log to clipboard")
        self.btn_copy_log.setAccessibleName("Copy Log")
        self.btn_copy_log.clicked.connect(self.copy_log)
        actions_layout.addWidget(self.btn_copy_log)

        build_layout.addLayout(actions_layout)
        build_layout.addWidget(QLabel("Build Log:"))

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

    def copy_log(self) -> None:
        """Copy the log content to clipboard with temporary visual feedback."""
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.console.toPlainText())

            # Provide immediate feedback on the button
            original_text = "Copy Log"
            self.btn_copy_log.setText("Copied! ✓")
            self.btn_copy_log.setStyleSheet(self.SUCCESS_BTN_STYLE)

            style = self.style()
            if style:
                self.btn_copy_log.setIcon(
                    style.standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
                )

            # Restore button after 2 seconds
            QTimer.singleShot(
                2000,
                lambda: self._restore_copy_button(original_text),
            )

    def _restore_copy_button(self, original_text: str) -> None:
        """Restore the copy button to its original state."""
        # Use try-except to handle potential race conditions if dialog is closed
        try:
            self.btn_copy_log.setText(original_text)
            self.btn_copy_log.setStyleSheet(self.DEFAULT_BTN_STYLE)
            self.btn_copy_log.setIcon(QIcon())  # Remove icon
        except RuntimeError:
            # Widget likely deleted
            pass

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
        mbox = QMessageBox(self)
        mbox.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        if success:
            mbox.setWindowTitle("Success")
            mbox.setText(msg)
            mbox.setIcon(QMessageBox.Icon.Information)
        else:
            mbox.setWindowTitle("Build Failed")
            mbox.setText(msg)
            mbox.setIcon(QMessageBox.Icon.Critical)
        mbox.exec()


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


class LayoutManagerDialog(QDialog):
    """Dialog allowing users to add or remove launcher tiles."""

    def __init__(
        self,
        available_models: dict[str, Any],
        active_models: list[str],
        parent: QWidget | None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Customize Launcher Tiles")
        self.resize(520, 520)

        layout = QVBoxLayout(self)

        description = QLabel(
            "Select which applications should appear on the launcher grid. "
            "Checked items will be visible while unchecked items will be hidden."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.list_widget = QListWidget()

        sorted_models = sorted(
            available_models.values(),
            key=lambda model: getattr(model, "name", "").lower(),
        )

        for model in sorted_models:
            item = QListWidgetItem(f"{model.name} — {model.description}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if model.id in active_models
                else Qt.CheckState.Unchecked
            )
            item.setData(Qt.ItemDataRole.UserRole, model.id)
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_ids(self) -> list[str]:
        """Return IDs of all checked models."""

        selections: list[str] = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                model_id = item.data(Qt.ItemDataRole.UserRole)
                if model_id:
                    selections.append(str(model_id))
        return selections


class ContextHelpDock(QDockWidget):
    """Context-aware help drawer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Quick Help", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )

        # Content
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet(
            "background-color: #252526; color: #cccccc; border: none; padding: 10px;"
        )
        self.setWidget(self.text_area)

        # Default content
        self.update_context(None)

    def update_context(self, model_id: str | None) -> None:
        """Update help content based on selected model."""
        if not model_id:
            self.text_area.setMarkdown(
                "### Context Aware Help\n\nSelect a model to view its documentation and quick start guide."
            )
            return

        # Map ID to doc file
        doc_file = self._get_doc_file(model_id)
        if doc_file and doc_file.exists():
            try:
                content = doc_file.read_text(encoding="utf-8")
                self.text_area.setMarkdown(content)
            except Exception as e:
                self.text_area.setText(f"Failed to load documentation: {e}")
        else:
            self.text_area.setMarkdown(
                f"### {model_id}\n\nNo specific documentation available."
            )

    def _get_doc_file(self, model_id: str) -> Path | None:
        docs_dir = REPOS_ROOT / "docs" / "engines"

        if "mujoco" in model_id:
            return docs_dir / "mujoco.md"
        elif "drake" in model_id:
            return docs_dir / "drake.md"
        elif "pinocchio" in model_id:
            return docs_dir / "pinocchio.md"
        elif "matlab" in model_id:
            return docs_dir / "matlab.md"
        elif "urdf" in model_id:
            return REPOS_ROOT / "tools" / "urdf_generator" / "README.md"
        elif "c3d" in model_id:
            # Fallback for c3d if no specific doc, or maybe it shares one?
            # Review: Added placeholder explanation.
            return None

        # Fallback to no doc
        return None


class GolfLauncher(QMainWindow):
    """Main application window for the launcher."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Golf Modeling Suite - GolfingRobot")
        self.resize(1400, 900)
        self.center_window()

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
        self.layout_edit_mode = False  # Track if layout editing is enabled
        self.running_processes: dict[str, subprocess.Popen] = (
            {}
        )  # Track running instances
        self.available_models: dict[str, Any] = {}
        self.special_app_lookup: dict[str, SpecialApp] = {}
        self.current_filter_text = ""

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

        self._build_available_models()
        self._initialize_model_order()

        self.init_ui()
        self.check_docker()

        # Load saved layout
        self._load_layout()

        # Set up periodic process cleanup
        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.timeout.connect(self._cleanup_processes)
        self.cleanup_timer.start(10000)  # Clean up every 10 seconds

    def _build_available_models(self) -> None:
        """Collect all known models and auxiliary applications."""

        if self.registry:
            for model in self.registry.get_all_models():
                self.available_models[model.id] = model

        for app in SPECIAL_APPS:
            self.available_models[app.id] = app
            self.special_app_lookup[app.id] = app

    def _initialize_model_order(self) -> None:
        """Set a sensible default grid ordering."""

        default_ids: list[str] = []
        if self.registry:
            default_ids.extend([m.id for m in self.registry.get_all_models()[:10]])

        default_ids.extend(
            [
                "urdf_generator",
                "c3d_viewer",
                "matlab_dataset_gui",
                "matlab_golf_gui",
                "matlab_code_analyzer",
            ]
        )

        self.model_order = [
            model_id for model_id in default_ids if model_id in self.available_models
        ]

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

    def _sync_model_cards(self) -> None:
        """Ensure widgets match the current model order."""

        # Remove cards that are no longer selected
        for model_id in list(self.model_cards.keys()):
            if model_id not in self.model_order:
                widget = self.model_cards.pop(model_id)
                widget.setParent(None)
                widget.deleteLater()

        # Create cards for any newly added models
        for model_id in self.model_order:
            if model_id not in self.model_cards:
                model = self._get_model(model_id)
                if model:
                    self.model_cards[model_id] = DraggableModelCard(model, self)

    def _apply_model_selection(self, selected_ids: list[str]) -> None:
        """Apply a new set of selected models from the layout dialog."""

        ordered_selection = [
            model_id for model_id in self.model_order if model_id in selected_ids
        ]

        for model_id in selected_ids:
            if model_id not in ordered_selection and model_id in self.available_models:
                ordered_selection.append(model_id)

        self.model_order = ordered_selection
        self._sync_model_cards()
        self._rebuild_grid()
        self._save_layout()

        if self.selected_model not in self.model_order:
            self.selected_model = self.model_order[0] if self.model_order else None

        # Copilot AI Review Change:
        # Start with the existing model_order filtered to the newly selected IDs so
        # that previously selected models keep their relative positions in the grid.
        # ordered_selection already handled this by iterating self.model_order first.
        # Append any newly selected models (not already in model_order) to the end.
        self.update_launch_button()

    def _get_model(self, model_id: str) -> Any | None:
        """Retrieve a model or application by ID."""

        if model_id in self.available_models:
            return self.available_models[model_id]

        if self.registry:
            return self.registry.get_model(model_id)

        return None

    def center_window(self) -> None:
        """Center the window on the primary screen."""
        screen = self.screen()
        if not screen:
            return

        geometry = self.frameGeometry()
        available_geometry = screen.availableGeometry()
        center_point = available_geometry.center()
        geometry.moveCenter(center_point)
        self.move(geometry.topLeft())

    def _load_layout(self) -> None:
        """Load the saved model layout from configuration file."""
        try:
            if not LAYOUT_CONFIG_FILE.exists():
                logger.info("No saved layout found, using default")
                return

            with open(LAYOUT_CONFIG_FILE, encoding="utf-8") as f:
                layout_data = json.load(f)

            # Restore model order if valid
            saved_order = [
                model_id
                for model_id in layout_data.get("model_order", [])
                if model_id in self.available_models
            ]
            if saved_order:
                self.model_order = saved_order
                self._sync_model_cards()
                self._rebuild_grid()
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

            self._rebuild_grid()  # Use _rebuild_grid as it exists
            logger.info("Layout loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load layout: {e}")
            self._center_window()

    def _center_window(self) -> None:
        """Center the window on the primary screen."""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            # Ensure width is treated as int, handling potential Mock objects from tests
            current_width = self.width()
            if hasattr(current_width, "return_value"):  # Handle MagicMock
                current_width = 1280
            width = (
                int(current_width) if isinstance(current_width, (int, float)) else 1280
            )

            w = width if width > 100 else 1280

            # Ensure height is treated as int, handling potential Mock objects from tests
            current_height = self.height()
            if hasattr(current_height, "return_value"):  # Handle MagicMock
                current_height = 800
            height = (
                int(current_height) if isinstance(current_height, (int, float)) else 800
            )
            h = height if height > 100 else 800

            # Handle Mock objects for screen geometry
            screen_x = screen_geo.x()
            if hasattr(screen_x, "return_value"):
                screen_x = 0
            screen_x = int(screen_x) if isinstance(screen_x, (int, float)) else 0

            screen_y = screen_geo.y()
            if hasattr(screen_y, "return_value"):
                screen_y = 0
            screen_y = int(screen_y) if isinstance(screen_y, (int, float)) else 0

            screen_width = screen_geo.width()
            if hasattr(screen_width, "return_value"):
                screen_width = 1920
            screen_width = (
                int(screen_width) if isinstance(screen_width, (int, float)) else 1920
            )

            screen_height = screen_geo.height()
            if hasattr(screen_height, "return_value"):
                screen_height = 1080
            screen_height = (
                int(screen_height) if isinstance(screen_height, (int, float)) else 1080
            )

            x = screen_x + (screen_width - w) // 2
            y = screen_y + (screen_height - h) // 2

            # Ensure not too high
            y = max(y, 50)

            self.setGeometry(x, y, w, h)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Handle window close event to save layout."""
        self._save_layout()

        # Stop cleanup timer
        if hasattr(self, "cleanup_timer"):
            self.cleanup_timer.stop()

        # Terminate running processes
        for key, process in list(self.running_processes.items()):
            if process.poll() is None:
                logger.info(f"Terminating child process: {key}")
                try:
                    process.terminate()
                except Exception as e:
                    logger.error(f"Failed to terminate {key}: {e}")

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

        # Search Bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search models...")
        self.search_input.setFixedWidth(200)
        self.search_input.setToolTip("Filter models by name or description (Ctrl+F)")
        self.search_input.setAccessibleName("Search models")
        self.search_input.setClearButtonEnabled(True)  # Add clear button
        self.search_input.textChanged.connect(self.update_search_filter)
        top_bar.addWidget(self.search_input)

        # Modify Layout toggle button
        self.btn_modify_layout = QPushButton("🔒 Layout Locked")
        self.btn_modify_layout.setCheckable(True)
        self.btn_modify_layout.setChecked(False)
        self.btn_modify_layout.setToolTip("Toggle to enable/disable tile rearrangement")
        self.btn_modify_layout.clicked.connect(self.toggle_layout_mode)
        self.btn_modify_layout.setStyleSheet(
            """
            QPushButton {
                background-color: #444444;
                color: #cccccc;
                padding: 8px 16px;
            }
            QPushButton:checked {
                background-color: #007acc;
                color: white;
            }
            """
        )
        top_bar.addWidget(self.btn_modify_layout)

        self.btn_customize_tiles = QPushButton("Edit Tiles")
        self.btn_customize_tiles.setEnabled(False)
        self.btn_customize_tiles.setToolTip("Add or remove launcher tiles in edit mode")
        self.btn_customize_tiles.clicked.connect(self.open_layout_manager)
        top_bar.addWidget(self.btn_customize_tiles)

        btn_env = QPushButton("Manage Environment")
        btn_env.clicked.connect(self.open_environment_manager)
        top_bar.addWidget(btn_env)

        btn_help = QPushButton("Help")
        btn_help.clicked.connect(self.open_help)
        top_bar.addWidget(btn_help)

        # AI Assistant Button (if available)
        if AI_AVAILABLE:
            self.btn_ai = QPushButton("🤖 AI Assistant")
            self.btn_ai.setToolTip("Open AI Assistant for help with analysis")
            self.btn_ai.setCheckable(True)
            self.btn_ai.clicked.connect(self.toggle_ai_assistant)
            self.btn_ai.setStyleSheet(
                """
                QPushButton {
                    background-color: #1976d2;
                    color: white;
                    padding: 8px 16px;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #1565c0;
                }
                QPushButton:checked {
                    background-color: #0d47a1;
                }
                """
            )
            top_bar.addWidget(self.btn_ai)

        main_layout.addLayout(top_bar)

        # --- Model Grid ---
        grid_area = QScrollArea()
        grid_area.setWidgetResizable(True)
        grid_area.setFrameShape(QFrame.Shape.NoFrame)
        grid_area.setStyleSheet("background: transparent;")

        grid_widget = QWidget()
        self.grid_layout = QGridLayout(grid_widget)
        self.grid_layout.setSpacing(20)

        self._sync_model_cards()

        row, col = 0, 0
        for model_id in self.model_order:
            card = self.model_cards.get(model_id)
            if not card:
                continue
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

        # --- Help Dock ---
        self.help_dock = ContextHelpDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.help_dock)
        self.help_dock.hide()  # Hidden by default

        # --- AI Assistant Dock ---
        if AI_AVAILABLE:
            self._setup_ai_dock()

        # Select first model by default if available
        if self.model_order:
            preferred_id = next(
                (mid for mid in self.model_order if mid == "mujoco_humanoid"),
                self.model_order[0],
            )
            self.select_model(preferred_id)

        # Keyboard Shortcut for Search
        self.shortcut_search = QShortcut(QKeySequence("Ctrl+F"), self)
        self.shortcut_search.activated.connect(self._focus_search)

    def _focus_search(self) -> None:
        """Focus and select all text in search bar."""
        self.search_input.setFocus()
        self.search_input.selectAll()

    def _setup_ai_dock(self) -> None:
        """Set up the AI Assistant dock widget."""
        if not AI_AVAILABLE:
            return

        # Create dock widget
        self.ai_dock = QDockWidget("AI Assistant", self)
        self.ai_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.ai_dock.setMinimumWidth(400)

        # Create AI panel
        self.ai_panel = AIAssistantPanel()
        self.ai_panel.settings_requested.connect(self._open_ai_settings)
        self.ai_dock.setWidget(self.ai_panel)

        # Add to right side
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.ai_dock)
        self.ai_dock.hide()  # Hidden by default

        # Connect dock visibility to button state
        self.ai_dock.visibilityChanged.connect(self._on_ai_dock_visibility_changed)

        # Load and apply saved AI settings
        try:
            settings = AISettings.load()
            self.ai_panel.apply_settings(settings)
        except Exception as e:
            logger.warning("Failed to load AI settings: %s", e)

    def toggle_ai_assistant(self, checked: bool) -> None:
        """Toggle the AI Assistant dock visibility.

        Args:
            checked: Whether the button is checked.
        """
        if not hasattr(self, "ai_dock"):
            return

        if checked:
            self.ai_dock.show()
        else:
            self.ai_dock.hide()

    def _on_ai_dock_visibility_changed(self, visible: bool) -> None:
        """Handle AI dock visibility change.

        Args:
            visible: Whether the dock is visible.
        """
        if hasattr(self, "btn_ai"):
            self.btn_ai.setChecked(visible)

    def _open_ai_settings(self) -> None:
        """Open the AI settings dialog."""
        if not AI_AVAILABLE:
            return

        dialog = AISettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            if hasattr(self, "ai_panel"):
                self.ai_panel.apply_settings(settings)
            logger.info("AI settings updated")

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

    def update_search_filter(self, text: str) -> None:
        """Update the search filter and rebuild grid."""
        self.current_filter_text = text.lower()
        self._rebuild_grid()

    def _rebuild_grid(self) -> None:
        """Rebuild the grid layout based on current model order."""
        self._sync_model_cards()

        # Clear the layout
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.itemAt(i)
            if item:
                child = item.widget()
                if child:
                    self.grid_layout.removeWidget(child)

        # Re-add widgets in new order
        row, col = 0, 0
        visible_count = 0
        for model_id in self.model_order:
            if model_id in self.model_cards:
                # Apply filter
                if self.current_filter_text:
                    model = self.model_cards[model_id].model
                    name = getattr(model, "name", "").lower()
                    desc = getattr(model, "description", "").lower()
                    if (
                        self.current_filter_text not in name
                        and self.current_filter_text not in desc
                    ):
                        continue

                visible_count += 1
                card = self.model_cards[model_id]
                self.grid_layout.addWidget(card, row, col)
                col += 1
                if col >= GRID_COLUMNS:
                    col = 0
                    row += 1

        # Show empty state if needed
        if visible_count == 0 and self.current_filter_text:
            lbl_empty = QLabel(f"No models found matching '{self.current_filter_text}'")
            lbl_empty.setStyleSheet(
                "color: #888; font-style: italic; font-size: 16px; margin-top: 50px;"
            )
            lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.grid_layout.addWidget(lbl_empty, 0, 0, 1, GRID_COLUMNS)

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
        # Check if already running
        if self._is_process_running("urdf_generator"):
            QMessageBox.information(
                self,
                "Already Running",
                "URDF Generator is already running.\n\n"
                "Please close the existing instance before launching a new one.",
            )
            return

        try:
            import sys
            from pathlib import Path

            # Path to URDF generator
            suite_root = Path(__file__).parent.parent.resolve()
            urdf_script = (
                suite_root / "tools" / "urdf_generator" / "launch_urdf_generator.py"
            ).resolve()

            # Security Check: Prevent path traversal
            if not urdf_script.is_relative_to(suite_root):
                raise ValueError(
                    f"Security violation: Script path {urdf_script} is outside suite root."
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

            # Launch in new process with security validation
            try:
                creation_flags = CREATE_NEW_CONSOLE if os.name == "nt" else 0
                process = secure_popen(
                    [sys.executable, str(urdf_script)],
                    cwd=str(suite_root),
                    suite_root=suite_root,
                    creationflags=creation_flags,
                )

                # Track the process
                self.running_processes["urdf_generator"] = process
                logger.info(f"URDF Generator launched with PID: {process.pid}")

            except SecureSubprocessError as e:
                logger.error(f"Security validation failed for URDF generator: {e}")
                QMessageBox.critical(
                    self, "Security Error", f"Cannot launch URDF generator: {e}"
                )
                return

        except Exception as e:
            logger.error(f"Failed to launch URDF Generator: {e}")
            QMessageBox.critical(
                self, "Launch Error", f"Failed to launch URDF Generator:\n{e}"
            )

    def _launch_c3d_viewer(self) -> None:
        """Launch the C3D motion viewer application."""
        # Check if already running
        if self._is_process_running("c3d_viewer"):
            QMessageBox.information(
                self,
                "Already Running",
                "C3D Motion Viewer is already running.\n\n"
                "Please close the existing instance before launching a new one.",
            )
            return

        try:
            import sys
            from pathlib import Path

            # Path to C3D viewer
            suite_root = Path(__file__).parent.parent.resolve()
            c3d_script = (
                suite_root
                / "engines"
                / "Simscape_Multibody_Models"
                / "3D_Golf_Model"
                / "python"
                / "src"
                / "apps"
                / "c3d_viewer.py"
            ).resolve()

            # Security Check: Prevent path traversal
            if not c3d_script.is_relative_to(suite_root):
                raise ValueError(
                    f"Security violation: Script path {c3d_script} is outside suite root."
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

            # Launch in new process with security validation
            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = (
                    str(suite_root) + os.pathsep + env.get("PYTHONPATH", "")
                )

                creation_flags = CREATE_NEW_CONSOLE if os.name == "nt" else 0
                process = secure_popen(
                    [sys.executable, "-m", "apps.c3d_viewer"],
                    cwd=str(c3d_script.parent.parent),
                    suite_root=suite_root,
                    creationflags=creation_flags,
                    env=env,
                )

                # Track the process
                self.running_processes["c3d_viewer"] = process
                logger.info(f"C3D Viewer launched with PID: {process.pid}")

            except SecureSubprocessError as e:
                logger.error(f"Security validation failed for C3D viewer: {e}")
                QMessageBox.critical(
                    self, "Security Error", f"Cannot launch C3D viewer: {e}"
                )
                return

            # Track the process
            self.running_processes["c3d_viewer"] = process
            logger.info(f"C3D Motion Viewer launched with PID: {process.pid}")

        except Exception as e:
            logger.error(f"Failed to launch C3D Viewer: {e}")
            QMessageBox.critical(
                self, "Launch Error", f"Failed to launch C3D Viewer:\n{e}"
            )

    def _launch_matlab_app(self, app: SpecialApp) -> None:
        """Launch a MATLAB-based application using batch mode."""

        app_id = getattr(app, "id", "matlab_app")
        if self._is_process_running(app_id):
            QMessageBox.information(
                self,
                "Already Running",
                f"{app.name} is already running.\n\n"
                "Please close the existing instance before launching a new one.",
            )
            return

        app_path = REPOS_ROOT / Path(app.path)
        if not app_path.exists():
            QMessageBox.warning(
                self,
                "MATLAB App Not Found",
                f"Unable to find the MATLAB app entry point at:\n{app_path}",
            )
            return

        working_dir = app_path.parent
        entrypoint = app_path.stem

        # Escape single quotes for safe insertion into MATLAB string literals
        working_dir_str = working_dir.as_posix()
        working_dir_escaped = working_dir_str.replace("'", "''")
        entrypoint_escaped = entrypoint.replace("'", "''")

        matlab_command = (
            f"cd('{working_dir_escaped}');"
            f"addpath(genpath('{working_dir_escaped}'));"
            f"{entrypoint_escaped}();"
        )

        # Get suite root for security validation
        suite_root = Path(__file__).parent.parent.resolve()

        try:
            process = secure_popen(
                ["matlab", "-batch", matlab_command],
                cwd=str(working_dir),
                suite_root=suite_root,
            )
            self.running_processes[app_id] = process
            logger.info("Launched MATLAB app %s with PID %s", app.name, process.pid)
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                "MATLAB Not Found",
                "MATLAB executable not found in PATH.\n"
                "Please verify your MATLAB installation and environment variables.",
            )
        except Exception as exc:
            logger.error("Failed to launch MATLAB app %s: %s", app.name, exc)
            QMessageBox.critical(
                self,
                "Launch Error",
                f"Failed to launch {app.name}:\n{exc}",
            )

    def select_model(self, model_id: str) -> None:
        """Select a model and update UI."""
        self.selected_model = model_id

        # Update Help Context
        if hasattr(self, "help_dock"):
            self.help_dock.update_context(model_id)

        # Get model name for display
        model_name = model_id
        model = self._get_model(model_id)
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
                    QFrame#ModelCard:focus {
                        border: 2px solid #ffffff;
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
                    QFrame#ModelCard:focus {
                        border: 2px solid #007acc;
                    }
                """
                )

        self.update_launch_button(model_name)

    def update_launch_button(self, model_name: str | None = None) -> None:
        """Update the launch button state."""
        model_type = None
        model = None
        if self.selected_model:
            model = self._get_model(self.selected_model)
            if model:
                model_name = model.name
                model_type = getattr(model, "type", None)

        if not self.selected_model:
            self.btn_launch.setEnabled(False)
            self.btn_launch.setText("SELECT A MODEL")
            return

        if model_type in {"matlab_app", "utility"} or (
            self.selected_model in self.special_app_lookup
        ):
            label_suffix = "(MATLAB)" if model_type == "matlab_app" else ""
            display_name = str(model_name).upper()
            self.btn_launch.setEnabled(True)
            self.btn_launch.setText(f"LAUNCH {display_name} {label_suffix}".strip())
            self.btn_launch.setStyleSheet("background-color: #28a745; color: white;")
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
        """Toggle the help drawer."""
        if self.help_dock.isVisible():
            self.help_dock.hide()
        else:
            self.help_dock.show()

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

        special_app = self.special_app_lookup.get(model_id)
        if special_app and getattr(special_app, "type", None) == "matlab_app":
            self._launch_matlab_app(special_app)
            return

        model = self._get_model(model_id)
        if not model:
            QMessageBox.critical(self, "Error", f"Model not found: {model_id}")
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
                # For opensim/myosim/openpose, force local true if we have a GUI script,
                # regardless of probe, to trigger the fallback/mock.
                if model.type in ["opensim", "myosim", "openpose"]:
                    is_local_fit = True
                elif probe and probe.is_available():
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
                elif model.type == "drake":
                    self._custom_launch_drake(path)
                elif model.type == "pinocchio":
                    self._custom_launch_pinocchio(path)
                elif model.type == "opensim":
                    self._custom_launch_opensim(path)
                elif model.type == "myosim":
                    self._custom_launch_myosim(path)
                elif model.type == "openpose":
                    self._custom_launch_openpose(path)
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
        # Check if already running
        if self._is_process_running("mujoco_humanoid"):
            QMessageBox.information(
                self,
                "Already Running",
                "MuJoCo Humanoid is already running.\n\n"
                "Please close the existing instance before launching a new one.",
            )
            return

        script = abs_repo_path / "python/humanoid_launcher.py"
        if not script.exists():
            raise FileNotFoundError(
                f"Script not found: {script}. "
                "Ensure the MuJoCo engine and selected model repository "
                "are properly installed."
            )

        logger.info(f"Launching Humanoid GUI: {script}")
        process = subprocess.Popen([sys.executable, str(script)], cwd=script.parent)
        self.running_processes["mujoco_humanoid"] = process
        logger.info(f"MuJoCo Humanoid launched with PID: {process.pid}")

    def _custom_launch_comprehensive(self, abs_repo_path: Path) -> None:
        """Launch the comprehensive dashboard directly."""
        # Check if already running
        if self._is_process_running("mujoco_dashboard"):
            QMessageBox.information(
                self,
                "Already Running",
                "MuJoCo Dashboard is already running.\n\n"
                "Please close the existing instance before launching a new one.",
            )
            return

        python_dir = abs_repo_path / "python"
        logger.info(f"Launching Comprehensive GUI from {python_dir}")
        process = subprocess.Popen(
            [sys.executable, "-m", "mujoco_humanoid_golf"], cwd=python_dir
        )
        self.running_processes["mujoco_dashboard"] = process
        logger.info(f"MuJoCo Dashboard launched with PID: {process.pid}")

    def _custom_launch_drake(self, abs_repo_path: Path) -> None:
        """Launch the Drake GUI directly."""
        # Use hardcoded path to ensure we hit the correct package root
        python_dir = REPOS_ROOT / "engines/physics_engines/drake/python"

        if not python_dir.exists():
            # Fallback to provided path if hardcoded fails
            python_dir = (
                abs_repo_path if abs_repo_path.is_dir() else abs_repo_path.parent
            )

        if not python_dir.exists():
            QMessageBox.critical(
                self, "Error", f"Drake Python directory not found: {python_dir}"
            )
            return

        logger.info(f"Launching Drake GUI from {python_dir}")
        creation_flags = CREATE_NEW_CONSOLE if os.name == "nt" else 0

        try:
            # Drake app is a module: src.drake_gui_app
            process = subprocess.Popen(
                [sys.executable, "-m", "src.drake_gui_app"],
                cwd=str(python_dir),
                creationflags=creation_flags,
            )
            self.running_processes["drake_gui"] = process
            logger.info(f"Drake GUI launched with PID: {process.pid}")
        except Exception as e:
            logger.error(f"Failed to launch Drake: {e}")
            QMessageBox.critical(self, "Launch Error", str(e))

    def _custom_launch_pinocchio(self, abs_repo_path: Path) -> None:
        """Launch the Pinocchio GUI directly."""
        python_dir = REPOS_ROOT / "engines/physics_engines/pinocchio/python"
        script = python_dir / "pinocchio_golf" / "gui.py"

        if not script.exists():
            QMessageBox.critical(
                self, "Error", f"Pinocchio GUI script not found: {script}"
            )
            return

        logger.info(f"Launching Pinocchio GUI: {script}")
        creation_flags = CREATE_NEW_CONSOLE if os.name == "nt" else 0

        try:
            process = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(python_dir),
                creationflags=creation_flags,
            )
            self.running_processes["pinocchio_gui"] = process
            logger.info(f"Pinocchio GUI launched with PID: {process.pid}")
        except Exception as e:
            logger.error(f"Failed to launch Pinocchio: {e}")
            QMessageBox.critical(self, "Launch Error", str(e))

    def _custom_launch_opensim(self, abs_repo_path: Path) -> None:
        """Launch the OpenSim GUI directly."""
        # Use our new GUI script
        script = REPOS_ROOT / "engines/physics_engines/opensim/python/opensim_gui.py"
        if not script.exists():
            QMessageBox.critical(
                self, "Error", f"OpenSim GUI script not found: {script}"
            )
            return

        logger.info(f"Launching OpenSim GUI: {script}")
        creation_flags = CREATE_NEW_CONSOLE if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(script.parent),
                creationflags=creation_flags,
            )
            self.running_processes["opensim_gui"] = process
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", str(e))

    def _custom_launch_myosim(self, abs_repo_path: Path) -> None:
        """Launch the MyoSim GUI directly."""
        script = REPOS_ROOT / "engines/physics_engines/myosim/python/myosim_gui.py"
        if not script.exists():
            QMessageBox.critical(
                self, "Error", f"MyoSim GUI script not found: {script}"
            )
            return

        logger.info(f"Launching MyoSim GUI: {script}")
        creation_flags = CREATE_NEW_CONSOLE if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(script.parent),
                creationflags=creation_flags,
            )
            self.running_processes["myosim_gui"] = process
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", str(e))

    def _custom_launch_openpose(self, abs_repo_path: Path) -> None:
        """Launch the OpenPose GUI directly."""
        script = REPOS_ROOT / "shared/python/pose_estimation/openpose_gui.py"
        if not script.exists():
            QMessageBox.critical(
                self, "Error", f"OpenPose GUI script not found: {script}"
            )
            return

        logger.info(f"Launching OpenPose GUI: {script}")
        creation_flags = CREATE_NEW_CONSOLE if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(script.parent),
                creationflags=creation_flags,
            )
            self.running_processes["openpose_gui"] = process
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", str(e))

    def _launch_docker_container(self, model: Any, abs_repo_path: Path) -> None:
        """Launch the simulation in a docker container."""
        # Parts of the command
        docker_base = ["docker", "run", "--rm", "-it"]
        cmd = list(docker_base)

        # Volumes - mount entire suite root to /workspace
        mount_path = str(REPOS_ROOT).replace("\\", "/")
        cmd.extend(["-v", f"{mount_path}:/workspace"])

        # Prepare dynamic working directory and entry command
        work_dir = "/workspace"
        entry_cmd = []
        host_port = None

        if model.type == "drake":
            work_dir = "/workspace/engines/physics_engines/drake/python"
            entry_cmd = ["python", "-m", "src.drake_gui_app"]
            host_port = 7000
            # Browser launch moved to end of method

        elif model.type == "custom_humanoid":
            work_dir = "/workspace/engines/physics_engines/mujoco/python"
            entry_cmd = ["python", "humanoid_launcher.py"]

        elif model.type == "custom_dashboard":
            work_dir = "/workspace/engines/physics_engines/mujoco/python"
            entry_cmd = ["python", "-m", "mujoco_humanoid_golf"]

        elif model.type == "pinocchio":
            work_dir = "/workspace/engines/physics_engines/pinocchio/python"
            entry_cmd = ["python", "pinocchio_golf/gui.py"]
            host_port = 7000
            # Browser launch moved to end of method

        # Set working directory
        cmd.extend(["-w", work_dir])

        # Environment variables for Python path and shared modules
        cmd.extend(
            ["-e", "PYTHONPATH=/workspace:/workspace/shared/python:/workspace/engines"]
        )

        # Mount 'shared' directory so that scripts can import shared modules
        shared_host_path = REPOS_ROOT / "shared"
        if shared_host_path.exists():
            mount_shared_str = str(shared_host_path).replace("\\", "/")
            cmd.extend(["-v", f"{mount_shared_str}:/shared"])

        # Display/X11
        if self.chk_live.isChecked():
            if os.name == "nt":
                cmd.extend(["-e", "DISPLAY=host.docker.internal:0"])
                cmd.extend(["-e", "MUJOCO_GL=glfw"])
                cmd.extend(["-e", "PYOPENGL_PLATFORM=glx"])
                cmd.extend(["-e", "QT_QPA_PLATFORM=xcb"])
            else:
                cmd.extend(["-e", f"DISPLAY={os.environ.get('DISPLAY', ':0')}"])
                cmd.extend(["-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw"])
        else:
            # Force headless OSMesa rendering if Live View is disabled
            cmd.extend(["-e", "MUJOCO_GL=osmesa"])

        # GPU
        if self.chk_gpu.isChecked():
            cmd.append("--gpus=all")

        # Network for Meshcat (Drake/Pinocchio)
        if host_port:
            # Find an available port on the host
            available_port = self._find_available_port(host_port)

            # Map Host(Available) -> Container(7000)
            # We assume the internal app always binds to 7000 as configured/default
            cmd.extend(["-p", f"{available_port}:7000"])
            cmd.extend(["-e", "MESHCAT_HOST=0.0.0.0"])

            # Update the browser launch to use the actual host port
            if host_port != available_port:
                logger.info(f"Port {host_port} busy, using {available_port} instead")

            host_port = available_port  # Update for browser launch below

            logger.info(
                f"Launching Meshcat browser at http://127.0.0.1:{host_port}/static/"
            )
            self._start_meshcat_browser(host_port)

        cmd.append(DOCKER_IMAGE_NAME)

        # Append entry command
        if entry_cmd:
            cmd.extend(entry_cmd)

        logger.info(f"Final Docker Command: {subprocess.list2cmdline(cmd)}")

        # Launch in Terminal
        if os.name == "nt":
            # Add a diagnostic echo and pause to the terminal so users can see errors
            # We must escape & characters for the echo command, otherwise cmd.exe interprets them
            command_str = " ".join(cmd)
            safe_echo_str = command_str.replace("&", "^&")

            diagnostic_cmd = [
                "cmd",
                "/k",
                f"echo Launching simulation container... && echo Command: {safe_echo_str} && {command_str}",
            ]
            logger.info("Starting new console for simulation...")
            subprocess.Popen(diagnostic_cmd, creationflags=CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(cmd)

    def _find_available_port(self, start_port: int, max_attempts: int = 100) -> int:
        """Find an available port starting from start_port."""
        import socket

        for port in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue

        raise RuntimeError(
            f"Could not find available port in range {start_port}-{start_port + max_attempts}"
        )

    def _start_meshcat_browser(self, port: int) -> None:
        """Start the meshcat browser."""

        def open_url() -> None:
            """Open the browser URL."""
            time.sleep(3)
            # Meshcat default UI is at /static/
            # Use 127.0.0.1 instead of localhost to avoid resolution issues
            webbrowser.open(f"http://127.0.0.1:{port}/static/")

        threading.Thread(target=open_url, daemon=True).start()

    def toggle_layout_mode(self) -> None:
        """Toggle layout editing mode."""
        self.layout_edit_mode = self.btn_modify_layout.isChecked()

        if self.layout_edit_mode:
            self.btn_modify_layout.setText("🔓 Layout Unlocked")
            logger.info("Layout editing enabled - tiles can be rearranged")
        else:
            self.btn_modify_layout.setText("🔒 Layout Locked")
            # Save layout when locking
            self._save_layout()
            logger.info("Layout editing disabled - layout saved")

        # Update all model cards to enable/disable dragging
        for card in self.model_cards.values():
            card.setAcceptDrops(self.layout_edit_mode)
            if self.layout_edit_mode:
                card.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                card.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_customize_tiles.setEnabled(self.layout_edit_mode)

    def open_layout_manager(self) -> None:
        """Allow users to add or remove available launcher tiles."""

        if not self.layout_edit_mode:
            QMessageBox.information(
                self,
                "Layout Locked",
                "Enable layout editing to add or remove tiles.",
            )
            return

        dialog = LayoutManagerDialog(self.available_models, self.model_order, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selections = dialog.selected_ids()
            self._apply_model_selection(selections)

    def _is_process_running(self, process_key: str) -> bool:
        """Check if a process is still running.

        Args:
            process_key: The key used to track the process

        Returns:
            True if the process is running, False otherwise
        """
        if process_key not in self.running_processes:
            return False

        process = self.running_processes[process_key]

        # Check if process is still running
        if process.poll() is None:
            return True

        # Process has terminated, remove it
        del self.running_processes[process_key]
        return False

    def _cleanup_processes(self) -> None:
        """Clean up terminated processes from the tracking dictionary."""
        terminated = []
        for key, process in self.running_processes.items():
            if process.poll() is not None:
                terminated.append(key)

        for key in terminated:
            del self.running_processes[key]
            logger.info(f"Process {key} has terminated")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Global Font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Show splash screen
    splash = GolfSplashScreen()
    splash.show()

    # Phase 1: Load application resources
    splash.show_message("Loading application resources...", 10)
    QApplication.processEvents()

    # Phase 2: Load model registry (actual work during splash for init + cache warming)
    splash.show_message("Loading model registry...", 25)
    QApplication.processEvents()
    try:
        # Initialize registry during splash to warm caches; instance discarded (GolfLauncher
        # creates its own instance later). This ensures model scanning happens during splash.
        _model_registry = ModelRegistry()
        logger.info(f"Loaded {len(_model_registry.get_all_models())} models")
    except Exception as e:
        logger.warning(f"Model registry load warning: {e}")

    # Phase 3: Initialize engine manager (actual work - probing can be slow)
    splash.show_message("Probing physics engines...", 45)
    QApplication.processEvents()
    try:
        # Probe engines during splash to warm import caches; instance discarded.
        # This front-loads the latency of importing MuJoCo/Drake/Pinocchio/OpenSim.
        _engine_manager = EngineManager()
        available_engines = _engine_manager.get_available_engines()
        logger.info(
            f"Found {len(available_engines)} available engines: {available_engines}"
        )
    except Exception as e:
        logger.warning(f"Engine manager init warning: {e}")

    # Phase 4: Check optional dependencies
    splash.show_message("Checking AI assistant...", 65)
    QApplication.processEvents()
    if AI_AVAILABLE:
        logger.info("AI Assistant module available")
    else:
        logger.info("AI Assistant not available (optional)")

    # Phase 5: Build UI
    splash.show_message("Building user interface...", 80)
    QApplication.processEvents()

    # Create main window
    window = GolfLauncher()

    # Phase 6: Final setup
    splash.show_message("Finalizing...", 95)
    QApplication.processEvents()

    splash.show_message("Ready!", 100)
    QApplication.processEvents()

    # Close splash and show main window
    splash.finish(window)
    window.show()

    sys.exit(app.exec())
