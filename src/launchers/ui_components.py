"""UI Components for the Golf Modeling Suite Launcher.

This module provides specialized widgets, dialogs, and background workers
to improve the modularity and maintainability of the launcher.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QMimeData, QPoint, Qt, QThread, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QDrag,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QMouseEvent,
    QPainter,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplashScreen,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_config import get_logger
from src.shared.python.secure_subprocess import (
    secure_run,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Constants (mirrored from main launcher)
REPOS_ROOT = Path(__file__).parent.parent.parent.resolve()
ASSETS_DIR = Path(__file__).parent / "assets"
DOCKER_IMAGE_NAME = "robotics_env"
LAUNCH_FEEDBACK_DURATION_MS = 2000

# Metadata (mirrored from main launcher)
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
    "C3D Motion Viewer": "c3d_icon.png",
    "Dataset Generator GUI": "simscape_multibody.png",
    "Golf Swing Analysis GUI": "opensim.png",
    "MATLAB Code Analyzer": "urdf_icon.png",
}

# Theme availability check
try:
    from src.shared.python.theme import (
        Colors,
        Sizes,
        Weights,
        get_display_font,
        get_qfont,
    )

    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False


class StartupResults:
    """Container for async startup results."""

    def __init__(self) -> None:
        self.registry: Any = None
        self.engine_manager: Any = None
        self.available_engines: list = []
        self.ai_available: bool = False
        self.docker_available: bool = False
        self.startup_time_ms: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> StartupResults:
        """Create StartupResults from worker results dict."""
        results = cls()
        results.registry = data.get("registry")
        results.engine_manager = data.get("engine_manager")
        results.available_engines = data.get("available_engines", [])
        results.ai_available = data.get("ai_available", False)
        results.docker_available = data.get("docker_available", False)
        results.startup_time_ms = data.get("startup_time_ms", 0)
        return results


class GolfSplashScreen(QSplashScreen):
    """Custom splash screen for Golf Modeling Suite."""

    SPLASH_WIDTH = 520
    SPLASH_HEIGHT = 340

    def __init__(self) -> None:
        splash_pix = QPixmap(self.SPLASH_WIDTH, self.SPLASH_HEIGHT)
        bg_color = Colors.BG_DEEP if THEME_AVAILABLE else "#0D0D0D"
        splash_pix.fill(QColor(bg_color))

        super().__init__(splash_pix)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        self.logo_pixmap: QPixmap | None = None
        logo_candidates = [
            ASSETS_DIR / "golf_robot_ultra_sharp.png",
            ASSETS_DIR / "golf_robot_windows_optimized.png",
            ASSETS_DIR / "golf_robot_icon_128.png",
        ]
        for logo_path in logo_candidates:
            if logo_path.exists():
                self.logo_pixmap = QPixmap(str(logo_path))
                if not self.logo_pixmap.isNull():
                    self.logo_pixmap = self.logo_pixmap.scaled(
                        64,
                        64,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    break

        self.loading_message = "Initializing Golf Modeling Suite..."
        self.progress = 0

    def drawContents(self, painter: QPainter | None) -> None:
        if painter is None:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        if THEME_AVAILABLE:
            text_primary = Colors.TEXT_PRIMARY
            text_secondary = Colors.TEXT_TERTIARY
            accent = Colors.PRIMARY
            bg_bar = Colors.BG_ELEVATED
            text_quaternary = Colors.TEXT_QUATERNARY
        else:
            text_primary = "#FFFFFF"
            text_secondary = "#A0A0A0"
            accent = "#0A84FF"
            bg_bar = "#2D2D2D"
            text_quaternary = "#666666"

        center_x = self.width() // 2
        logo_y = 50
        if self.logo_pixmap and not self.logo_pixmap.isNull():
            logo_x = center_x - self.logo_pixmap.width() // 2
            painter.drawPixmap(logo_x, logo_y, self.logo_pixmap)
            title_y = logo_y + self.logo_pixmap.height() + 20
        else:
            title_y = 80

        title_font = (
            get_display_font(size=Sizes.XXL, weight=Weights.BOLD)
            if THEME_AVAILABLE
            else QFont("Segoe UI", 24, QFont.Weight.Bold)
        )
        painter.setFont(title_font)
        painter.setPen(QColor(text_primary))
        painter.drawText(
            self.rect().adjusted(20, title_y, -20, 0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "Golf Modeling Suite",
        )

        subtitle_font = (
            get_qfont(size=Sizes.MD, weight=Weights.NORMAL)
            if THEME_AVAILABLE
            else QFont("Segoe UI", 11)
        )
        painter.setFont(subtitle_font)
        painter.setPen(QColor(text_secondary))
        painter.drawText(
            self.rect().adjusted(20, title_y + 38, -20, 0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "Professional Biomechanics & Robotics Platform",
        )

        status_font = (
            get_qfont(size=Sizes.SM, weight=Weights.MEDIUM)
            if THEME_AVAILABLE
            else QFont("Segoe UI", 9, QFont.Weight.Medium)
        )
        painter.setFont(status_font)
        painter.setPen(QColor(accent))

        status_y = self.height() - 90
        painter.drawText(
            self.rect().adjusted(20, status_y, -20, 0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            self.loading_message,
        )

        bar_width = 360
        bar_height = 4
        bar_x = (self.width() - bar_width) // 2
        bar_y = self.height() - 60

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(bg_bar))
        painter.drawRoundedRect(bar_x, bar_y, bar_width, bar_height, 2, 2)

        if self.progress > 0:
            painter.setBrush(QColor(accent))
            progress_width = int(bar_width * (self.progress / 100))
            painter.drawRoundedRect(bar_x, bar_y, progress_width, bar_height, 2, 2)

        version_font = (
            get_qfont(size=Sizes.XS, weight=Weights.NORMAL)
            if THEME_AVAILABLE
            else QFont("Segoe UI", 8)
        )
        painter.setFont(version_font)
        painter.setPen(QColor(text_quaternary))
        painter.drawText(
            self.rect().adjusted(20, 0, -16, -12),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
            "v1.0.0-beta",
        )
        painter.drawText(
            self.rect().adjusted(16, 0, -20, -12),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
            "\u00a9 2024-2025 Golf Modeling Suite",
        )

    def show_message(self, message: str, progress: int) -> None:
        self.loading_message = message
        self.progress = progress
        self.showMessage(
            message, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter
        )
        self.repaint()
        QApplication.processEvents()


class AsyncStartupWorker(QThread):
    """Background worker for async application startup."""

    progress_signal = pyqtSignal(str, int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, repos_root: Path):
        super().__init__()
        self.repos_root = repos_root
        self.results = StartupResults()

    def run(self) -> None:
        try:
            self.progress_signal.emit("Loading model registry...", 10)
            from src.shared.python.model_registry import ModelRegistry

            registry = ModelRegistry(self.repos_root / "config/models.yaml")
            self.results.registry = registry

            self.progress_signal.emit("Probing physics engines...", 30)
            from src.shared.python.engine_manager import EngineManager

            self.results.engine_manager = EngineManager(self.repos_root)
            self.results.engine_manager.probe_all_engines()

            self.progress_signal.emit("Checking Docker status...", 60)
            try:
                secure_run(["docker", "--version"], timeout=3.0, check=True)
                self.results.docker_available = True
            except Exception:
                self.results.docker_available = False

            self.progress_signal.emit("Ready", 100)
            time.sleep(0.5)
            self.finished_signal.emit(self.results)
        except Exception as e:
            logger.error(f"Startup failed: {e}")
            self.error_signal.emit(str(e))


class DraggableModelCard(QFrame):
    """Draggable model card widget with reordering support."""

    def __init__(self, model: Any, parent_launcher: Any):
        super().__init__(parent_launcher)
        self.model = model
        self.parent_launcher = parent_launcher

        # Match initial drag-and-drop state to the parent's mode
        self.setAcceptDrops(bool(getattr(parent_launcher, "layout_edit_mode", False)))
        self.setObjectName("ModelCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.drag_start_position = QPoint()

        self.setup_ui()

    def setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        img_name = MODEL_IMAGES.get(self.model.name)
        if not img_name:
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
        lbl_img.setStyleSheet(
            "QLabel { border: none; background: transparent; text-align: center; }"
        )

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
            lbl_img.setStyleSheet(
                "QLabel { color: #666; font-style: italic; border: none; background: transparent; text-align: center; }"
            )

        layout.addWidget(lbl_img)

        lbl_name = QLabel(self.model.name)
        lbl_name.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        lbl_name.setWordWrap(True)
        lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_name)

        lbl_desc = QLabel(self.model.description)
        lbl_desc.setFont(QFont("Segoe UI", 9))
        lbl_desc.setStyleSheet("color: #cccccc;")
        lbl_desc.setWordWrap(True)
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_desc)

        # Status Chip
        status_text, status_color, text_color = self._get_status_info()
        lbl_status = QLabel(status_text)
        lbl_status.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl_status.setStyleSheet(
            f"background-color: {status_color}; color: {text_color}; padding: 2px 6px; border-radius: 4px;"
        )
        lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_status.setFixedWidth(80)

        chip_layout = QHBoxLayout()
        chip_layout.addStretch()
        chip_layout.addWidget(lbl_status)
        chip_layout.addStretch()
        layout.addLayout(chip_layout)

    def _get_status_info(self) -> tuple[str, str, str]:
        t = getattr(self.model, "type", "").lower()
        if t in [
            "custom_humanoid",
            "custom_dashboard",
            "drake",
            "pinocchio",
            "openpose",
        ]:
            return "GUI Ready", "#28a745", "#000000"

        path_str = str(getattr(self.model, "path", ""))
        if t == "mjcf" or path_str.endswith(".xml"):
            return "Viewer", "#17a2b8", "#000000"
        elif t in ["opensim", "myosim"]:
            return "Engine Ready", "#28a745", "#000000"
        elif t in ["matlab", "matlab_app"]:
            return "External", "#6f42c1", "#ffffff"
        elif t in ["urdf_generator", "c3d_viewer"]:
            return "Utility", "#6c757d", "#ffffff"

        return "Unknown", "#6c757d", "#ffffff"

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event and event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_position = event.position().toPoint()
            if self.parent_launcher:
                self.parent_launcher.select_model(self.model.id)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if not event or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if not getattr(self.parent_launcher, "layout_edit_mode", False):
            return

        if (
            event.position().toPoint() - self.drag_start_position
        ).manhattanLength() < QApplication.startDragDistance():
            return

        drag = QDrag(self)
        mimeData = QMimeData()
        mimeData.setText(f"model_card:{self.model.id}")
        drag.setMimeData(mimeData)
        drag.setPixmap(self.grab())
        drag.setHotSpot(self.drag_start_position)
        drag.exec(Qt.DropAction.MoveAction)

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:
        if self.parent_launcher:
            self.parent_launcher.launch_model_direct(self.model.id)

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        if not event:
            return

        mime_data = event.mimeData()
        if (
            mime_data
            and mime_data.hasText()
            and mime_data.text().startswith("model_card:")
        ):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent | None) -> None:
        if not event:
            return

        mime_data = event.mimeData()
        if mime_data and mime_data.hasText():
            source_id = mime_data.text().split(":")[1]
            if self.parent_launcher and source_id != self.model.id:
                self.parent_launcher._swap_models(source_id, self.model.id)
            event.acceptProposedAction()


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
        except Exception:
            self.result.emit(False)


class DockerBuildThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, target_stage: str = "all") -> None:
        super().__init__()
        self.target_stage = target_stage

    def run(self) -> None:
        mujoco_path = REPOS_ROOT / "engines/physics_engines/mujoco"
        if not mujoco_path.exists():
            self.finished_signal.emit(False, f"Path not found: {mujoco_path}")
            return

        cmd = [
            "docker",
            "build",
            "-t",
            DOCKER_IMAGE_NAME,
            "--target",
            self.target_stage,
            "--progress=plain",
            ".",
        ]
        self.log_signal.emit(f"Starting build for {self.target_stage}...")

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(mujoco_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    self.log_signal.emit(line.strip())

            process.wait()
            self.finished_signal.emit(process.returncode == 0, "Build finished.")
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
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Build Tab
        tab_build = QWidget()
        build_layout = QVBoxLayout(tab_build)
        self.combo_stage = QComboBox()
        self.combo_stage.addItems(["all", "mujoco", "pinocchio", "drake", "base"])
        build_layout.addWidget(QLabel("Target Stage:"))
        build_layout.addWidget(self.combo_stage)

        btn_build = QPushButton("Build Environment")
        btn_build.clicked.connect(self.start_build)
        build_layout.addWidget(btn_build)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet(
            "background-color: #1e1e1e; color: #00ff00; font-family: Consolas;"
        )
        build_layout.addWidget(self.console)
        tabs.addTab(tab_build, "Build Docker")

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def start_build(self) -> None:
        self.console.clear()
        self.build_thread = DockerBuildThread(self.combo_stage.currentText())
        self.build_thread.log_signal.connect(self.console.append)
        self.build_thread.start()


class HelpDialog(QDialog):
    """Dialog to display help documentation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Golf Suite - Help")
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

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

        return None
