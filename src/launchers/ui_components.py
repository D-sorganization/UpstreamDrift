"""UI Components for the UpstreamDrift Launcher.

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
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplashScreen,
    QTabWidget,
    QTextBrowser,
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

# Tile image file names
_IMG_SIMSCAPE = "simscape_multibody.png"
_IMG_MATLAB = "matlab_logo.png"

# Metadata (mirrored from main launcher)
# Maps display names to tile image files in assets/
MODEL_IMAGES = {
    # Physics Engines - Current names from models.yaml
    "MuJoCo": "mujoco_humanoid.png",
    "Drake": "drake.png",
    "Pinocchio": "pinocchio.png",
    "OpenSim": "opensim.png",
    "MyoSuite": "myosim.png",
    # MATLAB/Simscape
    "Matlab Models": _IMG_MATLAB,
    # Tools
    "Motion Capture": "c3d_icon.png",
    "Model Explorer": "urdf_icon.png",
    "Putting Green": "putting_green.svg",
    "Video Analyzer": "video_analyzer.svg",
    "Data Explorer": "data_explorer.svg",
    # Legacy names (backward compatibility)
    "MuJoCo Humanoid": "mujoco_humanoid.png",
    "MuJoCo Dashboard": "mujoco_hand.png",
    "Drake Golf Model": "drake.png",
    "Pinocchio Golf Model": "pinocchio.png",
    "OpenSim Golf": "opensim.png",
    "MyoSim Suite": "myosim.png",
    "OpenPose Analysis": "openpose.jpg",
    "Matlab Simscape": _IMG_MATLAB,
    "Matlab Simscape 2D": _IMG_MATLAB,
    "Matlab Simscape 3D": _IMG_MATLAB,
    "Dataset Generator GUI": _IMG_MATLAB,
    "Golf Swing Analysis GUI": _IMG_MATLAB,
    "MATLAB Code Analyzer": _IMG_MATLAB,
    "URDF Generator": "urdf_icon.png",
    "C3D Motion Viewer": "c3d_icon.png",
    "Shot Tracer": "golf_icon.png",
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
    """Custom splash screen for UpstreamDrift."""

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
            ASSETS_DIR / "golf_logo.png",
            ASSETS_DIR / "golf_icon.png",
        ]
        for logo_path in logo_candidates:
            if logo_path.exists():
                self.logo_pixmap = QPixmap(str(logo_path))
                if not self.logo_pixmap.isNull():
                    self.logo_pixmap = self.logo_pixmap.scaled(
                        80,
                        80,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    break

        self.loading_message = "Initializing UpstreamDrift..."
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
            "UpstreamDrift",
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
            "UpstreamDrift",
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

            registry = ModelRegistry(self.repos_root / "src/config/models.yaml")
            self.results.registry = registry

            self.progress_signal.emit("Initializing engine manager...", 30)
            try:
                from src.shared.python.engine_manager import EngineManager

                self.results.engine_manager = EngineManager(self.repos_root)
                # Skip probing to avoid hanging - engines will be probed on demand
                # self.results.engine_manager.probe_all_engines()
            except Exception as e:
                logger.warning(f"Engine manager init failed: {e}")
                self.results.engine_manager = None

            self.progress_signal.emit("Checking Docker status...", 60)
            try:
                secure_run(["docker", "--version"], timeout=2.0, check=True)
                self.results.docker_available = True
            except Exception:
                self.results.docker_available = False
                logger.debug("Docker not available or timed out")

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
            # Fallback based on model ID
            model_id = self.model.id.lower()
            if "mujoco" in model_id:
                img_name = "mujoco_humanoid.png"
            elif "drake" in model_id:
                img_name = "drake.png"
            elif "pinocchio" in model_id:
                img_name = "pinocchio.png"
            elif "opensim" in model_id:
                img_name = "opensim.png"
            elif "myosim" in model_id or "myosuite" in model_id:
                img_name = "myosim.png"
            elif "matlab" in model_id:
                img_name = "matlab_logo.png"
            elif "motion" in model_id or "capture" in model_id or "c3d" in model_id:
                img_name = "c3d_icon.png"
            elif "model_explorer" in model_id or "urdf" in model_id:
                img_name = "urdf_icon.png"
            # Fallback for engine types if ID didn't match
            elif "engine_managed" in getattr(self.model, "type", ""):
                if getattr(self.model, "engine_type", "") == "mujoco":
                    img_name = "mujoco_humanoid.png"

        # Check assets dir first, then fall back to SVG logos dir
        img_path = None
        if img_name:
            img_path = ASSETS_DIR / img_name
            if not img_path.exists():
                # Fall back to SVG logos directory
                svg_logos_dir = Path(__file__).parent.parent.parent / "assets" / "logos"
                img_path = svg_logos_dir / img_name
                if not img_path.exists():
                    img_path = None
        lbl_img = QLabel()
        lbl_img.setFixedSize(200, 200)
        lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Enforce centering in the layout
        layout.setAlignment(lbl_img, Qt.AlignmentFlag.AlignCenter)

        # Use transparent background, remove text-align as it's handled by setAlignment
        lbl_img.setStyleSheet("QLabel { border: none; background: transparent; }")

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
                "QLabel { color: #666; font-style: italic; border: none; background: transparent; }"
            )

        img_container = QWidget()
        img_layout = QHBoxLayout(img_container)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.addStretch()
        img_layout.addWidget(lbl_img)
        img_layout.addStretch()

        layout.addWidget(img_container)

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
        lbl_status.setMinimumWidth(
            80
        )  # Use minimum width instead of fixed width to prevent text cutoff
        # lbl_status.setFixedWidth(80)

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
        mujoco_path = REPOS_ROOT / "src/engines/physics_engines/mujoco"
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
                creationflags=(
                    subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
                    if os.name == "nt"
                    else 0
                ),
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


class SettingsDialog(QDialog):
    """Settings dialog with Layout, Configuration, and Diagnostics tabs.

    Tab order:
        0 - Layout: tile arrangement, lock, reset
        1 - Configuration: execution env, simulation opts, Docker rebuild
        2 - Diagnostics: system checks, error logs, terminal output
    """

    reset_layout_requested = pyqtSignal()

    # Tab index constants for external callers
    TAB_LAYOUT = 0
    TAB_CONFIG = 1
    TAB_DIAGNOSTICS = 2

    def __init__(
        self,
        parent: QWidget | None = None,
        diagnostics_data: dict[str, Any] | None = None,
        initial_tab: int = 0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(850, 650)
        self._diagnostics_data = diagnostics_data
        self._setup_ui()
        self.tabs.setCurrentIndex(initial_tab)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        from src.shared.python.draggable_tabs import DraggableTabWidget

        self.tabs = DraggableTabWidget(
            core_tabs={"Layout", "Configuration", "Diagnostics"}
        )
        self.tabs.setTabsClosable(False)
        layout.addWidget(self.tabs)

        self.tabs.addTab(self._create_layout_tab(), "Layout")
        self.tabs.addTab(self._create_configuration_tab(), "Configuration")
        self.tabs.addTab(self._create_diagnostics_tab(), "Diagnostics")

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    # ── Layout tab ──────────────────────────────────────────────────

    def _create_layout_tab(self) -> QWidget:
        """Layout tab: tile lock, edit tiles, reset to defaults."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        group = QGroupBox("Tile Layout")
        inner = QVBoxLayout(group)

        self._btn_layout_lock = QPushButton("Layout: Locked")
        self._btn_layout_lock.setCheckable(True)
        self._btn_layout_lock.setChecked(False)
        self._btn_layout_lock.setStyleSheet(
            "QPushButton { background: #444; color: #ccc; padding: 8px 16px; }"
            "QPushButton:checked { background: #007acc; color: white; }"
        )
        inner.addWidget(self._btn_layout_lock)

        self._btn_edit_tiles = QPushButton("Edit Tiles (show/hide)")
        self._btn_edit_tiles.setEnabled(False)
        inner.addWidget(self._btn_edit_tiles)

        inner.addSpacing(12)

        btn_reset = QPushButton("Reset Layout to Defaults")
        btn_reset.setToolTip("Restore all tiles and default arrangement")
        btn_reset.clicked.connect(self._on_reset_layout)
        inner.addWidget(btn_reset)

        tab_layout.addWidget(group)

        # Sync with parent launcher
        launcher = self.parent()
        if launcher and hasattr(launcher, "btn_modify_layout"):
            self._btn_layout_lock.setChecked(launcher.btn_modify_layout.isChecked())
            self._btn_layout_lock.toggled.connect(launcher.btn_modify_layout.click)
            self._btn_edit_tiles.clicked.connect(launcher.open_layout_manager)
            self._btn_layout_lock.toggled.connect(self._btn_edit_tiles.setEnabled)

        tab_layout.addStretch()
        return tab

    # ── Configuration tab ───────────────────────────────────────────

    def _create_configuration_tab(self) -> QWidget:
        """Configuration tab: execution env + simulation opts + Docker rebuild."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # --- Execution environment ---
        env_group = QGroupBox("Execution Environment")
        env_inner = QVBoxLayout(env_group)

        self.chk_docker = QCheckBox("Docker mode")
        self.chk_docker.setToolTip(
            "Run physics engines in Docker containers (requires Docker Desktop)"
        )
        env_inner.addWidget(self.chk_docker)

        self.chk_wsl = QCheckBox("WSL mode")
        self.chk_wsl.setToolTip(
            "Run in WSL2 Ubuntu environment (full Pinocchio/Drake/Crocoddyl support)"
        )
        env_inner.addWidget(self.chk_wsl)

        tab_layout.addWidget(env_group)

        # --- Simulation options ---
        sim_group = QGroupBox("Simulation Options")
        sim_inner = QVBoxLayout(sim_group)

        self.chk_live_viz = QCheckBox("Live Visualization")
        self.chk_live_viz.setToolTip(
            "Enable real-time 3D visualization during simulation"
        )
        sim_inner.addWidget(self.chk_live_viz)

        self.chk_gpu = QCheckBox("GPU Acceleration")
        self.chk_gpu.setToolTip(
            "Use GPU for physics computation (requires supported hardware)"
        )
        sim_inner.addWidget(self.chk_gpu)

        tab_layout.addWidget(sim_group)

        # --- Rebuild Environment (Docker build) ---
        build_group = QGroupBox("Rebuild Environment")
        build_inner = QVBoxLayout(build_group)

        stage_row = QHBoxLayout()
        stage_row.addWidget(QLabel("Target Stage:"))
        self.combo_stage = QComboBox()
        self.combo_stage.addItems(["all", "mujoco", "pinocchio", "drake", "base"])
        stage_row.addWidget(self.combo_stage)
        build_inner.addLayout(stage_row)

        btn_build = QPushButton("Build Environment")
        btn_build.clicked.connect(self._start_build)
        build_inner.addWidget(btn_build)

        self.build_console = QTextEdit()
        self.build_console.setReadOnly(True)
        self.build_console.setMaximumHeight(150)
        self.build_console.setStyleSheet(
            "background-color: #1e1e1e; color: #00ff00;"
            "font-family: 'Cascadia Code', Consolas, monospace; font-size: 11px;"
        )
        build_inner.addWidget(self.build_console)

        tab_layout.addWidget(build_group)

        # Sync checkboxes with parent launcher state
        launcher = self.parent()
        if launcher and hasattr(launcher, "chk_docker"):
            self.chk_docker.setChecked(launcher.chk_docker.isChecked())
            self.chk_wsl.setChecked(launcher.chk_wsl.isChecked())
            self.chk_live_viz.setChecked(launcher.chk_live.isChecked())
            self.chk_gpu.setChecked(launcher.chk_gpu.isChecked())

            self.chk_docker.toggled.connect(launcher.chk_docker.setChecked)
            self.chk_wsl.toggled.connect(launcher.chk_wsl.setChecked)
            self.chk_live_viz.toggled.connect(launcher.chk_live.setChecked)
            self.chk_gpu.toggled.connect(launcher.chk_gpu.setChecked)

        return tab

    # ── Diagnostics tab ─────────────────────────────────────────────

    def _create_diagnostics_tab(self) -> QWidget:
        """Diagnostics tab: system checks, error log viewer, terminal output."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # System checks browser
        self._diag_browser = QTextBrowser()
        self._diag_browser.setOpenExternalLinks(False)
        self._diag_browser.setStyleSheet(
            "QTextBrowser {"
            "  background-color: #1e1e1e; color: #d4d4d4;"
            "  font-family: 'Segoe UI', sans-serif; font-size: 13px;"
            "  padding: 12px;"
            "}"
        )
        tab_layout.addWidget(self._diag_browser, stretch=3)

        if self._diagnostics_data:
            self._render_diagnostics(self._diagnostics_data)

        # Process output log viewer
        proc_group = QGroupBox("Process Output Log (recent)")
        proc_inner = QVBoxLayout(proc_group)
        self._proc_log_viewer = QTextEdit()
        self._proc_log_viewer.setReadOnly(True)
        self._proc_log_viewer.setMaximumHeight(180)
        self._proc_log_viewer.setStyleSheet(
            "QTextEdit {"
            "  background-color: #0d0d0d; color: #00ff00;"
            "  font-family: 'Cascadia Code', Consolas, monospace;"
            "  font-size: 11px;"
            "}"
        )
        proc_inner.addWidget(self._proc_log_viewer)
        tab_layout.addWidget(proc_group, stretch=1)

        # Application log viewer
        log_group = QGroupBox("Application Log (recent)")
        log_inner = QVBoxLayout(log_group)
        self._log_viewer = QTextEdit()
        self._log_viewer.setReadOnly(True)
        self._log_viewer.setMaximumHeight(160)
        self._log_viewer.setStyleSheet(
            "QTextEdit {"
            "  background-color: #0d0d0d; color: #d4d4d4;"
            "  font-family: 'Cascadia Code', Consolas, monospace;"
            "  font-size: 11px;"
            "}"
        )
        log_inner.addWidget(self._log_viewer)
        tab_layout.addWidget(log_group, stretch=1)

        # Load recent log lines
        self._load_process_log()
        self._load_app_log()

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn_refresh = QPushButton("Re-run Diagnostics")
        btn_refresh.setToolTip("Run all diagnostic checks again")
        btn_refresh.clicked.connect(self._refresh_diagnostics)
        btn_row.addWidget(btn_refresh)

        btn_refresh_log = QPushButton("Refresh Logs")
        btn_refresh_log.setToolTip("Reload all log files")
        btn_refresh_log.clicked.connect(self._refresh_all_logs)
        btn_row.addWidget(btn_refresh_log)

        tab_layout.addLayout(btn_row)
        return tab

    def _load_app_log(self) -> None:
        """Load recent lines from the application log file."""
        log_candidates = [
            Path.cwd() / "app_launch.log",
            Path.home() / ".golf_modeling_suite" / "launcher.log",
        ]
        for log_path in log_candidates:
            if log_path.exists():
                try:
                    text = log_path.read_text(encoding="utf-8", errors="replace")
                    lines = text.strip().splitlines()
                    recent = "\n".join(lines[-200:])
                    self._log_viewer.setPlainText(recent)
                    self._log_viewer.moveCursor(
                        self._log_viewer.textCursor().End  # type: ignore[arg-type]
                    )
                    return
                except Exception:
                    pass
        self._log_viewer.setPlainText("(No log file found)")

    def _load_process_log(self) -> None:
        """Load recent lines from the process output log file."""
        log_path = Path.home() / ".golf_modeling_suite" / "process_output.log"
        if log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                lines = text.strip().splitlines()
                recent = "\n".join(lines[-300:])
                self._proc_log_viewer.setPlainText(recent)
                self._proc_log_viewer.moveCursor(
                    self._proc_log_viewer.textCursor().End  # type: ignore[arg-type]
                )
                return
            except Exception:
                pass
        self._proc_log_viewer.setPlainText(
            "(No process output log yet — launch a model to generate output)"
        )

    def _refresh_all_logs(self) -> None:
        """Refresh both log viewers."""
        self._load_process_log()
        self._load_app_log()

    def _render_diagnostics(self, data: dict[str, Any]) -> None:
        """Render diagnostics results as styled HTML."""
        summary = data.get("summary", {})
        checks = data.get("checks", [])
        runtime = data.get("runtime_state", {})
        recommendations = data.get("recommendations", [])

        status = summary.get("status", "unknown").upper()
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        warnings = summary.get("warnings", 0)
        total = summary.get("total_checks", passed + failed + warnings)

        status_color = "#2da44e" if status == "HEALTHY" else "#d29922"
        html = f"""
        <div style="margin-bottom: 12px;">
            <h2 style="color:{status_color}; margin: 0;">Status: {status}</h2>
            <p><b>{total} checks:</b>
                <span style="color:#2da44e;">{passed} passed</span>,
                <span style="color:#f85149;">{failed} failed</span>,
                <span style="color:#d29922;">{warnings} warnings</span>
            </p>
        </div>
        """

        # All check details (pass and fail)
        html += "<h3>Check Results</h3><table style='width:100%;'>"
        for check in checks:
            icon = {"pass": "&#9989;", "fail": "&#10060;", "warning": "&#9888;"}.get(
                check["status"], "&#8226;"
            )
            color = {"pass": "#2da44e", "fail": "#f85149", "warning": "#d29922"}.get(
                check["status"], "#d4d4d4"
            )
            duration = check.get("duration_ms", 0)
            html += (
                f"<tr><td style='color:{color}; padding:2px 6px;'>{icon}</td>"
                f"<td style='padding:2px 6px;'><b>{check['name']}</b></td>"
                f"<td style='padding:2px 6px; color:#a0a0a0;'>{check['message']}</td>"
                f"<td style='padding:2px 6px; color:#666;'>{duration:.0f}ms</td></tr>"
            )
        html += "</table>"

        # Engine availability table (from engine_availability check details)
        engine_check = next(
            (c for c in checks if c["name"] == "engine_availability"), None
        )
        engines = (
            engine_check.get("details", {}).get("engines", []) if engine_check else []
        )
        if engines:
            html += "<h3>Physics Engines</h3>"
            html += (
                "<table style='width:100%; border-collapse:collapse;'>"
                "<tr style='border-bottom:1px solid #333;'>"
                "<th style='padding:4px 8px; text-align:left;'>Engine</th>"
                "<th style='padding:4px 8px; text-align:left;'>Status</th>"
                "<th style='padding:4px 8px; text-align:left;'>Version</th>"
                "<th style='padding:4px 8px; text-align:left;'>Details</th>"
                "</tr>"
            )
            for eng in engines:
                installed = eng.get("installed", False)
                icon = "&#9989;" if installed else "&#10060;"
                color = "#2da44e" if installed else "#f85149"
                name = eng.get("name", "?").replace("_", " ").title()
                version = eng.get("version") or "-"
                diag = eng.get("diagnostic", "")
                missing = eng.get("missing_deps", [])
                detail_str = diag
                if missing and not installed:
                    detail_str = f"Missing: {', '.join(missing[:3])}"
                html += (
                    f"<tr>"
                    f"<td style='padding:3px 8px;'><b>{name}</b></td>"
                    f"<td style='padding:3px 8px; color:{color};'>{icon} "
                    f"{'Installed' if installed else 'Not installed'}</td>"
                    f"<td style='padding:3px 8px; color:#a0a0a0;'>{version}</td>"
                    f"<td style='padding:3px 8px; color:#888;'>{detail_str}</td>"
                    f"</tr>"
                )
            html += "</table>"

        # Runtime state
        if runtime:
            html += "<h3>Runtime State</h3><ul>"
            html += f"<li>Available models: {runtime.get('available_models_count', '?')}</li>"
            html += f"<li>Tile order: {runtime.get('model_order_count', '?')}</li>"
            html += f"<li>Model cards: {runtime.get('model_cards_count', '?')}</li>"
            html += f"<li>Registry loaded: {runtime.get('registry_loaded', '?')}</li>"
            html += f"<li>Docker available: {runtime.get('docker_available', '?')}</li>"
            html += "</ul>"

        # Recommendations
        if recommendations:
            html += "<h3>Recommendations</h3><ul>"
            for rec in recommendations[:8]:
                html += f"<li>{rec}</li>"
            html += "</ul>"

        self._diag_browser.setHtml(html)

    def _refresh_diagnostics(self) -> None:
        """Re-run diagnostics and update the display."""
        try:
            from src.launchers.launcher_diagnostics import LauncherDiagnostics

            diag = LauncherDiagnostics()
            results = diag.run_all_checks()

            launcher = self.parent()
            if launcher and hasattr(launcher, "available_models"):
                results["runtime_state"] = {
                    "available_models_count": len(launcher.available_models),
                    "available_model_ids": list(launcher.available_models.keys()),
                    "model_order_count": len(launcher.model_order),
                    "model_order": launcher.model_order,
                    "model_cards_count": len(launcher.model_cards),
                    "selected_model": launcher.selected_model,
                    "docker_available": launcher.docker_available,
                    "registry_loaded": launcher.registry is not None,
                }

            self._diagnostics_data = results
            self._render_diagnostics(results)
        except Exception as e:
            self._diag_browser.setHtml(
                f"<p style='color:#f85149;'>Error running diagnostics: {e}</p>"
            )

    def _on_reset_layout(self) -> None:
        self.reset_layout_requested.emit()

    def _start_build(self) -> None:
        self.build_console.clear()
        self.build_thread = DockerBuildThread(self.combo_stage.currentText())
        self.build_thread.log_signal.connect(self.build_console.append)
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
