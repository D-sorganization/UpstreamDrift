"""Startup components for the UpstreamDrift Launcher.

Provides the splash screen, async startup worker, and startup result container.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.security.secure_subprocess import secure_run

if TYPE_CHECKING:
    from src.shared.python.theme.theme_manager import ThemeColors

logger = get_logger(__name__)

# Constants
REPOS_ROOT = Path(__file__).parent.parent.parent.resolve()
ASSETS_DIR = Path(__file__).parent / "assets"

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


def _get_theme_colors() -> ThemeColors:
    """Get current theme colors, with fallback to dark theme defaults."""
    try:
        from src.shared.python.theme import get_current_colors

        return get_current_colors()
    except ImportError:
        from src.shared.python.theme import DARK_THEME

        return DARK_THEME


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

    def __init__(self, repos_root: Path) -> None:
        super().__init__()
        self.repos_root = repos_root
        self.results = StartupResults()

    def run(self) -> None:
        try:
            self.progress_signal.emit("Loading model registry...", 10)
            from src.shared.python.config.model_registry import ModelRegistry

            registry = ModelRegistry(self.repos_root / "src/config/models.yaml")
            self.results.registry = registry

            self.progress_signal.emit("Initializing engine manager...", 30)
            try:
                from src.shared.python.engine_core.engine_manager import EngineManager

                self.results.engine_manager = EngineManager(self.repos_root)
            except ImportError as e:
                logger.warning(f"Engine manager init failed: {e}")
                self.results.engine_manager = None

            self.progress_signal.emit("Checking Docker status...", 60)
            try:
                secure_run(["docker", "--version"], timeout=2.0, check=True)
                self.results.docker_available = True
            except (RuntimeError, ValueError, OSError):
                self.results.docker_available = False
                logger.debug("Docker not available or timed out")

            self.progress_signal.emit("Ready", 100)
            time.sleep(0.5)
            self.finished_signal.emit(self.results)
        except ImportError as e:
            logger.error(f"Startup failed: {e}")
            self.error_signal.emit(str(e))
