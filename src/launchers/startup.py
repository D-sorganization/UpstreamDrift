"""Startup components for the UpstreamDrift Launcher.

Provides the splash screen, async startup worker, and startup result container.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PyQt6 import sip
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap, QMouseEvent, QKeyEvent
from PyQt6.QtWidgets import QApplication, QSplashScreen

from src.launchers.startup_phases import (
    OUTCOME_OK,
    ProviderProbeResult,
    StartupPhaseError,
    StartupPhaseRecord,
    StartupTimeline,
    format_phase_diagnostics,
    probe_tools_provider,
)
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.security.secure_subprocess import secure_run

logger = get_logger(__name__)

# Constants
REPOS_ROOT = Path(__file__).parent.parent.parent.resolve()
ASSETS_DIR = Path(__file__).parent / "assets"

# Named startup phases (issue #8360). One source for the worker, the splash
# text, the diagnostics and the tests.
PHASE_REGISTRY = "Loading model registry"
PHASE_ENGINES = "Initializing engine manager"
PHASE_DOCKER = "Checking Docker status"
PHASE_TOOLS_PROVIDER = "Checking Tools provider (Rate of Closure)"

# Per-phase hard bounds in seconds. Every phase that can block (imports,
# filesystem discovery, subprocess probes) settles within its bound; the
# registry is the only required phase, everything else degrades.
REGISTRY_PHASE_TIMEOUT_S = 20.0
ENGINE_PHASE_TIMEOUT_S = 20.0
DOCKER_PHASE_TIMEOUT_S = 15.0
PROVIDER_PHASE_TIMEOUT_S = 5.0

# Theme availability check
try:
    from src.shared.python.theme import (  # type: ignore[attr-defined]
        CSS_FONT_DISPLAY,
        CSS_FONT_UI,
        Colors,
        Sizes,
        Weights,
        get_display_font,
        get_qfont,
    )

    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False
    CSS_FONT_DISPLAY = (
        '"Outfit", "Inter", "SF Pro Display", "Segoe UI", "Roboto", sans-serif'
    )
    CSS_FONT_UI = '"Outfit", "Inter", "SF Pro Display", "Segoe UI", "Roboto", "Helvetica Neue", system-ui, sans-serif'


def _font_stack_to_families(font_stack: str) -> list[str]:
    """Convert a CSS-style font stack string to ordered Qt family names."""
    if font_stack is None:
        raise ValueError("font_stack must be provided")
    return [family.strip().strip('"') for family in font_stack.split(",")]


def _fallback_qfont(font_stack: str, size: int, weight: QFont.Weight) -> QFont:
    """Create a Qt font from a shared font stack when theme helpers are unavailable."""
    if font_stack is None:
        raise ValueError("font_stack must be provided")
    font = QFont()
    font.setFamilies(_font_stack_to_families(font_stack))
    font.setPointSize(size)
    font.setWeight(weight)
    return font


def _get_theme_colors() -> dict[str, str]:
    """Get the active theme's color mapping.

    Delegates to the canonical package-level accessor (issue #8972), which
    itself falls back to the built-in Dark palette when no theme manager is
    available. Consumers probe the mapping defensively via ``.get``.
    """
    try:
        from src.shared.python.theme.palette import get_current_colors
    except ImportError:
        # Accessor unavailable (partial install); fall back to the built-in
        # Dark palette so startup styling still has a complete mapping.
        from src.shared.python.theme.palette import DARK_THEME

        return DARK_THEME
    return get_current_colors()


class StartupResults:
    """Container for async startup results."""

    def __init__(self) -> None:
        self.registry: Any = None
        self.engine_manager: Any = None
        self.available_engines: list = []
        self.ai_available: bool = False
        self.docker_available: bool = False
        self.startup_time_ms: int = 0
        # Issue #8360: structured per-phase diagnostics and the optional
        # Tools/Rate provider probe, so the shell can report *which* phase
        # degraded instead of inferring it from splash behaviour.
        self.phases: tuple[StartupPhaseRecord, ...] = ()
        self.tools_provider: ProviderProbeResult | None = None

    @classmethod
    def from_dict(cls, data: dict) -> StartupResults:
        """Create StartupResults from worker results dict."""
        if data is None:
            raise ValueError("data must be provided")
        results = cls()
        results.registry = data.get("registry")
        results.engine_manager = data.get("engine_manager")
        results.available_engines = data.get("available_engines", [])
        results.ai_available = data.get("ai_available", False)
        results.docker_available = data.get("docker_available", False)
        results.startup_time_ms = data.get("startup_time_ms", 0)
        results.phases = tuple(data.get("phases", ()))
        results.tools_provider = data.get("tools_provider")
        return results

    @property
    def is_degraded(self) -> bool:
        """True when any recorded phase settled other than ``ok``."""
        return any(phase.outcome != OUTCOME_OK for phase in self.phases)

    def degraded_summary(self) -> str:
        """One-line, user-facing summary of the degraded phases (or ``""``)."""
        degraded = [p for p in self.phases if p.outcome != OUTCOME_OK]
        if not degraded:
            return ""
        parts = [
            f"{p.name}: {p.outcome}" + (f" [{p.category}]" if p.category else "")
            for p in degraded
        ]
        return "Startup degraded - " + "; ".join(parts)

    def format_diagnostics(self) -> str:
        """Copyable timestamped diagnostics for every recorded phase."""
        return format_phase_diagnostics(self.phases, total_ms=self.startup_time_ms)


class SplashScreen(QSplashScreen):
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

    @staticmethod
    def _resolve_theme_colors() -> tuple[str, str, str, str, str]:
        """Return primary, secondary, accent, bar-bg, and quaternary colors."""
        if THEME_AVAILABLE:
            return (
                Colors.TEXT_PRIMARY,
                Colors.TEXT_TERTIARY,
                Colors.PRIMARY,
                Colors.BG_ELEVATED,
                Colors.TEXT_QUATERNARY,
            )
        return ("#FFFFFF", "#A0A0A0", "#0A84FF", "#2D2D2D", "#666666")

    def _draw_logo_and_title(
        self, painter: QPainter, text_primary: str, text_secondary: str
    ) -> None:
        """Draw the logo image, title text, and subtitle."""
        if painter is None:
            raise ValueError("painter must be provided")
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
            else _fallback_qfont(CSS_FONT_DISPLAY, 24, QFont.Weight.Bold)
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
            else _fallback_qfont(CSS_FONT_UI, 11, QFont.Weight.Normal)
        )
        painter.setFont(subtitle_font)
        painter.setPen(QColor(text_secondary))
        painter.drawText(
            self.rect().adjusted(20, title_y + 38, -20, 0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "Biomechanics & Robotics Platform",
        )

    def _draw_progress_bar(self, painter: QPainter, accent: str, bg_bar: str) -> None:
        """Draw the loading status text and progress bar."""
        if painter is None:
            raise ValueError("painter must be provided")
        status_font = (
            get_qfont(size=Sizes.SM, weight=Weights.MEDIUM)
            if THEME_AVAILABLE
            else _fallback_qfont(CSS_FONT_UI, 9, QFont.Weight.Medium)
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

    def _draw_version_labels(self, painter: QPainter, text_quaternary: str) -> None:
        """Draw the version and branding labels at the bottom."""
        if painter is None:
            raise ValueError("painter must be provided")
        version_font = (
            get_qfont(size=Sizes.XS, weight=Weights.NORMAL)
            if THEME_AVAILABLE
            else _fallback_qfont(CSS_FONT_UI, 8, QFont.Weight.Normal)
        )
        painter.setFont(version_font)
        painter.setPen(QColor(text_quaternary))
        try:
            from src.launchers.about_dialog import _resolve_app_version

            version_str = f"v{_resolve_app_version()}"
        except ImportError:
            version_str = "v1.0.0-beta"
        painter.drawText(
            self.rect().adjusted(20, 0, -16, -12),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
            version_str,
        )
        painter.drawText(
            self.rect().adjusted(16, 0, -20, -12),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
            "UpstreamDrift",
        )

    def drawContents(self, painter: QPainter | None) -> None:
        """Paint the splash screen logo, title, progress bar, and version."""
        if painter is None:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        text_primary, text_secondary, accent, bg_bar, text_quaternary = (
            self._resolve_theme_colors()
        )

        self._draw_logo_and_title(painter, text_primary, text_secondary)
        self._draw_progress_bar(painter, accent, bg_bar)
        self._draw_version_labels(painter, text_quaternary)

    def is_alive(self) -> bool:
        """False once the underlying C++ splash widget has been deleted."""
        return not sip.isdeleted(self)

    def show_message(self, message: str, progress: int) -> None:
        """Update the displayed loading message and progress percentage.

        A late worker callback arriving after the splash was deleted is a
        no-op (issue #8360) instead of ``RuntimeError: wrapped C/C++ object
        ... has been deleted``.
        """
        if message is None:
            raise ValueError("message must be provided")
        if not self.is_alive():
            return
        self.loading_message = message
        self.progress = progress
        self.showMessage(
            message, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter
        )
        self.repaint()
        QApplication.processEvents()

    def show_degraded(self, summary: str) -> None:
        """Switch from "waiting" to an explicit degraded/failed state."""
        if not summary:
            raise ValueError("summary must be non-empty")
        self.show_message(f"Startup problem: {summary}", self.progress)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Allow the user to dismiss the splash screen by clicking it."""
        self.close()
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Allow the user to dismiss the splash screen by pressing Escape."""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        super().keyPressEvent(event)


class AsyncStartupWorker(QThread):
    """Background worker for async application startup.

    Every phase runs through :class:`StartupTimeline` with an explicit
    timeout (issue #8360). Only the model registry is required; the engine
    manager, Docker probe and optional Tools/Rate provider degrade to
    ``None``/``False``/unavailable and the shell still opens.
    """

    progress_signal = pyqtSignal(str, int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, repos_root: Path) -> None:
        if repos_root is None:
            raise ValueError("repos_root must be provided")
        super().__init__()
        self.repos_root = repos_root
        self.results = StartupResults()
        self.timeline = StartupTimeline()

    def diagnostics_text(self) -> str:
        """Timestamped diagnostics including any phase still running."""
        return self.timeline.format_diagnostics()

    # -- phase bodies (run on bounded helper threads) -------------------------

    def _load_registry(self) -> Any:
        from src.shared.python.config.model_registry import ModelRegistry

        return ModelRegistry(self.repos_root / "src/config/models.yaml")

    def _load_engine_manager(self) -> Any:
        from src.shared.python.engine_core.engine_manager import EngineManager

        return EngineManager(self.repos_root)

    def _probe_docker(self) -> bool:
        # Use the WSL-aware resolver so Windows hosts running Docker
        # inside WSL pass the probe instead of hitting WinError 2 on
        # a non-existent native ``docker.exe``.
        from src.launchers.docker_manager import get_docker_cmd

        docker_cmd = get_docker_cmd() + ["--version"]
        # check=False so an absent Docker (a normal non-zero exit) does
        # not travel through secure_run's ERROR-logging exception path:
        # "Docker not installed" is expected degradation, not a failure
        # (#6613). Timeouts/other errors still raise and settle the phase.
        probe = secure_run(docker_cmd, timeout=10.0, check=False)
        if probe.returncode != 0:
            logger.debug("Docker not available (probe exit code %s)", probe.returncode)
        return probe.returncode == 0

    def _probe_tools_provider(self) -> ProviderProbeResult:
        return probe_tools_provider(self.repos_root, os.environ.get("TOOLS_REPO_PATH"))

    @staticmethod
    def _docker_outcome(available: bool) -> tuple[str, str, str]:
        return OUTCOME_OK, "", "available" if available else "not available"

    # -- worker entry point -----------------------------------------------------

    def _finish_timeline(self) -> None:
        self.results.phases = self.timeline.records
        self.results.startup_time_ms = self.timeline.elapsed_ms()

    def run(self) -> None:
        """Execute bounded startup phases in a background thread."""
        timeline = self.timeline
        try:
            self.progress_signal.emit(f"{PHASE_REGISTRY}...", 10)
            self.results.registry = timeline.run_phase(
                PHASE_REGISTRY,
                self._load_registry,
                timeout_s=REGISTRY_PHASE_TIMEOUT_S,
                required=True,
            )

            self.progress_signal.emit(f"{PHASE_ENGINES}...", 30)
            self.results.engine_manager = timeline.run_phase(
                PHASE_ENGINES,
                self._load_engine_manager,
                timeout_s=ENGINE_PHASE_TIMEOUT_S,
            )

            self.progress_signal.emit(f"{PHASE_DOCKER}...", 55)
            self.results.docker_available = bool(
                timeline.run_phase(
                    PHASE_DOCKER,
                    self._probe_docker,
                    timeout_s=DOCKER_PHASE_TIMEOUT_S,
                    outcome_of=self._docker_outcome,
                )
            )

            self.progress_signal.emit(f"{PHASE_TOOLS_PROVIDER}...", 80)
            self.results.tools_provider = timeline.run_phase(
                PHASE_TOOLS_PROVIDER,
                self._probe_tools_provider,
                timeout_s=PROVIDER_PHASE_TIMEOUT_S,
                outcome_of=ProviderProbeResult.phase_outcome,
            )
        except StartupPhaseError as exc:
            self._finish_timeline()
            logger.error("Startup failed in phase %r: %s", exc.record.name, exc)
            self.error_signal.emit(str(exc))
            return

        self._finish_timeline()
        self.progress_signal.emit("Ready", 100)
        self.msleep(500)  # QThread.msleep: non-blocking within the Qt thread scheduler
        self.finished_signal.emit(self.results)
