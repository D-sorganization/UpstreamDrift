"""External tool adapters for cross-repo GUI embedding.

Provides a mechanism for the UpstreamDrift unified launcher to discover
and instantiate GUI tools that live in external repositories (e.g., the
``Tools`` repository). Each adapter wraps one external tool and exposes
the standard ``get_dockable_ui()`` protocol.

The adapters gracefully degrade to a status widget when the external
repository or its dependencies are not available.

Design by Contract
------------------
Pre:  External repo paths must be resolvable via ``TOOLS_REPO_PATH`` or
      auto-discovery from the sibling directory.
Post: ``get_dockable_ui()`` always returns a valid QMainWindow, even if
      the external tool is unavailable (shows error state).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# External repo discovery
# ---------------------------------------------------------------------------

_TOOLS_REPO: Path | None = None


def _find_tools_repo() -> Path | None:
    """Auto-discover the Tools repository as a sibling of UpstreamDrift."""
    global _TOOLS_REPO
    if _TOOLS_REPO is not None:
        return _TOOLS_REPO

    # Check environment variable first
    import os

    env_path = os.environ.get("TOOLS_REPO_PATH")
    if env_path and Path(env_path).is_dir():
        _TOOLS_REPO = Path(env_path)
        return _TOOLS_REPO

    # Walk up from this file to find the repository root, then check sibling
    p = Path(__file__).resolve()
    for _ in range(10):
        p = p.parent
        candidate = p / "Tools"
        if (candidate / "src").is_dir():
            _TOOLS_REPO = candidate
            return _TOOLS_REPO
        # Also check parent of parent (Repositories folder)
        if p.name in {"UpstreamDrift", "src"}:
            continue

    logger.warning("Tools repository not found via sibling discovery")
    return None


def _ensure_tools_on_path() -> bool:
    """Ensure the Tools/src directory is on sys.path.

    Returns True if the path was added or already present.
    """
    repo = _find_tools_repo()
    if repo is None:
        return False
    src_dir = str(repo / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        logger.info("Added Tools repo to sys.path: %s", src_dir)
    return True


# ---------------------------------------------------------------------------
# Unavailable-tool placeholder
# ---------------------------------------------------------------------------


class _UnavailableToolWidget(QWidget):
    """Placeholder widget shown when an external tool cannot be loaded."""

    def __init__(
        self, tool_name: str, error: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel(f"⚠ {tool_name}")
        title_font = title.font()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        msg = QLabel(f"This tool could not be loaded:\n\n{error}")
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setStyleSheet("color: #ff9800;")
        layout.addWidget(msg)

        hint = QLabel(
            "Ensure the Tools repository is available as a sibling directory\n"
            "or set the TOOLS_REPO_PATH environment variable."
        )
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)

    def cleanup(self) -> None:
        """No-op cleanup for the placeholder."""


class _UnavailableToolWindow(QMainWindow):
    """Window wrapper for the unavailable tool placeholder."""

    def __init__(self, tool_name: str, error: str) -> None:
        super().__init__()
        self.setWindowTitle(f"{tool_name} (Unavailable)")
        self.setMinimumSize(600, 400)
        self._widget = _UnavailableToolWidget(tool_name, error, self)
        self.setCentralWidget(self._widget)
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage(f"{tool_name}: not available")


# ---------------------------------------------------------------------------
# External tool adapters
# ---------------------------------------------------------------------------


def _wrap_external_widget(tool_name: str, import_func: Any) -> QMainWindow:
    """Attempt to import and wrap an external tool widget.

    Args:
        tool_name: Human-readable tool name.
        import_func: Callable that returns a QWidget instance.

    Returns:
        QMainWindow wrapping the tool widget, or an error placeholder.
    """
    if not _ensure_tools_on_path():
        return _UnavailableToolWindow(tool_name, "Tools repository not found.")
    try:
        widget = import_func()
        window = QMainWindow()
        window.setWindowTitle(tool_name)
        window.setMinimumSize(1000, 700)
        window.setCentralWidget(widget)
        status = QStatusBar()
        window.setStatusBar(status)
        status.showMessage(f"{tool_name} — loaded from Tools repository")
        return window
    except Exception as e:
        logger.exception("Failed to load external tool: %s", tool_name)
        return _UnavailableToolWindow(tool_name, str(e))


# --- Video Analyzer ---


def _import_video_analyzer() -> QWidget:
    from video_analyzer.launch_pyqt6 import VideoAnalyzerWidget  # type: ignore[import-untyped]

    return VideoAnalyzerWidget()


def get_video_analyzer_dockable_ui() -> QMainWindow:
    """Return the Video Analyzer window for docking."""
    return _wrap_external_widget("Video Analyzer", _import_video_analyzer)


# --- Data Explorer ---


def _import_data_explorer() -> QWidget:
    from data_explorer.gui import MainWidget  # type: ignore[import-untyped]

    return MainWidget()


def get_data_explorer_dockable_ui() -> QMainWindow:
    """Return the Data Explorer window for docking."""
    return _wrap_external_widget("Data Explorer", _import_data_explorer)


# --- Data Processor ---


def _import_data_processor() -> QWidget:
    from data_processing.data_processor.gui import (  # type: ignore[import-untyped]
        MainWidget,
    )

    return MainWidget()


def get_data_processor_dockable_ui() -> QMainWindow:
    """Return the Data Processor window for docking."""
    return _wrap_external_widget("Data Processor", _import_data_processor)


# ---------------------------------------------------------------------------
# Convenience registry
# ---------------------------------------------------------------------------

EXTERNAL_TOOLS: dict[str, Any] = {
    "video_analyzer": get_video_analyzer_dockable_ui,
    "data_explorer": get_data_explorer_dockable_ui,
    "data_processor": get_data_processor_dockable_ui,
}
