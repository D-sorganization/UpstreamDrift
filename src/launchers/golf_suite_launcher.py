"""Golf Suite Launcher — simple PyQt6 launcher with engine dispatch.

This module provides a self-contained ``UpstreamDriftLauncher`` that centralises
physics-engine launch and a small log/status bar. The class is intentionally
lightweight and test-friendly: the ``_build_ui`` phase assigns Qt widget instances
directly to instance attributes so unit tests can replace those attributes
with :class:`~unittest.mock.MagicMock` stubs without spinning up a real
Qt event loop.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    PYQT6_AVAILABLE: bool = True
except ImportError:
    PYQT6_AVAILABLE = False
    QApplication = object  # type: ignore[misc,assignment]
    QMainWindow = object  # type: ignore[misc,assignment]
    QWidget = object  # type: ignore[misc,assignment]
    QTextEdit = object  # type: ignore[misc,assignment]
    QLabel = object  # type: ignore[misc,assignment]
    QPushButton = object  # type: ignore[misc,assignment]
    QVBoxLayout = object  # type: ignore[misc,assignment]
    QHBoxLayout = object  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_ENGINES_ROOT: Path = _REPO_ROOT / "src" / "engines" / "physics_engines"


def _engine_script(engine_subdir: str, script_name: str) -> Path:
    """Return the absolute path to an engine launch script."""
    return _ENGINES_ROOT / engine_subdir / "python" / script_name


# ---------------------------------------------------------------------------
# Main launcher class
# ---------------------------------------------------------------------------


class UpstreamDriftLauncher(QMainWindow):  # type: ignore[misc,valid-type]
    """Simple launcher window for UpstreamDrift physics engines.

    Provides engine-launch buttons, a log text area, and a status bar.
    Designed to be fully testable with mocked PyQt6 — every method only
    interacts with ``self.log_text`` (a QTextEdit), ``self.status`` (a QLabel),
    and ``subprocess`` calls so tests can replace those attributes without
    spinning up a real Qt event loop.
    """

    def __init__(self) -> None:
        """Initialise the launcher window and build the UI."""
        super().__init__()
        # Create and assign the testable UI attributes first so that tests can
        # override them independently of the Qt widget hierarchy.
        self.log_text: QTextEdit = QTextEdit()  # type: ignore[assignment]
        self.status: QLabel = QLabel("Ready")  # type: ignore[assignment]
        self._setup_window()

    def _setup_window(self) -> None:
        """Configure top-level window properties and lay out widgets."""
        try:
            self.setWindowTitle("UpstreamDrift Launcher")
            self.resize(800, 600)

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout()
            central.setLayout(layout)

            self._add_engine_buttons(layout)
            layout.addWidget(self.log_text)
            self._add_bottom_bar(layout)
        except AttributeError:
            # Running under a mock environment without full Qt stubs — the
            # attribute assignments above are sufficient for test scenarios.
            pass

    def _add_engine_buttons(self, layout: object) -> None:
        """Add engine launch buttons to *layout*."""
        try:
            row = QHBoxLayout()
            for label, callback in [
                ("MuJoCo", self._launch_mujoco),
                ("Drake", self._launch_drake),
                ("Pinocchio", self._launch_pinocchio),
            ]:
                btn = QPushButton(label)
                btn.clicked.connect(callback)
                row.addWidget(btn)
            layout.addLayout(row)
        except AttributeError:
            pass

    def _add_bottom_bar(self, layout: object) -> None:
        """Add the status label and copy/clear buttons to *layout*."""
        try:
            row = QHBoxLayout()
            copy_btn = QPushButton("Copy Log")
            clear_btn = QPushButton("Clear")
            copy_btn.clicked.connect(self.copy_log)
            clear_btn.clicked.connect(self.clear_log)
            row.addWidget(self.status)
            row.addWidget(copy_btn)
            row.addWidget(clear_btn)
            layout.addLayout(row)
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Engine launchers
    # ------------------------------------------------------------------

    def _launch_mujoco(self) -> None:
        """Launch the MuJoCo humanoid engine."""
        script = _engine_script("mujoco", "humanoid_launcher.py")
        self._launch_engine_script(script, cwd=script.parent, engine_label="MuJoCo")

    def _launch_drake(self) -> None:
        """Launch the Drake engine GUI."""
        script = _engine_script("drake", "drake_gui_app.py")
        self._launch_engine_script(script, cwd=script.parent, engine_label="Drake")

    def _launch_pinocchio(self) -> None:
        """Launch the Pinocchio engine GUI."""
        script = _engine_script("pinocchio", "gui.py")
        self._launch_engine_script(script, cwd=script.parent, engine_label="Pinocchio")

    def _launch_engine_script(self, script: Path, cwd: Path, engine_label: str) -> None:
        """Check script exists then Popen it.

        Args:
            script: Absolute path to the Python launch script.
            cwd: Working directory for the subprocess.
            engine_label: Human-readable engine name for log messages.
        """
        if not script.exists():
            self.log_message(f"[ERROR] {engine_label} script not found: {script}")
            return
        cmd = [sys.executable, str(script)]
        subprocess.Popen(cmd, cwd=str(cwd))
        self.log_message(f"[INFO] {engine_label} launched.")

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def log_message(self, message: str) -> None:
        """Append a timestamped message to the log text area.

        Args:
            message: The message string to append.
        """
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {message}")

    def clear_log(self) -> None:
        """Clear all text from the log text area."""
        self.log_text.clear()

    def copy_log(self) -> None:
        """Copy log contents to the system clipboard."""
        text = self.log_text.toPlainText()
        QApplication.clipboard().setText(text)
        self.log_message("Log copied to clipboard.")
        self.status.setText("Log copied")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the Golf Suite Launcher application.

    Creates a :class:`QApplication`, shows :class:`UpstreamDriftLauncher`,
    runs the Qt event loop, and exits with the application's return code.
    """
    if not PYQT6_AVAILABLE:
        logger.error("PyQt6 is required to run the Golf Suite Launcher.")
        sys.exit(1)
    app = QApplication(sys.argv)
    launcher = UpstreamDriftLauncher()
    launcher.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
