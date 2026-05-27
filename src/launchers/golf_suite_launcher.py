"""Golf Suite Launcher — simple PyQt6 launcher with engine dispatch.

This module provides a self-contained ``UpstreamDriftLauncher`` that centralises
physics-engine launch and a small log/status bar.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import contextlib

logger = logging.getLogger(__name__)

try:
    import PyQt6.QtWidgets as QtWidgets
    import PyQt6.QtCore as QtCore
    import PyQt6.QtGui as QtGui
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
    QtWidgets = None
    QtCore = None
    QtGui = None
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
    """Simple launcher window for UpstreamDrift physics engines."""

    def __init__(self) -> None:
        """Initialise the launcher window and build the UI."""
        super().__init__()
        self.suite_root = _REPO_ROOT
        self.mujoco_path = _ENGINES_ROOT / "mujoco"
        self.drake_path = _ENGINES_ROOT / "drake"

        self.log_text: QTextEdit = QTextEdit()  # type: ignore[assignment]
        self.status: QLabel = QLabel("Ready")  # type: ignore[assignment]

        self.btn_mujoco = QPushButton("MuJoCo")
        self.btn_drake = QPushButton("Drake")
        self.btn_pinocchio = QPushButton("Pinocchio")
        self.btn_opensim = QPushButton("OpenSim")
        self.btn_myosim = QPushButton("MyoSim")
        self.btn_openpose = QPushButton("OpenPose")
        self.btn_urdf = QPushButton("URDF")
        self.btn_shot_tracer = QPushButton("Shot Tracer")
        self.copy_btn = QPushButton("Copy Log")
        self.clear_btn = QPushButton("Clear")

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Configure top-level window properties and lay out widgets."""
        try:
            self.setWindowTitle("UpstreamDrift Launcher")
            self.resize(800, 600)

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout()
            central.setLayout(layout)

            # Add buttons to layout
            row = QHBoxLayout()
            for btn in [
                self.btn_mujoco,
                self.btn_drake,
                self.btn_pinocchio,
                self.btn_opensim,
                self.btn_myosim,
                self.btn_openpose,
                self.btn_urdf,
                self.btn_shot_tracer,
            ]:
                row.addWidget(btn)
            layout.addLayout(row)

            # Connect callbacks
            self.btn_mujoco.clicked.connect(self._launch_mujoco)
            self.btn_drake.clicked.connect(self._launch_drake)
            self.btn_pinocchio.clicked.connect(self._launch_pinocchio)
            self.btn_opensim.clicked.connect(self._launch_opensim)
            self.btn_myosim.clicked.connect(self._launch_myosim)
            self.btn_openpose.clicked.connect(self._launch_openpose)
            self.btn_urdf.clicked.connect(self._launch_urdf)
            self.btn_shot_tracer.clicked.connect(self._launch_shot_tracer)

            layout.addWidget(self.log_text)

            bottom = QHBoxLayout()
            bottom.addWidget(self.status)
            bottom.addWidget(self.copy_btn)
            bottom.addWidget(self.clear_btn)
            layout.addLayout(bottom)

            self.copy_btn.clicked.connect(self.copy_log)
            self.clear_btn.clicked.connect(self.clear_log)
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Engine launchers
    # ------------------------------------------------------------------

    def _launch_mujoco(self) -> None:
        self._launch_script(
            "MuJoCo",
            _engine_script("mujoco", "humanoid_launcher.py"),
            _ENGINES_ROOT / "mujoco" / "python",
        )

    def _launch_drake(self) -> None:
        self._launch_script(
            "Drake",
            _engine_script("drake", "drake_gui_app.py"),
            _ENGINES_ROOT / "drake" / "python",
        )

    def _launch_pinocchio(self) -> None:
        self._launch_script(
            "Pinocchio",
            _engine_script("pinocchio", "gui.py"),
            _ENGINES_ROOT / "pinocchio" / "python",
        )

    def _launch_opensim(self) -> None:
        self._launch_script(
            "OpenSim",
            _engine_script("opensim", "gui.py"),
            _ENGINES_ROOT / "opensim" / "python",
        )

    def _launch_myosim(self) -> None:
        self._launch_script(
            "MyoSim",
            _engine_script("myosim", "gui.py"),
            _ENGINES_ROOT / "myosim" / "python",
        )

    def _launch_openpose(self) -> None:
        self._launch_script(
            "OpenPose",
            _engine_script("openpose", "gui.py"),
            _ENGINES_ROOT / "openpose" / "python",
        )

    def _launch_urdf(self) -> None:
        self._launch_script(
            "URDF Generator",
            _engine_script("urdf", "gui.py"),
            _ENGINES_ROOT / "urdf" / "python",
        )

    def _launch_shot_tracer(self) -> None:
        self._launch_script(
            "Shot Tracer",
            _engine_script("shot_tracer", "gui.py"),
            _ENGINES_ROOT / "shot_tracer" / "python",
        )

    def _launch_script(self, engine_name: str, script_path: Path, cwd: Path) -> None:
        if not script_path.exists():
            if QtWidgets is not None:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Script not found: {script_path}"
                )
            self.status.setText("Error: Script not found")
            return
        try:
            import os

            env = os.environ.copy()
            repo_root = Path(__file__).resolve().parents[2]
            pythonpath = env.get("PYTHONPATH", "")
            if pythonpath:
                env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{pythonpath}"
            else:
                env["PYTHONPATH"] = str(repo_root)
            cmd = [sys.executable, str(script_path)]
            subprocess.Popen(cmd, cwd=str(cwd), env=env)
            self.status.setText(f"{engine_name} Launched")
            self.log_message(f"[INFO] {engine_name} launched.")
        except Exception as e:  # noqa: BLE001
            if QtWidgets is not None:
                QtWidgets.QMessageBox.critical(self, "Error", str(e))
            self.status.setText("Error")

    def _restore_btn(self, btn: Any, text: str, icon: Any) -> None:
        if btn is not None:
            btn.setText(text)
            if icon is not None:
                btn.setIcon(icon)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def log_message(self, message: str) -> None:
        """Append a timestamped message to the log text area."""
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {message}")

    def clear_log(self) -> None:
        """Clear all text from the log text area."""
        self.log_text.clear()
        with contextlib.suppress(AttributeError):
            self.clear_btn.setText("Cleared!")

    def copy_log(self) -> None:
        """Copy log contents to the system clipboard."""
        try:
            text = self.log_text.toPlainText()
            clipboard = (
                QtWidgets.QApplication.clipboard() if QtWidgets is not None else None
            )
            if clipboard is not None:
                clipboard.setText(text)
                self.log_message("Log copied to clipboard.")
                self.status.setText("Log copied")
                self.copy_btn.setText("Copied!")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the Golf Suite Launcher application."""
    if not PYQT6_AVAILABLE:
        logger.error("PyQt6 is required to run the Golf Suite Launcher.")
        sys.exit(1)
    app = QtWidgets.QApplication(sys.argv)
    launcher = UpstreamDriftLauncher()
    launcher.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
