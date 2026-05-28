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
from typing import Any

from src.shared.python.engine_core.engine_availability import PYQT6_AVAILABLE

logger = logging.getLogger(__name__)

if PYQT6_AVAILABLE:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except ImportError:
        PYQT6_AVAILABLE = False

if not PYQT6_AVAILABLE:
    QtCore = None  # type: ignore[assignment]
    QtGui = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]

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


class UpstreamDriftLauncher(QtWidgets.QMainWindow if PYQT6_AVAILABLE else object):  # type: ignore[misc,valid-type]
    """Simple launcher window for UpstreamDrift physics engines.

    Provides engine-launch buttons, a log text area, and a status bar.
    Designed to be fully testable with mocked PyQt6.
    """

    def __init__(self) -> None:
        """Initialise the launcher window and build the UI."""
        super().__init__()
        self.suite_root = _REPO_ROOT
        self.script_dir = Path(__file__).resolve().parent
        self.mujoco_path = _engine_script("mujoco", "humanoid_launcher.py")
        self.drake_path = _engine_script("drake", "drake_gui_app.py")
        self.pinocchio_path = _engine_script("pinocchio", "gui.py")
        self.opensim_path = _engine_script("opensim", "gui.py")
        self.myosim_path = _engine_script("myosim", "gui.py")
        self.openpose_path = _REPO_ROOT / "src" / "tools" / "openpose" / "gui.py"
        self.urdf_path = _REPO_ROOT / "src" / "tools" / "urdf_generator" / "gui.py"
        self.pendulum_path = _engine_script("pendulums", "gui.py")

        # UI elements
        self.status = QtWidgets.QLabel("Ready")
        self.log_text = QtWidgets.QTextEdit()
        self.copy_btn = QtWidgets.QPushButton("Copy Log")
        self.clear_btn = QtWidgets.QPushButton("Clear")

        self.btn_mujoco = QtWidgets.QPushButton("MuJoCo")
        self.btn_drake = QtWidgets.QPushButton("Drake")
        self.btn_pinocchio = QtWidgets.QPushButton("Pinocchio")
        self.btn_opensim = QtWidgets.QPushButton("OpenSim")
        self.btn_myosim = QtWidgets.QPushButton("MyoSim")
        self.btn_openpose = QtWidgets.QPushButton("OpenPose")
        self.btn_urdf = QtWidgets.QPushButton("URDF Generator")
        self.btn_shot_tracer = QtWidgets.QPushButton("Shot Tracer")

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Configure top-level window properties and lay out widgets."""
        try:
            self.setWindowTitle("UpstreamDrift - Local Launcher (DEPRECATED)")
            self.resize(800, 600)

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            layout = QtWidgets.QVBoxLayout()
            central.setLayout(layout)

            row = QtWidgets.QHBoxLayout()
            row.addWidget(self.btn_mujoco)
            row.addWidget(self.btn_drake)
            row.addWidget(self.btn_pinocchio)
            row.addWidget(self.btn_opensim)
            row.addWidget(self.btn_myosim)
            row.addWidget(self.btn_openpose)
            row.addWidget(self.btn_urdf)
            row.addWidget(self.btn_shot_tracer)
            layout.addLayout(row)

            layout.addWidget(self.log_text)

            bottom = QtWidgets.QHBoxLayout()
            bottom.addWidget(self.status)
            bottom.addWidget(self.copy_btn)
            bottom.addWidget(self.clear_btn)
            layout.addLayout(bottom)

            # Connections
            self.btn_mujoco.clicked.connect(self._launch_mujoco)
            self.btn_drake.clicked.connect(self._launch_drake)
            self.btn_pinocchio.clicked.connect(self._launch_pinocchio)
            self.btn_opensim.clicked.connect(self._launch_opensim)
            self.btn_myosim.clicked.connect(self._launch_myosim)
            self.btn_openpose.clicked.connect(self._launch_openpose)
            self.btn_urdf.clicked.connect(self._launch_urdf)
            self.btn_shot_tracer.clicked.connect(self._launch_shot_tracer)
            self.copy_btn.clicked.connect(self.copy_log)
            self.clear_btn.clicked.connect(self.clear_log)
        except AttributeError:
            # Running under a mock environment without full Qt stubs
            pass

    def _launch_mujoco(self) -> None:
        """Launch MuJoCo engine."""
        self._launch_script("MuJoCo", self.mujoco_path, self.mujoco_path.parent.parent)

    def _launch_drake(self) -> None:
        """Launch Drake engine."""
        self._launch_script("Drake", self.drake_path, self.drake_path.parent.parent)

    def _launch_pinocchio(self) -> None:
        """Launch Pinocchio engine."""
        self._launch_script(
            "Pinocchio", self.pinocchio_path, self.pinocchio_path.parent.parent
        )

    def _launch_opensim(self) -> None:
        """Launch OpenSim engine."""
        self._launch_script(
            "OpenSim", self.opensim_path, self.opensim_path.parent.parent
        )

    def _launch_myosim(self) -> None:
        """Launch MyoSim engine."""
        self._launch_script("MyoSim", self.myosim_path, self.myosim_path.parent.parent)

    def _launch_openpose(self) -> None:
        """Launch OpenPose tool."""
        self._launch_script("OpenPose", self.openpose_path, self.openpose_path.parent)

    def _launch_urdf(self) -> None:
        """Launch URDF Generator tool."""
        self._launch_script("URDF Generator", self.urdf_path, self.urdf_path.parent)

    def _launch_shot_tracer(self) -> None:
        """Launch Shot Tracer tool."""
        shot_tracer_path = self.script_dir / "shot_tracer.py"
        self._launch_script("Shot Tracer", shot_tracer_path, self.suite_root)

    def _launch_script(self, name: str, path: Path, cwd: Path) -> None:
        """Check script exists then Popen it."""
        if not path.exists():
            try:
                if QtWidgets is not None and hasattr(QtWidgets, "QMessageBox"):
                    QtWidgets.QMessageBox.critical(
                        self, "Error", f"Script not found: {path}"
                    )
            except AttributeError:
                pass
            self.status.setText("Error: Script not found")
            return
        try:
            import os

            cmd = [sys.executable, str(path)]
            env = os.environ.copy()
            pythonpath = env.get("PYTHONPATH", "")
            root_str = str(cwd)
            if root_str not in pythonpath:
                env["PYTHONPATH"] = (
                    f"{root_str}{os.pathsep}{pythonpath}" if pythonpath else root_str
                )
            subprocess.Popen(cmd, cwd=str(cwd), env=env)
            self.status.setText(f"{name} Launched")
        except (OSError, subprocess.SubprocessError) as e:
            try:
                if QtWidgets is not None and hasattr(QtWidgets, "QMessageBox"):
                    QtWidgets.QMessageBox.critical(
                        self, "Error", f"Failed to launch: {e}"
                    )
            except AttributeError:
                pass
            self.status.setText("Error")

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
        self.clear_btn.setText("Cleared!")

    def copy_log(self) -> None:
        """Copy log contents to the system clipboard."""
        text = self.log_text.toPlainText()
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard:
            clipboard.setText(text)
            self.copy_btn.setText("Copied!")
            self.status.setText("Log copied")

    def _restore_btn(self, btn: Any, label: str, icon: Any = None) -> None:
        """Restore button label and icon."""
        if btn is not None:
            btn.setText(label)
            if icon is not None:
                btn.setIcon(icon)


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
