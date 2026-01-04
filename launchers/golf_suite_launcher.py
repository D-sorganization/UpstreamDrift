"""Unified Golf Suite Launcher (Local Python Version) - Golf Modeling Suite.

Launches the MuJoCo, Drake, and Pinocchio golf model GUIs from a single interface.
This version assumes all dependencies are installed in the local Python environment
or accessible via `sys.executable`.

It does NOT use Docker. For Docker support, use `golf_launcher.py`.
"""

import logging
import subprocess
import sys
from pathlib import Path

try:
    from PyQt6 import QtCore, QtWidgets

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QtWidgets = None  # type: ignore
    QtCore = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GolfSuiteLauncher")


class GolfLauncher(QtWidgets.QMainWindow if PYQT_AVAILABLE else object):  # type: ignore[misc]
    def __init__(self) -> None:
        if not PYQT_AVAILABLE:
            raise ImportError("PyQt6 is required to run this launcher.")
        super().__init__()
        self.setWindowTitle("Golf Modeling Suite - Local Launcher")
        self.resize(400, 300)

        # Paths - UPDATED FOR GOLF_MODELING_SUITE
        # Script location: Golf_Modeling_Suite/launchers/golf_suite_launcher.py
        # Root: Golf_Modeling_Suite/

        self.script_dir = Path(__file__).parent.resolve()
        # launchers/ -> Golf_Modeling_Suite/
        self.suite_root = self.script_dir.parent

        # Define paths to the GUI scripts in the new structure
        self.mujoco_path = (
            self.suite_root
            / "engines/physics_engines/mujoco/python/mujoco_humanoid_golf"
            / "advanced_gui.py"
        )
        self.drake_path = (
            self.suite_root / "engines/physics_engines/drake/python/src/golf_gui.py"
        )
        self.pinocchio_path = (
            self.suite_root
            / "engines/physics_engines/pinocchio/python/pinocchio_golf/gui.py"
        )

        self._setup_ui()

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        title = QtWidgets.QLabel("Golf Modeling Suite (Local)")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Launches physics engines using local python environment"
        )
        subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        # Buttons
        self.btn_mujoco = QtWidgets.QPushButton("Launch MuJoCo Engine")
        self.btn_mujoco.setMinimumHeight(40)
        self.btn_mujoco.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.btn_mujoco.setToolTip("Launch the MuJoCo physics engine interface")
        self.btn_mujoco.setAccessibleName("Launch MuJoCo")
        self.btn_mujoco.clicked.connect(self._launch_mujoco)
        layout.addWidget(self.btn_mujoco)

        self.btn_drake = QtWidgets.QPushButton("Launch Drake Engine")
        self.btn_drake.setMinimumHeight(40)
        self.btn_drake.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.btn_drake.setToolTip("Launch the Drake physics engine interface")
        self.btn_drake.setAccessibleName("Launch Drake")
        self.btn_drake.clicked.connect(self._launch_drake)
        layout.addWidget(self.btn_drake)

        self.btn_pinocchio = QtWidgets.QPushButton("Launch Pinocchio Engine")
        self.btn_pinocchio.setMinimumHeight(40)
        self.btn_pinocchio.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.btn_pinocchio.setToolTip("Launch the Pinocchio physics engine interface")
        self.btn_pinocchio.setAccessibleName("Launch Pinocchio")
        self.btn_pinocchio.clicked.connect(self._launch_pinocchio)
        layout.addWidget(self.btn_pinocchio)

        layout.addSpacing(20)

        # Log area
        log_group = QtWidgets.QGroupBox("Simulation Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setMaximumHeight(120)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "background-color: #2b2b2b; "
            "color: #ffffff; "
            "font-family: 'Consolas', monospace;"
        )
        log_layout.addWidget(self.log_text)

        log_controls = QtWidgets.QHBoxLayout()
        log_controls.addStretch()

        copy_btn = QtWidgets.QPushButton("Copy Log")
        copy_btn.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            )
        )
        copy_btn.setToolTip("Copy log contents to clipboard")
        copy_btn.setAccessibleName("Copy Log")
        copy_btn.clicked.connect(self.copy_log)
        log_controls.addWidget(copy_btn)

        clear_btn = QtWidgets.QPushButton("Clear Log")
        clear_btn.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogResetButton
            )
        )
        clear_btn.setToolTip("Clear the simulation log output")
        clear_btn.setAccessibleName("Clear Log")
        clear_btn.clicked.connect(self.clear_log)
        log_controls.addWidget(clear_btn)

        log_layout.addLayout(log_controls)
        layout.addWidget(log_group)

        layout.addStretch()

        self.status = QtWidgets.QLabel("Ready")
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)

    def log_message(self, message: str) -> None:
        """Add a timestamped message to the log area."""
        import datetime

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def clear_log(self) -> None:
        """Clear the log text area."""
        self.log_text.clear()
        self.log_message("Log cleared.")

    def copy_log(self) -> None:
        """Copy log text to clipboard."""
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard:
            clipboard.setText(self.log_text.toPlainText())
            self.log_message("Log copied to clipboard.")

    def _launch_script(self, name: str, path: Path, cwd: Path) -> None:
        self.status.setText(f"Launching {name}...")
        self.log_message(f"Starting {name} engine...")
        self.log_message(f"Script path: {path}")
        self.log_message(f"Working directory: {cwd}")
        logger.info("Launching %s from %s", name, path)

        if not path.exists():
            error_msg = f"Could not find script:\n{path}"
            self.log_message(f"ERROR: Script not found at {path}")
            QtWidgets.QMessageBox.critical(self, "Error", error_msg)
            self.status.setText("Error: Script not found")
            return

        try:
            # Launch detached process
            # Use same python interpreter
            process = subprocess.Popen(
                [sys.executable, str(path)], cwd=str(cwd)
            )  # noqa: S603
            self.log_message(f"{name} launched successfully (PID: {process.pid})")
            self.status.setText(f"{name} Launched")
        except (OSError, subprocess.SubprocessError) as e:
            error_msg = f"Failed to launch {name}:\n{e}"
            self.log_message(f"ERROR: {error_msg}")
            QtWidgets.QMessageBox.critical(self, "Error", error_msg)
            self.status.setText("Error")

    def _launch_mujoco(self) -> None:
        # CWD should be python root of mujoco engine
        cwd = self.mujoco_path.parent.parent
        self._launch_script("MuJoCo", self.mujoco_path, cwd)

    def _launch_drake(self) -> None:
        # CWD should be python/ directory of drake engine
        cwd = self.drake_path.parent.parent
        self._launch_script("Drake", self.drake_path, cwd)

    def _launch_pinocchio(self) -> None:
        # CWD should be python/ directory of pinocchio engine
        cwd = self.pinocchio_path.parent.parent
        self._launch_script("Pinocchio", self.pinocchio_path, cwd)


def main() -> None:
    if not PYQT_AVAILABLE:
        print(
            "Error: PyQt6 is not installed. Please install it to use the GUI launcher."
        )
        print("Try: pip install PyQt6")
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    window = GolfLauncher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
