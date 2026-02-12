"""Unified Golf Suite Launcher (Local Python Version) - Golf Modeling Suite.

.. deprecated::
    This launcher is deprecated. Use `golf_launcher.py` instead, which supports
    both local and Docker modes. This file is kept for backwards compatibility
    but will be removed in a future version.

Launches the MuJoCo, Drake, and Pinocchio golf model GUIs from a single interface.
This version assumes all dependencies are installed in the local Python environment
or accessible via `sys.executable`.

It does NOT use Docker. For Docker support, use `golf_launcher.py`.
"""

import subprocess
import sys
from pathlib import Path

from src.shared.python.engine_core.engine_availability import PYQT6_AVAILABLE
from src.shared.python.logging_pkg.logging_config import (
    configure_gui_logging,
    get_logger,
)

# Configure logging for GUI application
configure_gui_logging()
logger = get_logger("GolfSuiteLauncher")

if PYQT6_AVAILABLE:
    from PyQt6 import QtCore, QtGui, QtWidgets
else:
    QtWidgets = None  # type: ignore
    QtCore = None  # type: ignore
    QtGui = None  # type: ignore

# UI feedback timing constants
LAUNCH_FEEDBACK_DURATION_MS = (
    2000  # Duration to show feedback messages (e.g., "Copied!")
)


class GolfLauncher(QtWidgets.QMainWindow if PYQT6_AVAILABLE else object):  # type: ignore[misc]
    def __init__(self) -> None:
        if not PYQT6_AVAILABLE:
            raise ImportError("PyQt6 is required to run this launcher.")
        super().__init__()
        self.setWindowTitle("Golf Modeling Suite - Local Launcher")
        self.resize(400, 300)

        # Paths - UPDATED FOR GOLF_MODELING_SUITE
        # Script location: UpstreamDrift/launchers/golf_suite_launcher.py
        # Root: UpstreamDrift/

        self.script_dir = Path(__file__).parent.resolve()
        # launchers/ -> UpstreamDrift/
        self.suite_root = self.script_dir.parent

        # Define paths to the GUI scripts in the new structure
        self.mujoco_path = (
            self.suite_root
            / "src/engines/physics_engines/mujoco/python/humanoid_launcher.py"
        )
        self.drake_path = (
            self.suite_root
            / "src/engines/physics_engines/drake/python/src/drake_gui_app.py"
        )
        self.pinocchio_path = (
            self.suite_root
            / "src/engines/physics_engines/pinocchio/python/pinocchio_golf/gui.py"
        )
        self.opensim_path = (
            self.suite_root
            / "src/engines/physics_engines/opensim/python/opensim_gui.py"
        )
        self.myosim_path = (
            self.suite_root
            / "src/engines/physics_engines/myosuite/python/myosuite_physics_engine.py"
        )
        self.openpose_path = (
            self.suite_root / "src/shared/python/pose_estimation/openpose_gui.py"
        )
        self.urdf_path = (
            self.suite_root / "tools/urdf_generator/launch_urdf_generator.py"
        )

        self._setup_ui()

    def _create_engine_button(
        self,
        label: str,
        tooltip: str,
        accessible_name: str,
        slot: object,
        icon_pixmap: QtWidgets.QStyle.StandardPixmap = QtWidgets.QStyle.StandardPixmap.SP_MediaPlay,
    ) -> QtWidgets.QPushButton:
        """Create a standard engine launch button."""
        btn = QtWidgets.QPushButton(label)
        btn.setMinimumHeight(40)
        btn.setIcon(self.style().standardIcon(icon_pixmap))
        btn.setToolTip(tooltip)
        btn.setAccessibleName(accessible_name)
        btn.clicked.connect(slot)
        return btn

    def _setup_engine_buttons(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Create and add all engine launch buttons to the layout."""
        self.btn_mujoco = self._create_engine_button(
            "Launch &MuJoCo Engine",
            "Launch the MuJoCo physics engine interface",
            "Launch MuJoCo",
            self._launch_mujoco,
        )
        layout.addWidget(self.btn_mujoco)

        self.btn_drake = self._create_engine_button(
            "Launch &Drake Engine",
            "Launch the Drake physics engine interface",
            "Launch Drake",
            self._launch_drake,
        )
        layout.addWidget(self.btn_drake)

        self.btn_pinocchio = self._create_engine_button(
            "Launch &Pinocchio Engine",
            "Launch the Pinocchio physics engine interface",
            "Launch Pinocchio",
            self._launch_pinocchio,
        )
        layout.addWidget(self.btn_pinocchio)

        self.btn_opensim = self._create_engine_button(
            "Launch &OpenSim Golf",
            "Launch OpenSim musculoskeletal modeling",
            "Launch OpenSim",
            self._launch_opensim,
        )
        layout.addWidget(self.btn_opensim)

        self.btn_myosim = self._create_engine_button(
            "Launch &MyoSim Suite",
            "Launch MyoSuite muscle-actuated simulation",
            "Launch MyoSim",
            self._launch_myosim,
        )
        layout.addWidget(self.btn_myosim)

        self.btn_openpose = self._create_engine_button(
            "Launch Open&Pose Analysis",
            "Launch Pose estimation and motion capture",
            "Launch OpenPose",
            self._launch_openpose,
        )
        layout.addWidget(self.btn_openpose)

        self.btn_urdf = self._create_engine_button(
            "Launch &URDF Generator",
            "Launch Interactive URDF model builder",
            "Launch URDF Generator",
            self._launch_urdf,
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_ToolBarHorizontalExtensionButton,
        )
        layout.addWidget(self.btn_urdf)

    def _setup_shot_tracer_section(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Add separator and shot tracer button to the layout."""
        layout.addSpacing(10)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        layout.addSpacing(10)

        self.btn_shot_tracer = self._create_engine_button(
            "Launch &Shot Tracer",
            "Launch the ball flight visualization (Waterloo/Penner model)",
            "Launch Shot Tracer",
            self._launch_shot_tracer,
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_ArrowForward,
        )
        layout.addWidget(self.btn_shot_tracer)

    def _setup_log_area(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Create the simulation log group box with copy/clear controls."""
        layout.addSpacing(20)

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

        self.copy_btn = QtWidgets.QPushButton("&Copy Log")
        self.copy_btn.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            )
        )
        self.copy_btn.setToolTip("Copy the simulation log to clipboard")
        self.copy_btn.setAccessibleName("Copy Log")
        self.copy_btn.clicked.connect(self.copy_log)
        log_controls.addWidget(self.copy_btn)

        self.clear_btn = QtWidgets.QPushButton("C&lear Log")
        self.clear_btn.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogResetButton
            )
        )
        self.clear_btn.setToolTip("Clear the simulation log output")
        self.clear_btn.setAccessibleName("Clear Log")
        self.clear_btn.clicked.connect(self.clear_log)
        log_controls.addWidget(self.clear_btn)

        log_layout.addLayout(log_controls)
        layout.addWidget(log_group)

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

        # Engine launch buttons
        self._setup_engine_buttons(layout)

        # Shot tracer section with separator
        self._setup_shot_tracer_section(layout)

        # Log area with copy/clear controls
        self._setup_log_area(layout)

        layout.addStretch()

        self.status = QtWidgets.QLabel("Ready")
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)

    def log_message(self, message: str) -> None:
        """Add a timestamped message to the log area."""
        import datetime

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def copy_log(self) -> None:
        """Copy the log content to clipboard."""
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.log_text.toPlainText())
            self.log_message("Log copied to clipboard.")

            # Provide immediate feedback on the button
            # We use hardcoded defaults for restoration to prevent race conditions
            # where rapid clicks capture "Copied!" as the original text.
            default_text = "&Copy Log"
            default_icon = self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            )

            self.copy_btn.setText("Copied!")
            self.copy_btn.setIcon(
                self.style().standardIcon(
                    QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton
                )
            )

            # Restore button after feedback duration
            QtCore.QTimer.singleShot(
                LAUNCH_FEEDBACK_DURATION_MS,
                lambda: self._restore_btn(self.copy_btn, default_text, default_icon),
            )

            # Status bar update (clear after 3 seconds)
            self.status.setText("Log copied")
            QtCore.QTimer.singleShot(3000, lambda: self.status.setText("Ready"))

    def _restore_btn(self, btn: QtWidgets.QPushButton, text: str, icon: object) -> None:
        if btn:
            btn.setText(text)
            if icon is not None:
                btn.setIcon(icon)  # type: ignore

    def clear_log(self) -> None:
        """Clear the log text area."""
        self.log_text.clear()
        self.log_message("Log cleared.")

        # Provide immediate feedback on the button
        default_text = "C&lear Log"
        default_icon = self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_DialogResetButton
        )

        self.clear_btn.setText("Cleared!")
        self.clear_btn.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton
            )
        )

        # Restore button after feedback duration
        QtCore.QTimer.singleShot(
            LAUNCH_FEEDBACK_DURATION_MS,
            lambda: self._restore_btn(self.clear_btn, default_text, default_icon),
        )

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
            process = subprocess.Popen([sys.executable, str(path)], cwd=str(cwd))  # noqa: S603
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

    def _launch_opensim(self) -> None:
        cwd = self.opensim_path.parent.parent
        self._launch_script("OpenSim", self.opensim_path, cwd)

    def _launch_myosim(self) -> None:
        cwd = self.myosim_path.parent.parent
        self._launch_script("MyoSim", self.myosim_path, cwd)

    def _launch_openpose(self) -> None:
        cwd = self.openpose_path.parent
        self._launch_script("OpenPose", self.openpose_path, cwd)

    def _launch_urdf(self) -> None:
        cwd = self.urdf_path.parent
        self._launch_script("URDF Generator", self.urdf_path, cwd)

    def _launch_shot_tracer(self) -> None:
        # Shot tracer is in launchers/ directory
        shot_tracer_path = self.script_dir / "shot_tracer.py"
        # CWD should be the suite root for imports to work
        self._launch_script("Shot Tracer", shot_tracer_path, self.suite_root)


def main() -> None:
    if not PYQT6_AVAILABLE:
        # If logger is configured (basic config above), this goes to stderr/stdout
        logger.error(
            "Error: PyQt6 is not installed. Please install it to use the GUI launcher."
        )
        logger.error("Try: pip install PyQt6")
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    window = GolfLauncher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
