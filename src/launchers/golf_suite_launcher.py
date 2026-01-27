"""Unified Golf Suite Launcher (Local Python Version) - Golf Modeling Suite.

Launches the MuJoCo, Drake, and Pinocchio golf model GUIs from a single interface.
This version assumes all dependencies are installed in the local Python environment
or accessible via `sys.executable`.

It does NOT use Docker. For Docker support, use `golf_launcher.py`.
"""

import subprocess
import sys
from pathlib import Path

from src.shared.python.engine_availability import PYQT6_AVAILABLE
from src.shared.python.logging_config import configure_gui_logging, get_logger

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
        # Script location: Golf_Modeling_Suite/launchers/golf_suite_launcher.py
        # Root: Golf_Modeling_Suite/

        self.script_dir = Path(__file__).parent.resolve()
        # launchers/ -> Golf_Modeling_Suite/
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
        self.btn_mujoco = QtWidgets.QPushButton("Launch &MuJoCo Engine")
        self.btn_mujoco.setMinimumHeight(40)
        self.btn_mujoco.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.btn_mujoco.setToolTip("Launch the MuJoCo physics engine interface")
        self.btn_mujoco.setAccessibleName("Launch MuJoCo")
        self.btn_mujoco.clicked.connect(self._launch_mujoco)
        layout.addWidget(self.btn_mujoco)

        self.btn_drake = QtWidgets.QPushButton("Launch &Drake Engine")
        self.btn_drake.setMinimumHeight(40)
        self.btn_drake.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.btn_drake.setToolTip("Launch the Drake physics engine interface")
        self.btn_drake.setAccessibleName("Launch Drake")
        self.btn_drake.clicked.connect(self._launch_drake)
        layout.addWidget(self.btn_drake)

        self.btn_pinocchio = QtWidgets.QPushButton("Launch &Pinocchio Engine")
        self.btn_pinocchio.setMinimumHeight(40)
        self.btn_pinocchio.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.btn_pinocchio.setToolTip("Launch the Pinocchio physics engine interface")
        self.btn_pinocchio.setAccessibleName("Launch Pinocchio")
        self.btn_pinocchio.clicked.connect(self._launch_pinocchio)
        layout.addWidget(self.btn_pinocchio)

        self.btn_opensim = QtWidgets.QPushButton("Launch &OpenSim Golf")
        self.btn_opensim.setMinimumHeight(40)
        self.btn_opensim.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.btn_opensim.setToolTip("Launch OpenSim musculoskeletal modeling")
        self.btn_opensim.setAccessibleName("Launch OpenSim")
        self.btn_opensim.clicked.connect(self._launch_opensim)
        layout.addWidget(self.btn_opensim)

        self.btn_myosim = QtWidgets.QPushButton("Launch &MyoSim Suite")
        self.btn_myosim.setMinimumHeight(40)
        self.btn_myosim.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.btn_myosim.setToolTip("Launch MyoSuite muscle-actuated simulation")
        self.btn_myosim.setAccessibleName("Launch MyoSim")
        self.btn_myosim.clicked.connect(self._launch_myosim)
        layout.addWidget(self.btn_myosim)

        self.btn_openpose = QtWidgets.QPushButton("Launch Open&Pose Analysis")
        self.btn_openpose.setMinimumHeight(40)
        self.btn_openpose.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.btn_openpose.setToolTip("Launch Pose estimation and motion capture")
        self.btn_openpose.setAccessibleName("Launch OpenPose")
        self.btn_openpose.clicked.connect(self._launch_openpose)
        layout.addWidget(self.btn_openpose)

        self.btn_urdf = QtWidgets.QPushButton("Launch &URDF Generator")
        self.btn_urdf.setMinimumHeight(40)
        self.btn_urdf.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_ToolBarHorizontalExtensionButton
            )
        )
        self.btn_urdf.setToolTip("Launch Interactive URDF model builder")
        self.btn_urdf.setAccessibleName("Launch URDF Generator")
        self.btn_urdf.clicked.connect(self._launch_urdf)
        layout.addWidget(self.btn_urdf)

        layout.addSpacing(10)

        # Separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        layout.addSpacing(10)

        # Shot Tracer - Ball Flight Visualization
        self.btn_shot_tracer = QtWidgets.QPushButton("Launch &Shot Tracer")
        self.btn_shot_tracer.setMinimumHeight(40)
        self.btn_shot_tracer.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowForward)
        )
        self.btn_shot_tracer.setToolTip(
            "Launch the ball flight visualization (Waterloo/Penner model)"
        )
        self.btn_shot_tracer.setAccessibleName("Launch Shot Tracer")
        self.btn_shot_tracer.clicked.connect(self._launch_shot_tracer)
        layout.addWidget(self.btn_shot_tracer)

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
