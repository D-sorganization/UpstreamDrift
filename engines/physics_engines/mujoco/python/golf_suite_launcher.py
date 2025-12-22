"""Unified Golf Suite Launcher.

Launches the MuJoCo, Drake, and Pinocchio golf model GUIs from a single interface.
"""

import logging
import subprocess
import sys
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GolfSuiteLauncher")


class GolfLauncher(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        """Initialize the launcher."""
        super().__init__()
        self.setWindowTitle("Golf Model Suite Launcher")
        self.resize(400, 300)

        # Paths
        # Accessing sibling repositories assuming standard checkout structure
        # Current: .../MuJoCo_Golf_Swing_Model/python/
        # Root: .../MuJoCo_Golf_Swing_Model/
        # Sibling: .../Drake_Golf_Model/

        # We are at CWD likely.
        # Let's try to resolve paths relative to this script
        self.script_dir = Path(__file__).parent.resolve()
        self.repo_root = self.script_dir.parent  # python/ -> root
        self.repos_dir = self.repo_root.parent  # Repositories/

        self.mujoco_path = self.script_dir / "mujoco_humanoid_golf/advanced_gui.py"
        self.drake_path = self.repos_dir / "Drake_Golf_Model/python/src/golf_gui.py"
        self.pinocchio_path = (
            self.repos_dir / "Pinocchio_Golf_Model/python/pinocchio_golf/gui.py"
        )

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        title = QtWidgets.QLabel("Golf Model Suite")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        layout.addSpacing(20)

        # Buttons
        self.btn_mujoco = QtWidgets.QPushButton("Launch MuJoCo Model")
        self.btn_mujoco.setMinimumHeight(40)
        self.btn_mujoco.clicked.connect(self._launch_mujoco)
        layout.addWidget(self.btn_mujoco)

        self.btn_drake = QtWidgets.QPushButton("Launch Drake Model")
        self.btn_drake.setMinimumHeight(40)
        self.btn_drake.clicked.connect(self._launch_drake)
        layout.addWidget(self.btn_drake)

        self.btn_pinocchio = QtWidgets.QPushButton("Launch Pinocchio Model")
        self.btn_pinocchio.setMinimumHeight(40)
        self.btn_pinocchio.clicked.connect(self._launch_pinocchio)
        layout.addWidget(self.btn_pinocchio)

        layout.addStretch()

        self.status = QtWidgets.QLabel("Ready")
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)

    def _launch_script(self, name: str, path: Path, cwd: Path) -> None:
        """Launch a script in a subprocess."""
        self.status.setText(f"Launching {name}...")
        logger.info("Launching %s from %s", name, path)

        if not path.exists():
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Could not find script:\n{path}"
            )
            self.status.setText("Error")
            return

        try:
            # Launch detached process
            # Use same python interpreter
            subprocess.Popen([sys.executable, str(path)], cwd=str(cwd))  # noqa: S603
            self.status.setText(f"{name} Launched")
        except (OSError, subprocess.SubprocessError) as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to launch {name}:\n{e}"
            )
            self.status.setText("Error")

    def _launch_mujoco(self) -> None:
        """Launch the MuJoCo model GUI."""
        # CWD should be python root of mujoco repo for imports to work usually
        cwd = self.mujoco_path.parent.parent
        self._launch_script("MuJoCo", self.mujoco_path, cwd)

    def _launch_drake(self) -> None:
        """Launch the Drake model GUI inside Docker container."""
        logger.info("Launching Drake GUI in Docker...")
        self.status.setText("Launching Drake (Docker)...")

        # Docker command to run the GUI
        # Assumes:
        # 1. 'robotics_env:latest' image is built
        # 2. X Server (VcXsrv) is running on host at :0
        # 3. Host Repositories are mounted to /workspace
        
        # We need to map the repository root from host to /workspace in container
        repo_root_host = self.repos_dir.resolve()
        
        docker_cmd = [
            "docker", "run", "--rm",
            "-it",
            # Port mapping for Meshcat (Standard Port 7000)
            "-p", "7000:7000",
            # Environment variables
            "-e", "DISPLAY=host.docker.internal:0",
            "-e", "QT_X11_NO_MITSHM=1",
            "-e", "MESHCAT_HOST=0.0.0.0",  # Bind to all interfaces in container
            # Mount workspace
            "-v", f"{repo_root_host}:/workspace",
            # Image
            "robotics_env:latest",
            # Command inside container
            "python", "Golf_Modeling_Suite/engines/physics_engines/drake/python/src/golf_gui.py"
        ]

        logger.info(f"Docker command: {' '.join(str(c) for c in docker_cmd)}")

        try:
            # We use Popen to run it detached
            subprocess.Popen(docker_cmd, cwd=str(self.repo_root))
            self.status.setText("Drake Launched (Docker)")
        except (OSError, subprocess.SubprocessError) as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to launch Drake Docker container:\n{e}"
            )
            self.status.setText("Error")

    def _launch_pinocchio(self) -> None:
        """Launch the Pinocchio model GUI."""
        cwd = self.pinocchio_path.parent.parent  # python/
        self._launch_script("Pinocchio", self.pinocchio_path, cwd)


def main() -> None:
    """Run the application."""
    app = QtWidgets.QApplication(sys.argv)
    window = GolfLauncher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
