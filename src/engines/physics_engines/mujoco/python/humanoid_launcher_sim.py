"""SimulationMixin -- Simulation, Docker, and result methods for HumanoidLauncher."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox

from src.shared.python.ui.qt.process_worker import ProcessWorker

logger = logging.getLogger(__name__)


class SimulationMixin:
    """Simulation launch, Docker, and result-viewing methods for HumanoidLauncher."""

    def _get_docker_internal_command(self) -> tuple[list[str], dict[str, str] | None]:
        cmd = ["python", "-m", "humanoid_golf.sim"]
        env = {"PYTHONPATH": "../docker/src"}
        return cmd, env

    def _get_docker_base_cmd(self, abs_repo_path: str) -> tuple[list[str], str]:
        is_windows = platform.system() == "Windows"
        mount_path = abs_repo_path

        if is_windows:
            drive, tail = os.path.splitdrive(abs_repo_path)
            if drive:
                drive_letter = drive[0].lower()
                rel_path = tail.replace("\\", "/")
                wsl_path = f"/mnt/{drive_letter}{rel_path}"
                return ["wsl", "docker", "run"], wsl_path
            else:
                logging.warning(
                    "Repository path '%s' does not start with a drive letter; "
                    "using absolute path directly for Docker mount.",
                    abs_repo_path,
                )
                return ["docker", "run"], abs_repo_path.replace("\\", "/")

        return ["docker", "run"], mount_path

    def _append_display_env(self, cmd: list[str]) -> None:
        is_windows = platform.system() == "Windows"

        if not self.config.live_view:
            cmd.extend(["-e", "MUJOCO_GL=osmesa"])
            return

        if is_windows:
            cmd.extend(["-e", "DISPLAY=host.docker.internal:0"])
            cmd.extend(["-e", "MUJOCO_GL=glfw"])
            cmd.extend(["-e", "PYOPENGL_PLATFORM=glx"])
            cmd.extend(["-e", "QT_AUTO_SCREEN_SCALE_FACTOR=0"])
            cmd.extend(["-e", "QT_SCALE_FACTOR=1"])
            cmd.extend(["-e", "QT_QPA_PLATFORM=xcb"])
        else:
            cmd.extend(["-e", f"DISPLAY={os.environ.get('DISPLAY', ':0')}"])
            cmd.extend(["-e", "MUJOCO_GL=glfw"])
            cmd.extend(["-e", "PYOPENGL_PLATFORM=glx"])
            cmd.extend(["-v", "/tmp/.X11-unix:/tmp/.X11-unix"])  # nosec B108

    def get_simulation_command(self) -> tuple[list[str], dict[str, str] | None]:
        """Construct the command to run the simulation.

        Returns:
            Tuple of (command_list, environment_dict).
        """

        if Path("/.dockerenv").exists():
            return self._get_docker_internal_command()

        abs_repo_path = str(self.repo_path.resolve())
        cmd, mount_path = self._get_docker_base_cmd(abs_repo_path)

        cmd.extend(
            ["--rm", "-v", f"{mount_path}:/workspace", "-w", "/workspace/docker/src"]
        )

        self._append_display_env(cmd)

        cmd.extend(["robotics_env", "/opt/mujoco-env/bin/python", "-u"])
        cmd.extend(["-m", "humanoid_golf.sim"])

        return cmd, None

    def start_simulation(self) -> None:
        """Save config and launch the simulation in a worker process."""
        self.save_config()

        self.log("Starting simulation...")

        # Reset recorder data for new run

        self.recorder.reset()

        cmd, env = self.get_simulation_command()

        self.simulation_thread = ProcessWorker(cmd, env=env)

        self.simulation_thread.log_signal.connect(self.log)

        self.simulation_thread.finished_signal.connect(self.on_simulation_finished)

        self.simulation_thread.start()

        self.btn_run.setEnabled(False)

        self.btn_stop.setEnabled(True)

    def stop_simulation(self) -> None:
        """Terminate the running simulation worker process."""
        if self.simulation_thread:
            self.log("Stopping simulation...")

            self.simulation_thread.stop()

    def on_simulation_finished(self, code: int, stderr: str) -> None:
        """Handle simulation completion and update UI state."""
        if code == 0:
            self.log("Simulation finished successfully.")

            self.btn_video.setEnabled(True)

            self.btn_data.setEnabled(True)

        elif code == 139:
            self.log(f"Simulation failed with code {code} (Segmentation Fault).")

            self.log(
                "⚠️ COMMON CAUSE: X11 Display Server not found or "
                "configured incorrectly."
            )

            self.log("1. Ensure VcXsrv (XLaunch) is running.")

            self.log("2. Ensure 'Disable access control' is CHECKED in VcXsrv.")

            self.log(
                "3. If you don't need the live GUI, uncheck 'Live Interactive View'."
            )

        else:
            self.log(f"Simulation failed with code {code}.")

        self.btn_run.setEnabled(True)

        self.btn_stop.setEnabled(False)

        # Handle segmentation fault with user prompt for headless mode

        if code == 139:
            reply = QMessageBox.question(
                self,
                "Simulation Crashed (X11 Error)",
                "The simulation crashed due to a display error "
                "(Segmentation Fault).\n\n"
                "This usually means the X11 server (VcXsrv) is not running or "
                "blocked.\n\n"
                "Would you like to try running in Headless Mode instead?\n"
                "(This will disable the live view but still generate video results)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.log("Switching to Headless Mode and retrying...")

                self.chk_live.setChecked(False)  # Uncheck box

                # Config is saved automatically in start_simulation

                self.start_simulation()

    def rebuild_docker(self) -> None:
        """Rebuild the Docker simulation environment after confirmation."""
        reply = QMessageBox.question(
            self,
            "Rebuild Environment",
            "This will rebuild the Docker environment. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.log(
                "Rebuilding Docker environment... "
                "(This functionality is simplified here, check terminal)"
            )

            docker_dir = self.repo_path / "docker"

            cmd = ["docker", "build", "-t", "robotics_env", "."]

            # Start worker for build

            self.build_thread = ProcessWorker(cmd, cwd=str(docker_dir))

            self.build_thread.log_signal.connect(self.log)

            self.build_thread.finished_signal.connect(
                lambda c, e: self.log(f"Build complete with code {c}")
            )

            self.build_thread.start()

    def open_video(self) -> None:
        """Open the recorded simulation video in the default player."""
        vid_path = self.repo_path / "docker" / "src" / "humanoid_golf.mp4"

        self._open_file(vid_path)

    def open_data(self) -> None:
        """Open the recorded simulation CSV data file."""
        csv_path = self.repo_path / "docker" / "src" / "golf_data.csv"

        self._open_file(csv_path)

    def _open_file(self, path: Path) -> None:
        if not path.exists():
            QMessageBox.warning(self, "Error", f"File not found: {path}")

            return

        if platform.system() == "Windows" and hasattr(os, "startfile"):
            # Ensure path exists before opening

            if path.exists():
                os.startfile(str(path))

            else:
                logging.error(f"Cannot open non-existent file: {path}")

        elif platform.system() == "Darwin":
            subprocess.run(["open", str(path)], check=False)

        else:
            subprocess.run(["xdg-open", str(path)], check=False)
