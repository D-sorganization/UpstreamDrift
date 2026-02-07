"""Docker management components for the Golf Modeling Suite Launcher.

This module encapsulates Docker build and check threads to improve the
orthogonality of the main launcher application.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

# Reuse existing subprocess utilities
from src.shared.python.secure_subprocess import SecureSubprocessError, secure_run


class DockerCheckThread(QThread):
    """Asynchronous thread to check for Docker availability."""

    result = pyqtSignal(bool)

    def run(self) -> None:
        """Run docker check."""
        try:
            secure_run(
                ["docker", "--version"],
                timeout=5.0,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.result.emit(True)
        except (SecureSubprocessError, FileNotFoundError):
            self.result.emit(False)


class DockerBuildThread(QThread):
    """Asynchronous thread to perform Docker builds with real-time logging."""

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(
        self,
        target_stage: str = "all",
        image_name: str = "robotics_env",
        context_path: Path | None = None,
    ) -> None:
        """Initialize the build thread."""
        super().__init__()
        self.target_stage = target_stage
        self.image_name = image_name
        self.context_path = context_path

    def run(self) -> None:
        """Run the docker build command."""
        if self.context_path is None or not self.context_path.exists():
            self.finished_signal.emit(
                False, f"Invalid Docker context path: {self.context_path}"
            )
            return

        cmd = [
            "docker",
            "build",
            "-t",
            self.image_name,
            "--target",
            self.target_stage,
            "--progress=plain",
            ".",
        ]

        self.log_signal.emit(f"Starting build for target: {self.target_stage}")
        self.log_signal.emit(f"Context: {self.context_path}")
        self.log_signal.emit(f"Command: {' '.join(cmd)}")

        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            # Windows-specific process flags
            creation_flags = 0
            if os.name == "nt":
                creation_flags = 0x08000000  # CREATE_NO_WINDOW

            process = subprocess.Popen(
                cmd,
                cwd=str(self.context_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creation_flags,
            )

            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    if line:
                        self.log_signal.emit(line.strip())
                process.stdout.close()

            process.wait()

            if process.returncode == 0:
                self.finished_signal.emit(True, "Build successful.")
            else:
                self.finished_signal.emit(
                    False, f"Build failed with code {process.returncode}"
                )

        except Exception as e:
            self.finished_signal.emit(False, str(e))


class DockerLauncher:
    """Handles Docker container launching for simulation models.

    This class encapsulates the logic for launching physics simulations
    in Docker containers, including display configuration, GPU support,
    and model-specific launch commands.
    """

    def __init__(
        self, repo_root: Path, image_name: str = "robotics_env:latest"
    ) -> None:
        """Initialize the Docker launcher.

        Args:
            repo_root: Root directory of the repository.
            image_name: Docker image name to use for containers.
        """
        self.repo_root = repo_root
        self.image_name = image_name
        from src.shared.python.logging_config import get_logger

        self.logger = get_logger(__name__)

    def check_image_exists(self) -> bool:
        """Check if the Docker image exists.

        Returns:
            True if the image exists, False otherwise.
        """
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self.image_name],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.warning(f"Failed to check Docker image: {e}")
            return False

    def build_launch_command(
        self,
        model_type: str,
        repo_path: Path,
        use_gpu: bool = False,
    ) -> list[str]:
        """Build the Docker launch command for a model.

        Args:
            model_type: Type of the model (drake, pinocchio, custom_humanoid, etc.)
            repo_path: Path to the model within the repository.
            use_gpu: Whether to enable GPU support.

        Returns:
            List of command arguments for docker run.
        """
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self.repo_root}:/workspace",
            "-e",
            "PYTHONPATH=/workspace:/workspace/src:/workspace/src/shared/python",
        ]

        # Display configuration for GUI apps
        if os.name == "nt":  # Windows
            cmd.extend(
                [
                    "-e",
                    "DISPLAY=host.docker.internal:0",
                    "-e",
                    "MUJOCO_GL=glfw",
                    "-e",
                    "PYOPENGL_PLATFORM=glx",
                    "-e",
                    "QT_QPA_PLATFORM=xcb",
                ]
            )
        else:  # Linux
            disp = os.environ.get("DISPLAY", ":0")
            cmd.extend(
                [
                    "-e",
                    f"DISPLAY={disp}",
                    "-v",
                    "/tmp/.X11-unix:/tmp/.X11-unix",  # nosec B108 - Docker X11 socket mount
                ]
            )

        # GPU Support
        if use_gpu:
            cmd.extend(["--gpus=all"])

        # Port mapping for MeshCat (Drake/Pinocchio)
        if model_type in ("drake", "pinocchio"):
            cmd.extend(["-p", "7000:7000", "-e", "MESHCAT_HOST=0.0.0.0"])

        # Working Directory
        work_dir = (
            f"/workspace/{repo_path.parent.relative_to(self.repo_root).as_posix()}"
        )
        cmd.extend(["-w", work_dir])

        # Python command - determine correct launch command based on model type
        if model_type == "drake":
            cmd.extend([self.image_name, "python", "-m", "src.drake_gui_app"])
        elif model_type == "pinocchio":
            cmd.extend([self.image_name, "python", "pinocchio_golf/gui.py"])
        elif model_type in ("custom_humanoid", "custom_dashboard"):
            cmd.extend([self.image_name, "python", repo_path.name])
        else:
            cmd.extend([self.image_name, "python", repo_path.name])

        return cmd

    def launch_container(
        self,
        model_type: str,
        model_name: str,
        repo_path: Path,
        use_gpu: bool = False,
    ) -> subprocess.Popen[bytes] | None:
        """Launch a Docker container for the given model.

        Args:
            model_type: Type of the model.
            model_name: Display name of the model.
            repo_path: Path to the model within the repository.
            use_gpu: Whether to enable GPU support.

        Returns:
            The process object if successful, None otherwise.
        """
        cmd = self.build_launch_command(model_type, repo_path, use_gpu)
        self.logger.info(f"Docker Launch: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
            )
            return process
        except Exception as e:
            self.logger.error(f"Failed to launch Docker container: {e}")
            return None
