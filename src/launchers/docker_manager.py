"""Docker management components for the Golf Modeling Suite Launcher.

This module encapsulates Docker build and check threads to improve the
orthogonality of the main launcher application.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def get_docker_cmd() -> list[str]:
    """Get the base docker command, using WSL fallback on Windows if needed.

    Resolution order:
        1. Native ``docker``/``docker.exe`` on PATH (Docker Desktop / Linux)
        2. ``wsl docker`` on Windows when Docker lives inside a WSL distro
        3. Bare ``docker`` as a last resort (subsequent calls will surface
           a clear FileNotFoundError if neither is available)

    The selected command is logged at DEBUG so failed Docker probes are
    diagnosable from launcher logs.
    """
    if shutil.which("docker"):
        logger.debug("Docker resolved to native 'docker' on PATH")
        return ["docker"]
    if os.name == "nt" and shutil.which("wsl"):
        logger.debug("Docker resolved to 'wsl docker' fallback")
        return ["wsl", "docker"]
    logger.debug(
        "Neither 'docker' nor 'wsl' on PATH; falling back to bare 'docker' "
        "(probe will fail)"
    )
    return ["docker"]


from PyQt6.QtCore import QThread, pyqtSignal  # noqa: E402

from src.launchers.launcher_constants import validate_docker_stage  # noqa: E402
from src.shared.python.docker_config import (  # noqa: E402
    DOCKER_IMAGE_ENGINE,
)
from src.shared.python.docker_config import (  # noqa: E402
    LEGACY_DOCKER_ALIASES as LEGACY_DOCKER_IMAGE_ALIASES,
)

# Reuse existing subprocess utilities
from src.shared.python.security.secure_subprocess import (  # noqa: E402
    SecureSubprocessError,
    secure_run,
)


class DockerCheckThread(QThread):
    """Asynchronous thread to check for Docker availability."""

    result = pyqtSignal(bool)

    def run(self) -> None:
        """Run docker check.

        Logs the resolved command and failure cause so the user has visible
        diagnostics when Docker is unavailable. Previously this silently
        emitted ``False`` for any failure, which made the
        ``wsl``-not-whitelisted regression invisible until log inspection.
        """
        cmd = get_docker_cmd() + ["--version"]
        try:
            secure_run(
                cmd,
                timeout=15.0,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.result.emit(True)
        except SecureSubprocessError as exc:
            logger.warning(
                "Docker check rejected by secure_subprocess gate (cmd=%s): %s",
                cmd,
                exc,
            )
            self.result.emit(False)
        except FileNotFoundError:
            logger.warning(
                "Docker check failed: %s not found on PATH (cmd=%s)",
                cmd[0],
                cmd,
            )
            self.result.emit(False)


class DockerBuildThread(QThread):
    """Asynchronous thread to perform Docker builds with real-time logging."""

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(
        self,
        target_stage: str = "standard",
        image_name: str = DOCKER_IMAGE_ENGINE,
        context_path: Path | None = None,
        dockerfile_path: Path | None = None,
        build_args: dict[str, str] | None = None,
    ) -> None:
        """Initialize the build thread."""
        if target_stage is None:
            raise ValueError("target_stage must be provided")
        super().__init__()
        self.target_stage = validate_docker_stage(target_stage)
        self.image_name = image_name
        self.context_path = context_path
        self.dockerfile_path = dockerfile_path
        self.build_args = build_args or {}

    def run(self) -> None:
        """Run the docker build command."""
        if self.context_path is None or not self.context_path.exists():
            self.finished_signal.emit(
                False, f"Invalid Docker context path: {self.context_path}"
            )
            return

        cmd = get_docker_cmd() + [
            "build",
            "-t",
            self.image_name,
        ]

        if self.dockerfile_path:
            try:
                # If the dockerfile is inside the context, make it relative so both Windows and WSL can read it
                rel_path = self.dockerfile_path.relative_to(self.context_path)
                cmd.extend(["-f", rel_path.as_posix()])
            except ValueError:
                # Fallback to posix path (forward slashes)
                cmd.extend(["-f", self.dockerfile_path.as_posix()])

        for k, v in self.build_args.items():
            cmd.extend(["--build-arg", f"{k}={v}"])

        if "PROFILE" not in self.build_args and "FEATURES" not in self.build_args:
            cmd.extend(["--target", self.target_stage])

        cmd.extend(["--progress=plain", "."])

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
                    if line.strip():
                        self.log_signal.emit(line.strip())
                process.stdout.close()

            # Add timeout to prevent indefinite hangs (issue #2715)
            try:
                process.wait(timeout=3600)  # 1-hour limit
            except subprocess.TimeoutExpired:
                self.log_signal.emit(
                    "Build timed out after 1 hour; terminating process"
                )
                process.kill()
                self.finished_signal.emit(
                    False, "Build timed out (exceeded 1 hour limit)"
                )
                return

            if process.returncode == 0:
                self.finished_signal.emit(True, "Build successful.")
            else:
                self.finished_signal.emit(
                    False, f"Build failed with code {process.returncode}"
                )

        except (FileNotFoundError, PermissionError, OSError) as e:
            self.finished_signal.emit(False, str(e))


class DockerLauncher:
    """Handles Docker container launching for simulation models.

    This class encapsulates the logic for launching physics simulations
    in Docker containers, including display configuration, GPU support,
    and model-specific launch commands.
    """

    def __init__(self, repo_root: Path, image_name: str = DOCKER_IMAGE_ENGINE) -> None:
        """Initialize the Docker launcher.

        Args:
            repo_root: Root directory of the repository.
            image_name: Docker image name to use for containers.
        """
        if repo_root is None:
            raise ValueError("repo_root must be provided")
        self.repo_root = repo_root
        self.image_name = image_name
        from src.shared.python.logging_pkg.logging_config import get_logger

        self.logger = get_logger(__name__)

    def check_image_exists(self) -> bool:
        """Check if the Docker image exists.

        Returns:
            True if the image exists, False otherwise.
        """
        try:
            result = subprocess.run(
                get_docker_cmd() + ["image", "inspect", self.image_name],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                return True

            for legacy_image in LEGACY_DOCKER_IMAGE_ALIASES:
                legacy_result = subprocess.run(
                    get_docker_cmd() + ["image", "inspect", legacy_image],
                    capture_output=True,
                    timeout=10,
                )
                if legacy_result.returncode == 0:
                    self.logger.warning(
                        "Using legacy Docker image '%s'. Retag to '%s' when convenient.",
                        legacy_image,
                        self.image_name,
                    )
                    self.image_name = legacy_image
                    return True
            return False
        except (OSError, ValueError) as e:
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
        if model_type is None:
            raise ValueError("model_type must be provided")

        repo_root_str = str(self.repo_root)
        if "wsl" in get_docker_cmd():
            import re

            match = re.match(r"^([a-zA-Z]):(.*)", repo_root_str)
            if match:
                drive = match.group(1).lower()
                path_part = match.group(2).replace("\\", "/")
                repo_root_str = f"/mnt/{drive}{path_part}"
            else:
                repo_root_str = repo_root_str.replace("\\", "/")

        cmd = get_docker_cmd() + [
            "run",
            "--rm",
            "-v",
            f"{repo_root_str}:/workspace",
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
            cmd.extend(["-p", "7000:7000", "-e", "MESHCAT_HOST=0.0.0.0"])  # nosec: Docker container networking requires 0.0.0.0

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
        capture_output: bool = False,
    ) -> subprocess.Popen[bytes] | None:
        """Launch a Docker container for the given model.

        Args:
            model_type: Type of the model.
            model_name: Display name of the model.
            repo_path: Path to the model within the repository.
            use_gpu: Whether to enable GPU support.
            capture_output: If True, pipe stdout/stderr for unified console
                capture instead of opening a separate terminal window.

        Returns:
            The process object if successful, None otherwise.
        """
        if model_type is None:
            raise ValueError("model_type must be provided")
        cmd = self.build_launch_command(model_type, repo_path, use_gpu)
        self.logger.info(f"Docker Launch: {' '.join(cmd)}")

        try:
            if capture_output:
                # Unified console mode: pipe output for streaming
                creationflags = 0
                if os.name == "nt":
                    creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )
            else:
                # Legacy: separate terminal window
                process = subprocess.Popen(
                    cmd,
                    creationflags=(
                        subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0  # type: ignore[attr-defined]
                    ),
                )
            return process
        except (FileNotFoundError, PermissionError, OSError) as e:
            self.logger.error(f"Failed to launch Docker container: {e}")
            return None
