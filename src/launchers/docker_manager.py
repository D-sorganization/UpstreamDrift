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
