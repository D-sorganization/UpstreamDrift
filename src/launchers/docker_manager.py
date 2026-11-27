"""Docker management components for the Golf Modeling Suite Launcher.

This module encapsulates Docker build and check threads to improve the
orthogonality of the main launcher application.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from src.shared.python.core.process_safety import is_pid_alive

logger = logging.getLogger(__name__)

CONTAINER_LABEL_MANAGED = "upstreamdrift.managed=true"
CONTAINER_LABEL_PID_KEY = "upstreamdrift.launcher_pid"
DEFAULT_MESHCAT_PORT = 7000


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
from src.shared.python.core.process_safety import managed_popen  # noqa: E402

# Reuse existing subprocess utilities
from src.shared.python.security.secure_subprocess import (  # noqa: E402
    SecureSubprocessError,
    secure_run,
)
from src.shared.python.security.subprocess_utils import kill_process_tree  # noqa: E402

DOCKER_BUILD_TIMEOUT_SECONDS = 3600.0
DOCKER_BUILD_WAIT_POLL_SECONDS = 0.1
DOCKER_BUILD_STDOUT_JOIN_TIMEOUT_SECONDS = 5.0
DOCKER_BUILD_CANCEL_GRACE_SECONDS = 2.0
WINDOWS_CREATE_NO_WINDOW = 0x08000000
WINDOWS_CREATE_NEW_PROCESS_GROUP = 0x00000200


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
                timeout=10.0,
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
        target_stage: str = "all",
        image_name: str = DOCKER_IMAGE_ENGINE,
        context_path: Path | None = None,
        dockerfile_path: Path | None = None,
        build_args: dict[str, str] | None = None,
        build_timeout_seconds: float = DOCKER_BUILD_TIMEOUT_SECONDS,
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
        self.build_timeout_seconds = build_timeout_seconds
        self._cancelled = False
        self._process: subprocess.Popen[str] | None = None
        self._process_lock = threading.Lock()

    def _process_has_exited(self, process: subprocess.Popen[str]) -> bool:
        """Return whether the child process has already exited."""
        if process.returncode is not None:
            return True
        poll = getattr(process, "poll", None)
        if not callable(poll):
            return False
        return poll() is not None

    def _wait_after_cancel(self, process: subprocess.Popen[str]) -> bool:
        """Return True if ``process`` exits during the cancel grace window."""
        try:
            process.wait(timeout=DOCKER_BUILD_CANCEL_GRACE_SECONDS)
            return True
        except subprocess.TimeoutExpired:
            return False

    def _cancel_process(self, process: subprocess.Popen[str]) -> None:
        """Ask Docker to stop, then escalate if the CLI keeps running."""
        if self._process_has_exited(process):
            return

        if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                if self._wait_after_cancel(process):
                    return
            except (AttributeError, OSError, ProcessLookupError):
                logger.debug("Unable to send CTRL_BREAK_EVENT to docker build")

        try:
            process.terminate()
            if self._wait_after_cancel(process):
                return
        except (OSError, ProcessLookupError):
            return

        try:
            process.kill()
        except (OSError, ProcessLookupError):
            return

    def cancel(self) -> None:
        """Cancel an in-progress Docker build and unblock the worker thread."""
        self._cancelled = True
        with self._process_lock:
            process = self._process
        if process is not None:
            self._cancel_process(process)

    def _stream_process_output(
        self, process: subprocess.Popen[str]
    ) -> tuple[threading.Thread, queue.Queue[str]] | None:
        """Stream Docker stdout without blocking timeout enforcement."""
        stdout_pipe = process.stdout
        if stdout_pipe is None:
            return None

        output_queue: queue.Queue[str] = queue.Queue()

        def read_stdout() -> None:
            try:
                for line in iter(stdout_pipe.readline, ""):
                    stripped = line.strip()
                    if stripped:
                        output_queue.put(stripped)
            finally:
                stdout_pipe.close()

        reader = threading.Thread(
            target=read_stdout,
            name="docker-build-stdout-reader",
            daemon=True,
        )
        reader.start()
        return reader, output_queue

    def _emit_queued_output(self, output_queue: queue.Queue[str] | None) -> None:
        """Emit all Docker output currently collected by the reader thread."""
        if output_queue is None:
            return
        while True:
            try:
                line = output_queue.get_nowait()
            except queue.Empty:
                return
            self.log_signal.emit(line)

    def _wait_for_build_process(
        self,
        process: subprocess.Popen[str],
        output_queue: queue.Queue[str] | None,
    ) -> None:
        """Wait for the Docker process while continuing to flush output."""
        deadline = time.monotonic() + self.build_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    cmd=getattr(process, "args", ["docker", "build"]),
                    timeout=self.build_timeout_seconds,
                )
            try:
                process.wait(timeout=min(DOCKER_BUILD_WAIT_POLL_SECONDS, remaining))
                return
            except subprocess.TimeoutExpired:
                self._emit_queued_output(output_queue)

    def _terminate_timed_out_build(self, process: subprocess.Popen[str]) -> None:
        """Terminate the Docker build process tree after a build timeout."""
        if not kill_process_tree(process.pid):
            process.kill()

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
                creation_flags = WINDOWS_CREATE_NO_WINDOW
                creation_flags |= WINDOWS_CREATE_NEW_PROCESS_GROUP

            with managed_popen(
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
                kill_timeout=DOCKER_BUILD_CANCEL_GRACE_SECONDS,
            ) as process:
                with self._process_lock:
                    self._process = process
                stdout_stream = self._stream_process_output(process)
                stdout_reader = stdout_stream[0] if stdout_stream is not None else None
                output_queue = stdout_stream[1] if stdout_stream is not None else None

                # Add timeout to prevent indefinite hangs (issue #2715)
                try:
                    self._wait_for_build_process(process, output_queue)
                except subprocess.TimeoutExpired:
                    self._emit_queued_output(output_queue)
                    if self._cancelled:
                        self.finished_signal.emit(False, "Build cancelled by user")
                    else:
                        self.log_signal.emit(
                            "Build timed out after 1 hour; terminating process"
                        )
                        self._terminate_timed_out_build(process)
                        self.finished_signal.emit(
                            False, "Build timed out (exceeded 1 hour limit)"
                        )
                    return
                finally:
                    if stdout_reader is not None:
                        stdout_reader.join(DOCKER_BUILD_STDOUT_JOIN_TIMEOUT_SECONDS)
                    self._emit_queued_output(output_queue)
                    with self._process_lock:
                        self._process = None

                if self._cancelled:
                    self.finished_signal.emit(False, "Build cancelled by user")
                elif process.returncode == 0:
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
        self._launcher_pid = os.getpid()
        self._container_lock = threading.Lock()
        self._active_containers: dict[str, int | None] = {}
        self._reserved_ports: set[int] = set()

    def _make_container_name(self, model_type: str, model_name: str = "") -> str:
        raw = model_name or model_type or "sim"
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw).strip("-").lower()[:24] or "sim"
        return f"ud-{slug}-{uuid.uuid4().hex[:8]}"

    def _select_host_port(
        self,
        preferred: int = DEFAULT_MESHCAT_PORT,
        max_probes: int = 25,
    ) -> int:
        with self._container_lock:
            reserved = set(self._reserved_ports)
        for offset in range(max_probes):
            candidate = preferred + offset
            if candidate in reserved:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind(("127.0.0.1", candidate))
                except OSError:
                    continue
                return candidate
        return preferred

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
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            self.logger.warning(f"Failed to check Docker image: {e}")
            return False

    def _docker_workspace_root(self, docker_cmd: list[str]) -> str:
        """Convert repo_root to a WSL mount path when launching via ``wsl docker``."""
        repo_root_str = str(self.repo_root)
        if "wsl" not in docker_cmd:
            return repo_root_str
        match = re.match(r"^([a-zA-Z]):(.*)", repo_root_str)
        if match:
            drive = match.group(1).lower()
            path_part = match.group(2).replace("\\", "/")
            return f"/mnt/{drive}{path_part}"
        return repo_root_str.replace("\\", "/")

    @staticmethod
    def _display_args() -> list[str]:
        """Build host-OS display environment and X11 mount flags for Docker GUI apps."""
        if os.name == "nt":
            return [
                "-e",
                "DISPLAY=host.docker.internal:0",
                "-e",
                "MUJOCO_GL=glfw",
                "-e",
                "PYOPENGL_PLATFORM=glx",
                "-e",
                "QT_QPA_PLATFORM=xcb",
            ]
        disp = os.environ.get("DISPLAY", ":0")
        return [
            "-e",
            f"DISPLAY={disp}",
            "-v",
            "/tmp/.X11-unix:/tmp/.X11-unix",  # nosec B108 - Docker X11 socket mount
        ]

    def build_launch_command(
        self,
        model_type: str,
        repo_path: Path,
        use_gpu: bool = False,
        *,
        container_name: str | None = None,
        host_port: int = DEFAULT_MESHCAT_PORT,
    ) -> list[str]:
        """Build the Docker launch command for a model.

        Args:
            model_type: Type of the model (drake, pinocchio, custom_humanoid, etc.)
            repo_path: Path to the model within the repository.
            use_gpu: Whether to enable GPU support.
            container_name: Explicit container name (auto-generated when omitted).
            host_port: Host port mapped to MeshCat port 7000.

        Returns:
            List of command arguments for docker run.
        """
        if model_type is None:
            raise ValueError("model_type must be provided")

        resolved_name = container_name or self._make_container_name(model_type)
        docker_cmd = get_docker_cmd()
        repo_root_str = self._docker_workspace_root(docker_cmd)

        cmd = docker_cmd + [
            "run",
            "--rm",
            "--name",
            resolved_name,
            "--label",
            CONTAINER_LABEL_MANAGED,
            "--label",
            f"{CONTAINER_LABEL_PID_KEY}={self._launcher_pid}",
            "-v",
            f"{repo_root_str}:/workspace",
            "-e",
            "PYTHONPATH=/workspace:/workspace/src:/workspace/src/shared/python",
        ]
        cmd.extend(self._display_args())

        if use_gpu:
            cmd.extend(["--gpus=all"])

        if model_type in ("drake", "pinocchio"):
            cmd.extend(["-p", f"{host_port}:7000", "-e", "MESHCAT_HOST=0.0.0.0"])  # nosec: Docker container networking requires 0.0.0.0

        work_dir = (
            f"/workspace/{repo_path.parent.relative_to(self.repo_root).as_posix()}"
        )
        cmd.extend(["-w", work_dir])

        if model_type == "drake":
            cmd.extend([self.image_name, "python", "-m", "src.drake_gui_app"])
        elif model_type == "pinocchio":
            cmd.extend([self.image_name, "python", "pinocchio_golf/gui.py"])
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
        container_name = self._make_container_name(model_type, model_name)
        allocated_port: int | None = None
        if model_type in ("drake", "pinocchio"):
            allocated_port = self._select_host_port(DEFAULT_MESHCAT_PORT)
            cmd = self.build_launch_command(
                model_type,
                repo_path,
                use_gpu,
                container_name=container_name,
                host_port=allocated_port,
            )
        else:
            cmd = self.build_launch_command(
                model_type,
                repo_path,
                use_gpu,
                container_name=container_name,
            )
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
            with self._container_lock:
                self._active_containers[container_name] = allocated_port
                if allocated_port is not None:
                    self._reserved_ports.add(allocated_port)
            return process
        except (FileNotFoundError, PermissionError, OSError) as e:
            self.logger.error(f"Failed to launch Docker container: {e}")
            return None

    def stop_container(self, container_name: str) -> bool:
        """Stop a managed Docker container by name and release its reserved port."""
        if not container_name:
            return False
        cmd = get_docker_cmd() + ["stop", "-t", "2", container_name]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            stopped = result.returncode == 0
        except (subprocess.SubprocessError, OSError) as exc:
            self.logger.warning("Failed to stop container %s: %s", container_name, exc)
            stopped = False
        with self._container_lock:
            port = self._active_containers.pop(container_name, None)
            if port is not None:
                self._reserved_ports.discard(port)
        return stopped

    def stop_all_containers(self) -> list[str]:
        """Stop all containers started by this launcher instance."""
        with self._container_lock:
            names = list(self._active_containers.keys())
        stopped: list[str] = []
        for container_name in names:
            if self.stop_container(container_name):
                stopped.append(container_name)
        return stopped

    def sweep_orphan_containers(
        self,
        *,
        pid_alive_fn: Callable[[int], bool] | None = None,
    ) -> list[str]:
        """Stop running UpstreamDrift containers whose launcher PID is gone."""
        is_alive = pid_alive_fn or is_pid_alive
        ps_cmd = get_docker_cmd() + [
            "ps",
            "--filter",
            f"label={CONTAINER_LABEL_MANAGED}",
            "--format",
            '{{.Names}}\t{{.Label "' + CONTAINER_LABEL_PID_KEY + '"}}',
        ]
        try:
            result = subprocess.run(ps_cmd, capture_output=True, text=True, timeout=10)
        except (subprocess.SubprocessError, OSError) as exc:
            self.logger.debug("Orphan container sweep skipped: %s", exc)
            return []
        if result.returncode != 0 or not result.stdout.strip():
            return []

        stopped: list[str] = []
        for line in result.stdout.splitlines():
            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                continue
            cname, pid_text = parts[0].strip(), parts[1].strip()
            if not cname:
                continue
            try:
                owner_pid = int(pid_text)
            except ValueError:
                continue
            if not is_alive(owner_pid) and self.stop_container(cname):
                stopped.append(cname)
        return stopped

    def sweep_orphan_containers_async(self) -> threading.Thread:
        """Run orphan container cleanup on a daemon thread at startup."""
        thread = threading.Thread(
            target=self.sweep_orphan_containers,
            name="docker-orphan-sweep",
            daemon=True,
        )
        thread.start()
        return thread
