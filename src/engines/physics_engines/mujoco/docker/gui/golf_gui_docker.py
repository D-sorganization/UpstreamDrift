"""
Docker operations mixin for the Golf Simulation GUI.

Extracted from GolfSimulationGUI to respect SRP:
Docker build/run logic is independent of GUI layout and styling.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import tempfile
import threading
from typing import Protocol, TYPE_CHECKING, cast

if TYPE_CHECKING:
    import queue
    import tkinter as tk
    from tkinter import messagebox

    class DockerProtocol(Protocol):
        """Protocol for Docker host class."""

        root: tk.Tk
        is_windows: bool
        wsl_path: str
        repo_path: str
        live_view_var: tk.BooleanVar
        stop_event: threading.Event
        process: subprocess.Popen
        btn_run: tk.Button
        btn_stop: tk.Button
        btn_rebuild: tk.Button
        btn_open_video: tk.Button
        btn_open_data: tk.Button

        def log(self, message: str) -> None: ...
        def on_sim_success(self) -> None: ...

logger = logging.getLogger(__name__)

try:
    import queue  # noqa: F401
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    pass


class DockerMixin:
    """Docker build and simulation process management for the Golf Simulation GUI.

    Requires host class to provide:
        root: tk.Tk
        is_windows: bool
        wsl_path: str
        repo_path: str
        live_view_var: tk.BooleanVar
        stop_event: threading.Event
        process: subprocess.Popen (set during run)
        btn_run, btn_stop, btn_rebuild: tk.Button
        btn_open_video, btn_open_data: tk.Button
        log(message: str) -> None
    """

    @staticmethod
    def _generate_update_dockerfile() -> str:
        """Generate a minimal Dockerfile to add missing dependencies."""
        return (
            "# Add missing dependencies to existing robotics_env\n"
            "FROM robotics_env:latest\n\n"
            "# Install missing dependencies in the existing virtual "
            "environment\n"
            'RUN /opt/mujoco-env/bin/pip install "defusedxml>=0.7.1" '
            '"PyQt6>=6.6.0"\n\n'
            "# Update PATH to use robotics_env by default\n"
            'ENV PATH="/opt/mujoco-env/bin:$PATH"\n'
            'ENV VIRTUAL_ENV="/opt/mujoco-env"\n'
        )

    def _run_docker_build(self, temp_dir: str, cmd: list[str]) -> int:
        """Execute the docker build command and return the exit code."""
        host = cast("DockerProtocol", self)
        if host.is_windows:
            create_new_console = 0x00000010
            result = subprocess.run(
                ["cmd", "/k", *cmd],
                cwd=temp_dir,
                creationflags=create_new_console,  # type: ignore[call-arg]
            )
        else:
            result = subprocess.run(cmd, cwd=temp_dir, check=True)
        return result.returncode

    def _verify_docker_update(self) -> None:
        """Run a quick container test to verify defusedxml is available."""
        host = cast("DockerProtocol", self)
        test_cmd = [
            "docker",
            "run",
            "--rm",
            "robotics_env",
            "python",
            "-c",
            "import defusedxml; print('defusedxml confirmed working')",
        ]
        test_result = subprocess.run(test_cmd, capture_output=True, text=True)
        if test_result.returncode == 0:
            host.root.after(0, host.log, test_result.stdout.strip())
        else:
            host.root.after(0, host.log, "Update completed but test failed")

    def rebuild_docker(self) -> None:
        """Add missing dependencies to the existing robotics_env Docker image."""
        host = cast("DockerProtocol", self)
        msg = (
            "This will add missing dependencies (like defusedxml) to the existing "
            "robotics_env.\n"
            "This should be quick since we're just adding packages. Continue?"
        )
        result = messagebox.askyesno(
            "Update Robotics Environment",
            msg,
        )

        if not result:
            return

        host.log("Updating robotics_env with missing dependencies...")
        host.btn_rebuild.config(state=tk.DISABLED)

        def run_update() -> None:
            """Rebuild the Docker image with missing dependencies."""
            try:
                dockerfile_content = self._generate_update_dockerfile()

                with tempfile.TemporaryDirectory() as temp_dir:
                    dockerfile_path = os.path.join(temp_dir, "Dockerfile")
                    with open(dockerfile_path, "w") as f:
                        f.write(dockerfile_content)

                    cmd = ["docker", "build", "-t", "robotics_env", "."]
                    host.root.after(0, host.log, f"Running: {' '.join(cmd)}")
                    host.root.after(0, host.log, "Adding defusedxml to robotics_env...")

                    returncode = self._run_docker_build(temp_dir, cmd)

                    if returncode == 0:
                        host.root.after(
                            0, host.log, "robotics_env updated successfully!"
                        )
                        host.root.after(
                            0,
                            host.log,
                            "defusedxml and other dependencies are now available.",
                        )
                        self._verify_docker_update()
                    else:
                        host.root.after(
                            0,
                            host.log,
                            f"Update failed with code {returncode}",
                        )

            except ImportError as e:
                host.root.after(0, host.log, f"Update failed: {e}")
            finally:
                host.root.after(0, lambda: host.btn_rebuild.config(state=tk.NORMAL))

        threading.Thread(target=run_update, daemon=True).start()

    def _build_docker_command(self) -> list[str]:
        """Build the docker run command for the simulation subprocess."""
        host = cast("DockerProtocol", self)
        if host.is_windows:
            cmd = [
                "wsl",
                "docker",
                "run",
                "--rm",
                "-v",
                f"{host.wsl_path}:/workspace",
                "-w",
                "/workspace/python",
            ]

            if host.live_view_var.get():
                # Allow GUI to display on host Windows X Server (VcXsrv)
                cmd.extend(["-e", "DISPLAY=host.docker.internal:0"])
                cmd.extend(["-e", "MUJOCO_GL=glfw"])
                cmd.extend(["-e", "PYOPENGL_PLATFORM=glx"])
            else:
                cmd.extend(["-e", "MUJOCO_GL=osmesa"])

            cmd.extend(
                [
                    "robotics_env",
                    "/opt/mujoco-env/bin/python",
                    "-u",
                    "-m",
                    "mujoco_humanoid_golf",
                ]
            )
        else:
            cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{host.repo_path}:/workspace",
                "-w",
                "/workspace/python",
            ]

            if host.live_view_var.get():
                cmd.extend(["-e", f"DISPLAY={os.environ.get('DISPLAY', ':0')}"])
                cmd.extend(["-e", "MUJOCO_GL=glfw"])
                cmd.extend(["-e", "PYOPENGL_PLATFORM=glx"])
                cmd.extend(["-v", "/tmp/.X11-unix:/tmp/.X11-unix"])  # nosec B108
            else:
                cmd.extend(["-e", "MUJOCO_GL=osmesa"])

            cmd.extend(
                [
                    "robotics_env",
                    "/opt/mujoco-env/bin/python",
                    "-m",
                    "mujoco_humanoid_golf",
                ]
            )

        return cmd

    def _stream_process_output(self) -> None:
        """Read subprocess stdout via a queue and log lines to the GUI."""
        host = cast("DockerProtocol", self)
        q: queue.Queue[str | None] = queue.Queue()

        def enqueue_output(out, output_queue) -> None:
            """Enqueue output from subprocess."""
            try:
                for line in iter(out.readline, ""):
                    output_queue.put(line)
                out.close()
            except (RuntimeError, ValueError, OSError) as e:
                with contextlib.suppress(RuntimeError, ValueError, AttributeError):
                    host.root.after(0, host.log, f"Exception in enqueue_output: {e}")
            output_queue.put(None)  # Sentinel

        t = threading.Thread(
            target=enqueue_output, args=(host.process.stdout, q), daemon=True
        )
        t.start()

        while True:
            # Check user stop
            if host.stop_event.is_set() and host.process.poll() is None:
                host.process.terminate()

            try:
                output = q.get(timeout=0.1)
            except queue.Empty:
                if host.process.poll() is not None and not t.is_alive():
                    break
                continue

            if output is None:  # Sentinel
                break

            host.root.after(0, host.log, output.strip())

    def _handle_process_failure(self, rc) -> None:
        """Log error details and suggest solutions for common failures."""
        host = cast("DockerProtocol", self)
        host.root.after(0, host.log, f"Process exited with code {rc}")
        if host.process.stderr:
            err = host.process.stderr.read()
            if err:
                host.root.after(0, host.log, f"ERROR: {err}")
            # Check for specific common errors and provide solutions
            if "defusedxml" in err:
                host.root.after(
                    0,
                    host.log,
                    "SOLUTION: Missing defusedxml dependency. "
                    "Please rebuild Docker image.",
                )
                host.root.after(0, host.log, "Run: docker build -t robotics_env .")
            elif "ModuleNotFoundError" in err:
                host.root.after(
                    0,
                    host.log,
                    "SOLUTION: Missing Python dependency. "
                    "Check Dockerfile and rebuild.",
                )
            elif "DISPLAY" in err or "X11" in err:
                host.root.after(
                    0,
                    host.log,
                    "SOLUTION: X11/Display issue. "
                    "Try disabling 'Live Interactive View'.",
                )

    def _reset_buttons_state(self) -> None:
        """Reset run/stop buttons to their default enabled states."""
        host = cast("DockerProtocol", self)
        host.root.after(0, lambda: host.btn_run.config(state=tk.NORMAL))
        host.root.after(0, lambda: host.btn_stop.config(state=tk.DISABLED))

    def _run_docker_process(self) -> None:
        """Run the simulation in a subprocess."""
        host = cast("DockerProtocol", self)
        cmd = self._build_docker_command()

        try:
            host.log(f"Running command: {' '.join(cmd)}")

            # Race condition fix: Check stop event before starting
            if host.stop_event.is_set():
                host.log("Simulation cancelled.")
                self._reset_buttons_state()
                return

            host.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            self._stream_process_output()

            rc = host.process.poll()

            # Check if stopped by user
            if host.stop_event.is_set():
                host.root.after(0, host.log, "Simulation stopped by user.")
                self._reset_buttons_state()
            elif rc == 0:
                host.root.after(0, host.on_sim_success)
            else:
                self._handle_process_failure(rc)
                self._reset_buttons_state()

        except ImportError as e:
            host.root.after(0, host.log, f"Failed to run subprocess: {e}")
            self._reset_buttons_state()
