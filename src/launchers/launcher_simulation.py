"""Simulation launching mixin for GolfLauncher.

Contains methods for launching simulations, MJCF viewers, Docker containers,
script processes, module processes, URDF generator, C3D viewer, shot tracer,
MATLAB apps, and dependency checking.
"""

# mypy: disable-error-code="attr-defined,arg-type"

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QEventLoop
from PyQt6.QtWidgets import QApplication, QMessageBox

from src.launchers.launcher_constants import (
    CREATE_NO_WINDOW,
    REPOS_ROOT,
)
from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.security.secure_subprocess import secure_popen
from src.shared.python.theme.style_constants import Styles

logger = get_logger(__name__)


class LauncherSimulationMixin:
    """Mixin for GolfLauncher simulation launching.

    Provides methods for launching various simulation types,
    dependency checking, and subprocess management.
    """

    def _get_subprocess_env(self) -> dict[str, str]:
        """Get environment dict with PYTHONPATH set for subprocess launches."""
        env = os.environ.copy()
        pythonpath = str(REPOS_ROOT)
        if "PYTHONPATH" in env:
            pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        env["PYTHONPATH"] = pythonpath

        # Fix for MuJoCo DLL loading issue on Windows with Python 3.13
        if "MUJOCO_PLUGIN_PATH" not in env:
            env["MUJOCO_PLUGIN_PATH"] = ""

        return env

    @precondition(
        lambda self, model_type: model_type is not None and len(model_type.strip()) > 0,
        "Model type must be a non-empty string",
    )
    def _check_module_dependencies(self, model_type: str) -> tuple[bool, str]:
        """Check if required dependencies for a module type are available.

        Args:
            model_type: The type of model to check dependencies for.

        Returns:
            Tuple of (success, error_message). If success is True, error_message is empty.
        """
        # Map model types to their required imports
        dependency_checks = {
            "custom_humanoid": ("mujoco", "MuJoCo"),
            "custom_dashboard": ("mujoco", "MuJoCo"),
            "mjcf": ("mujoco", "MuJoCo"),
            "drake": ("pydrake", "Drake (pydrake)"),
            "pinocchio": ("pinocchio", "Pinocchio"),
            "opensim": ("opensim", "OpenSim"),
            "myosim": ("myosuite", "MyoSuite"),
        }

        check = dependency_checks.get(model_type)
        if not check:
            return True, ""  # No specific dependency check needed

        module_name, display_name = check

        import_check_code = f"""
import sys
import os
try:
    import {module_name}
    print("OK")
except ImportError as e:
    print(f"ImportError: {{e}}")
except OSError as e:
    print(f"OSError: {{e}}")
except ImportError as e:
    print(f"Error: {{type(e).__name__}}: {{e}}")
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", import_check_code],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(REPOS_ROOT),
                env=self._get_subprocess_env(),
            )
            output = result.stdout.strip()
            if output == "OK":
                return True, ""
            else:
                return False, f"{display_name} dependency check failed:\n{output}"
        except subprocess.TimeoutExpired:
            return False, f"{display_name} dependency check timed out"
        except (OSError, ValueError) as e:
            return False, f"Failed to check {display_name} dependencies: {e}"

    def _show_dependency_error(self, model_name: str, error_msg: str) -> None:
        """Show a dialog with dependency error information and suggestions."""
        detailed_msg = f"Cannot launch {model_name}.\n\n{error_msg}\n\n"

        if "DLL" in error_msg or "OSError" in error_msg:
            detailed_msg += (
                "Suggestions:\n"
                "- Try reinstalling the package: pip install --force-reinstall mujoco\n"
                "- Ensure Visual C++ Redistributable is installed\n"
                "- Check Python version compatibility"
            )
        elif "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
            detailed_msg += (
                "Suggestions:\n"
                "- Install the missing package using pip\n"
                "- Check that you're using the correct Python environment"
            )

        QMessageBox.warning(self, "Dependency Error", detailed_msg)

    def _try_launch_special_app(self, model_id: str) -> bool:
        if "urdf_generator" in model_id or "model_explorer" in model_id:
            self._launch_urdf_generator()
            return True
        elif "c3d_viewer" in model_id:
            self._launch_c3d_viewer()
            return True
        elif "shot_tracer" in model_id:
            self._launch_shot_tracer()
            return True
        return False

    def _try_launch_docker(self, model: Any) -> bool:
        use_docker = hasattr(self, "chk_docker") and self.chk_docker.isChecked()
        if not (use_docker and self.docker_available):
            return False

        self.lbl_status.setText(f"> Launching {model.name} in Docker...")
        self.lbl_status.setStyleSheet(Styles.STATUS_INFO)
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

        try:
            repo_path = getattr(model, "path", None)
            if repo_path:
                self._launch_docker_container(model, REPOS_ROOT / repo_path)
            else:
                self.show_toast("Model path missing for Docker launch.", "error")
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Docker launch failed: {e}")
            self.show_toast(f"Docker Launch Failed: {e}", "error")
            self.lbl_status.setText("> Ready")
            self.lbl_status.setStyleSheet(Styles.STATUS_INACTIVE)
        return True

    def _check_local_dependencies(self, model: Any) -> bool:
        use_wsl = hasattr(self, "chk_wsl") and self.chk_wsl.isChecked()
        if use_wsl:
            return True

        self.lbl_status.setText(f"> Checking {model.name} dependencies...")
        self.lbl_status.setStyleSheet(Styles.STATUS_WARNING)
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

        deps_ok, deps_error = self._check_module_dependencies(model.type)
        if deps_ok:
            return True

        if self.docker_available:
            response = QMessageBox.question(
                self,
                "Local Dependencies Missing",
                f"{deps_error}\n\n"
                "Would you like to try launching in Docker mode instead?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if response == QMessageBox.StandardButton.Yes:
                self.chk_docker.setChecked(True)
                self.launch_simulation()
                return False
        self._show_dependency_error(model.name, deps_error)
        self.lbl_status.setText("! Dependency Error")
        self.lbl_status.setStyleSheet(Styles.STATUS_ERROR)
        return False

    def _execute_local_launch(self, model: Any) -> None:
        repo_path = getattr(model, "path", None)
        if not repo_path:
            self.show_toast("Model path missing.", "error")
            return

        abs_repo_path = REPOS_ROOT / repo_path
        handler = self.model_handler_registry.get_handler(model.type)
        if handler:
            success = handler.launch(model, REPOS_ROOT, self.process_manager)
            if success:
                self.show_toast(f"{model.name} Launched", "success")
                self.lbl_status.setText(f"* {model.name} Running")
                self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
            else:
                self.show_toast(f"Failed to launch {model.name}", "error")
                self.lbl_status.setText("* Launch Error")
                self.lbl_status.setStyleSheet(Styles.STATUS_ERROR)
        elif model.type == "mjcf" or str(repo_path).endswith(".xml"):
            self._launch_generic_mjcf(abs_repo_path)
        else:
            self.show_toast(f"Unknown launch type: {model.type}", "warning")

    def launch_simulation(self) -> None:
        """Launch the selected simulation."""
        if not self.selected_model:
            return

        model_id = self.selected_model

        if self._try_launch_special_app(model_id):
            return

        model = self._get_model(model_id)
        if not model:
            self.show_toast("Model configuration not found.", "error")
            return

        if model.type == "matlab_app":
            self._launch_matlab_app(model)
            return

        if self._try_launch_docker(model):
            return

        if not self._check_local_dependencies(model):
            return

        self.lbl_status.setText(f"> Launching {model.name}...")
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

        try:
            self._execute_local_launch(model)
        except (ValueError, RuntimeError) as e:
            logger.error(f"Launch failed: {e}")
            self.show_toast(f"Launch Failed: {e}", "error")
            self.lbl_status.setText("> Ready")
            self.lbl_status.setStyleSheet(Styles.STATUS_INACTIVE)

    @precondition(
        lambda self, path: path is not None and str(path).strip() != "",
        "MJCF path must be a non-empty Path",
    )
    def _launch_generic_mjcf(self, path: Path) -> None:
        """Launch generic MJCF file in passive viewer."""
        import mujoco
        import mujoco.viewer

        try:
            m = mujoco.MjModel.from_xml_path(str(path))
            d = mujoco.MjData(m)

            viewer_script = (
                REPOS_ROOT
                / "engines"
                / "physics_engines"
                / "mujoco"
                / "python"
                / "passive_viewer.py"
            )

            if viewer_script.exists():
                process = self.process_manager.launch_script(
                    path.name, viewer_script, viewer_script.parent
                )
                if not process:
                    raise RuntimeError("ProcessManager returned None")
                self.show_toast("Launched Passive Viewer", "success")
            else:
                self.show_toast(
                    "Viewer script missing, attempting direct launch...", "warning"
                )
                mujoco.viewer.launch(m, d)

        except (RuntimeError, TypeError, ValueError) as e:
            raise RuntimeError(f"Failed to launch MJCF: {e}") from e

    def _launch_docker_container(self, model: Any, repo_path: Path) -> None:
        """Launch the model in a Docker container.

        Delegates to DockerLauncher for container orchestration while
        handling UI feedback (prompts, status updates, error dialogs).
        """
        from src.launchers.launcher_process_manager import start_vcxsrv

        try:
            # Auto-start VcXsrv on Windows for GUI support
            if os.name == "nt":
                if not start_vcxsrv():
                    response = QMessageBox.question(
                        self,
                        "X Server Not Available",
                        "VcXsrv X server is not running and could not be started.\n\n"
                        "Docker GUI apps require an X server.\n\n"
                        "Install VcXsrv from: https://vcxsrv.com\n\n"
                        "Continue anyway?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if response != QMessageBox.StandardButton.Yes:
                        return

            # Check if Docker image exists
            if not self.docker_launcher.check_image_exists():
                QMessageBox.warning(
                    self,
                    "Docker Image Not Found",
                    f"The Docker image '{self.docker_launcher.image_name}' is not available.\n\n"
                    "Build it first using:\n"
                    "  docker build -t robotics_env .\n\n"
                    "Or use the Environment dialog to build.",
                )
                return

            # Launch container via DockerLauncher
            use_gpu = hasattr(self, "chk_gpu") and self.chk_gpu.isChecked()
            process = self.docker_launcher.launch_container(
                model_type=model.type,
                model_name=model.name,
                repo_path=repo_path,
                use_gpu=use_gpu,
                capture_output=True,
            )

            if process:
                # Route Docker output through the unified console
                self.process_manager.attach_process(model.name, process)
                self.show_toast(f"{model.name} Launched (Docker)", "success")
                self.lbl_status.setText(f"* {model.name} Running (Docker)")
                self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
            else:
                self.lbl_status.setText("* Docker Error")
                self.lbl_status.setStyleSheet(Styles.STATUS_ERROR)
                QMessageBox.critical(
                    self,
                    "Docker Launch Error",
                    f"Failed to launch {model.name} in Docker",
                )

        except (ValueError, RuntimeError) as e:
            logger.error(f"Failed to launch Docker container: {e}")
            QMessageBox.critical(
                self,
                "Docker Launch Error",
                f"Failed to launch {model.name} in Docker:\n\n{e}",
            )
            self.lbl_status.setText("* Docker Error")
            self.lbl_status.setStyleSheet(Styles.STATUS_ERROR)

    @precondition(
        lambda self, name, script_path, cwd: name is not None and len(name.strip()) > 0,
        "Process name must be a non-empty string",
    )
    @precondition(
        lambda self, name, script_path, cwd: script_path is not None,
        "Script path must not be None",
    )
    def _launch_script_process(self, name: str, script_path: Path, cwd: Path) -> None:
        """Helper to launch python script with error visibility.

        On Windows, uses cmd /k to keep the terminal open if the script crashes.
        If WSL mode is enabled, launches the script in WSL2 Ubuntu environment.
        """
        # Check if WSL mode is enabled
        use_wsl = hasattr(self, "chk_wsl") and self.chk_wsl.isChecked()

        if use_wsl:
            success = self.process_manager.launch_in_wsl(str(script_path))
            if success:
                self.lbl_status.setText(f"* {name} Running (WSL)")
                self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
                self.show_toast(f"{name} Launched in WSL", "success")
            else:
                QMessageBox.critical(
                    self, "Launch Error", f"Failed to launch {name} in WSL"
                )
            return

        # Delegate to ProcessManager with keep_terminal_open=True for error visibility
        process = self.process_manager.launch_script(
            name, script_path, cwd, keep_terminal_open=True
        )

        if process:
            self.show_toast(f"{name} Launched", "success")
            self.lbl_status.setText(f"* {name} Running")
            self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
        else:
            QMessageBox.critical(self, "Launch Error", f"Failed to launch {name}")

    @precondition(
        lambda self, name, module_name, cwd: name is not None and len(name.strip()) > 0,
        "Process name must be a non-empty string",
    )
    @precondition(
        lambda self, name, module_name, cwd: module_name is not None
        and len(module_name.strip()) > 0,
        "Module name must be a non-empty string",
    )
    def _launch_module_process(self, name: str, module_name: str, cwd: Path) -> None:
        """Helper to launch python module with error visibility.

        Similar to _launch_script_process but uses -m to run a module.
        If WSL mode is enabled, launches in WSL2 Ubuntu environment.
        """
        # Check if WSL mode is enabled
        use_wsl = hasattr(self, "chk_wsl") and self.chk_wsl.isChecked()

        if use_wsl:
            success = self.process_manager.launch_module_in_wsl(module_name, cwd)
            if success:
                self.lbl_status.setText(f"* {name} Running (WSL)")
                self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
                self.show_toast(f"{name} Launched in WSL", "success")
            else:
                QMessageBox.critical(
                    self, "Launch Error", f"Failed to launch {name} in WSL"
                )
            return

        # Delegate to ProcessManager with keep_terminal_open=True for error visibility
        process = self.process_manager.launch_module(
            name, module_name, cwd, keep_terminal_open=True
        )

        if process:
            self.show_toast(f"{name} Launched", "success")
            self.lbl_status.setText(f"* {name} Running")
            self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
        else:
            QMessageBox.critical(self, "Launch Error", f"Failed to launch {name}")

    def _launch_urdf_generator(self) -> None:
        """Launch the URDF generator / Model Explorer application."""
        from src.shared.python.core.constants import URDF_GENERATOR_SCRIPT

        script_path = REPOS_ROOT / URDF_GENERATOR_SCRIPT

        # Check if already running
        if "urdf_generator" in self.running_processes:
            proc = self.running_processes["urdf_generator"]
            if proc.poll() is None:
                self.show_toast("URDF Generator is already running.", "warning")
                return

        self.lbl_status.setText("> Launching URDF Generator...")
        self.lbl_status.setStyleSheet(Styles.STATUS_WARNING)
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

        try:
            logger.info("Launching URDF Generator: %s", script_path)

            process = self.process_manager.launch_script(
                "urdf_generator", script_path, REPOS_ROOT
            )
            if not process:
                raise RuntimeError("ProcessManager returned None")
            self.show_toast("URDF Generator launched.", "success")
            self.lbl_status.setText("> URDF Generator Running")
            self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)

        except (ValueError, RuntimeError, OSError) as e:
            logger.error(f"Failed to launch URDF Generator: {e}")
            self.show_toast(f"Launch failed: {e}", "error")
            self.lbl_status.setText("! Launch Error")
            self.lbl_status.setStyleSheet(Styles.STATUS_ERROR)

    def _launch_c3d_viewer(self) -> None:
        """Launch the C3D motion viewer application."""
        c3d_script = REPOS_ROOT / "tools" / "c3d_viewer" / "c3d_viewer.py"

        if not c3d_script.exists():
            c3d_script = REPOS_ROOT / "tools" / "c3d_viewer_app.py"

        if not c3d_script.exists():
            self.show_toast("C3D Viewer script not found.", "error")
            return

        if "c3d_viewer" in self.running_processes:
            if self.running_processes["c3d_viewer"].poll() is None:
                self.show_toast("C3D Viewer is already running.", "warning")
                return

        try:
            logger.info("Launching C3D Viewer: %s", c3d_script)
            process = self.process_manager.launch_script(
                "c3d_viewer", c3d_script, c3d_script.parent
            )
            if not process:
                raise RuntimeError("ProcessManager returned None")
            self.show_toast("C3D Viewer launched.", "success")

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Failed to launch C3D Viewer: {e}")
            self.show_toast(f"Launch failed: {e}", "error")

    def _launch_shot_tracer(self) -> None:
        """Launch the Shot Tracer ball flight visualization."""
        shot_tracer_script = REPOS_ROOT / "src" / "launchers" / "shot_tracer.py"

        if not shot_tracer_script.exists():
            self.show_toast("Shot Tracer script not found.", "error")
            return

        if "shot_tracer" in self.running_processes:
            if self.running_processes["shot_tracer"].poll() is None:
                self.show_toast("Shot Tracer is already running.", "warning")
                return

        try:
            logger.info("Launching Shot Tracer: %s", shot_tracer_script)
            process = self.process_manager.launch_script(
                "shot_tracer", shot_tracer_script, REPOS_ROOT
            )
            if not process:
                raise RuntimeError("ProcessManager returned None")
            self.show_toast("Shot Tracer launched.", "success")

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Failed to launch Shot Tracer: {e}")
            self.show_toast(f"Launch failed: {e}", "error")

    def _launch_matlab_app(self, app: Any) -> None:
        """Launch a MATLAB-based application with proper desktop GUI."""
        app_path = getattr(app, "path", None)
        if not app_path:
            self.show_toast("Invalid MATLAB configuration.", "error")
            return

        self.show_toast(f"Launching MATLAB: {app.name}...", "info")

        try:
            abs_path = REPOS_ROOT / app_path
            path_str = str(abs_path).replace("\\", "/")

            # Check if using batch script wrapper
            if str(app_path).endswith(".bat") or str(app_path).endswith(".sh"):
                cmd = [str(abs_path)]
                process = secure_popen(
                    cmd,
                    cwd=str(abs_path.parent),
                    creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            else:
                # Determine the appropriate MATLAB command based on file type
                if str(app_path).endswith(".slx"):
                    matlab_cmd = f"open_system('{path_str}')"
                elif str(app_path).endswith(".m"):
                    matlab_cmd = f"cd('{str(abs_path.parent).replace(chr(92), '/')}'); run('{abs_path.name}')"
                else:
                    matlab_cmd = f"open('{path_str}')"

                cmd = ["matlab", "-nosplash", "-r", matlab_cmd]

                process = secure_popen(
                    cmd,
                    cwd=str(abs_path.parent),
                    creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
                )

            self.running_processes[app.id] = process
            self.show_toast(f"{app.name} launch initiated.", "success")

        except FileNotFoundError:
            self.show_toast("MATLAB executable not found in PATH.", "error")
        except (PermissionError, OSError) as e:
            logger.error(f"Failed to launch MATLAB app: {e}")
            self.show_toast(f"Launch failed: {e}", "error")
