# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Simulation launching mixin for UpstreamDriftLauncher.

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
from src.launchers.launcher_model_sources import (
    get_model_source_root,
    resolve_model_artifact_path,
)
from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.security.secure_subprocess import secure_popen
from src.shared.python.theme.style_constants import Styles

logger = get_logger(__name__)


DEPENDENCY_MAP: dict[str, dict[str, str]] = {
    "mujoco_unified": {
        "module": "mujoco",
        "display_name": "MuJoCo",
        "install_cmd": "pip install mujoco",
        "doc_url": "https://mujoco.org",
    },
    "custom_humanoid": {
        "module": "mujoco",
        "display_name": "MuJoCo",
        "install_cmd": "pip install mujoco",
        "doc_url": "https://mujoco.org",
    },
    "custom_dashboard": {
        "module": "mujoco",
        "display_name": "MuJoCo",
        "install_cmd": "pip install mujoco",
        "doc_url": "https://mujoco.org",
    },
    "mjcf": {
        "module": "mujoco",
        "display_name": "MuJoCo",
        "install_cmd": "pip install mujoco",
        "doc_url": "https://mujoco.org",
    },
    "drake_golf": {
        "module": "pydrake",
        "display_name": "Drake (pydrake)",
        "install_cmd": "pip install drake",
        "doc_url": "https://drake.mit.edu/python_bindings.html",
    },
    "drake": {
        "module": "pydrake",
        "display_name": "Drake (pydrake)",
        "install_cmd": "pip install drake",
        "doc_url": "https://drake.mit.edu/python_bindings.html",
    },
    "pinocchio_golf": {
        "module": "pinocchio",
        "display_name": "Pinocchio",
        "install_cmd": "pip install pin-project",
        "doc_url": "https://github.com/stack-of-tasks/pinocchio",
    },
    "pinocchio": {
        "module": "pinocchio",
        "display_name": "Pinocchio",
        "install_cmd": "pip install pin-project",
        "doc_url": "https://github.com/stack-of-tasks/pinocchio",
    },
    "opensim_golf": {
        "module": "opensim",
        "display_name": "OpenSim",
        "install_cmd": "conda install -c opensim opensim",
        "doc_url": "https://opensim.stanford.edu",
    },
    "opensim": {
        "module": "opensim",
        "display_name": "OpenSim",
        "install_cmd": "conda install -c opensim opensim",
        "doc_url": "https://opensim.stanford.edu",
    },
    "myosim_suite": {
        "module": "myosuite",
        "display_name": "MyoSuite",
        "install_cmd": "pip install myosuite",
        "doc_url": "https://github.com/facebookresearch/myosuite",
    },
    "myosim": {
        "module": "myosuite",
        "display_name": "MyoSuite",
        "install_cmd": "pip install myosuite",
        "doc_url": "https://github.com/facebookresearch/myosuite",
    },
    "mediapipe_analysis": {
        "module": "mediapipe",
        "display_name": "MediaPipe",
        "install_cmd": "pip install mediapipe",
        "doc_url": "https://google.github.io/mediapipe/",
    },
    "openpose_analysis": {
        "module": "pyopenpose",
        "display_name": "OpenPose (pyopenpose)",
        "install_cmd": "pip install pyopenpose",
        "doc_url": "https://github.com/CMU-Perceptual-Computing-Lab/openpose",
    },
    "bunker_shot": {
        "module": "pychrono",
        "display_name": "Project Chrono (pychrono)",
        "install_cmd": "conda install -c projectchrono pychrono",
        "doc_url": "https://projectchrono.org",
    },
    "bunkershot3d": {
        "module": "pyqtgraph",
        "display_name": "PyQtGraph",
        "install_cmd": "pip install pyqtgraph PyOpenGL",
        "doc_url": "https://www.pyqtgraph.org/",
    },
    "pinn_hybrid": {
        "module": "jax",
        "display_name": "JAX / Equinox",
        "install_cmd": "pip install jax jaxlib equinox",
        "doc_url": "https://github.com/google/jax",
    },
    "physics_informed": {
        "module": "jax",
        "display_name": "JAX / Equinox",
        "install_cmd": "pip install jax jaxlib equinox",
        "doc_url": "https://github.com/google/jax",
    },
}


class SimulationManager:
    def __init__(self, launcher):
        self.launcher = launcher

    def __getattr__(self, name):
        if name == "launcher":
            raise AttributeError("launcher not initialized")
        launcher = self.__dict__.get("launcher")
        if launcher is None:
            raise AttributeError("launcher not initialized")
        return getattr(launcher, name)

    def __setattr__(self, name, value):
        if name == "launcher" or hasattr(type(self), name) or name in self.__dict__:
            super().__setattr__(name, value)
        else:
            launcher = self.__dict__.get("launcher")
            if launcher is not None and hasattr(launcher, name):
                setattr(launcher, name, value)
            else:
                super().__setattr__(name, value)

    """Mixin for UpstreamDriftLauncher simulation launching.

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
        lambda self, key: key is not None and len(key.strip()) > 0,
        "Dependency key must be a non-empty string",
    )
    def _check_module_dependencies(self, key: str) -> tuple[bool, str]:
        """Check if required dependencies for a module type or ID are available.

        Args:
            key: The type or ID of model to check dependencies for.

        Returns:
            Tuple of (success, error_message). If success is True, error_message is empty.
        """
        if key is None:
            raise ValueError("key must be provided")

        check = DEPENDENCY_MAP.get(key)
        if not check:
            return True, ""  # No specific dependency check needed

        module_name = check["module"]
        display_name = check["display_name"]

        import_check_code = f"""
import sys
import os
try:
    import {module_name}
    sys.stdout.write("OK\\n")
except ImportError as e:
    sys.stdout.write(f"ImportError: {{e}}\\n")
except OSError as e:
    sys.stdout.write(f"OSError: {{e}}\\n")
except (RuntimeError, TypeError, AttributeError) as e:
    sys.stdout.write(f"Error: {{type(e).__name__}}: {{e}}\\n")
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", import_check_code],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(REPOS_ROOT),
                env=self._get_subprocess_env(),
                creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            output = result.stdout.strip()
            if output == "OK":
                return True, ""
            return False, f"{display_name} dependency check failed:\n{output}"
        except subprocess.TimeoutExpired:
            return False, f"{display_name} dependency check timed out"
        except (OSError, ValueError) as e:
            return False, f"Failed to check {display_name} dependencies: {e}"

    def _show_dependency_error(self, model_name: str, error_msg: str) -> None:
        """Show a dialog with dependency error information and suggestions."""
        if model_name is None:
            raise ValueError("model_name must be provided")
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

        QMessageBox.warning(self.launcher, "Dependency Error", detailed_msg)

    def _try_launch_special_app(self, model_id: str) -> bool:
        if model_id is None:
            raise ValueError("model_id must be provided")
        if "urdf_generator" in model_id or "model_explorer" in model_id:
            self._launch_urdf_generator()
            return True
        if "c3d_viewer" in model_id:
            self._launch_c3d_viewer()
            return True
        if "shot_tracer" in model_id:
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
            model_path = getattr(model, "path", None)
            if model_path:
                self._launch_docker_container(
                    model,
                    resolve_model_artifact_path(model, REPOS_ROOT),
                )
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

        # Retrieve or check dependencies with caching
        if not hasattr(self.launcher, "_dependency_status_cache"):
            self.launcher._dependency_status_cache = {}

        key = model.id if model.id in DEPENDENCY_MAP else model.type

        if model.id not in self.launcher._dependency_status_cache:
            deps_ok, deps_error = self._check_module_dependencies(key)
            self.launcher._dependency_status_cache[model.id] = (deps_ok, deps_error)
        else:
            deps_ok, deps_error = self.launcher._dependency_status_cache[model.id]

        if deps_ok:
            return True

        if self.docker_available:
            response = QMessageBox.question(
                self.launcher,
                "Local Dependencies Missing",
                f"{deps_error}\n\n"
                "Would you like to try launching in Docker mode instead?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if response == QMessageBox.StandardButton.Yes:
                self.chk_docker.setChecked(True)
                self.launch_simulation()
                return False

        # Show the custom dependency error dialog
        dep_info = DEPENDENCY_MAP.get(key, {})
        dep_name = dep_info.get("display_name", key)
        install_cmd = dep_info.get("install_cmd", "")
        doc_url = dep_info.get("doc_url", "")

        if hasattr(self.launcher, "show_dependency_error"):
            self.launcher.show_dependency_error(
                model.name,
                dep_name,
                install_cmd,
                doc_url,
                deps_error,
            )
        else:
            self._show_dependency_error(model.name, deps_error)

        self.lbl_status.setText("! Dependency Error")
        self.lbl_status.setStyleSheet(Styles.STATUS_ERROR)
        return False

    def _execute_local_launch(self, model: Any) -> None:
        try:
            abs_model_path = resolve_model_artifact_path(model, REPOS_ROOT)
        except ValueError:
            self.show_toast("Model path missing.", "error")
            return

        handler = self.model_handler_registry.get_handler(model.type)
        if handler:
            # Unified Architecture: Check if handler supports docking
            dockable_factory = getattr(type(handler), "get_dockable_ui", None)
            if callable(dockable_factory):
                try:
                    ui_widget = handler.get_dockable_ui(model, REPOS_ROOT)
                    if ui_widget:
                        if getattr(
                            self, "sidekick_sidebar", None
                        ) is not None and hasattr(ui_widget, "set_sidekick_session"):
                            # The sidebar widget might have a direct session attribute or IS the session.
                            session = getattr(
                                self.sidekick_sidebar, "session", self.sidekick_sidebar
                            )
                            try:
                                ui_widget.set_sidekick_session(session)
                            except Exception as e:  # noqa: BLE001
                                logger.warning(
                                    "Failed to inject Sidekick session: %s", e
                                )

                        # Check user preference or model default; for now default to docking
                        launcher = getattr(model, "launcher", None)
                        if isinstance(launcher, dict):
                            launch_mode = launcher.get("default_launch", "tab")
                        else:
                            launch_mode = (
                                getattr(launcher, "default_launch", "tab")
                                if launcher
                                else "tab"
                            )
                        if launch_mode == "window" and hasattr(self, "popout_widget"):
                            self.popout_widget(ui_widget, model.name)
                            self.show_toast(f"{model.name} Popped Out", "success")
                        elif hasattr(self, "dock_widget_as_tab"):
                            self.dock_widget_as_tab(ui_widget, model.name)
                            self.show_toast(f"{model.name} Docked", "success")

                        self.lbl_status.setText(f"* {model.name} Running")
                        self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
                        return
                except Exception as e:  # noqa: BLE001
                    logger.error("Failed to load dockable UI for %s: %s", model.name, e)

            try:
                success = handler.launch(model, REPOS_ROOT, self.process_manager)
            except Exception as e:
                logger.error(
                    "Launch exception for %s (type=%s, path=%s, handler=%s): %s",
                    model.name,
                    model.type,
                    getattr(model, "path", "N/A"),
                    type(handler).__name__,
                    e,
                    exc_info=True,
                )
                if hasattr(self, "_append_console_line"):
                    import traceback

                    tb_str = "".join(
                        traceback.format_exception(type(e), e, e.__traceback__)
                    )
                    self._append_console_line(
                        "Launcher", f"Failed to launch {model.name}:\n{tb_str}"
                    )
                success = False

            if success:
                self.show_toast(f"{model.name} Launched", "success")
                self.lbl_status.setText(f"* {model.name} Running")
                self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
            else:
                # Diagnostic: log why launch failed for debugging silent failures
                logger.error(
                    "Launch failed for %s (type=%s, path=%s, handler=%s)",
                    model.name,
                    model.type,
                    getattr(model, "path", "N/A"),
                    type(handler).__name__,
                )
                if hasattr(self, "_append_console_line"):
                    self._append_console_line(
                        "Launcher",
                        f"Failed to launch {model.name} (type={model.type}, path={getattr(model, 'path', 'N/A')}). See logs above.",
                    )
                self.show_toast(
                    f"Failed to launch {model.name} — check console", "error"
                )
                self.lbl_status.setText("* Launch Error")
                self.lbl_status.setStyleSheet(Styles.STATUS_ERROR)

                if hasattr(self, "_console_dock"):
                    self._console_dock.show()
                    if hasattr(self, "_action_console"):
                        self._action_console.setChecked(True)
        elif model.type == "mjcf" or str(abs_model_path).endswith(".xml"):
            self._launch_generic_mjcf(abs_model_path)
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

        if model.type == "matlab_suite":
            from src.launchers.matlab_suite_dialog import MatlabSuiteDialog

            dialog = MatlabSuiteDialog(self.launcher)
            dialog.exec()
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
        if path is None:
            raise ValueError("path must be provided")
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
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        from src.launchers.launcher_process_manager import start_vcxsrv

        try:
            # Auto-start VcXsrv on Windows for GUI support
            if os.name == "nt" and not start_vcxsrv():
                response = QMessageBox.question(
                    self.launcher,
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
                    self.launcher,
                    "Docker Image Not Found",
                    f"The Docker image '{self.docker_launcher.image_name}' is not available.\n\n"
                    "Build it first using:\n"
                    "  docker build -t upstream-drift:engine .\n\n"
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
                    self.launcher,
                    "Docker Launch Error",
                    f"Failed to launch {model.name} in Docker",
                )

        except (ValueError, RuntimeError) as e:
            logger.error(f"Failed to launch Docker container: {e}")
            QMessageBox.critical(
                self.launcher,
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
        if name is None:
            raise ValueError("name must be provided")
        use_wsl = hasattr(self, "chk_wsl") and self.chk_wsl.isChecked()

        if use_wsl:
            success = self.process_manager.launch_in_wsl(str(script_path))
            if success:
                self.lbl_status.setText(f"* {name} Running (WSL)")
                self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
                self.show_toast(f"{name} Launched in WSL", "success")
            else:
                QMessageBox.critical(
                    self.launcher, "Launch Error", f"Failed to launch {name} in WSL"
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
            QMessageBox.critical(
                self.launcher, "Launch Error", f"Failed to launch {name}"
            )

    @precondition(
        lambda self, name, module_name, cwd: name is not None and len(name.strip()) > 0,
        "Process name must be a non-empty string",
    )
    @precondition(
        lambda self, name, module_name, cwd: (
            module_name is not None and len(module_name.strip()) > 0
        ),
        "Module name must be a non-empty string",
    )
    def _launch_module_process(self, name: str, module_name: str, cwd: Path) -> None:
        """Helper to launch python module with error visibility.

        Similar to _launch_script_process but uses -m to run a module.
        If WSL mode is enabled, launches in WSL2 Ubuntu environment.
        """
        # Check if WSL mode is enabled
        if name is None:
            raise ValueError("name must be provided")
        use_wsl = hasattr(self, "chk_wsl") and self.chk_wsl.isChecked()

        if use_wsl:
            success = self.process_manager.launch_module_in_wsl(module_name, cwd)
            if success:
                self.lbl_status.setText(f"* {name} Running (WSL)")
                self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
                self.show_toast(f"{name} Launched in WSL", "success")
            else:
                QMessageBox.critical(
                    self.launcher, "Launch Error", f"Failed to launch {name} in WSL"
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
            QMessageBox.critical(
                self.launcher, "Launch Error", f"Failed to launch {name}"
            )

    def _launch_urdf_generator(self) -> None:
        """Launch the URDF generator / Model Explorer application."""
        # Try to load embedded URDF Generator (Model Explorer) first
        from src.shared.python.launcher_embed import get_embeddable_tool

        tool = get_embeddable_tool("model_explorer")
        if tool:
            try:
                # Check if already open
                for idx in range(self.workspace_tabs.count()):
                    if self.workspace_tabs.tabText(idx) == "Model Explorer":
                        self.workspace_tabs.setCurrentIndex(idx)
                        return

                ui_widget = tool.create_main_widget(self.launcher)
                if ui_widget:
                    self.dock_widget_as_tab(ui_widget, "Model Explorer")
                    self.show_toast("Model Explorer loaded as tab.", "success")
                    self.lbl_status.setText("> Model Explorer Running")
                    self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
                    return
            except Exception as e:
                logger.exception("Failed to launch Model Explorer embedded: %s", e)

        # Fallback to separate process launch if tool is not registered or failed
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
        """Launch the C3D motion viewer application.

        Searches for a C3D viewer entry-point script in (in order):
        1. The in-repo Simscape 3D viewer wrapper
           (``src/engines/.../python/src/apps/run_c3d_viewer.py``).
        2. The fleet-shared vendor viewer
           (``vendor/ud-tools/src/c3d_viewer/launch_pyqt6.py``).
        3. Legacy locations under a sibling ``tools/`` directory, kept for
           backwards-compatibility with installations that pre-date the
           in-repo viewers.
        """
        candidates = [
            REPOS_ROOT
            / "src"
            / "engines"
            / "Simscape_Multibody_Models"
            / "3D_Golf_Model"
            / "python"
            / "src"
            / "apps"
            / "run_c3d_viewer.py",
            REPOS_ROOT
            / "vendor"
            / "ud-tools"
            / "src"
            / "c3d_viewer"
            / "launch_pyqt6.py",
            REPOS_ROOT / "tools" / "c3d_viewer" / "c3d_viewer.py",
            REPOS_ROOT / "tools" / "c3d_viewer_app.py",
        ]
        c3d_script = next((p for p in candidates if p.exists()), None)

        if c3d_script is None:
            logger.error(
                "C3D Viewer script not found. Searched: %s",
                ", ".join(str(p) for p in candidates),
            )
            self.show_toast("C3D Viewer script not found.", "error")
            return

        if (
            "c3d_viewer" in self.running_processes
            and self.running_processes["c3d_viewer"].poll() is None
        ):
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

        if (
            "shot_tracer" in self.running_processes
            and self.running_processes["shot_tracer"].poll() is None
        ):
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
            abs_path = resolve_model_artifact_path(app, REPOS_ROOT)
            model_root = get_model_source_root(app, REPOS_ROOT)
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
                    cwd=str(model_root),
                    creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
                )

            self.running_processes[app.id] = process
            self.show_toast(f"{app.name} launch initiated.", "success")

        except FileNotFoundError:
            self.show_toast("MATLAB executable not found in PATH.", "error")
        except (PermissionError, OSError) as e:
            logger.error(f"Failed to launch MATLAB app: {e}")
            self.show_toast(f"Launch failed: {e}", "error")
