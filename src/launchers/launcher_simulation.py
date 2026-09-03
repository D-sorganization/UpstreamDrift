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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QEventLoop, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMessageBox

from src.launchers.launcher_constants import (
    CREATE_NO_WINDOW,
    REPOS_ROOT,
)
from src.launchers.launcher_model_sources import (
    get_model_source_root,
    resolve_model_artifact_path,
)
from src.shared.python.ui.tile_help import attach_tile_help
from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.security.secure_subprocess import (
    SecureSubprocessError,
    secure_popen,
)
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


def dependency_cache_key(model: Any) -> str:
    """Return the stable cache key shared by selection and launch paths."""
    model_id = getattr(model, "id", None)
    if not model_id:
        raise ValueError("model.id must be provided")
    return str(model_id)


def dependency_probe_key(model: Any) -> str:
    """Return the dependency-map key for a model."""
    model_id = dependency_cache_key(model)
    if model_id in DEPENDENCY_MAP:
        return model_id
    model_type = getattr(model, "type", None)
    return str(model_type) if model_type is not None else model_id


class DependencyProbeThread(QThread):
    """Run a dependency probe away from the GUI thread."""

    probe_finished = pyqtSignal(str, bool, str)

    def __init__(
        self,
        model_id: str,
        probe_key: str,
        checker: Callable[[str], tuple[bool, str]],
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        if not model_id:
            raise ValueError("model_id must be provided")
        if not probe_key:
            raise ValueError("probe_key must be provided")
        self._model_id = model_id
        self._probe_key = probe_key
        self._checker = checker

    def run(self) -> None:
        ok, error = self._checker(self._probe_key)
        self.probe_finished.emit(self._model_id, ok, error)


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

    def _launcher_ref(self) -> Any:
        """Return the concrete launcher for manager or delegated calls."""
        return self.__dict__.get("launcher", self)

    def _dependency_cache(self) -> dict[str, tuple[bool, str]]:
        """Return the launcher's dependency cache, creating it if needed."""
        launcher = self._launcher_ref()
        if "_dependency_status_cache" not in launcher.__dict__:
            launcher._dependency_status_cache = {}
        return launcher._dependency_status_cache

    def _dependency_probe_threads(self) -> dict[str, DependencyProbeThread]:
        """Return active dependency probes, creating storage if needed."""
        launcher = self._launcher_ref()
        if "_dependency_probe_workers" not in launcher.__dict__:
            launcher._dependency_probe_workers = {}
        return launcher._dependency_probe_workers

    def _start_dependency_probe(self, model_id: str, model: Any) -> bool:
        """Start or reuse an async dependency probe for ``model``.

        Returns True when a probe is in-flight after this call. The result is
        recorded through ``_on_dependency_probe_finished`` on the GUI thread.
        """
        probe_key = dependency_probe_key(model)
        if probe_key not in DEPENDENCY_MAP:
            return False

        threads = self._dependency_probe_threads()
        if model_id in threads:
            return True

        launcher = self._launcher_ref()
        thread = DependencyProbeThread(
            model_id,
            probe_key,
            self._check_module_dependencies,
            launcher if hasattr(launcher, "thread") else None,
        )
        threads[model_id] = thread

        def handle_probe_finished(
            finished_model_id: str, deps_ok: bool, deps_error: str
        ) -> None:
            try:
                self._on_dependency_probe_finished(
                    finished_model_id, deps_ok, deps_error
                )
            finally:
                threads.pop(finished_model_id, None)

        thread.probe_finished.connect(handle_probe_finished)
        thread.start()
        return True

    def _on_dependency_probe_finished(
        self, model_id: str, deps_ok: bool, deps_error: str
    ) -> None:
        """Record an async dependency probe and refresh selected-model UI."""
        launcher = self._launcher_ref()
        self._dependency_cache()[model_id] = (deps_ok, deps_error)
        if getattr(launcher, "selected_model", None) != model_id:
            return

        model = launcher._get_model(model_id)
        if model is None:
            return

        if deps_ok:
            if hasattr(launcher, "_set_dependency_success_status"):
                launcher._set_dependency_success_status()
            if hasattr(launcher, "update_launch_button"):
                launcher.update_launch_button(getattr(model, "name", model_id))
            return

        if hasattr(launcher, "_set_dependency_error_status"):
            probe_key = dependency_probe_key(model)
            launcher._set_dependency_error_status(
                model, probe_key, deps_error, DEPENDENCY_MAP
            )

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
        if "training_controller" in model_id:
            self._launch_training_controller()
            return True
        if "library_tool" in model_id:
            if hasattr(self, "_open_library_tab"):
                self._open_library_tab()
            return True
        if model_id == "sidekick":
            if getattr(self, "sidekick_sidebar", None) is None:
                self._install_sidekick_sidebar()
            self._toggle_sidekick(True)
            self.open_sidekick_tab("chat")
            return True
        return False

    def _try_launch_docker(self, model: Any) -> bool:
        if getattr(model, "embed_adapter", None):
            return False
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

        model_id = dependency_cache_key(model)
        key = dependency_probe_key(model)
        if key not in DEPENDENCY_MAP:
            return True

        cache = self._dependency_cache()
        if model_id not in cache:
            self.lbl_status.setText(f"> Checking {model.name} dependencies...")
            self.lbl_status.setStyleSheet(Styles.STATUS_WARNING)
            if hasattr(self, "btn_launch"):
                self.btn_launch.setEnabled(False)
            self._start_dependency_probe(model_id, model)
            return False

        deps_ok, deps_error = cache[model_id]

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

    def _try_dockable_launch(self, model: Any, handler: Any) -> bool:
        """Launch ``model`` through its handler's dockable UI, if it has one.

        Extracted verbatim from :meth:`_execute_local_launch`, which exceeded
        the 100-line function budget the moment it was touched (issue #9413).
        Behaviour is unchanged apart from the help wiring noted below.

        Args:
            model: The launcher model being started.
            handler: The registered handler for ``model.type``.

        Returns:
            ``True`` when a dockable UI was created and surfaced, so the
            caller must not fall through to the process launch. ``False``
            when the handler has no dockable UI, returned none, or raised.
        """
        dockable_factory = getattr(type(handler), "get_dockable_ui", None)
        if not callable(dockable_factory):
            return False

        try:
            ui_widget = handler.get_dockable_ui(model, REPOS_ROOT)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load dockable UI for %s: %s", model.name, e)
            return False
        if not ui_widget:
            return False

        # Per-tile help (F1 + Help menu) for handler-provided dockable UIs.
        attach_tile_help(ui_widget, getattr(model, "id", None))

        if hasattr(self, "sidekick_sidebar") and hasattr(
            ui_widget, "set_sidekick_session"
        ):
            # The sidebar widget might have a direct session attribute or IS
            # the session.
            session = getattr(self.sidekick_sidebar, "session", self.sidekick_sidebar)
            try:
                ui_widget.set_sidekick_session(session)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to inject Sidekick session: %s", e)

        available = getattr(ui_widget, "is_tool_available", True)

        # Always dock as tab first; pop-out can be triggered post-launch from
        # DraggableTabWidget.
        if hasattr(self, "dock_widget_as_tab"):
            self.dock_widget_as_tab(ui_widget, model.name)
            self.show_toast(
                f"{model.name} Docked" if available else f"Failed to load {model.name}",
                "success" if available else "error",
            )
        elif hasattr(self, "popout_widget"):
            self.popout_widget(ui_widget, model.name)
            self.show_toast(
                (
                    f"{model.name} Popped Out"
                    if available
                    else f"Failed to load {model.name}"
                ),
                "success" if available else "error",
            )

        if available:
            self.lbl_status.setText(f"* {model.name} Running")
            self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
        else:
            self.lbl_status.setText("* Launch Error")
            self.lbl_status.setStyleSheet(Styles.STATUS_ERROR)
        return True

    def _execute_local_launch(self, model: Any) -> None:
        try:
            abs_model_path = resolve_model_artifact_path(model, REPOS_ROOT)
        except ValueError:
            self.show_toast("Model path missing.", "error")
            return

        handler = self.model_handler_registry.get_handler(model.type)
        if handler:
            if self._try_dockable_launch(model, handler):
                return

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
                reason = None
                status_message = getattr(handler, "status_message", None)
                if callable(status_message):
                    reason_text = status_message(model)
                    if isinstance(reason_text, str) and reason_text.strip():
                        reason = reason_text.strip()

                if reason is None:
                    logger.error(
                        "Launch failed for %s (type=%s, path=%s, handler=%s)",
                        model.name,
                        model.type,
                        getattr(model, "path", "N/A"),
                        type(handler).__name__,
                    )
                    toast_message = f"Failed to launch {model.name} — check console"
                    console_message = f"Failed to launch {model.name} (type={model.type}, path={getattr(model, 'path', 'N/A')}). See logs above."
                else:
                    logger.error("Launch unavailable for %s: %s", model.name, reason)
                    toast_message = reason
                    console_message = reason

                if hasattr(self, "_append_console_line"):
                    self._append_console_line("Launcher", console_message)
                self.show_toast(toast_message, "error")
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

        if hasattr(self, "layout_manager") and hasattr(
            self.layout_manager, "record_launch"
        ):
            self.layout_manager.record_launch(model_id)
            if hasattr(self, "_save_layout"):
                self._save_layout()

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
            from src.launchers.matlab_suite_dialog import MatlabSuiteWidget

            widget = MatlabSuiteWidget(self.launcher)

            # Always dock as tab first; pop-out can be triggered post-launch from DraggableTabWidget
            if hasattr(self, "dock_widget_as_tab"):
                self.dock_widget_as_tab(widget, model.name)
                self.show_toast(f"{model.name} Docked", "success")
            elif hasattr(self, "popout_widget"):
                self.popout_widget(widget, model.name)
                self.show_toast(f"{model.name} Popped Out", "success")

            self.lbl_status.setText(f"* {model.name} Running")
            self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
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
                / "src"
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
                self.launcher._passive_mjcf_viewer = mujoco.viewer.launch_passive(m, d)

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
                    attach_tile_help(ui_widget, getattr(tool, "tool_id", None))
                    ui_widget.destroyed.connect(tool.cleanup)
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

    def _launch_training_controller(self) -> None:
        """Launch or focus the Training Controller tab."""
        from src.shared.python.launcher_embed import get_embeddable_tool

        tool = get_embeddable_tool("training_controller")
        if tool:
            try:
                # Check if already open
                for idx in range(self.workspace_tabs.count()):
                    if self.workspace_tabs.tabText(idx) == "Training":
                        self.workspace_tabs.setCurrentIndex(idx)
                        return

                ui_widget = tool.create_main_widget(self.launcher)
                if ui_widget:
                    attach_tile_help(ui_widget, getattr(tool, "tool_id", None))
                    ui_widget.destroyed.connect(tool.cleanup)
                    self.dock_widget_as_tab(ui_widget, "Training")
                    self.show_toast("Training Controller loaded as tab.", "success")
                    self.lbl_status.setText("> Training Controller Running")
                    self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
                    btn = getattr(self, "btn_training_sidebar", None)
                    if btn is not None:
                        btn.setChecked(True)
                    return
            except Exception as e:
                logger.exception("Failed to launch Training Controller embedded: %s", e)

        self.show_toast("Training Controller is unavailable.", "error")

    def _launch_c3d_viewer(self) -> None:
        """Launch the C3D motion viewer application.

        First attempts to load the C3D Viewer embedded as a tab using the
        shared EmbeddableTool registry. If the embeddable tool is not registered,
        falls back to spawning a standalone subprocess.
        """
        try:
            from src.shared.python.launcher_embed import get_embeddable_tool

            # The adapter registers automatically via embedded_tool_bootstrap.py
            tool = get_embeddable_tool("c3d_viewer")
            if tool:
                # Check if already open as tab
                for idx in range(self.workspace_tabs.count()):
                    if self.workspace_tabs.tabText(idx) == "C3D Viewer":
                        self.workspace_tabs.setCurrentIndex(idx)
                        return

                ui_widget = tool.create_main_widget(self.launcher)
                if ui_widget:
                    attach_tile_help(ui_widget, getattr(tool, "tool_id", None))
                    ui_widget.destroyed.connect(tool.cleanup)
                    self.dock_widget_as_tab(ui_widget, "C3D Viewer")
                    self.show_toast("C3D Viewer loaded as tab.", "success")
                    self.lbl_status.setText("> C3D Viewer Running")
                    self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
                    return
        except Exception as e:
            logger.exception("Failed to launch C3D Viewer embedded: %s", e)

        # Fallback to subprocess if not registered or failed
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
                "c3d_viewer", c3d_script, c3d_script.parent, keep_terminal_open=True
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
                "shot_tracer", shot_tracer_script, REPOS_ROOT, keep_terminal_open=True
            )
            if not process:
                raise RuntimeError("ProcessManager returned None")
            self.show_toast("Shot Tracer launched.", "success")

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Failed to launch Shot Tracer: {e}")
            self.show_toast(f"Launch failed: {e}", "error")

    def _launch_matlab_app(self, app: Any) -> bool:
        """Launch a MATLAB application and report whether the launch was initiated."""
        app_path = getattr(app, "path", None)
        if not app_path:
            self.show_toast("Invalid MATLAB configuration.", "error")
            return False

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
            return True

        except FileNotFoundError:
            self.show_toast("MATLAB executable not found in PATH.", "error")
            return False
        except SecureSubprocessError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                self.show_toast("MATLAB executable not found in PATH.", "error")
                return False
            logger.exception("Failed to launch MATLAB app")
            self.show_toast(
                "MATLAB could not be started. Verify its installation and PATH.",
                "error",
            )
            return False
        except (PermissionError, OSError) as e:
            logger.error(f"Failed to launch MATLAB app: {e}")
            self.show_toast(f"Launch failed: {e}", "error")
            return False
