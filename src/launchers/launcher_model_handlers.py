"""Model-specific launch handlers for the Golf Launcher.

This module provides specialized launch logic for different physics engines
and simulation types (MuJoCo, Drake, Pinocchio, OpenSim, etc.).

DRY refactoring: Consolidated 7 near-identical handler classes into
a data-driven ScriptHandler with a handler registry table.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from src.launchers.launcher_model_sources import (
    get_model_python_paths,
    get_model_working_directory,
    resolve_model_artifact_path,
)
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from src.launchers.launcher_process_manager import ProcessManager

logger = get_logger(__name__)


class ModelHandler(Protocol):
    """Protocol for model launch handlers."""

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler can handle the given model type."""
        ...

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the model."""
        ...


class ModuleHandler:
    """Handler that launches a Python module via process_manager.launch_module.

    DRY replacement for HumanoidMuJoCoHandler and ComprehensiveModelHandler.
    """

    def __init__(
        self, model_types: set[str], module_name: str, display_name: str
    ) -> None:
        if model_types is None:
            raise ValueError("model_types must be provided")
        self.model_types = model_types
        self.module_name = module_name
        self.display_name = display_name

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.model_types

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the module."""
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        process = process_manager.launch_module(
            name=self.display_name,
            module_name=self.module_name,
            cwd=repo_path,
            extra_python_paths=get_model_python_paths(model, repo_path),
        )
        return process is not None


class ScriptHandler:
    """Handler that launches a Python script via process_manager.launch_script.

    DRY replacement for DrakeHandler, PinocchioHandler, OpenSimHandler,
    MyoSimHandler, and OpenPoseHandler.
    """

    def __init__(
        self,
        model_types: set[str],
        script_path: str,
        display_name: str,
        cwd_path: str | None = None,
    ) -> None:
        if model_types is None:
            raise ValueError("model_types must be provided")
        self.model_types = model_types
        self._script_path = script_path
        self.display_name = display_name
        self._cwd_path = cwd_path

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.model_types

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the script."""
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        script_path = repo_path / self._script_path
        cwd = repo_path / self._cwd_path if self._cwd_path else repo_path

        process = process_manager.launch_script(
            name=self.display_name,
            script_path=script_path,
            cwd=cwd,
            extra_python_paths=get_model_python_paths(model, repo_path),
        )
        return process is not None


class SpecialAppHandler:
    """Handler for launching special applications (tools, utilities).

    Handles model types: special_app
    Covers: c3d_viewer, openpose, mediapipe, model_explorer, video_analyzer,
    data_explorer, and any future tool/utility tiles.

    Design by Contract:
        Precondition: model.path must be a valid relative path to a Python script
        Postcondition: script is launched as a subprocess
    """

    MODEL_TYPES = {"special_app"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch a special application by running its script.

        Args:
            model: Model configuration with 'path' and 'name' attrs.
            repo_path: Path to the repository root.
            process_manager: Process manager for subprocess handling.

        Returns:
            True if launch succeeded, False otherwise.
        """
        # DBC Precondition: model must have a path
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        model_path = getattr(model, "path", None) or ""
        if not model_path:
            logger.error(
                "SpecialAppHandler: model '%s' has no path",
                getattr(model, "id", "unknown"),
            )
            return False

        script_path = resolve_model_artifact_path(model, repo_path)
        model_name = getattr(model, "name", model_path)

        if not script_path.exists():
            logger.warning("SpecialAppHandler: script not found: %s", script_path)
            return False

        process = process_manager.launch_script(
            name=model_name,
            script_path=script_path,
            cwd=get_model_working_directory(model, repo_path),
            extra_python_paths=get_model_python_paths(model, repo_path),
        )
        return process is not None


class PuttingGreenHandler:
    """Handler for launching the Putting Green simulator.

    Design by Contract:
        Precondition: model.path must point to the putting green simulator
        Postcondition: putting green simulator subprocess is running
    """

    MODEL_TYPES = {"putting_green"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the Putting Green simulation.

        Args:
            model: Model configuration with 'path' attr.
            repo_path: Path to the repository root.
            process_manager: Process manager for subprocess handling.

        Returns:
            True if launch succeeded, False otherwise.
        """
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        model_path = getattr(model, "path", None) or ""
        if not model_path:
            logger.error("PuttingGreenHandler: model has no path")
            return False

        script_path = resolve_model_artifact_path(model, repo_path)
        if not script_path.exists():
            logger.warning("PuttingGreenHandler: script not found: %s", script_path)
            return False

        process = process_manager.launch_script(
            name="Putting Green Simulator",
            script_path=script_path,
            cwd=get_model_working_directory(model, repo_path, script_path.parent),
            extra_python_paths=get_model_python_paths(model, repo_path),
        )
        return process is not None

class BiomechExerciseHandler:
    """Handler for launching biomechanics exercise dashboards."""

    MODEL_TYPES = {"biomech_exercise"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the biomechanics exercise dashboard."""
        if repo_path is None:
            raise ValueError("repo_path must be provided")

        exercise_name = getattr(model, "exercise", "gait")
        model_name = getattr(model, "name", f"Biomechanics Exercise: {exercise_name.title()}")
        script_path = repo_path / "src" / "launchers" / "exercise_dashboard.py"

        # Pass the exercise as a CLI argument through an environment variable or module?
        # Actually, since process_manager.launch_script doesn't take args natively,
        # we can launch it using launch_module with the right module_name, but that doesn't take args either.
        # Let's use secure_popen manually or just set an environment variable.
        env = process_manager.get_subprocess_env(get_model_python_paths(model, repo_path))
        env["BIOMECH_EXERCISE"] = exercise_name

        # Wait, if we use launch_script, we can't pass args. 
        # But we can modify launch_script to take args, or just use the environment variable fallback.
        # Let's use secure_popen if needed, but launch_script is safer. Let's see if we can pass args?
        # Let's use launch_script and let the script read sys.argv or env.
        # I'll modify exercise_dashboard.py to read BIOMECH_EXERCISE env var if --exercise is not passed.
        
        process = process_manager.launch_script(
            name=model_name,
            script_path=script_path,
            cwd=repo_path,
            env=env,
            extra_python_paths=get_model_python_paths(model, repo_path),
        )
        return process is not None


def _open_with_system_app(file_path: Path, handler_name: str) -> bool:
    """Open a file with the system default application.

    DRY helper: eliminates the duplicated platform-detection code in
    MatlabFileHandler and DocumentHandler.

    Args:
        file_path: Path to the file to open.
        handler_name: Name of the calling handler (for log messages).

    Returns:
        True if the file was opened, False otherwise.
    """
    try:
        if platform.system() == "Windows":
            os.startfile(str(file_path))  # type: ignore[attr-defined]  # noqa: S606
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(file_path)])  # noqa: S603, S607
        else:
            subprocess.Popen(["xdg-open", str(file_path)])  # noqa: S603, S607
        logger.info("%s: opened %s", handler_name, file_path.name)
        return True
    except (FileNotFoundError, PermissionError, OSError):
        logger.exception("%s: failed to open %s", handler_name, file_path)
        return False


class _SystemFileHandler:
    """Base handler for opening files with system applications.

    DRY base for MatlabFileHandler and DocumentHandler.
    """

    MODEL_TYPES: set[str] = set()
    HANDLER_NAME: str = "SystemFileHandler"

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Open a file with the system default application."""
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        model_path = getattr(model, "path", None) or ""
        if not model_path:
            logger.error(
                "%s: model '%s' has no path",
                self.HANDLER_NAME,
                getattr(model, "id", "unknown"),
            )
            return False

        file_path = resolve_model_artifact_path(model, repo_path)
        if not file_path.exists():
            logger.warning("%s: file not found: %s", self.HANDLER_NAME, file_path)
            return False

        return _open_with_system_app(file_path, self.HANDLER_NAME)


class MatlabFileHandler(_SystemFileHandler):
    """Handler for opening MATLAB files (.slx, .m) with system MATLAB."""

    MODEL_TYPES = {"matlab_file"}
    HANDLER_NAME = "MatlabFileHandler"


class DocumentHandler:
    """Handler for opening document files (.md, .pdf, etc.) securely using a proxy runner."""

    MODEL_TYPES = {"document"}
    HANDLER_NAME = "DocumentHandler"

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: Any,
    ) -> bool:
        """Open a document securely using the approved document_proxy."""
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        model_path = getattr(model, "path", None) or ""
        if not model_path:
            logger.error(
                "%s: model '%s' has no path",
                self.HANDLER_NAME,
                getattr(model, "id", "unknown"),
            )
            return False

        file_path = resolve_model_artifact_path(model, repo_path)
        if not file_path.exists():
            logger.warning("%s: file not found: %s", self.HANDLER_NAME, file_path)
            return False

        proxy_script = repo_path / "src" / "launchers" / "document_proxy.py"
        if not proxy_script.exists():
            logger.error(
                "%s: Proxy script not found: %s", self.HANDLER_NAME, proxy_script
            )
            return False

        try:
            from src.shared.python.security.secure_subprocess import secure_popen
            import sys

            process = secure_popen(
                [sys.executable, str(proxy_script), str(file_path)],
                cwd=str(repo_path),
                suite_root=repo_path,
            )
            # Attach the process so it can be tracked/stopped if necessary, though it likely exits immediately.
            process_manager.attach_process(f"Document_{file_path.name}", process)
            logger.info("%s: launched proxy for %s", self.HANDLER_NAME, file_path.name)
            return True
        except Exception as e:
            logger.error(
                "%s: failed to launch document proxy: %s", self.HANDLER_NAME, e
            )
            return False


# ============================================================
# Handler Registry Table (DRY: data-driven registration)
# ============================================================

_MODULE_HANDLERS = [
    ModuleHandler(
        model_types={"humanoid_mujoco", "humanoid", "custom_humanoid"},
        module_name="src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf",
        display_name="MuJoCo Humanoid Golf",
    ),
    ModuleHandler(
        model_types={"comprehensive", "comprehensive_mujoco", "custom_dashboard"},
        module_name="src.engines.physics_engines.mujoco.python.humanoid_launcher",
        display_name="Comprehensive Golf Model",
    ),
    ModuleHandler(
        model_types={"drake", "drake_golf"},
        module_name="src.engines.physics_engines.drake.python.src.drake_gui_app",
        display_name="Drake Golf Model",
    ),
]

_SCRIPT_HANDLERS = [
    ScriptHandler(
        model_types={"pinocchio", "pinocchio_golf"},
        script_path="src/engines/physics_engines/pinocchio/python/pinocchio_golf/main.py",
        display_name="Pinocchio Golf Model",
        cwd_path="src/engines/physics_engines/pinocchio/python",
    ),
    ScriptHandler(
        model_types={"opensim", "opensim_golf"},
        script_path="src/engines/physics_engines/opensim/python/opensim_golf.py",
        display_name="OpenSim Golf Model",
        cwd_path="src/engines/physics_engines/opensim/python",
    ),
    ScriptHandler(
        model_types={"myosim", "myosim_golf", "musculoskeletal"},
        script_path="src/engines/physics_engines/myosim/python/main.py",
        display_name="MyoSim Golf Model",
        cwd_path="src/engines/physics_engines/myosim/python",
    ),
    ScriptHandler(
        model_types={"openpose", "pose_estimation"},
        script_path="src/shared/python/pose_estimation/openpose_gui.py",
        display_name="OpenPose",
    ),
]


# Backward-compatible aliases for the old per-engine handler classes.
# These were replaced by the data-driven ModuleHandler/ScriptHandler tables
# but some tests import them by name.
HumanoidMuJoCoHandler = type(
    "HumanoidMuJoCoHandler",
    (ModuleHandler,),
    {
        "__init__": lambda self: ModuleHandler.__init__(
            self,
            model_types={"humanoid_mujoco", "humanoid", "custom_humanoid"},
            module_name="src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf",
            display_name="MuJoCo Humanoid Golf",
        ),
    },
)


class ModelHandlerRegistry:
    """Registry for model launch handlers.

    This class implements the Strategy pattern for model launching,
    allowing new handlers to be added without modifying existing code.
    """

    def __init__(self) -> None:
        """Initialize the handler registry with default handlers."""
        self._handlers: list[ModelHandler] = [
            *_MODULE_HANDLERS,
            *_SCRIPT_HANDLERS,
            SpecialAppHandler(),
            PuttingGreenHandler(),
            BiomechExerciseHandler(),
            MatlabFileHandler(),
            DocumentHandler(),
        ]

    def register_handler(self, handler: ModelHandler) -> None:
        """Register a new model handler.

        Args:
            handler: Handler instance implementing the ModelHandler protocol.
        """
        self._handlers.append(handler)

    def get_handler(self, model_type: str) -> ModelHandler | None:
        """Get a handler that can launch the given model type.

        Args:
            model_type: The type of model to launch.

        Returns:
            A handler that can launch the model, or None if not found.
        """
        if model_type is None:
            raise ValueError("model_type must be provided")
        for handler in self._handlers:
            if handler.can_handle(model_type):
                return handler
        return None

    def launch_model(
        self,
        model_type: str,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch a model using the appropriate handler.

        Args:
            model_type: The type of model to launch.
            model: Model configuration object.
            repo_path: Path to the repository root.
            process_manager: Process manager for subprocess handling.

        Returns:
            True if launch succeeded, False otherwise.
        """
        if model_type is None:
            raise ValueError("model_type must be provided")
        handler = self.get_handler(model_type)
        if handler is None:
            logger.warning(f"No handler found for model type: {model_type}")
            return False

        return handler.launch(model, repo_path, process_manager)
