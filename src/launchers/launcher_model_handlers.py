"""Model-specific launch handlers for the Golf Launcher.

This module provides specialized launch logic for different physics engines
and simulation types (MuJoCo, Drake, Pinocchio, OpenSim, etc.).
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

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


class HumanoidMuJoCoHandler:
    """Handler for launching MuJoCo humanoid golf simulations."""

    MODEL_TYPES = {"humanoid_mujoco", "humanoid", "custom_humanoid"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the MuJoCo humanoid simulation.

        Args:
            model: Model configuration object.
            repo_path: Path to the repository root.
            process_manager: Process manager for subprocess handling.

        Returns:
            True if launch succeeded, False otherwise.
        """
        module_name = "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf"
        cwd = repo_path

        process = process_manager.launch_module(
            name="MuJoCo Humanoid Golf",
            module_name=module_name,
            cwd=cwd,
        )
        return process is not None


class ComprehensiveModelHandler:
    """Handler for launching the comprehensive golf model."""

    MODEL_TYPES = {"comprehensive", "comprehensive_mujoco", "custom_dashboard"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the comprehensive golf model simulation.

        Args:
            model: Model configuration object.
            repo_path: Path to the repository root.
            process_manager: Process manager for subprocess handling.

        Returns:
            True if launch succeeded, False otherwise.
        """
        module_name = "src.engines.physics_engines.mujoco.python.humanoid_launcher"
        cwd = repo_path

        process = process_manager.launch_module(
            name="Comprehensive Golf Model",
            module_name=module_name,
            cwd=cwd,
        )
        return process is not None


class DrakeHandler:
    """Handler for launching Drake physics simulations."""

    MODEL_TYPES = {"drake", "drake_golf"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the Drake simulation.

        Args:
            model: Model configuration object.
            repo_path: Path to the repository root.
            process_manager: Process manager for subprocess handling.

        Returns:
            True if launch succeeded, False otherwise.
        """
        script_path = (
            repo_path / "src/engines/physics_engines/drake/python/src/drake_gui_app.py"
        )
        cwd = repo_path / "src/engines/physics_engines/drake/python"

        process = process_manager.launch_script(
            name="Drake Golf Model",
            script_path=script_path,
            cwd=cwd,
        )
        return process is not None


class PinocchioHandler:
    """Handler for launching Pinocchio physics simulations."""

    MODEL_TYPES = {"pinocchio", "pinocchio_golf"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the Pinocchio simulation.

        Args:
            model: Model configuration object.
            repo_path: Path to the repository root.
            process_manager: Process manager for subprocess handling.

        Returns:
            True if launch succeeded, False otherwise.
        """
        script_path = (
            repo_path
            / "src/engines/physics_engines/pinocchio/python/pinocchio_golf/main.py"
        )
        cwd = repo_path / "src/engines/physics_engines/pinocchio/python"

        process = process_manager.launch_script(
            name="Pinocchio Golf Model",
            script_path=script_path,
            cwd=cwd,
        )
        return process is not None


class OpenSimHandler:
    """Handler for launching OpenSim simulations."""

    MODEL_TYPES = {"opensim", "opensim_golf"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the OpenSim simulation.

        Args:
            model: Model configuration object.
            repo_path: Path to the repository root.
            process_manager: Process manager for subprocess handling.

        Returns:
            True if launch succeeded, False otherwise.
        """
        script_path = (
            repo_path / "src/engines/physics_engines/opensim/python/opensim_golf.py"
        )
        cwd = repo_path / "src/engines/physics_engines/opensim/python"

        process = process_manager.launch_script(
            name="OpenSim Golf Model",
            script_path=script_path,
            cwd=cwd,
        )
        return process is not None


class MyoSimHandler:
    """Handler for launching MyoSim (musculoskeletal) simulations."""

    MODEL_TYPES = {"myosim", "myosim_golf", "musculoskeletal"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the MyoSim simulation.

        Args:
            model: Model configuration object.
            repo_path: Path to the repository root.
            process_manager: Process manager for subprocess handling.

        Returns:
            True if launch succeeded, False otherwise.
        """
        script_path = repo_path / "src/engines/physics_engines/myosim/python/main.py"
        cwd = repo_path / "src/engines/physics_engines/myosim/python"

        process = process_manager.launch_script(
            name="MyoSim Golf Model",
            script_path=script_path,
            cwd=cwd,
        )
        return process is not None


class OpenPoseHandler:
    """Handler for launching OpenPose pose estimation GUI."""

    MODEL_TYPES = {"openpose", "pose_estimation"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the OpenPose GUI.

        Args:
            model: Model configuration object.
            repo_path: Path to the repository root.
            process_manager: Process manager for subprocess handling.

        Returns:
            True if launch succeeded, False otherwise.
        """
        script_path = repo_path / "src/shared/python/pose_estimation/openpose_gui.py"
        cwd = repo_path

        process = process_manager.launch_script(
            name="OpenPose",
            script_path=script_path,
            cwd=cwd,
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
        model_path = getattr(model, "path", None) or ""
        if not model_path:
            logger.error(
                "SpecialAppHandler: model '%s' has no path",
                getattr(model, "id", "unknown"),
            )
            return False

        script_path = repo_path / model_path
        model_name = getattr(model, "name", model_path)

        if not script_path.exists():
            logger.warning("SpecialAppHandler: script not found: %s", script_path)
            return False

        process = process_manager.launch_script(
            name=model_name,
            script_path=script_path,
            cwd=repo_path,
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
        model_path = getattr(model, "path", None) or ""
        if not model_path:
            logger.error("PuttingGreenHandler: model has no path")
            return False

        script_path = repo_path / model_path
        if not script_path.exists():
            logger.warning("PuttingGreenHandler: script not found: %s", script_path)
            return False

        process = process_manager.launch_script(
            name="Putting Green Simulator",
            script_path=script_path,
            cwd=script_path.parent,
        )
        return process is not None


class MatlabFileHandler:
    """Handler for opening MATLAB files (.slx, .m) with system MATLAB.

    Design by Contract:
        Precondition: model.path must point to a .slx or .m file
        Postcondition: file is opened with the system MATLAB installation
    """

    MODEL_TYPES = {"matlab_file"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Open a MATLAB file with the system MATLAB installation.

        Args:
            model: Model configuration with 'path' and 'name' attrs.
            repo_path: Path to the repository root.
            process_manager: Process manager (unused, opens via OS).

        Returns:
            True if launch succeeded, False otherwise.
        """
        model_path = getattr(model, "path", None) or ""
        if not model_path:
            logger.error(
                "MatlabFileHandler: model '%s' has no path",
                getattr(model, "id", "unknown"),
            )
            return False

        file_path = repo_path / model_path
        if not file_path.exists():
            logger.warning("MatlabFileHandler: file not found: %s", file_path)
            return False

        try:
            if platform.system() == "Windows":
                os.startfile(str(file_path))  # noqa: S606
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(file_path)])  # noqa: S603, S607
            else:
                subprocess.Popen(["xdg-open", str(file_path)])  # noqa: S603, S607
            logger.info("Opened MATLAB file: %s", file_path.name)
            return True
        except (FileNotFoundError, PermissionError, OSError):
            logger.exception("Failed to open MATLAB file: %s", file_path)
            return False


class DocumentHandler:
    """Handler for opening document files (.md, .pdf, etc.) with the system viewer.

    Design by Contract:
        Precondition: model.path must point to a document file
        Postcondition: file is opened with the system default application
    """

    MODEL_TYPES = {"document"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Open a document file with the system default application.

        Args:
            model: Model configuration with 'path' and 'name' attrs.
            repo_path: Path to the repository root.
            process_manager: Process manager (unused, opens via OS).

        Returns:
            True if launch succeeded, False otherwise.
        """
        model_path = getattr(model, "path", None) or ""
        if not model_path:
            logger.error(
                "DocumentHandler: model '%s' has no path",
                getattr(model, "id", "unknown"),
            )
            return False

        file_path = repo_path / model_path
        if not file_path.exists():
            logger.warning("DocumentHandler: file not found: %s", file_path)
            return False

        try:
            if platform.system() == "Windows":
                os.startfile(str(file_path))  # noqa: S606
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(file_path)])  # noqa: S603, S607
            else:
                subprocess.Popen(["xdg-open", str(file_path)])  # noqa: S603, S607
            logger.info("Opened document: %s", file_path.name)
            return True
        except (FileNotFoundError, PermissionError, OSError):
            logger.exception("Failed to open document: %s", file_path)
            return False


class ModelHandlerRegistry:
    """Registry for model launch handlers.

    This class implements the Strategy pattern for model launching,
    allowing new handlers to be added without modifying existing code.
    """

    def __init__(self) -> None:
        """Initialize the handler registry with default handlers."""
        self._handlers: list[ModelHandler] = [
            HumanoidMuJoCoHandler(),
            ComprehensiveModelHandler(),
            DrakeHandler(),
            PinocchioHandler(),
            OpenSimHandler(),
            MyoSimHandler(),
            OpenPoseHandler(),
            SpecialAppHandler(),
            PuttingGreenHandler(),
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
        handler = self.get_handler(model_type)
        if handler is None:
            logger.warning(f"No handler found for model type: {model_type}")
            return False

        return handler.launch(model, repo_path, process_manager)
