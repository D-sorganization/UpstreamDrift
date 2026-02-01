"""Model-specific launch handlers for the Golf Launcher.

This module provides specialized launch logic for different physics engines
and simulation types (MuJoCo, Drake, Pinocchio, OpenSim, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from src.shared.python.logging_config import get_logger

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

    MODEL_TYPES = {"humanoid_mujoco", "humanoid"}

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
        module_name = (
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.main"
        )
        cwd = repo_path

        process = process_manager.launch_module(
            name="MuJoCo Humanoid Golf",
            module_name=module_name,
            cwd=cwd,
        )
        return process is not None


class ComprehensiveModelHandler:
    """Handler for launching the comprehensive golf model."""

    MODEL_TYPES = {"comprehensive", "comprehensive_mujoco"}

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
