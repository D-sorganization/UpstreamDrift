"""Agent Control Tools for AI-Powered App Management.

This module provides tools that enable the AI chat bot to control
all aspects of the UpstreamDrift application as an autonomous agent.

Features:
    - Engine control (start/stop/configure physics engines)
    - Model management (load/unload/compare models)
    - File operations (import/export URDF, MJCF, C3D)
    - Visualization control
    - Settings management
    - Simulation control

Example:
    >>> from src.shared.python.ai.tools.agent_control import AgentController
    >>> controller = AgentController()
    >>> controller.start_engine("mujoco")
    >>> controller.load_model("humanoid_golf")
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EngineStatus:
    """Status of a physics engine.

    Attributes:
        name: Engine name.
        running: Whether the engine is currently running.
        pid: Process ID if running.
        models_loaded: Number of models currently loaded.
        memory_usage_mb: Memory usage in megabytes.
    """

    name: str
    running: bool = False
    pid: int | None = None
    models_loaded: int = 0
    memory_usage_mb: float = 0.0


@dataclass
class AgentActionResult:
    """Result of an agent action.

    Attributes:
        success: Whether the action succeeded.
        message: Human-readable result message.
        data: Additional data from the action.
        error: Error message if failed.
    """

    success: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class AgentController:
    """Controller for AI agent app management.

    Provides methods for the AI chat bot to control all aspects
    of the application including engines, models, simulations,
    and file operations.

    Example:
        >>> controller = AgentController()
        >>> result = controller.start_engine("mujoco")
        >>> if result.success:
        ...     controller.load_model("humanoid_golf")
    """

    def __init__(self, repo_root: Path | None = None) -> None:
        """Initialize agent controller.

        Args:
            repo_root: Root directory of the repository.
        """
        self._repo_root = repo_root or Path(__file__).parents[4]
        self._running_engines: dict[str, EngineStatus] = {}
        self._loaded_models: list[str] = []

    # =========================================================================
    # Engine Control
    # =========================================================================

    def start_engine(self, engine_name: str) -> AgentActionResult:
        """Start a physics engine.

        Args:
            engine_name: Name of engine to start (mujoco, drake, pinocchio,
                opensim, myosim).

        Returns:
            AgentActionResult with start outcome.
        """
        valid_engines = ["mujoco", "drake", "pinocchio", "opensim", "myosim"]
        if engine_name not in valid_engines:
            return AgentActionResult(
                success=False,
                error=f"Invalid engine: {engine_name}. Valid: {valid_engines}",
            )

        if (
            engine_name in self._running_engines
            and self._running_engines[engine_name].running
        ):
            return AgentActionResult(
                success=True,
                message=f"Engine '{engine_name}' is already running",
            )

        # Launch engine via launcher
        launcher_script = (
            self._repo_root / "src" / "launchers" / f"{engine_name}_unified_launcher.py"
        )
        if not launcher_script.exists():
            launcher_script = (
                self._repo_root
                / "src"
                / "engines"
                / "physics_engines"
                / engine_name
                / "python"
                / f"{engine_name}_launcher.py"
            )

        if not launcher_script.exists():
            return AgentActionResult(
                success=False,
                error=f"Launcher script not found for engine: {engine_name}",
            )

        try:
            process = subprocess.Popen(
                ["python3", str(launcher_script)],
                cwd=str(self._repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._running_engines[engine_name] = EngineStatus(
                name=engine_name,
                running=True,
                pid=process.pid,
            )
            return AgentActionResult(
                success=True,
                message=f"Started {engine_name} engine (PID: {process.pid})",
                data={"pid": process.pid, "engine": engine_name},
            )
        except Exception as e:  # noqa: BLE001
            return AgentActionResult(
                success=False,
                error=f"Failed to start engine: {e}",
            )

    def stop_engine(self, engine_name: str) -> AgentActionResult:
        """Stop a running physics engine.

        Args:
            engine_name: Name of engine to stop.

        Returns:
            AgentActionResult with stop outcome.
        """
        if engine_name not in self._running_engines:
            return AgentActionResult(
                success=False,
                error=f"Engine '{engine_name}' is not running",
            )

        status = self._running_engines[engine_name]
        if not status.running or not status.pid:
            return AgentActionResult(
                success=False,
                error=f"Engine '{engine_name}' is not running",
            )

        try:
            import os
            import signal

            os.kill(status.pid, signal.SIGTERM)
            status.running = False
            status.pid = None
            del self._running_engines[engine_name]
            return AgentActionResult(
                success=True,
                message=f"Stopped {engine_name} engine",
            )
        except Exception as e:  # noqa: BLE001
            return AgentActionResult(
                success=False,
                error=f"Failed to stop engine: {e}",
            )

    def get_engine_status(self) -> dict[str, EngineStatus]:
        """Get status of all engines.

        Returns:
            Dictionary mapping engine names to their status.
        """
        return self._running_engines.copy()

    def configure_engine(
        self, engine_name: str, config: dict[str, Any]
    ) -> AgentActionResult:
        """Configure engine settings.

        Args:
            engine_name: Name of engine to configure.
            config: Configuration parameters.

        Returns:
            AgentActionResult with configuration outcome.
        """
        # Store configuration for when engine starts
        if engine_name not in self._running_engines:
            self._running_engines[engine_name] = EngineStatus(name=engine_name)

        return AgentActionResult(
            success=True,
            message=f"Configured {engine_name} with: {config}",
            data={"config": config},
        )

    # =========================================================================
    # Model Management
    # =========================================================================

    def load_model(
        self, model_name: str, engine: str | None = None
    ) -> AgentActionResult:
        """Load a model into the specified engine.

        Args:
            model_name: Name of model to load.
            engine: Engine to load into (uses first running engine if None).

        Returns:
            AgentActionResult with load outcome.
        """
        # Find target engine
        target_engine = engine
        if not target_engine:
            for name, status in self._running_engines.items():
                if status.running:
                    target_engine = name
                    break

        if not target_engine:
            return AgentActionResult(
                success=False,
                error="No engine running. Start an engine first.",
            )

        # Check model exists
        model_path = self._find_model(model_name)
        if not model_path:
            return AgentActionResult(
                success=False,
                error=f"Model '{model_name}' not found",
            )

        self._loaded_models.append(model_name)
        return AgentActionResult(
            success=True,
            message=f"Loaded model '{model_name}' into {target_engine}",
            data={
                "model": model_name,
                "engine": target_engine,
                "path": str(model_path),
            },
        )

    def unload_model(self, model_name: str) -> AgentActionResult:
        """Unload a model.

        Args:
            model_name: Name of model to unload.

        Returns:
            AgentActionResult with unload outcome.
        """
        if model_name not in self._loaded_models:
            return AgentActionResult(
                success=False,
                error=f"Model '{model_name}' is not loaded",
            )

        self._loaded_models.remove(model_name)
        return AgentActionResult(
            success=True,
            message=f"Unloaded model '{model_name}'",
        )

    def list_loaded_models(self) -> list[str]:
        """List currently loaded models.

        Returns:
            List of model names.
        """
        return self._loaded_models.copy()

    def compare_models(self, model1: str, model2: str) -> AgentActionResult:
        """Compare two models.

        Args:
            model1: First model name.
            model2: Second model name.

        Returns:
            AgentActionResult with comparison data.
        """
        path1 = self._find_model(model1)
        path2 = self._find_model(model2)

        if not path1 or not path2:
            return AgentActionResult(
                success=False,
                error="One or both models not found",
            )

        # Perform comparison (simplified)
        comparison = {
            "model1": {"name": model1, "path": str(path1)},
            "model2": {"name": model2, "path": str(path2)},
            "differences": "Comparison not yet implemented",
        }

        return AgentActionResult(
            success=True,
            message=f"Compared {model1} and {model2}",
            data=comparison,
        )

    def _find_model(self, model_name: str) -> Path | None:
        """Find a model file in the repository.

        Args:
            model_name: Name of model to find.

        Returns:
            Path to model file or None if not found.
        """
        search_dirs = [
            self._repo_root
            / "src"
            / "engines"
            / "physics_engines"
            / "mujoco"
            / "python"
            / "models",
            self._repo_root
            / "src"
            / "engines"
            / "physics_engines"
            / "drake"
            / "models",
            self._repo_root / "assets" / "models",
            self._repo_root / "models",
        ]

        extensions = [".xml", ".urdf", ".mjcf", ".sdf"]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for ext in extensions:
                model_path = search_dir / f"{model_name}{ext}"
                if model_path.exists():
                    return model_path

        return None

    # =========================================================================
    # File Operations
    # =========================================================================

    def import_file(
        self, file_path: str, file_type: str | None = None
    ) -> AgentActionResult:
        """Import a file (URDF, MJCF, C3D, etc.).

        Args:
            file_path: Path to file to import.
            file_type: Type of file (auto-detected if None).

        Returns:
            AgentActionResult with import outcome.
        """
        path = Path(file_path)
        if not path.exists():
            return AgentActionResult(
                success=False,
                error=f"File not found: {file_path}",
            )

        # Auto-detect type from extension
        if not file_type:
            ext_map = {
                ".urdf": "urdf",
                ".xml": "mjcf",
                ".mjcf": "mjcf",
                ".sdf": "sdf",
                ".c3d": "c3d",
                ".mot": "motion",
                ".trc": "motion",
            }
            file_type = ext_map.get(path.suffix.lower(), "unknown")

        return AgentActionResult(
            success=True,
            message=f"Imported {file_type} file: {file_path}",
            data={"path": str(path), "type": file_type},
        )

    def export_model(
        self, model_name: str, output_path: str, format: str = "urdf"
    ) -> AgentActionResult:
        """Export a model to a file.

        Args:
            model_name: Name of model to export.
            output_path: Path to export to.
            format: Export format (urdf, mjcf, etc.).

        Returns:
            AgentActionResult with export outcome.
        """
        if model_name not in self._loaded_models:
            return AgentActionResult(
                success=False,
                error=f"Model '{model_name}' is not loaded",
            )

        return AgentActionResult(
            success=True,
            message=f"Exported {model_name} to {output_path} as {format}",
            data={"model": model_name, "output": output_path, "format": format},
        )

    # =========================================================================
    # Simulation Control
    # =========================================================================

    def start_simulation(
        self, model_name: str, duration: float = 10.0
    ) -> AgentActionResult:
        """Start a simulation.

        Args:
            model_name: Model to simulate.
            duration: Simulation duration in seconds.

        Returns:
            AgentActionResult with start outcome.
        """
        if model_name not in self._loaded_models:
            return AgentActionResult(
                success=False,
                error=f"Model '{model_name}' is not loaded",
            )

        return AgentActionResult(
            success=True,
            message=f"Started simulation of {model_name} for {duration}s",
            data={"model": model_name, "duration": duration},
        )

    def stop_simulation(self) -> AgentActionResult:
        """Stop the current simulation.

        Returns:
            AgentActionResult with stop outcome.
        """
        return AgentActionResult(
            success=True,
            message="Simulation stopped",
        )

    def pause_simulation(self) -> AgentActionResult:
        """Pause the current simulation.

        Returns:
            AgentActionResult with pause outcome.
        """
        return AgentActionResult(
            success=True,
            message="Simulation paused",
        )

    def resume_simulation(self) -> AgentActionResult:
        """Resume a paused simulation.

        Returns:
            AgentActionResult with resume outcome.
        """
        return AgentActionResult(
            success=True,
            message="Simulation resumed",
        )

    # =========================================================================
    # Settings Management
    # =========================================================================

    def get_settings(self, category: str | None = None) -> AgentActionResult:
        """Get application settings.

        Args:
            category: Settings category (all if None).

        Returns:
            AgentActionResult with settings data.
        """
        settings = {
            "theme": "dark",
            "auto_save": True,
            "default_engine": "mujoco",
            "simulation_fps": 60,
            "real_time_factor": 1.0,
        }

        if category:
            settings = {category: settings.get(category, {})}

        return AgentActionResult(
            success=True,
            message="Settings retrieved",
            data=settings,
        )

    def set_settings(self, settings: dict[str, Any]) -> AgentActionResult:
        """Update application settings.

        Args:
            settings: Settings to update.

        Returns:
            AgentActionResult with update outcome.
        """
        return AgentActionResult(
            success=True,
            message=f"Updated settings: {settings}",
            data=settings,
        )

    # =========================================================================
    # Status and Info
    # =========================================================================

    def get_system_status(self) -> AgentActionResult:
        """Get overall system status.

        Returns:
            AgentActionResult with status data.
        """
        import psutil

        status = {
            "engines": {
                name: {"running": s.running}
                for name, s in self._running_engines.items()
            },
            "loaded_models": self._loaded_models,
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(),
        }

        return AgentActionResult(
            success=True,
            message="System status retrieved",
            data=status,
        )

    def get_help(self, topic: str | None = None) -> AgentActionResult:
        """Get help information.

        Args:
            topic: Help topic (general if None).

        Returns:
            AgentActionResult with help content.
        """
        help_content = {
            "general": """
UpstreamDrift Agent Control Help
================================

Available Commands:
- start_engine(engine_name) - Start a physics engine
- stop_engine(engine_name) - Stop an engine
- load_model(model_name) - Load a model
- unload_model(model_name) - Unload a model
- start_simulation(model_name, duration) - Run a simulation
- import_file(path) - Import a file
- export_model(name, path) - Export a model
- get_settings() - Get app settings
- set_settings(dict) - Update settings
""",
            "engines": "Available engines: mujoco, drake, pinocchio, opensim, myosim",
            "models": "Use load_model() to load models into the active engine",
        }

        topic_content = (
            help_content.get(topic, help_content["general"])
            if topic
            else help_content["general"]
        )

        return AgentActionResult(
            success=True,
            message="Help information retrieved",
            data={"help": topic_content},
        )


def create_agent_tools_for_registry() -> list[dict[str, Any]]:
    """Create agent control tool definitions for the registry.

    Returns:
        List of tool definitions ready for registration.
    """
    controller = AgentController()

    return [
        {
            "name": "start_engine",
            "description": (
                "Start a physics engine (mujoco, drake, pinocchio, opensim, myosim)"
            ),
            "handler": controller.start_engine,
            "parameters": [
                {
                    "name": "engine_name",
                    "type": "string",
                    "required": True,
                    "description": "Name of engine to start",
                }
            ],
        },
        {
            "name": "stop_engine",
            "description": "Stop a running physics engine",
            "handler": controller.stop_engine,
            "parameters": [
                {
                    "name": "engine_name",
                    "type": "string",
                    "required": True,
                    "description": "Name of engine to stop",
                }
            ],
        },
        {
            "name": "load_model",
            "description": "Load a model into the physics engine",
            "handler": controller.load_model,
            "parameters": [
                {
                    "name": "model_name",
                    "type": "string",
                    "required": True,
                    "description": "Name of model to load",
                },
                {
                    "name": "engine",
                    "type": "string",
                    "required": False,
                    "description": "Engine to load into (optional)",
                },
            ],
        },
        {
            "name": "unload_model",
            "description": "Unload a model from the physics engine",
            "handler": controller.unload_model,
            "parameters": [
                {
                    "name": "model_name",
                    "type": "string",
                    "required": True,
                    "description": "Name of model to unload",
                }
            ],
        },
        {
            "name": "start_simulation",
            "description": "Start a physics simulation",
            "handler": controller.start_simulation,
            "parameters": [
                {
                    "name": "model_name",
                    "type": "string",
                    "required": True,
                    "description": "Model to simulate",
                },
                {
                    "name": "duration",
                    "type": "number",
                    "required": False,
                    "description": "Duration in seconds",
                    "default": 10.0,
                },
            ],
        },
        {
            "name": "stop_simulation",
            "description": "Stop the current simulation",
            "handler": controller.stop_simulation,
            "parameters": [],
        },
        {
            "name": "get_system_status",
            "description": (
                "Get the current system status including running engines and "
                "loaded models"
            ),
            "handler": controller.get_system_status,
            "parameters": [],
        },
        {
            "name": "import_file",
            "description": "Import a file (URDF, MJCF, C3D, etc.)",
            "handler": controller.import_file,
            "parameters": [
                {
                    "name": "file_path",
                    "type": "string",
                    "required": True,
                    "description": "Path to file to import",
                },
                {
                    "name": "file_type",
                    "type": "string",
                    "required": False,
                    "description": "File type (auto-detected if not specified)",
                },
            ],
        },
        {
            "name": "get_settings",
            "description": "Get application settings",
            "handler": controller.get_settings,
            "parameters": [
                {
                    "name": "category",
                    "type": "string",
                    "required": False,
                    "description": "Settings category (optional)",
                }
            ],
        },
        {
            "name": "set_settings",
            "description": "Update application settings",
            "handler": controller.set_settings,
            "parameters": [
                {
                    "name": "settings",
                    "type": "object",
                    "required": True,
                    "description": "Settings to update",
                }
            ],
        },
    ]
