"""Launcher service for the API layer.

Provides process management and model handler functionality for the API
without creating a direct module-level dependency on ``src.launchers``.

This service layer uses lazy imports to access launcher code only when
endpoints are actually called, breaking the ``api -> launchers`` circular
dependency (launchers also imports from shared, and the API layer should
be agnostic of the GUI launcher).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class LauncherService:
    """Facade for launcher functionality used by the API layer.

    Lazily initializes ProcessManager and ModelHandlerRegistry from
    ``src.launchers`` on first use.
    """

    def __init__(self, repo_root: Path) -> None:
        if repo_root is None:
            raise ValueError("repo_root must not be None")
        if not repo_root.is_dir():
            raise FileNotFoundError(
                f"repo_root does not exist or is not a directory: {repo_root}"
            )
        self._repo_root = repo_root
        self._process_manager: Any = None
        self._handler_registry: Any = None

    @property
    def process_manager(self) -> Any:
        """Lazily initialize ProcessManager."""
        if self._process_manager is None:
            from src.launchers.launcher_process_manager import ProcessManager

            self._process_manager = ProcessManager(repo_root=self._repo_root)
        return self._process_manager

    @property
    def handler_registry(self) -> Any:
        """Lazily initialize ModelHandlerRegistry."""
        if self._handler_registry is None:
            from src.launchers.launcher_model_handlers import ModelHandlerRegistry

            self._handler_registry = ModelHandlerRegistry()
        return self._handler_registry

    @precondition(
        lambda self, model_type: model_type is not None and len(model_type) > 0,
        "Model type must be a non-empty string",
    )
    def get_handler(self, model_type: str) -> Any:
        """Get a handler for the given model type.

        Args:
            model_type: The type of model to launch.

        Returns:
            A handler instance, or None if no handler found.
        """
        return self.handler_registry.get_handler(model_type)

    def get_running_processes(self) -> dict[str, dict[str, Any]]:
        """Get information about running processes.

        Returns:
            Dictionary mapping process name to status info.
        """
        processes = {}
        for name, proc in self.process_manager.running_processes.items():
            poll = proc.poll()
            processes[name] = {
                "pid": proc.pid,
                "running": poll is None,
                "exit_code": poll,
            }
        return processes

    @precondition(
        lambda self, name: name is not None and len(name) > 0,
        "Process name must be a non-empty string",
    )
    def stop_process(self, name: str) -> bool:
        """Stop a running process by name.

        Args:
            name: Process name.

        Returns:
            True if process was found and stopped, False if not found.
        """
        from src.shared.python.security.subprocess_utils import kill_process_tree

        proc = self.process_manager.running_processes.get(name)
        if proc is None:
            return False

        logger.info("[stop] Killing process tree for %s (pid=%s)", name, proc.pid)
        kill_process_tree(proc.pid)
        del self.process_manager.running_processes[name]
        logger.info("[stop] Process %s stopped and removed", name)
        return True
