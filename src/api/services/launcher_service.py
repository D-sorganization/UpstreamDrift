"""Launcher service for the API layer.

Provides process management and model handler functionality for the API
without creating a direct module-level dependency on ``src.launchers``.

This service layer uses lazy imports to access launcher code only when
endpoints are actually called, breaking the ``api -> launchers`` circular
dependency (launchers also imports from shared, and the API layer should
be agnostic of the GUI launcher).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)
STOP_PROCESS_TIMEOUT_SECONDS = 5.0


class StopProcessStatus(str, Enum):
    """Outcome categories for a launcher process-stop request."""

    STOPPED = "stopped"
    NOT_FOUND = "not_found"
    FAILED = "failed"


@dataclass(frozen=True)
class StopProcessResult:
    """Structured result for stopping a launcher process."""

    status: StopProcessStatus
    name: str
    detail: str = ""

    def __bool__(self) -> bool:
        """Preserve truthiness for legacy callers that expect a bool."""
        return self.status is StopProcessStatus.STOPPED


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
        running = self.process_manager.running_processes
        processes = {}
        for name, proc in running.items():
            poll = proc.poll()
            processes[name] = {
                "pid": proc.pid,
                "running": poll is None,
                "exit_code": poll,
            }
        return processes

    def _fallback_stop_process(self, name: str, proc: Any) -> bool:
        """Attempt bounded terminate/kill fallback after tree-kill failure."""
        if proc.poll() is not None:
            return True
        try:
            logger.warning("[stop] Falling back to terminate() for %s", name)
            proc.terminate()
            try:
                proc.wait(timeout=STOP_PROCESS_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                logger.warning("[stop] terminate() timed out for %s; killing", name)
                proc.kill()
                proc.wait(timeout=STOP_PROCESS_TIMEOUT_SECONDS)
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            logger.exception("[stop] Failed to stop process %s", name)
            return False
        return proc.poll() is not None

    @precondition(
        lambda self, name: name is not None and len(name) > 0,
        "Process name must be a non-empty string",
    )
    def stop_process(self, name: str) -> StopProcessResult:
        """Stop a running process by name.

        Args:
            name: Process name.

        Returns:
            Structured stop outcome for API/UI status mapping.
        """
        from src.shared.python.security.subprocess_utils import kill_process_tree

        running = self.process_manager.running_processes
        proc = running.get(name)
        if proc is None:
            return StopProcessResult(StopProcessStatus.NOT_FOUND, name)

        logger.info("[stop] Killing process tree for %s (pid=%s)", name, proc.pid)
        tree_stopped = kill_process_tree(proc.pid)
        if not tree_stopped and not self._fallback_stop_process(name, proc):
            detail = f"Failed to stop process: {name}"
            logger.error("[stop] %s remains running after stop attempts", name)
            return StopProcessResult(StopProcessStatus.FAILED, name, detail)
        del running[name]
        logger.info("[stop] Process %s stopped and removed", name)
        return StopProcessResult(StopProcessStatus.STOPPED, name)
