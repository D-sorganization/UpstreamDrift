"""Shared workflow adapter for common engine orchestration operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.shared.python.engine_core.engine_manager import EngineManager
from src.shared.python.engine_core.engine_registry import EngineType
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

_ENGINE_NAME_MAP: dict[str, EngineType] = {
    "mujoco": EngineType.MUJOCO,
    "drake": EngineType.DRAKE,
    "pinocchio": EngineType.PINOCCHIO,
    "opensim": EngineType.OPENSIM,
    "myosuite": EngineType.MYOSIM,
    "putting_green": EngineType.PUTTING_GREEN,
}


@dataclass(frozen=True)
class EngineWorkflowResult:
    """Result wrapper for shared workflow adapter operations."""

    ok: bool
    payload: dict[str, Any]
    status_code: int = 200


class EngineWorkflowAdapter:
    """Adapter for common probe/load/unload engine workflows."""

    def __init__(self, engine_manager: EngineManager) -> None:
        self._engine_manager = engine_manager

    @staticmethod
    def resolve_engine(engine_name: str) -> EngineType | None:
        """Resolve an external engine name to an EngineType."""
        return _ENGINE_NAME_MAP.get(engine_name.lower())

    @staticmethod
    def parse_engine_identifier(engine_name: str) -> EngineType | None:
        """Resolve either public names or EngineType identifiers."""
        resolved = EngineWorkflowAdapter.resolve_engine(engine_name)
        if resolved is not None:
            return resolved

        try:
            return EngineType(engine_name.lower())
        except ValueError:
            try:
                return EngineType[engine_name.upper()]
            except KeyError:
                return None

    def probe(self, engine_name: str) -> EngineWorkflowResult:
        """Probe availability for a named engine."""
        engine_type = self.resolve_engine(engine_name)
        if engine_type is None:
            return EngineWorkflowResult(
                ok=False,
                payload={"available": False, "error": f"Unknown engine: {engine_name}"},
            )

        available_engines = self._engine_manager.get_available_engines()
        is_available = engine_type in available_engines
        return EngineWorkflowResult(
            ok=True,
            payload={
                "available": is_available,
                "version": "1.0.0" if is_available else None,
                "capabilities": ["physics"] if is_available else [],
            },
        )

    def load(self, engine_name: str) -> EngineWorkflowResult:
        """Load a named engine through EngineManager."""
        engine_type = self.resolve_engine(engine_name)
        if engine_type is None:
            return EngineWorkflowResult(
                ok=False,
                status_code=400,
                payload={"detail": f"Unknown engine: {engine_name}"},
            )

        success = self._engine_manager.switch_engine(engine_type)
        if not success:
            return EngineWorkflowResult(
                ok=False,
                status_code=400,
                payload={"detail": f"Failed to load {engine_name}"},
            )

        return EngineWorkflowResult(
            ok=True,
            payload={
                "status": "loaded",
                "engine": engine_name,
                "version": "1.0.0",
                "capabilities": ["physics"],
                "message": f"{engine_name} loaded successfully",
            },
        )

    def unload(self, engine_name: str) -> EngineWorkflowResult:
        """Unload currently active engine when it matches the target."""
        engine_type = self.parse_engine_identifier(engine_name)
        if engine_type is None:
            return EngineWorkflowResult(
                ok=False,
                status_code=400,
                payload={"detail": f"Invalid engine type: {engine_name}"},
            )

        current = self._engine_manager.get_current_engine()
        if current == engine_type:
            logger.info("Unloading active engine %s", engine_name)
            self._engine_manager.cleanup()

        return EngineWorkflowResult(
            ok=True,
            payload={"status": "unloaded", "engine": engine_name},
        )
