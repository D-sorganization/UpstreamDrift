"""Engine management routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from src.shared.python.engine_manager import EngineManager
from src.shared.python.engine_registry import EngineType

from ..models.responses import EngineStatusResponse
from ..utils.path_validation import validate_model_path

router = APIRouter()

_engine_manager: EngineManager | None = None
_logger: Any = None


def configure(engine_manager: EngineManager | None, logger: Any) -> None:
    """Configure dependencies for engine routes."""
    global _engine_manager, _logger
    _engine_manager = engine_manager
    _logger = logger


@router.get("/engines", response_model=list[EngineStatusResponse])
async def get_engines() -> list[EngineStatusResponse]:
    """Get status of all available physics engines."""
    if not _engine_manager:
        raise HTTPException(status_code=500, detail="Engine manager not initialized")

    engines = []
    available_engines = _engine_manager.get_available_engines()
    for engine_type in EngineType:
        status = _engine_manager.get_engine_status(engine_type)
        is_available = engine_type in available_engines

        engines.append(
            EngineStatusResponse(
                engine_type=engine_type.value,
                status=status.value,
                is_available=is_available,
                description=f"{engine_type.value} physics engine",
            )
        )

    return engines


@router.post("/engines/{engine_type}/load")
async def load_engine(
    engine_type: str, model_path: str | None = None
) -> dict[str, str]:
    """Load a specific physics engine with optional model."""
    if not _engine_manager:
        raise HTTPException(status_code=500, detail="Engine manager not initialized")

    try:
        engine_enum = EngineType(engine_type.upper())
        _engine_manager._load_engine(engine_enum)

        engine = _engine_manager.get_active_physics_engine()
        if not engine:
            raise HTTPException(
                status_code=400, detail=f"Failed to load engine: {engine_type}"
            )

        if model_path:
            engine = _engine_manager.get_active_physics_engine()
            if engine:
                validated_path = validate_model_path(model_path)
                engine.load_from_path(validated_path)

        return {"message": f"Engine {engine_type} loaded successfully"}

    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"Unknown engine type: {engine_type}"
        ) from exc
    except Exception as exc:
        if _logger:
            _logger.error("Error loading engine: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Error loading engine: {str(exc)}"
        ) from exc
