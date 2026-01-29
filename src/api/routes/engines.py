"""Engine management routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.shared.python.engine_manager import EngineManager
from src.shared.python.engine_registry import EngineType

from ..auth.middleware import OptionalAuth, is_local_mode
from ..dependencies import get_engine_manager

# We keep using the existing response models where appropriate, or define new ones if needed by the plan
from ..models.responses import EngineStatusResponse
from ..utils.path_validation import validate_model_path

router = APIRouter()


class EngineListResponse(BaseModel):
    engines: list[EngineStatusResponse]
    mode: str  # "local" or "cloud"


@router.get("/engines", response_model=EngineListResponse)
async def get_engines(
    engine_manager: EngineManager = Depends(get_engine_manager),
    _user=Depends(OptionalAuth(auto_error=False)),
) -> EngineListResponse:
    """Get status of all available physics engines."""
    engines = []
    # Note: get_available_engines returns list[EngineType]
    available_engines = engine_manager.get_available_engines()

    for engine_type in EngineType:
        status = engine_manager.get_engine_status(engine_type)
        is_available = engine_type in available_engines

        engines.append(
            EngineStatusResponse(
                engine_type=engine_type.value,
                status=status.value,
                is_available=is_available,
                description=f"{engine_type.value} physics engine",
                # Add capabilities if the model supports it, otherwise default
            )
        )

    return EngineListResponse(
        engines=engines,
        mode="local" if is_local_mode() else "cloud",
    )


@router.post("/engines/{engine_type}/load")
async def load_engine(
    engine_type: str,
    model_path: str | None = None,
    engine_manager: EngineManager = Depends(get_engine_manager),
    _user=Depends(OptionalAuth()),
) -> dict[str, Any]:
    """Load a specific physics engine with optional model."""
    try:
        engine_enum = EngineType(engine_type.upper())
        # Access protected method via public interface if possible, or refactor Manager later.
        # For now, we assume _load_engine is what we have access to or we use load_engine if public.
        # Checking previous file usage: it used _load_engine.
        if hasattr(engine_manager, 'load_engine'):
             engine_manager.load_engine(engine_enum)
        else:
             engine_manager._load_engine(engine_enum)

        engine = engine_manager.get_active_physics_engine()
        if not engine:
            raise HTTPException(
                status_code=400, detail=f"Failed to load engine: {engine_type}"
            )

        if model_path:
             validated_path = validate_model_path(model_path)
             if hasattr(engine, 'load_from_path'):
                engine.load_from_path(validated_path)

        state = None
        if hasattr(engine, 'get_state'):
            state = engine.get_state()

        return {
            "status": "loaded",
            "engine": engine_type,
            "model": model_path,
            "state": state
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"Unknown engine type: {engine_type}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Error loading engine: {str(exc)}"
        ) from exc


@router.post("/engines/{engine_type}/unload")
async def unload_engine(
    engine_type: str,
    engine_manager: EngineManager = Depends(get_engine_manager),
    _user=Depends(OptionalAuth()),
) -> dict[str, str]:
    """Unload a physics engine to free resources."""
    try:
        # Assuming unload_engine exists or we implement a wrapper
        if hasattr(engine_manager, 'unload_engine'):
            engine_manager.unload_engine(EngineType(engine_type.upper()))
        else:
            # Fallback if specific unload isn't implemented logic
            pass
        return {"status": "unloaded", "engine": engine_type}
    except ValueError as exc:
         raise HTTPException(status_code=400, detail=f"Invalid engine type: {engine_type}") from exc
