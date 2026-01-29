from fastapi import Request

from src.shared.python.engine_manager import EngineManager


def get_engine_manager(request: Request) -> EngineManager:
    """Retrieve the EngineManager from app state."""
    return request.app.state.engine_manager
