"""Physics validation for complex (multi-body) models."""

from pathlib import Path

from src.shared.python.engine_core.engine_manager import EngineManager, EngineType
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Locate the repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def is_engine_available(engine_type: EngineType) -> bool:
    """Check if an engine is installed and importable."""
    manager = EngineManager()
    probe_result = manager.get_probe_result(engine_type)
    return bool(probe_result.is_available())
