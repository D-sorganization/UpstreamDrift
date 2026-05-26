"""Physics validation tests verifying momentum conservation."""

from src.shared.python.engine_core.engine_manager import EngineManager, EngineType
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


def is_engine_available(engine_type: EngineType) -> bool:
    """Check if an engine is installed and importable."""
    manager = EngineManager()
    probe_result = manager.get_probe_result(engine_type)
    return bool(probe_result.is_available())
