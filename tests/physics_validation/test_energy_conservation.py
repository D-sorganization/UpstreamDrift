"""Physics validation tests verifying energy conservation."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from src.shared.python.core.constants import GRAVITY_M_S2
from src.shared.python.engine_core.engine_manager import EngineManager, EngineType
from src.shared.python.logging_pkg.logging_config import get_logger

from tests.physics_validation.analytical import AnalyticalBallistic

logger = get_logger(__name__)


def is_engine_available(engine_type: EngineType) -> bool:
    """Check if an engine is installed and importable."""
    manager = EngineManager()
    probe_result = manager.get_probe_result(engine_type)
    return bool(probe_result.is_available())
