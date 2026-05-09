"""Integration test for verifying consistency across physics engines.

Uses shared fixtures from tests/fixtures/conftest.py to load
gold-standard models (simple pendulum, double pendulum) into
available physics engines and compare results.

Per Guideline M2/P3: Cross-engine validation with explicit tolerances.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from src.shared.python.engine_core.cross_engine_validator import CrossEngineValidator
from src.shared.python.logging_pkg.logging_config import get_logger

from tests.fixtures.fixtures_lib import (
    TOLERANCE_ACCELERATION_M_S2,
    _check_drake_available,
    _check_mujoco_available,
    _check_pinocchio_available,
    compute_accelerations,
    set_identical_state,
    skip_if_insufficient_engines,
)

logger = get_logger(__name__)

# Tolerance multiplier for triangulation outlier detection
# A relaxed 10x threshold is used to identify engines with systematic deviations
TRIANGULATION_TOLERANCE_MULTIPLIER = 10.0


def _get_available_engine_count() -> int:
    """Count available physics engines."""
    count = 0
    if _check_mujoco_available():
        count += 1
    if _check_drake_available():
        count += 1
    if _check_pinocchio_available():
        count += 1
    return count
