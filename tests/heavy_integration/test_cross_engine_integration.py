from typing import Any

import numpy as np
import pytest
from src.shared.python.engine_core.cross_engine_validator import CrossEngineValidator
from src.shared.python.logging_pkg.logging_config import get_logger

from tests.fixtures.fixtures_lib import (
    TOLERANCE_ACCELERATION_M_S2,
    TOLERANCE_CLOSURE_RAD_S2,
    TOLERANCE_JACOBIAN,
    compute_accelerations,
    set_identical_state,
    skip_if_insufficient_engines,
)

logger = get_logger(__name__)


pytestmark = pytest.mark.live_simulation
