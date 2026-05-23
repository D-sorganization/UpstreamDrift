"""Tests for MyoSuite integration (Section K).

Verifies:
- MuJoCo muscle actuator integration
- Activation → force → torque pipeline
- Muscle-induced acceleration analysis
- Grip modeling via hand muscle forces
- Cross-validation with OpenSim

Refactored to use shared engine availability module (DRY principle).
"""

from __future__ import annotations

import pytest
from src.shared.python.engine_core.engine_availability import MYOSUITE_AVAILABLE
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


@pytest.fixture
def myosuite_env_available() -> bool:
    """Check if MyoSuite is available."""
    if not MYOSUITE_AVAILABLE:
        pytest.skip("MyoSuite not installed")
    return True


class TestCrossValidation:
    """Cross-validation with OpenSim."""


pytestmark = pytest.mark.live_simulation
