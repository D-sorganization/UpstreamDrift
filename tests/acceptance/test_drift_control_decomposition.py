"""Tests for drift-control decomposition (Section F).

Verifies that drift + control = full dynamics for all physics engines.
Refactored for DRY compliance using parameterized engine tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# TOLERANCE for superposition test
SUPERPOSITION_TOLERANCE = 1e-5


def _get_engine(engine_name: str) -> Any:
    """Factory to get the requested physics engine, skipping if not available."""
    if engine_name == "pinocchio":
        try:
            import pinocchio as pin

            if not hasattr(pin, "__version__"):
                pytest.skip("Pinocchio mocked")
            from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
                PinocchioPhysicsEngine,
            )

            return PinocchioPhysicsEngine()
        except ImportError:
            pytest.skip("Pinocchio not installed")

    elif engine_name == "mujoco":
        try:
            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
                MuJoCoPhysicsEngine,
            )

            return MuJoCoPhysicsEngine()
        except ImportError:
            pytest.skip("MuJoCo not installed")

    pytest.skip(f"Engine {engine_name} not available")
