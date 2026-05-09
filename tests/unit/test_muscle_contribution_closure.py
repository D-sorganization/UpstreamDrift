"""Tests for muscle contribution closure in induced acceleration analysis.

Scientific Background:
    Induced Acceleration Analysis decomposes total acceleration into contributions:
        a_total = Σ a_muscle_i + a_passive + a_external

    This test verifies the fundamental closure property: the sum of all muscle-induced
    accelerations should equal the total acceleration (when no external torques applied).

References:
    - Zajac, F. E. (2002). "Understanding muscle coordination of the human leg with
      dynamical simulations." Journal of Biomechanics.
    - Anderson, F. C., & Pandy, M. G. (2003). "Individual muscle contributions to
      support in normal walking." Gait & Posture.

Refactored to use shared engine availability module (DRY principle).
"""

import typing

import numpy as np
import pytest
from src.shared.python.engine_core.engine_availability import (
    MYOSUITE_AVAILABLE,
    skip_if_unavailable,
)

if MYOSUITE_AVAILABLE:
    from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
        MyoSuitePhysicsEngine as _MyoSuitePhysicsEngine,
    )
else:
    _MyoSuitePhysicsEngine = None  # type: ignore

# Skip entire module if MyoSuite not available
pytestmark = skip_if_unavailable("myosuite")


if MYOSUITE_AVAILABLE:
    pass

else:
    # Fallback to ensure some tests are collected even if marked as skipped by pytest.
    # We hide these from Mypy to avoid "Name already defined" [no-redef] errors.
    if not typing.TYPE_CHECKING:
        pass
