"""Integration test configuration and shared fixtures.

This conftest.py makes fixtures from tests/fixtures/fixtures_lib.py available
to all integration tests via pytest's automatic fixture discovery.
"""

# mypy: ignore-errors
# The dynamic sys.path import cannot be resolved by mypy statically

from __future__ import annotations

import sys
from pathlib import Path
from src.shared.python.path_utils import get_repo_root, get_src_root


# Add fixtures directory to path
FIXTURES_DIR = get_src_root() / "fixtures"
sys.path.insert(0, str(FIXTURES_DIR))

# Re-export all fixtures from the fixtures library
# This makes them available to all tests in this directory
from fixtures_lib import (  # noqa: F401, E402
    TOLERANCE_ACCELERATION_M_S2,
    TOLERANCE_CLOSURE_RAD_S2,
    TOLERANCE_JACOBIAN,
    TOLERANCE_POSITION_M,
    TOLERANCE_TORQUE_NM,
    TOLERANCE_VELOCITY_M_S,
    EngineInstance,
    all_available_pendulum_engines,
    available_engines,
    compute_accelerations,
    double_pendulum_path,
    drake_pendulum,
    get_states,
    mujoco_pendulum,
    pinocchio_pendulum,
    set_identical_state,
    simple_pendulum_path,
)
