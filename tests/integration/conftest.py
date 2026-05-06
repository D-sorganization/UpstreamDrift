"""Integration test configuration and shared fixtures.

This conftest.py makes fixtures from tests/fixtures/fixtures_lib.py available
to all integration tests via pytest's automatic fixture discovery.
"""

# mypy: ignore-errors

from __future__ import annotations

from src.shared.python.data_io.path_utils import get_tests_root

# Fixtures directory for test shared utilities
FIXTURES_DIR = get_tests_root() / "fixtures"

# Re-export all fixtures from the fixtures library
# This makes them available to all tests in this directory
import numpy as np  # noqa: E402
import pytest  # noqa: E402
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


@pytest.fixture
def synthetic_demonstration() -> object:
    """Tiny synthetic :class:`Demonstration` used by sysid integration tests.

    Sized to be cheap (5 frames, 2 dof) — only used to verify code paths,
    not numerical correctness.
    """
    from src.learning.imitation.dataset import Demonstration

    n_frames = 5
    n_dof = 2
    return Demonstration(
        timestamps=np.linspace(0.0, 0.04, n_frames, dtype=np.float64),
        joint_positions=np.zeros((n_frames, n_dof), dtype=np.float64),
        joint_velocities=np.zeros((n_frames, n_dof), dtype=np.float64),
        actions=np.zeros((n_frames, n_dof), dtype=np.float64),
    )
