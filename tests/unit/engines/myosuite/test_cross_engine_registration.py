"""MyoSuite registration in shared cross-engine replay (MS-52, #10345)."""

from __future__ import annotations

import pytest

from src.shared.python.motion_matching.cross_engine_replay import (
    KINEMATIC_ONLY_ENGINES,
    VALID_ENGINES,
)

pytestmark = pytest.mark.unit


def test_myosuite_registered_for_kinematic_comparison() -> None:
    assert "myosuite" in VALID_ENGINES
    assert frozenset({"myosuite"}) == KINEMATIC_ONLY_ENGINES
