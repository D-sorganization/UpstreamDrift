"""Closure residual over the canned swing for every engine (OSV-2, #11728)."""

from __future__ import annotations

import pytest

from scripts.grip_closure_report import CLUBS, swing_closure_report
from src.engines.physics_engines.opensim.python.tour_matching.address import (
    FROZEN_ADDRESS_TOLERANCE_PROFILE,
)

pytestmark = pytest.mark.integration

TOLERANCE_M = FROZEN_ADDRESS_TOLERANCE_PROFILE.max_grip_closure_m


@pytest.mark.parametrize("club", CLUBS)
def test_closure_stays_within_the_frozen_tolerance_where_available(club: str) -> None:
    report = swing_closure_report(club)
    assert set(report) == {"mujoco", "drake", "pinocchio", "opensim", "myosuite"}
    available = {k: v for k, v in report.items() if v["available"]}
    for engine, doc in available.items():
        assert doc["max_m"] is not None, engine
        assert doc["max_m"] <= TOLERANCE_M, (engine, doc)
    for engine, doc in report.items():
        if not doc["available"]:
            # Unavailable is reported as None with a reason, never as zero.
            assert doc["max_m"] is None and doc["reason"], engine
    # At least the engines whose bindings ship in the base environment report.
    assert report["opensim"]["available"] or report["opensim"]["reason"]
