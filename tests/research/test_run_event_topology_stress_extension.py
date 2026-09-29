"""Preregistered Phase B runner contracts for issue #9125."""

from __future__ import annotations

import pytest

from scripts.research.proximal_distal_energy.run_event_topology_stress_extension import (
    PREREGISTRATION_COMMENT,
    registered_stress_scenarios,
)

pytestmark = pytest.mark.unit


def test_phase_b_scale_ladder_and_stop_rule_are_fixed() -> None:
    scenarios = registered_stress_scenarios()

    assert tuple(item.name for item in scenarios) == (
        "fraction_0p02",
        "fraction_0p05",
        "fraction_0p1",
        "fraction_0p2",
        "fraction_0p5",
    )
    assert tuple(item.scale_fraction for item in scenarios) == (
        0.02,
        0.05,
        0.10,
        0.20,
        0.50,
    )
    assert all(item.replicate_count == 192 for item in scenarios)
    assert PREREGISTRATION_COMMENT.endswith("issuecomment-5431171920")
