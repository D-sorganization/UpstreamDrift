"""Tests for the human range-of-motion table and violation flags."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching import range_of_motion as module

pytestmark = pytest.mark.unit


def test_table_covers_legs_and_hinge_like_upper_coordinates() -> None:
    assert "hip_flexion_r" in module.HUMAN_RANGES_DEG
    assert "knee_angle_l" in module.HUMAN_RANGES_DEG
    assert module.HUMAN_RANGES_DEG["LEInput"] == (-150.0, 5.0)
    assert "LSInputZ" not in module.HUMAN_RANGES_DEG  # Euler component, no range
    for lo, hi in module.HUMAN_RANGES_DEG.values():
        assert lo < 0 < hi or lo <= 0 <= hi


def test_violations_flag_only_excursions_beyond_tolerance() -> None:
    order = ["TranslationInputX", "LEInput", "knee_angle_r", "LSInputZ"]
    q = np.radians(
        np.array(
            [
                [0.0, -20.0, -30.0, 400.0],
                [0.0, 10.0, -130.0, 400.0],  # elbow hyperextended, knee past range
                [0.0, 5.3, 0.0, 400.0],  # within the 0.5 deg tolerance
            ]
        )
    )
    found = module.violations(q, order)
    assert set(found) == {"LEInput", "knee_angle_r"}
    assert found["LEInput"].max_excess_deg == pytest.approx(5.0)
    assert found["LEInput"].frames == 1 and found["LEInput"].fraction == pytest.approx(
        1 / 3
    )
    assert found["knee_angle_r"].max_excess_deg == pytest.approx(10.0)
    assert module.violations(q[:1], order) == {}
    # A whole turn added to a coordinate is not an excursion.
    turned = q.copy()
    turned[:, 1] += 2 * np.pi
    assert module.violations(turned, order)["LEInput"].max_excess_deg == pytest.approx(
        5.0
    )
    with pytest.raises(ValueError):
        module.violations(q[:, :2], order)
    with pytest.raises(ValueError):
        module.violations(q, order, tolerance_deg=-1.0)
    doc = module.as_document(module.HUMAN_RANGES_DEG)
    assert doc["REInput"] == [-150.0, 5.0]
