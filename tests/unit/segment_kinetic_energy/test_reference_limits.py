"""Reference swing segment KE envelope used for fixed shading limits (#15396)."""

from __future__ import annotations

import pytest

from src.shared.python.segment_kinetic_energy.reference_swing_limits import (
    IMPACT_EXCLUDE_RADIUS_FRAMES,
    WHOLE_BODY_KE_SANITY_J,
    assert_reference_constant_matches_swing,
    compute_reference_swing_peak_segment_ke_j,
)
from src.shared.python.segment_kinetic_energy.shading import (
    REFERENCE_SWING_PEAK_SEGMENT_KE_J,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]


def test_reference_nominal_default_about_500_j() -> None:
    assert pytest.approx(486.293, abs=1e-2) == REFERENCE_SWING_PEAK_SEGMENT_KE_J


def test_reference_peak_impact_gated_pre_post_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ball impact at frames 661–664; filters must not bridge the impact window."""
    monkeypatch.setenv("MUJOCO_GL", "disable")
    peak = compute_reference_swing_peak_segment_ke_j()
    assert IMPACT_EXCLUDE_RADIUS_FRAMES == 25
    assert 300.0 < peak < REFERENCE_SWING_PEAK_SEGMENT_KE_J
    assert peak == pytest.approx(467.097, rel=0, abs=0.02)


def test_reference_constant_guard() -> None:
    assert_reference_constant_matches_swing()


def test_preprocessed_swing_whole_body_ke_stays_below_sanity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "disable")
    peak = compute_reference_swing_peak_segment_ke_j()
    assert peak < WHOLE_BODY_KE_SANITY_J