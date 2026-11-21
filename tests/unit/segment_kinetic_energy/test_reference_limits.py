"""Reference swing segment KE envelope used for fixed shading limits (#15396)."""

from __future__ import annotations

import pytest

from src.shared.python.segment_kinetic_energy.reference_swing_limits import (
    assert_reference_constant_matches_swing,
    compute_reference_swing_peak_segment_ke_j,
)
from src.shared.python.segment_kinetic_energy.shading import (
    REFERENCE_SWING_PEAK_SEGMENT_KE_J,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]


def test_reference_peak_matches_recorded_driver_swing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "disable")
    peak = compute_reference_swing_peak_segment_ke_j()
    assert peak == pytest.approx(REFERENCE_SWING_PEAK_SEGMENT_KE_J, rel=0, abs=1e-3)


def test_reference_constant_guard() -> None:
    assert_reference_constant_matches_swing()
