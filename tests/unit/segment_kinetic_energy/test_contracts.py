"""Contract tests for segment kinetic-energy frames (#15389)."""

from __future__ import annotations

import pytest

from src.shared.python.segment_kinetic_energy import (
    SegmentKineticEnergyFrame,
    SegmentKineticEnergySample,
    SEGMENT_KE_PROXY_DISCLAIMER,
)

pytestmark = pytest.mark.unit


def test_sample_enforces_trans_plus_rot_equals_total() -> None:
    with pytest.raises(ValueError, match="translational_j \\+ rotational_j"):
        SegmentKineticEnergySample(
            body="upper",
            body_index=1,
            com_m=(0.0, 0.0, 1.0),
            translational_j=2.0,
            rotational_j=1.0,
            total_j=10.0,
        )


def test_disclaimer_does_not_claim_energy_transfer() -> None:
    lowered = SEGMENT_KE_PROXY_DISCLAIMER.lower()
    assert "not" in lowered
    assert "inter-segment energy flow" in lowered or "energy flow" in lowered
