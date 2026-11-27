"""Flight visualization must respect both verdicts at the outcome boundary."""

from __future__ import annotations

import pytest

from bunkershot3d.geometry import DeliveredGeometry, GeometricBounce
from bunkershot3d.solvers import (
    EnvelopeStatus,
    FidelityTier,
    ValidityVerdict,
    dimensionless_groups,
)
from src.tools.bunker_shot_gui.flight import BunkerFlight
from src.tools.bunker_shot_gui.outcomes import ShotOutcome

pytestmark = pytest.mark.unit


def _outcome(
    shot_status: EnvelopeStatus,
    carry_status: EnvelopeStatus,
    *,
    refused: bool = False,
    with_flight: bool = True,
) -> ShotOutcome:
    """Use a coherent trace so the verdict is the only rejection reason."""
    groups = (
        dimensionless_groups(
            speed_m_s=25.0,
            feature_length_m=0.1,
            grain_diameter_m=0.5e-3,
            element_size_m=2.0e-3,
        ),
    )
    flight = BunkerFlight(
        flight_m=((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 0.0)),
        rollout_m=((2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
        carry_m=2.0,
    )
    return ShotOutcome(
        verdict=ValidityVerdict(shot_status, groups=groups),
        fidelity_tier=FidelityTier.F0,
        refused=refused,
        delivered=DeliveredGeometry(58.0, GeometricBounce(12.0), 0.0, 12.0),
        carry_m=2.0 if with_flight else None,
        carry_verdict=(
            ValidityVerdict(carry_status, groups=groups) if with_flight else None
        ),
        flight=flight if with_flight else None,
    )


@pytest.mark.parametrize("shot_status", list(EnvelopeStatus))
@pytest.mark.parametrize("carry_status", list(EnvelopeStatus))
def test_flight_requires_nonrefused_shot_and_carry_verdicts(
    shot_status: EnvelopeStatus, carry_status: EnvelopeStatus
) -> None:
    if EnvelopeStatus.REFUSED in (shot_status, carry_status):
        with pytest.raises(ValueError, match="flight.*refused"):
            _outcome(shot_status, carry_status)
    else:
        outcome = _outcome(shot_status, carry_status)
        assert outcome.flight is not None
        assert outcome.carry_m == outcome.flight.carry_m
        assert outcome.status is shot_status
        assert outcome.carry_verdict is not None
        assert outcome.carry_verdict.status is carry_status


def test_explicitly_refused_shot_still_rejects_flight() -> None:
    with pytest.raises(ValueError):
        _outcome(EnvelopeStatus.EXTRAPOLATED, EnvelopeStatus.EXTRAPOLATED, refused=True)


def test_refused_shot_can_still_carry_its_verdict_without_a_trace() -> None:
    outcome = _outcome(
        EnvelopeStatus.REFUSED,
        EnvelopeStatus.REFUSED,
        refused=True,
        with_flight=False,
    )
    assert outcome.status is EnvelopeStatus.REFUSED
    assert outcome.flight is None
    assert outcome.carry_m is None
