"""Generated OpenSim club hand STLs track the canonical geometry copies (#15141)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_GEOMETRY = _REPO / "src/engines/physics_engines/opensim/models/geometry/club"
_GENERATED = _REPO / "src/engines/physics_engines/opensim/models/generated/club"


@pytest.mark.parametrize(
    "name",
    [
        "club_driver_lead_hand.stl",
        "club_driver_trail_hand.stl",
        "club_iron7_lead_hand.stl",
        "club_iron7_trail_hand.stl",
    ],
)
def test_generated_hand_stl_matches_geometry(name: str) -> None:
    """Verify generated hand stl matches geometry."""
    canonical = _GEOMETRY / name
    generated = _GENERATED / name
    assert canonical.is_file(), name
    assert generated.is_file(), name
    assert generated.read_bytes() == canonical.read_bytes(), name
