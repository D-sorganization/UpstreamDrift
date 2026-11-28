"""Published dimensions and portable appearances for three classic golfers."""

import pytest

from src.shared.python.model_appearance.player_models import (
    player_appearance,
    player_profile,
)
from src.shared.python.model_appearance.schema import document_to_dict

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("player", "height_m", "mass_kg"),
    [
        ("tom_watson", 1.7526, 72.5747792),
        ("greg_norman_1986", 1.8542, 83.91458845),
        ("george_knudson", 1.778, 72.5747792),
    ],
)
def test_classic_profile_dimensions_and_sources(player, height_m, mass_kg):
    """Keep cited contemporaneous dimensions and a schema-valid appearance."""
    profile = player_profile(player)
    assert (profile["stature_m"], profile["mass_kg"]) == (height_m, mass_kg)
    assert any("pgatourmedia.pgatourhq.com" in x["url"] for x in profile["sources"])
    assert any("youtube.com" in x["url"] for x in profile["sources"])
    assert profile["limitations"]
    assert (
        document_to_dict(player_appearance(player))["schema_version"] == "appearance-v1"
    )
