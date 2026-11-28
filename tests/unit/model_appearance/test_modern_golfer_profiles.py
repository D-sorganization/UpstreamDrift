"""Sourced current-era golfer appearance catalog contracts."""

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
        ("rory_mcilroy", 1.778, 72.5747792),
        ("jon_rahm", 1.8796, 99.7903214),
        ("dustin_johnson", 1.9304, 86.1825503),
        ("sergio_garcia", 1.778, 81.6466266),
    ],
)
def test_modern_profiles_are_sourced_and_valid(player, height_m, mass_kg):
    """Preserve exact listed-unit conversions and an explicit visual document."""
    profile = player_profile(player)
    assert (profile["stature_m"], profile["mass_kg"]) == (height_m, mass_kg)
    assert any("pgatour.com" in x["url"] for x in profile["sources"])
    assert any("youtube.com" in x["url"] for x in profile["sources"])
    assert profile["limitations"]
    assert (
        document_to_dict(player_appearance(player))["schema_version"] == "appearance-v1"
    )
