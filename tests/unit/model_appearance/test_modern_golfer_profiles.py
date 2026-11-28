"""Sourced current-era golfer appearance catalog contracts."""

import pytest

from src.shared.python.model_appearance.player_models import (
    build_player_spec,
    player_appearance,
    player_profile,
)
from src.shared.python.model_appearance.schema import (
    document_to_dict,
    physics_spec_sha256,
)

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


@pytest.mark.parametrize(
    ("player", "body_build", "polo_color"),
    [
        ("rory_mcilroy", 0.12, [0.10, 0.14, 0.20, 1.0]),
        ("jon_rahm", 0.25, [0.54, 0.66, 0.72, 1.0]),
        ("dustin_johnson", -0.18, [0.12, 0.16, 0.23, 1.0]),
        ("sergio_garcia", -0.08, [0.36, 0.08, 0.12, 1.0]),
    ],
)
def test_modern_visual_presets_keep_body_build_and_palette_appearance_only(
    player, body_build, polo_color
):
    """Keep sourced palette/build changes outside every physical spec."""
    profile = player_profile(player)
    appearance = document_to_dict(player_appearance(player))
    spec = build_player_spec(player)

    assert appearance["body_build"] == body_build
    assert appearance["materials"]["polo_white"]["base_color"] == polo_color
    assert any("selected" in source["values"].lower() for source in profile["sources"])
    assert "body_build" not in spec
    assert "appearance" not in spec
    assert physics_spec_sha256({**spec, "appearance": appearance}) == (
        physics_spec_sha256(spec)
    )
    assert any(
        "body_build" in limitation and "roster mass" in limitation
        for limitation in profile["limitations"]
    )
