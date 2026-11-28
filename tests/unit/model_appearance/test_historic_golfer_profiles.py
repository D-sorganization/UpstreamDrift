"""Historical source and uncertainty contracts for four player candidates."""

import pytest

from src.shared.python.model_appearance.player_models import (
    player_appearance,
    player_profile,
)
from src.shared.python.model_appearance.schema import document_to_dict

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("player", "height", "mass"),
    [
        ("moe_norman", 1.7272, 80.0),
        ("john_schlee", 1.905, 77.1107029),
        ("johnny_miller", 1.8796, 77.1107029),
        ("lee_trevino", 1.7018, 83.91458845),
    ],
)
def test_historic_catalog_entry(player, height, mass):
    """Record the chosen candidate priors and a loadable appearance."""
    profile = player_profile(player)
    assert (profile["stature_m"], profile["mass_kg"]) == (height, mass)
    assert len(profile["sources"]) >= 2 and profile["limitations"]
    assert (
        document_to_dict(player_appearance(player))["schema_version"] == "appearance-v1"
    )


def test_moe_mass_is_explicitly_provisional():
    """Never present the assumed Moe Norman mass as a sourced measurement."""
    profile = player_profile("moe_norman")
    assert any(
        "80 kg" in note and "estimate" in note for note in profile["limitations"]
    )
