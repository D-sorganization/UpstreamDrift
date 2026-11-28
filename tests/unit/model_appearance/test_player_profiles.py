"""Source and appearance contracts for the five added golf legends."""

import pytest

from src.shared.python.model_appearance.player_models import (
    player_appearance,
    player_ids,
    player_profile,
)
from src.shared.python.model_appearance.schema import document_to_dict

pytestmark = pytest.mark.unit


PROFILES = {
    "seve_ballesteros": (1.83, 80.0),
    "jack_nicklaus": (1.778, 83.91458845),
    "bobby_jones": (1.7272, 74.84274105),
    "sam_snead": (1.8034, 86.1825503),
    "byron_nelson": (1.8542, 79.37866475),
    "mac_ogrady": (1.8288, 74.84274105),
}


@pytest.mark.parametrize(("player", "dimensions"), PROFILES.items())
def test_profile_ids_dimensions_sources_and_appearance(player, dimensions):
    """Keep each catalog entry sourced, sized and schema-valid."""
    profile = player_profile(player)
    assert (profile["stature_m"], profile["mass_kg"]) == dimensions
    assert profile["display_name"]
    assert any(
        "pgatour" in source["url"]
        or "jgto" in source["url"]
        or "encyclopedia" in source["url"]
        for source in profile["sources"]
    )
    assert any(
        "youtube.com" in source["url"] or "/videos/" in source["url"]
        for source in profile["sources"]
    )
    assert profile["limitations"]
    appearance = player_appearance(player)
    assert document_to_dict(appearance)["schema_version"] == "appearance-v1"
    assert appearance.head.enabled
    assert appearance.body_model == "ellipsoid"


def test_public_catalog_ids_include_legacy_and_new_profiles():
    """Expose every configured candidate to callers without a duplicate list."""
    assert player_ids() == (
        "ben_hogan",
        "bobby_jones",
        "byron_nelson",
        "dustin_johnson",
        "jack_nicklaus",
        "jon_rahm",
        "mac_ogrady",
        "rory_mcilroy",
        "sam_snead",
        "sergio_garcia",
        "seve_ballesteros",
        "tiger_woods",
    )


def test_profiles_have_distinct_face_and_clothing_treatments():
    """Retain intentional color and face-shape variation across legends."""
    profiles = [player_profile(player) for player in PROFILES]
    heads = [profile["appearance"]["head"] for profile in profiles]
    assert len({head["headwear_material"] for head in heads}) >= 3
    assert len({tuple(head["morph"].values()) for head in heads}) == len(heads)
    assert (
        len(
            {
                tuple(
                    profile["appearance"]["materials"]
                    .get("polo_white", {})
                    .get("base_color", ())
                )
                for profile in profiles
            }
        )
        >= 2
    )


def test_sam_snead_profile_uses_the_shared_straw_hat_contract():
    """Use the separately shared straw-hat style for Snead's silhouette."""
    head = player_appearance("sam_snead").head
    assert head.cap_style == "straw_hat"
    assert head.cap_material == "hat_straw"


def test_seve_profile_keeps_the_reviewed_dark_side_part_visible():
    """Preserve the reviewed compilation close-up's uncovered hairstyle."""
    profile = player_profile("seve_ballesteros")
    head = profile["appearance"]["head"]
    shirt = profile["appearance"]["materials"]["polo_white"]["base_color"]
    assert head["cap_style"] == "none"
    assert head["hair_part"] == "left"
    assert head["headwear_material"] == "hair_dark_brown"
    assert shirt[:3] == [0.10, 0.16, 0.29]

def test_mac_ogrady_profile_uses_period_dimensions_and_cautious_visuals():
    """Record the 1986 Tour-guide values and avoid treating the portrait as color evidence."""
    profile = player_profile("mac_ogrady")
    assert (profile["stature_m"], profile["mass_kg"]) == (1.8288, 74.84274105)
    assert any("1986_pgatour.pdf" in source["url"] for source in profile["sources"])
    assert any("1984_pgatour.pdf" in source["url"] for source in profile["sources"])
    assert any("viewer was inaccessible" in note for note in profile["limitations"])
    head = player_appearance("mac_ogrady").head
    assert head.hair_style == "swept_back"
    assert head.cap_style == "none"
    assert head.morph.face_length == 0.10
    assert head.morph.jaw < 0 and head.morph.nose_length > 0
