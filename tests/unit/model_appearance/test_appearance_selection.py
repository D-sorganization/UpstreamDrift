"""Appearance selection: overrides, precedence and the shared document writer (HRG-B2)."""

from __future__ import annotations

import pytest

from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.appearance_selection import (
    AppearanceOverrides,
    apply_overrides,
    hex_to_rgb,
    resolve_appearance,
    rgb_to_hex,
)
from src.shared.python.model_appearance.defaults import default_document_dict
from src.shared.python.model_appearance.schema import document_from_dict

pytestmark = pytest.mark.unit


def _rgba(doc: dict, name: str) -> tuple[float, ...]:
    return library.library_materials(document_from_dict(doc))[name].rgba


def test_hex_round_trip_and_validation() -> None:
    assert hex_to_rgb("#FF8000") == pytest.approx((1.0, 128 / 255, 0.0))
    assert rgb_to_hex(hex_to_rgb("#12ab9f")) == "#12ab9f"
    for bad in ("red", "#12345", "#gggggg", "123456", ""):
        with pytest.raises(ValueError, match="hex"):
            hex_to_rgb(bad)


def test_custom_hair_colour_becomes_a_document_material() -> None:
    doc = apply_overrides(default_document_dict(), AppearanceOverrides(hair="#336699"))
    assert doc["head"]["headwear"] == "hair"
    name = doc["head"]["headwear_material"]
    assert name in doc["materials"]
    assert _rgba(doc, name) == pytest.approx((0.2, 0.4, 0.6, 1.0))


def test_hair_id_selects_the_palette_swatch_without_a_material() -> None:
    doc = apply_overrides(
        default_document_dict(), AppearanceOverrides(hair="hair_blond")
    )
    assert doc["head"]["headwear_material"] == "hair_blond"
    assert "materials" not in doc or not doc["materials"]


def test_custom_skin_and_skin_id() -> None:
    custom = apply_overrides(
        default_document_dict(), AppearanceOverrides(skin_tone="#c68642")
    )
    assert _rgba(custom, custom["skin_tone"])[:3] == pytest.approx(
        hex_to_rgb("#c68642")
    )
    plain = apply_overrides(
        default_document_dict(), AppearanceOverrides(skin_tone="skin_6")
    )
    assert plain["skin_tone"] == "skin_6"


def test_top_and_bottom_colours_recolour_the_preset_garments() -> None:
    doc = apply_overrides(
        default_document_dict(),
        AppearanceOverrides(
            clothing="golf_polo_shorts", top_color="#aa0000", bottom_color="#00aa00"
        ),
    )
    assert doc["clothing"] == "golf_polo_shorts"
    parts = library.CLOTHING["golf_polo_shorts"]
    assert _rgba(doc, parts["torso"])[:3] == pytest.approx(hex_to_rgb("#aa0000"))
    assert _rgba(doc, parts[library.PANTS_PART])[:3] == pytest.approx(
        hex_to_rgb("#00aa00")
    )
    # the shoes keep their library material
    assert _rgba(doc, parts["foot"]) == library.MATERIALS[parts["foot"]].rgba


def test_garment_colour_without_garment_is_rejected() -> None:
    with pytest.raises(ValueError, match="no top"):
        apply_overrides(
            default_document_dict(),
            AppearanceOverrides(clothing="none", top_color="#ffffff"),
        )


def test_unknown_names_fail_closed() -> None:
    with pytest.raises(ValueError):
        apply_overrides(default_document_dict(), AppearanceOverrides(hair="hair_teal"))
    with pytest.raises(ValueError):
        apply_overrides(
            default_document_dict(), AppearanceOverrides(skin_tone="skin_99")
        )
    with pytest.raises(ValueError):
        apply_overrides(default_document_dict(), AppearanceOverrides(clothing="toga"))


def test_apply_does_not_mutate_its_input() -> None:
    base = default_document_dict()
    before = repr(base)
    apply_overrides(base, AppearanceOverrides(hair="#336699", top_color="#aa0000"))
    assert repr(base) == before


def test_precedence_cli_over_document_over_defaults() -> None:
    document = {
        "schema_version": "appearance-v1",
        "skin_tone": "skin_5",
        "clothing": "golf_polo_shorts",
    }
    # defaults only
    assert resolve_appearance(None, None)["skin_tone"] == "skin_2"
    # document beats defaults, and fields it does not set stay at the defaults
    resolved = resolve_appearance(None, document)
    assert resolved["skin_tone"] == "skin_5"
    assert resolved["clothing"] == "golf_polo_shorts"
    assert resolved["club_finish"] == default_document_dict()["club_finish"]
    # CLI beats document, field by field
    cli = AppearanceOverrides(skin_tone="skin_7")
    resolved = resolve_appearance(cli, document)
    assert resolved["skin_tone"] == "skin_7"
    assert resolved["clothing"] == "golf_polo_shorts"
    # an empty CLI changes nothing
    assert resolve_appearance(AppearanceOverrides(), document) == resolve_appearance(
        None, document
    )


def test_precedence_is_pure_and_result_validates() -> None:
    document = {"schema_version": "appearance-v1", "skin_tone": "skin_5"}
    snapshot = repr(document)
    out = resolve_appearance(AppearanceOverrides(hair="#101010"), document)
    assert repr(document) == snapshot
    document_from_dict(out)


def test_explicit_defaults_argument_is_the_base_layer() -> None:
    base = {**default_document_dict(), "skin_tone": "skin_8"}
    assert resolve_appearance(None, None, base)["skin_tone"] == "skin_8"
