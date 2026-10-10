"""CMB-7b (#11658): character appearance service (material library + build)."""

from __future__ import annotations

import pytest

from src.api.services.character_appearance_service import (
    GROUND_MATERIALS,
    appearance_library,
    appearance_sidecar_filename,
    build_appearance,
)
from src.shared.python.model_appearance import document_from_dict
from src.shared.python.model_appearance.library import (
    CLOTHING,
    CLUB_FINISHES,
    MATERIALS,
    SKIN_TONES,
)

pytestmark = pytest.mark.unit


def test_appearance_library_lists_every_choice() -> None:
    lib = appearance_library()
    assert set(lib["skin_tones"]) == set(SKIN_TONES)
    assert set(lib["clothing"]) == set(CLOTHING)
    assert set(lib["club_finishes"]) == set(CLUB_FINISHES)
    assert set(lib["headwear"]) == {"none", "hair", "cap"}
    assert set(lib["ground_materials"]) == set(GROUND_MATERIALS)
    assert set(lib["materials"]) == set(MATERIALS)


def test_appearance_library_material_dicts_have_pbr_fields() -> None:
    lib = appearance_library()
    for material in lib["materials"].values():
        assert {"base_color", "roughness", "metallic"} <= set(material)
        assert isinstance(material["base_color"], list)


def test_ground_materials_are_known_materials() -> None:
    assert set(GROUND_MATERIALS) <= set(MATERIALS)


def test_build_appearance_with_no_choices_validates_and_uses_defaults() -> None:
    doc_dict = build_appearance({}, None)
    doc = document_from_dict(doc_dict)
    assert doc.skin_tone == "skin_medium"
    assert doc.clothing == "golf_polo_shorts"
    assert doc.club_finish == "satin_steel"
    assert doc.spec_sha256 is None


def test_build_appearance_applies_every_pick() -> None:
    choices = {
        "skin_tone": "skin_tan",
        "clothing": "golf_polo_trousers",
        "club_finish": "black_pvd",
        "headwear": "cap",
        "headwear_material": "cap_navy",
        "ground_material": "studio_floor",
        "name": "picked",
    }
    doc_dict = build_appearance(choices, "a" * 64)
    doc = document_from_dict(doc_dict)
    assert doc.skin_tone == "skin_tan"
    assert doc.clothing == "golf_polo_trousers"
    assert doc.club_finish == "black_pvd"
    assert doc.head.headwear == "cap"
    assert doc.head.headwear_material == "cap_navy"
    assert doc.environment.ground_material == "studio_floor"
    assert doc.name == "picked"
    assert doc.spec_sha256 == "a" * 64


def test_build_appearance_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_dict = build_appearance({"skin_tone": "skin_light"}, None)
    again = document_from_dict(doc_dict)
    assert document_from_dict(doc_dict) == again


@pytest.mark.parametrize(
    "choices",
    [
        {"skin_tone": "skin_green"},
        {"clothing": "space_suit"},
        {"club_finish": "gold"},
        {"headwear": "cap", "headwear_material": "not_a_material"},
        {"ground_material": "lava"},
    ],
)
def test_build_appearance_rejects_unknown_names(choices: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        build_appearance(choices, None)


def test_build_appearance_rejects_non_mapping_choices() -> None:
    with pytest.raises(TypeError):
        build_appearance("not-a-mapping", None)  # type: ignore[arg-type]


def test_build_appearance_rejects_non_string_spec_sha() -> None:
    with pytest.raises(TypeError):
        build_appearance({}, 12345)  # type: ignore[arg-type]


def test_appearance_sidecar_filename_without_spec() -> None:
    name = appearance_sidecar_filename(
        {"name": "custom_doc"}, preset_id=None, spec_sha256=None
    )
    assert name == "custom_doc.appearance.json"


def test_appearance_sidecar_filename_with_preset_and_spec() -> None:
    name = appearance_sidecar_filename(
        {"name": "ignored"}, preset_id="junior", spec_sha256="deadbeef" + "0" * 56
    )
    assert name == "junior_deadbeef.appearance.json"


def test_appearance_sidecar_filename_falls_back_to_document_name() -> None:
    name = appearance_sidecar_filename({}, preset_id=None, spec_sha256=None)
    assert name == "character.appearance.json"
