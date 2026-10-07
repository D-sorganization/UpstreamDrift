"""CMB-2 (#11653): character presets library."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.shared.python.humanoid_character_builder.presets import loader
from src.shared.python.humanoid_character_builder.spec_params import (
    SpecCharacterParameters,
    compile_full_body_spec,
    params_from_spec,
)

pytestmark = pytest.mark.unit

EXPECTED = {
    "tour_average_male",
    "tour_average_female",
    "junior",
    "senior",
    "anthro_driver",
    "anthro_iron7",
}


def test_library_ships_expected_presets() -> None:
    assert EXPECTED <= set(loader.list_character_presets())


def test_listing_is_sorted_and_stable() -> None:
    names = loader.list_character_presets()
    assert names == sorted(names)


@pytest.mark.parametrize("preset_id", sorted(EXPECTED))
def test_every_shipped_file_validates_against_schema(preset_id: str) -> None:
    path = loader.CHARACTER_PRESET_DIR / f"{preset_id}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    loader.validate_character_preset(document)
    assert document["id"] == preset_id


def test_every_json_file_in_library_validates() -> None:
    files = sorted(loader.CHARACTER_PRESET_DIR.glob("*.json"))
    assert len(files) >= len(EXPECTED)
    for path in files:
        loader.validate_character_preset(json.loads(path.read_text("utf-8")))


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda d: d.pop("provenance"), "provenance"),
        (lambda d: d.__setitem__("schema_version", "v0"), "schema_version"),
        (lambda d: d["parameters"].__setitem__("stature_m", 9.0), "stature_m"),
        (lambda d: d["parameters"].__setitem__("club", "putter"), "club"),
        (lambda d: d["parameters"].__setitem__("bogus", 1.0), "bogus"),
        (lambda d: d.__setitem__("extra", 1), "extra"),
    ],
)
def test_schema_rejects_bad_documents(mutate, match: str) -> None:
    document = loader.load_character_preset("junior").document
    broken = copy.deepcopy(document)
    mutate(broken)
    with pytest.raises(ValueError, match=match):
        loader.validate_character_preset(broken)


def test_load_returns_parameters_and_metadata() -> None:
    preset = loader.load_character_preset("anthro_driver")
    assert preset.parameters == SpecCharacterParameters(
        stature_m=1.71,
        mass_kg=78.0,
        trunk_scale=1.15,
        arm_scale=1.1,
        shoulder_scale=1.0,
        club="driver",
    )
    assert preset.name and preset.limitations


def test_load_is_case_insensitive_and_rejects_unknown() -> None:
    assert loader.load_character_preset("JUNIOR").id == "junior"
    with pytest.raises(ValueError, match="Unknown character preset"):
        loader.load_character_preset("nobody")


def test_ids_cannot_escape_the_library_directory() -> None:
    with pytest.raises(ValueError, match="Unknown character preset"):
        loader.load_character_preset("../presets/loader")


def test_overrides_replace_parameters() -> None:
    preset = loader.load_character_preset("junior", mass_kg=50.0)
    assert preset.parameters.mass_kg == 50.0
    assert preset.parameters.stature_m == 1.52


def test_overrides_reject_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown"):
        loader.load_character_preset("junior", wingspan=3.0)


@pytest.mark.parametrize("preset_id", sorted(EXPECTED))
def test_each_preset_compiles_via_cmb1(preset_id: str) -> None:
    preset = loader.load_character_preset(preset_id)
    spec = compile_full_body_spec(preset.parameters)
    assert params_from_spec(spec) == preset.parameters


def test_anthro_presets_reproduce_committed_specs() -> None:
    """The anthro presets are the subjects of the committed driver and iron specs."""
    root = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"

    def committed_subject(club: str) -> dict:
        path = root / f"full_body_spec_anthro_{club}.json"
        return json.loads(path.read_text(encoding="utf-8"))["subject"]

    for preset_id, club in (("anthro_driver", "driver"), ("anthro_iron7", "iron7")):
        preset = loader.load_character_preset(preset_id)
        subject = committed_subject(club)
        assert subject["stature_m"] == preset.parameters.stature_m
        assert subject["mass_kg"] == preset.parameters.mass_kg
        assert subject["trunk_scale"] == preset.parameters.trunk_scale
        assert subject["arm_scale"] == preset.parameters.arm_scale
