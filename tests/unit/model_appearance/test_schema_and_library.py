"""Appearance schema, material library and spec-hash independence."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import (
    CLOTHING,
    MATERIALS,
    classify_body,
    document_from_dict,
    document_to_dict,
    load_appearance,
    physics_spec_sha256,
    save_appearance,
    validate_document,
)
from src.shared.python.model_appearance.library import (
    garment_name_for,
    material_name_for,
)
from src.shared.python.model_appearance.schema import appearance_path_for

pytestmark = pytest.mark.unit
SPEC = Path(__file__).resolve().parents[3] / (
    "docs/development/full_body_models/full_body_spec_v1.json"
)
MINIMAL = {"schema_version": "appearance-v1"}


def test_minimal_document_parses_with_defaults() -> None:
    doc = document_from_dict(MINIMAL)
    assert doc.skin_tone == "skin_medium"
    assert doc.clothing in CLOTHING and doc.club_finish in MATERIALS


def test_round_trip_is_schema_valid_and_stable(tmp_path: Path) -> None:
    doc = document_from_dict(
        {
            **MINIMAL,
            "skin_tone": "skin_dark",
            "materials": {
                "my_cloth": {
                    "base_color": [0.5, 0.1, 0.1],
                    "roughness": 0.9,
                    "texture": {"kind": "checker", "repeat": 4},
                }
            },
            "segments": [{"match": "toes_*", "material": "my_cloth"}],
            "environment": {"lighting": "daylight"},
        }
    )
    again = document_from_dict(document_to_dict(doc))
    assert again == doc
    path = tmp_path / "x.appearance.json"
    save_appearance(doc, path)
    assert load_appearance(path) == doc


@pytest.mark.parametrize(
    "bad",
    [
        {"schema_version": "appearance-v2"},
        {**MINIMAL, "unknown": 1},
        {**MINIMAL, "skin_tone": "Not A Name"},
        {**MINIMAL, "materials": {"a": {"base_color": [2, 0, 0]}}},
        {**MINIMAL, "materials": {"a": {"base_color": [0, 0, 0], "roughness": -1}}},
        {**MINIMAL, "segments": [{"mesh": "smooth"}]},
        {**MINIMAL, "spec_sha256": "abc"},
    ],
)
def test_schema_rejects_malformed_documents(bad: dict) -> None:
    with pytest.raises(ValueError, match="Invalid appearance document"):
        validate_document(bad)


@pytest.mark.parametrize(
    "bad",
    [
        {**MINIMAL, "skin_tone": "skin_unobtainium"},
        {**MINIMAL, "clothing": "space_suit"},
        {**MINIMAL, "club_finish": "missing"},
        {**MINIMAL, "segments": [{"match": "*", "material": "missing"}]},
        {
            **MINIMAL,
            "materials": {"t": {"base_color": [0, 0, 0], "texture": {"kind": "file"}}},
        },
    ],
)
def test_references_must_resolve_by_name(bad: dict) -> None:
    with pytest.raises(ValueError):
        document_from_dict(bad)


def test_non_mapping_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        validate_document([])  # type: ignore[arg-type]


def test_classification_covers_the_full_body_spec() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    parts = {b["name"]: classify_body(b["name"]) for b in spec["bodies"][1:]}
    assert "other" not in parts.values()
    assert {"torso", "upper_arm", "forearm", "thigh", "shin", "foot", "club"} <= set(
        parts.values()
    )
    with pytest.raises(ValueError):
        classify_body("")


def test_materials_and_garments_resolve_by_name() -> None:
    doc = document_from_dict({**MINIMAL, "skin_tone": "skin_dark"})
    assert material_name_for(doc, "femur_r") == "skin_dark"
    assert garment_name_for(doc, "femur_r") == CLOTHING[doc.clothing]["thigh"]
    assert material_name_for(doc, "Club/Clubface Vector") == doc.club_finish
    override = document_from_dict(
        {**MINIMAL, "segments": [{"match": "femur_*", "material": "chrome"}]}
    )
    assert material_name_for(override, "femur_r") == "chrome"
    assert garment_name_for(override, "femur_r") is None


def test_appearance_never_changes_the_physics_spec_hash() -> None:
    spec = json.loads(SPEC.read_bytes())
    base = physics_spec_sha256(spec)
    assert physics_spec_sha256(SPEC.read_bytes()) == base
    visual = copy.deepcopy(spec)
    visual["visual_hints"] = {"capsule_radius_m": {"femur_r": 0.09}}
    visual["appearance"] = {"skin_tone": "skin_dark"}
    assert physics_spec_sha256(visual) == base
    physical = copy.deepcopy(spec)
    physical["gravity_m_s2"] = [0.0, 0.0, -1.0]
    assert physics_spec_sha256(physical) != base


def test_appearance_document_lives_beside_the_spec() -> None:
    assert appearance_path_for("a/b/spec.json") == Path("a/b/spec.appearance.json")


def test_bound_spec_hash_round_trips() -> None:
    digest = physics_spec_sha256(SPEC.read_bytes())
    doc = document_from_dict({**MINIMAL, "spec_sha256": digest})
    assert document_to_dict(doc)["spec_sha256"] == digest
    assert np.isclose(sum(MATERIALS["turf"].base_color[:3]), 1.04)


def test_shipped_appearance_beside_the_full_body_spec_is_bound_and_valid() -> None:
    doc = load_appearance(appearance_path_for(SPEC))
    assert doc.spec_sha256 == physics_spec_sha256(SPEC.read_bytes())
