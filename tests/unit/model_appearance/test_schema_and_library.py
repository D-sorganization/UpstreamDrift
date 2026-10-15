"""Appearance schema, material library and spec-hash independence."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import (
    CLOTHING,
    MATERIALS,
    BallSettings,
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
    assert doc.ball == BallSettings(True, None, "address_geometry")


def test_ball_round_trips_with_explicit_position() -> None:
    doc = document_from_dict(
        {**MINIMAL, "ball": {"enabled": False, "source": "measured"}}
    )
    assert doc.ball == BallSettings(False, None, "measured")
    doc = document_from_dict(
        {
            **MINIMAL,
            "ball": {"position_m": [1.0, 2.0, 0.021335], "source": "measured"},
        }
    )
    assert doc.ball.position_m == pytest.approx((1.0, 2.0, 0.021335))
    again = document_from_dict(document_to_dict(doc))
    assert again == doc


def test_ball_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="ball source"):
        BallSettings(True, None, "guessed")


def test_ball_rejects_bad_position() -> None:
    with pytest.raises(ValueError, match="ball position_m"):
        BallSettings(True, (1.0, float("nan"), 0.0), "measured")


def test_schema_rejects_malformed_ball_block() -> None:
    with pytest.raises(ValueError, match="Invalid appearance document"):
        validate_document({**MINIMAL, "ball": {"source": "guessed"}})
    with pytest.raises(ValueError, match="Invalid appearance document"):
        validate_document({**MINIMAL, "ball": {"position_m": [1.0, 2.0]}})


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


# --- R2 (#12681): pelvis wears the pants; order-independent classification ---

ANTHRO_SPECS = tuple(
    SPEC.parent / f"full_body_spec_anthro_{club}.json" for club in ("driver", "iron7")
)
_GS3D = "solid_reference:GolfSwing3D_Kinetic"
# Golden part for every body of the anthro specs (and full_body_spec_v1).
GOLDEN_PARTS = {
    f"{_GS3D}/Hips and Torso Inputs/LowerTorso": "pelvis",
    f"{_GS3D}/Hips and Torso Inputs/UpperTorsoBase": "torso",
    f"{_GS3D}/Hips and Torso Inputs/COMRod": "torso",
    "GolfSwing3D_Kinetic/Head": "head",
    f"{_GS3D}/HubtoLS": "shoulder",
    f"{_GS3D}/HubtoRS": "shoulder",
    f"{_GS3D}/LUpperArm": "upper_arm",
    f"{_GS3D}/RUpperArm": "upper_arm",
    f"{_GS3D}/Left Elbow Joint/Spherical Solid": "forearm",
    f"{_GS3D}/Right Elbow Joint/Spherical Solid1": "forearm",
    f"{_GS3D}/Left Forearm/LLowerForearm": "forearm",
    f"{_GS3D}/Right Forearm/RLowerForearm": "forearm",
    f"{_GS3D}/Club/Clubface Vector": "club",
    f"{_GS3D}/Grip/RHandStandoff": "hand",
    **{f"femur_{s}": "thigh" for s in "lr"},
    **{f"tibia_{s}": "shin" for s in "lr"},
    **{f"{bone}_{s}": "foot" for bone in ("talus", "calcn", "toes") for s in "lr"},
}
PELVIS = f"{_GS3D}/Hips and Torso Inputs/LowerTorso"


def _spec_body_names(path: Path) -> list[str]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    return [b["name"] for b in spec["bodies"] if b["name"] != "world"]


@pytest.mark.parametrize("spec_path", [*ANTHRO_SPECS, SPEC], ids=lambda p: p.stem)
def test_every_spec_body_matches_the_golden_classification(spec_path: Path) -> None:
    names = _spec_body_names(spec_path)
    assert set(names) <= set(GOLDEN_PARTS), "golden table must cover every body"
    wrong = {
        n: (classify_body(n), GOLDEN_PARTS[n])
        for n in names
        if classify_body(n) != GOLDEN_PARTS[n]
    }
    assert wrong == {}, f"mis-classified (got, want): {wrong}"


def test_pelvis_is_classified_by_its_segment_not_its_subsystem_folder() -> None:
    # "Hips and Torso Inputs" matches *torso*; the leaf segment must decide.
    assert classify_body(PELVIS) == "pelvis"
    assert classify_body("LowerTorso") == "pelvis"


def test_classification_is_independent_of_rule_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.shared.python.model_appearance import library

    names = list(GOLDEN_PARTS)
    expected = [classify_body(n) for n in names]
    rng = np.random.default_rng(12681)
    for _ in range(5):
        order = rng.permutation(len(library.PART_RULES))
        monkeypatch.setattr(
            library, "PART_RULES", tuple(library.PART_RULES[i] for i in order)
        )
        assert [classify_body(n) for n in names] == expected
    monkeypatch.setattr(library, "PART_RULES", tuple(reversed(library.PART_RULES)))
    assert [classify_body(n) for n in names] == expected


def test_leaf_segment_rule_beats_a_parent_subsystem_rule() -> None:
    # Specific (leaf) before general (folder): a leaf hit is never overridden.
    assert classify_body("Left Forearm/LUpperArm") == "upper_arm"
    assert classify_body("Grip/RHandStandoff") == "hand"
    # A leaf with no rule falls back to its nearest classified folder.
    assert classify_body("Left Elbow Joint/Spherical Solid") == "forearm"


def test_ambiguous_tie_between_different_parts_raises() -> None:
    # The folder matches both *hips* (pelvis) and *torso* at equal specificity.
    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        classify_body(f"{_GS3D}/Hips and Torso Inputs/Spherical Solid")


@pytest.mark.parametrize("preset", sorted(CLOTHING))
def test_pelvis_garment_is_the_active_outfit_pants(preset: str) -> None:
    from src.shared.python.model_appearance.library import pants_material

    doc = document_from_dict({**MINIMAL, "clothing": preset})
    pants = pants_material(preset)
    assert pants == CLOTHING[preset].get("thigh")
    assert garment_name_for(doc, PELVIS) == pants
    if pants is not None:
        assert garment_name_for(doc, PELVIS) != CLOTHING[preset].get("torso")


def test_default_outfit_pelvis_matches_its_thigh_garment() -> None:
    doc = document_from_dict(MINIMAL)
    assert garment_name_for(doc, PELVIS) == garment_name_for(doc, "femur_r")


def test_check_clothing_rejects_a_pelvis_that_is_not_the_pants() -> None:
    from src.shared.python.model_appearance.library import check_clothing

    check_clothing(CLOTHING)  # every shipped preset satisfies the pants rule
    bad = {"odd": {**CLOTHING["golf_polo_shorts"], "pelvis": "polo_navy"}}
    with pytest.raises(ValueError, match="pelvis"):
        check_clothing(bad)
    with pytest.raises(ValueError, match="material"):
        check_clothing({"odd": {"thigh": "nope", "pelvis": "nope"}})
