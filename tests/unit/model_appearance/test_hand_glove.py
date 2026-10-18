"""Bare skin-coloured grip hands by default and an optional glove (HRG-G1, #13186).

One semantic resolver (``library.hand_material_name_for``) decides every grip
hand's material: a bare hand takes the document skin tone, a gloved hand the
white golf glove. ``glove`` (none / lead / trail / both) names a hand by its
golf role and ``handedness`` maps the role to the anatomical side, so the lead
hand is the left one for a right-handed golfer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.model_appearance import club_assembly, library
from src.shared.python.model_appearance.appearance_selection import (
    AppearanceOverrides,
    apply_overrides,
    resolve_appearance,
)
from src.shared.python.model_appearance.body_meshes import (
    body_meshes,
    club_meshes,
)
from src.shared.python.model_appearance.body_meshes import (
    resolve_appearance as resolve_meshes,
)
from src.shared.python.model_appearance.club_assembly import assembly_from_spec
from src.shared.python.model_appearance.defaults import default_document_dict
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.schema import (
    document_from_dict,
    document_to_dict,
)

pytestmark = pytest.mark.unit

MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
SPEC = json.loads((MODELS / "full_body_spec_anthro_driver.json").read_text("utf-8"))
GLOVES = ("none", "lead", "trail", "both")
HANDEDNESS = ("right", "left")
GLOVED = {
    "none": set(),
    "lead": {Hand.LEAD},
    "trail": {Hand.TRAIL},
    "both": {Hand.LEAD, Hand.TRAIL},
}
LEFT_OF = {"right": Hand.LEAD, "left": Hand.TRAIL}  # which role is the left hand


def _doc(**fields: object):
    return document_from_dict({**default_document_dict(), **fields})


def _hand_materials(doc) -> dict[str, str]:
    club = assembly_from_spec(SPEC)
    return {
        label: name
        for label, _mesh, name in club_meshes(club, doc.club_finish, doc)
        if label.endswith("_hand")
    }


def test_default_document_has_bare_hands() -> None:
    doc = _doc()
    assert doc.glove == "none"
    assert doc.handedness == "right"
    names = _hand_materials(doc)
    assert names == {"lead_hand": "skin_2", "trail_hand": "skin_2"}
    skin = library.MATERIALS["skin_2"].rgba
    for hand in Hand:
        assert library.library_materials(doc)[
            library.hand_material_name_for(hand, doc)
        ].rgba == pytest.approx(skin)


@pytest.mark.parametrize("clothing", sorted(library.CLOTHING))
@pytest.mark.parametrize("handedness", HANDEDNESS)
@pytest.mark.parametrize("glove", GLOVES)
def test_exactly_the_right_hands_are_gloved(clothing, handedness, glove) -> None:
    doc = _doc(clothing=clothing, handedness=handedness, glove=glove)
    gloved = {
        Hand.LEAD if label == "lead_hand" else Hand.TRAIL
        for label, name in _hand_materials(doc).items()
        if name == "glove_white"
    }
    assert gloved == GLOVED[glove]
    for hand in set(Hand) - GLOVED[glove]:
        assert library.hand_material_name_for(hand, doc) == doc.skin_tone
    # the anatomical sides: lead is the left hand of a right-hander
    left_gloved = LEFT_OF[handedness] in GLOVED[glove]
    sides = library.gloved_sides(doc)
    assert ("left" in sides) is left_gloved
    right_role = Hand.TRAIL if handedness == "right" else Hand.LEAD
    assert ("right" in sides) is (right_role in GLOVED[glove])


@pytest.mark.parametrize("clothing", sorted(library.CLOTHING))
def test_no_clothing_preset_carries_a_glove_policy(clothing) -> None:
    assert "hand" not in library.CLOTHING[clothing]
    assert "lead_hand" not in club_assembly.PART_MATERIALS
    assert "trail_hand" not in club_assembly.PART_MATERIALS


@pytest.mark.parametrize("glove", GLOVES)
def test_no_hand_mesh_is_steel_and_none_is_glove_white_by_default(glove) -> None:
    club = assembly_from_spec(SPEC)
    resolved = resolve_meshes(SPEC, _doc(glove=glove), club)
    names = {m.material_name for m in resolved.meshes if m.label.endswith("_hand")}
    assert "satin_steel" not in names
    assert ("glove_white" in names) is (glove != "none")
    assert "skin_2" in names or glove == "both"


def test_custom_skin_colours_a_bare_hand() -> None:
    doc = apply_overrides(
        default_document_dict(), AppearanceOverrides(skin_tone="#336699")
    )
    parsed = document_from_dict(doc)
    resolved = resolve_meshes(SPEC, parsed, assembly_from_spec(SPEC))
    hands = [m for m in resolved.meshes if m.label.endswith("_hand")]
    assert {m.material_name for m in hands} == {"custom_skin"}


def test_no_duplicate_standoff_loft_when_grip_hands_are_drawn() -> None:
    doc = _doc()
    standoff = [
        m
        for m in body_meshes(SPEC, doc, assembly_from_spec(SPEC))
        if m.body.endswith("RHandStandoff")
    ]
    assert standoff == []
    # without the club there is no grip hand, so the generic hand loft stays
    bare = [m for m in body_meshes(SPEC, doc) if m.body.endswith("RHandStandoff")]
    assert bare


def test_old_document_without_glove_resolves_to_none() -> None:
    legacy = {"schema_version": "appearance-v1", "skin_tone": "skin_5"}
    doc = document_from_dict(legacy)
    assert (doc.glove, doc.handedness) == ("none", "right")
    assert "glove" not in document_to_dict(doc)  # default is omitted


def test_glove_round_trips_through_the_document() -> None:
    data = document_to_dict(_doc(glove="lead", handedness="left"))
    assert (data["glove"], data["handedness"]) == ("lead", "left")
    again = document_from_dict(data)
    assert (again.glove, again.handedness) == ("lead", "left")


def test_schema_rejects_unknown_glove_and_handedness() -> None:
    for bad in ({"glove": "left"}, {"glove": "all"}, {"handedness": "both"}):
        with pytest.raises(ValueError):
            document_from_dict({**default_document_dict(), **bad})


def test_glove_override_precedence() -> None:
    document = {"schema_version": "appearance-v1", "glove": "trail"}
    assert resolve_appearance(None, document)["glove"] == "trail"
    resolved = resolve_appearance(AppearanceOverrides(glove="lead"), document)
    assert resolved["glove"] == "lead"
    assert "glove" not in resolve_appearance(None, None)
    with pytest.raises(ValueError):
        resolve_appearance(AppearanceOverrides(glove="sock"), None)


def test_cli_glove_flag_threads_through_the_resolver() -> None:
    from src.tools.native_viewer_export import cli

    argv = ["--bundle", "b", "--out", "o", "--swing", "s"]
    parsed = cli.build_parser().parse_args(argv + ["--glove", "lead"])
    assert cli.build_appearance(parsed)["glove"] == "lead"
    assert cli.build_appearance(cli.build_parser().parse_args(argv)) is None
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv + ["--glove", "sock"])


def test_web_and_desktop_writers_carry_the_glove() -> None:
    from src.api.services.character_appearance_service import (
        appearance_library,
        build_appearance,
    )
    from src.tools.character_builder.core import CharacterBuilderModel

    assert build_appearance({"glove": "both"}, None)["glove"] == "both"
    assert "glove" not in build_appearance({"glove": "none"}, None)
    assert appearance_library()["glove"] == list(GLOVES)
    model = CharacterBuilderModel()
    model.set_appearance("glove", "trail")
    assert model.appearance_document()["glove"] == "trail"
    with pytest.raises(ValueError):
        model.set_appearance("glove", "sock")
    assert model.appearance_choices()["glove"] == "trail"


@pytest.mark.parametrize("glove", GLOVES)
def test_opensim_hand_meshes_follow_the_document(glove) -> None:
    from src.engines.physics_engines.opensim.python import club_visuals

    doc = _doc(glove=glove)
    elements = club_visuals.club_mesh_elements(
        "driver", "", parts=club_visuals.HAND_PARTS, appearance=doc
    )
    skin = library.library_materials(doc)[doc.skin_tone].rgba[:3]
    white = library.MATERIALS["glove_white"].rgba[:3]
    colours = {}
    for el in elements:
        text = el.find("Appearance/color").text
        colours[el.get("name")] = tuple(float(x) for x in text.split())
    for hand, name in (
        ("lead", "club_lead_hand_geom"),
        ("trail", "club_trail_hand_geom"),
    ):
        expected = white if Hand(hand) in GLOVED[glove] else skin
        assert colours[name] == pytest.approx(expected)


def test_glove_is_visual_only() -> None:
    from src.shared.python.model_appearance.schema import physics_spec_sha256

    assert physics_spec_sha256(SPEC) == physics_spec_sha256(
        {**SPEC, "appearance": document_to_dict(_doc(glove="both"))}
    )
