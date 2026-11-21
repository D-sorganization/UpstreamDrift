"""Score lines, face colour and smoothing of the drawn club head (#15150)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import club_face_marking as fm
from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.club_head_mesh import (
    face_centre,
    load_club_head,
    measured_face_normal,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


def _on(**over: object) -> fm.FaceMarkingSettings:
    base = dataclasses.replace(fm.load_face_marking_settings(), face_marking=True)
    return dataclasses.replace(base, **over)


def _club(marking: fm.FaceMarkingSettings | None) -> ca.ClubAssembly:
    return ca.ClubAssembly(
        "driver", 1.15, 0.265, 0.0065, face_roll_deg=-37.92, face_marking=marking
    )


def test_defaults_leave_the_head_as_built() -> None:
    assert not fm.load_face_marking_settings().face_marking
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert ca.assembly_from_spec(spec).face_marking is None
    meshes = ca.assembly_meshes(_club(None))
    assert set(meshes) == {"shaft", "grip", "head"}
    assert len(meshes["head"].faces) == len(load_club_head("driver").mesh.faces)


@pytest.mark.parametrize(
    "field,value",
    [
        ("score_line_count", -1),
        ("score_line_spacing_m", 0.0),
        ("score_line_width_m", 0.004),
        ("face_rgba", (1.0, 2.0, 0.0, 1.0)),
        ("face_marking", "yes"),
    ],
)
def test_bad_settings_raise_value_error(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(fm.load_face_marking_settings(), **{field: value})


def test_marking_adds_face_and_line_parts_without_moving_the_contact_face() -> None:
    plain = ca.assembly_meshes(_club(None))["head"]
    marked = ca.assembly_meshes(_club(_on(centre_clear_width_m=0.02)))
    assert set(marked) == {"shaft", "grip", "head", "head_face", "face_lines"}
    assert len(marked["head"].faces) + len(marked["head_face"].faces) == len(
        plain.faces
    )
    both = np.vstack([marked["head"].vertices, marked["head_face"].vertices])
    for point in plain.vertices[::7]:
        assert np.linalg.norm(both - point, axis=1).min() < 1e-12


def test_every_line_triangle_set_lies_on_the_face_within_tolerance() -> None:
    head = load_club_head("driver")
    settings = _on()
    frame = fm.face_frame(head.head_mesh)
    assert np.allclose(frame.normal, measured_face_normal(head.head_mesh))
    assert np.allclose(frame.centre, face_centre(head.head_mesh))
    lines = fm.marked_head_parts(head, settings)["face_lines"]
    rel = lines.vertices - frame.centre
    toe, up, depth = rel @ frame.toe, rel @ frame.up, rel @ frame.normal
    # the driver face bulges: lines stay within 6 mm of the centre tangent plane
    assert np.abs(depth).max() < 0.006
    # and sit on the real surface: 0.15 mm lift, 1 micron tolerance
    mask = fm.face_triangle_mask(head.head_mesh, frame, fm.FACE_NORMAL_COS)
    tri = head.head_mesh.vertices[head.head_mesh.faces[mask]]
    start = lines.vertices + 0.03 * frame.normal
    above = fm._ray_heights(tri, start, -frame.normal) - 0.03
    assert np.abs(above - settings.score_line_height_m).max() < 1e-6
    assert np.abs(up).max() < 0.5 * settings.score_line_count * 0.0035 + 0.001
    assert np.abs(toe).min() >= 0.5 * settings.centre_clear_width_m - 1e-9


def test_line_count_and_clear_zone_are_obeyed() -> None:
    head = load_club_head("driver")
    assert "face_lines" not in fm.marked_head_parts(head, _on(score_line_count=0))
    few = fm.marked_head_parts(head, _on(score_line_count=3))["face_lines"]
    many = fm.marked_head_parts(head, _on(score_line_count=9))["face_lines"]
    assert len(many.faces) > 2.5 * len(few.faces)
    wide = fm.marked_head_parts(head, _on(centre_clear_width_m=0.04))
    narrow = fm.marked_head_parts(head, _on(centre_clear_width_m=0.0))
    assert len(wide["face_lines"].faces) < len(narrow["face_lines"].faces)


def test_materials_exist_and_differ_from_the_crown() -> None:
    assert library.MATERIALS["club_face"] != library.MATERIALS["satin_steel"]
    assert (
        ca.club_part_material("face_lines", "satin_steel", None) == "club_score_lines"
    )


def test_spec_override_turns_marking_on() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    spec["club"] = {
        **spec["club"],
        "face_marking": {"face_marking": True, "score_line_count": 5},
    }
    assert "face_lines" in ca.assembly_meshes(ca.assembly_from_spec(spec))


def test_marking_parts_follow_the_head_onto_the_chain_tip_link() -> None:
    from src.shared.python.model_appearance import document_from_dict
    from src.shared.python.model_appearance.club_chain_meshes import (
        chain_layout,
        club_meshes_by_body,
    )
    from src.shared.python.motion_matching.club_models import DRIVER
    from src.shared.python.motion_matching.club_shaft_chain import (
        apply_shaft_chain,
        build_shaft_chain,
    )

    path = (
        "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
    )
    spec = json.loads(
        (ROOT / path / "full_body_spec_hipcal_scaled.json").read_text("utf-8")
    )
    spec = apply_shaft_chain(spec, build_shaft_chain(DRIVER))
    club = dataclasses.replace(ca.assembly_from_spec(spec), face_marking=_on())
    layout = chain_layout(spec)
    doc = document_from_dict({"schema_version": "appearance-v1"})
    by_body = club_meshes_by_body(spec, club, "satin_steel", doc)
    tip = [p[0] for p in by_body[layout.tip_link]]
    others = [p[0] for b, ps in by_body.items() if b != layout.tip_link for p in ps]
    for label in ("head", "head_face", "face_lines"):
        assert label in tip and label not in others
