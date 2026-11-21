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
    settings = fm.load_face_marking_settings()
    assert not settings.active
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert ca.assembly_from_spec(spec).face_marking is None
    meshes = ca.assembly_meshes(_club(None))
    assert set(meshes) == {"shaft", "grip", "head"}
    head = load_club_head("driver")
    assert len(meshes["head"].faces) == len(head.mesh.faces)


@pytest.mark.parametrize(
    "field,value",
    [
        ("score_line_count", -1),
        ("score_line_count", 2.5),
        ("score_line_spacing_m", 0.0),
        ("score_line_width_m", 0.004),
        ("centre_clear_width_m", -0.001),
        ("face_rgba", (1.0, 2.0, 0.0, 1.0)),
        ("face_normal_cos", 1.0),
        ("subdivisions", 4),
        ("max_triangles", -5),
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
    head = load_club_head("driver")
    assert len(head.head_mesh.faces) == 1792  # the shared head is untouched


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
    mask = fm.face_triangle_mask(head.head_mesh, frame, settings.face_normal_cos)
    tri = head.head_mesh.vertices[head.head_mesh.faces[mask]]
    start = lines.vertices + 0.03 * frame.normal
    above = fm._ray_heights(tri, start, -frame.normal) - 0.03
    assert np.abs(above - settings.score_line_height_m).max() < 1e-6
    # ten rows of two ribbons plus the middle one: 11 rows, split by the zone
    rows = np.unique(np.round(up / settings.score_line_spacing_m, 1))
    assert len(rows) >= settings.score_line_count
    assert np.abs(up).max() < 0.5 * settings.score_line_count * 0.0035 + 0.001
    assert np.abs(toe).min() >= 0.5 * settings.centre_clear_width_m - 1e-9
    assert float(np.mean(fm_normals_dot(lines, frame.normal))) > 0.9


def fm_normals_dot(mesh: object, normal: np.ndarray) -> np.ndarray:
    fn, _ = fm._face_normals(mesh)
    return fn @ normal


def test_line_count_and_clear_zone_are_obeyed() -> None:
    head = load_club_head("driver")
    none = fm.marked_head_parts(head, _on(score_line_count=0))
    assert "face_lines" not in none
    few = fm.marked_head_parts(head, _on(score_line_count=3))["face_lines"]
    many = fm.marked_head_parts(head, _on(score_line_count=9))["face_lines"]
    assert len(many.faces) > 2.5 * len(few.faces)
    wide = fm.marked_head_parts(head, _on(centre_clear_width_m=0.04))["face_lines"]
    narrow = fm.marked_head_parts(head, _on(centre_clear_width_m=0.0))["face_lines"]
    assert len(wide.faces) < len(narrow.faces)


def test_materials_exist_and_differ_from_the_crown() -> None:
    face = library.MATERIALS["club_face"].rgba
    assert face != library.MATERIALS["satin_steel"].rgba
    for label in ("head_face", "face_lines"):
        assert ca.club_part_material(label, "satin_steel", None) in library.MATERIALS


def test_smoothing_keeps_vertices_and_respects_the_triangle_budget() -> None:
    head = load_club_head("driver")
    base = len(head.head_mesh.faces)
    once = fm.smooth_head_body(head.head_mesh, _on(subdivisions=1))
    assert len(once.faces) == 4 * base
    assert np.array_equal(
        once.vertices[: len(head.head_mesh.vertices)], head.head_mesh.vertices
    )
    assert abs(once.volume() / head.head_mesh.volume() - 1.0) < 0.02
    capped = fm.smooth_head_body(head.head_mesh, _on(subdivisions=3))
    assert len(capped.faces) <= 12000
    tight = fm.smooth_head_body(head.head_mesh, _on(subdivisions=3, max_triangles=base))
    assert len(tight.faces) == base


def test_spec_override_turns_marking_on() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    spec["club"] = {
        **spec["club"],
        "face_marking": {"face_marking": True, "score_line_count": 5},
    }
    club = ca.assembly_from_spec(spec)
    assert club.face_marking.score_line_count == 5
    assert "face_lines" in ca.assembly_meshes(club)
