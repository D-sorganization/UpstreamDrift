"""Open eyes, shaped brows and a smooth hairline on the CC0 head (#15299).

Every bound is read from ``assets/head_cc0/face_settings.json``.
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import head_features, head_hair, head_mesh

pytestmark = pytest.mark.unit
CHECKS = head_features.face_settings("checks")
SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)


@pytest.fixture(scope="module")
def skin():
    return head_mesh.load_head_asset().skin


@pytest.fixture(scope="module")
def parts(skin):
    return {name: mesh for name, mesh, _ in head_features.face_parts(skin)}


def _hidden(target, occluders, elevation_deg: float) -> float:
    """Share of the target's vertices nearest the viewer that lie behind an occluder triangle,
    seen from ``elevation_deg`` above straight ahead (parallel rays)."""
    a = np.radians(elevation_deg)
    toward = np.array([np.cos(a), 0.0, np.sin(a)])
    plane = np.array([[0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]]).T
    ahead = target.vertices @ toward
    points = target.vertices[ahead >= np.quantile(ahead, CHECKS["view_hidden_share"])]
    hidden = np.zeros(len(points), dtype=bool)
    for occluder in occluders:
        tri = occluder.vertices[occluder.faces]
        flat, depth = tri @ plane, tri @ toward
        for i, p in enumerate(points):
            weights = head_features.barycentric(flat, p @ plane)
            inside = (weights >= 0.0).all(axis=1)
            behind = (weights * depth).sum(axis=1) > p @ toward + 1e-6
            hidden[i] |= bool((inside & behind).any())
    return float(hidden.mean())


def test_face_numbers_come_from_the_settings_file() -> None:
    raw = json.loads(head_features.SETTINGS_PATH.read_text(encoding="utf-8"))
    marks = head_features.load_face_landmarks()
    for name, value in raw["face"].items():
        assert np.allclose(getattr(marks, name), value), name
    crop = head_hair.load_hair_crop()
    for name, value in raw["hair"].items():
        assert np.allclose(getattr(crop, name), value), name
    with pytest.raises(ValueError):
        replace(marks, iris_half=(0.003, -0.02, 0.02))
    with pytest.raises(ValueError):
        replace(marks, eye_inset=float("nan"))
    with pytest.raises(ValueError):
        head_features.face_settings("no_such_section")


def test_open_eyes_are_seen_from_ahead_and_from_above(skin, parts) -> None:
    for side in "lr":
        cover = [skin, parts[f"lid_{side}"], parts[f"brow_{side}"]]
        iris, pupil = parts[f"iris_{side}"], parts[f"pupil_{side}"]
        for k, elevation in enumerate(CHECKS["view_elevations_deg"]):
            dark = _hidden(pupil, cover, elevation)
            assert dark <= CHECKS["max_pupil_hidden"][k], (side, elevation, dark)
            ring = _hidden(iris, cover, elevation)
            assert ring <= CHECKS["max_iris_hidden"][k], (side, elevation, ring)
        white = 1.0 - _hidden(parts[f"eye_{side}"], [skin, iris], 0.0)
        assert white >= CHECKS["min_white_seen"], (side, white)


def test_eye_layers_are_never_coincident_and_sit_behind_the_lids(skin, parts) -> None:
    marks = head_features.load_face_landmarks()
    for sign, side in ((1.0, "l"), (-1.0, "r")):
        at = np.array([[sign * marks.eye_y, marks.eye_z - marks.iris_drop]])
        layers = [parts[f"{n}_{side}"] for n in ("eye", "iris", "pupil")]
        depth = [float(head_features.front_depth(m, at)[0]) for m in layers]
        around = at + marks.eye_cover * np.array([[0.0, 1.0], [0.0, -1.0]])
        depth.append(float(head_features.front_depth(skin, around).max()))  # the lids
        assert np.diff(depth).min() >= CHECKS["min_layer_gap"], (side, depth)
        assert parts[f"eye_{side}"].volume() > 0.0


def test_brow_is_a_thin_arched_strip_that_tapers_outward(skin, parts) -> None:
    for sign, side in ((1.0, "l"), (-1.0, "r")):
        v = parts[f"brow_{side}"].vertices
        proud = v[:, 0] - head_features.front_depth(skin, v[:, 1:3])
        assert np.nanmax(proud) <= CHECKS["brow_max_proud"], side
        along = sign * v[:, 1]
        inner, outer = (
            along < np.quantile(along, 0.25),
            along > np.quantile(along, 0.75),
        )
        ratio = np.ptp(v[outer, 2]) / np.ptp(v[inner, 2])
        assert ratio <= CHECKS["brow_max_end_ratio"], (side, ratio)
        rings = [v[np.isclose(along, a), 2].mean() for a in np.unique(along.round(6))]
        assert max(rings) > max(rings[0], rings[-1]), "the brow arches upward"


def test_hairline_has_no_corners(skin) -> None:
    for part in ("left", "none"):
        turns = head_hair.hairline_turns_deg(skin, part, CHECKS["hairline_step_rad"])
        assert len(turns) > 20
        assert turns.max() <= CHECKS["hairline_max_turn_deg"], (part, turns.max())
    with pytest.raises(ValueError):
        head_hair.hairline_turns_deg(skin, "left", 0.0)


def test_refinement_is_a_setting_and_zero_leaves_the_scalp_as_it_was(skin) -> None:
    crop = head_hair.load_hair_crop()
    coarse = head_hair.build_short_crop(skin, "left", replace(crop, refine_splits=0))
    assert len(coarse.faces) == CHECKS["unrefined_hair_triangles"]
    fine = head_hair.build_short_crop(skin, "left", crop)
    assert crop.refine_splits > 0 and len(fine.faces) > 4 * len(coarse.faces)
    for bad in ({"refine_splits": -1}, {"refine_edge": -0.1}, {"refine_curve": 1.5}):
        with pytest.raises(ValueError):
            head_hair.build_short_crop(skin, "left", replace(crop, **bad))


def test_refined_hair_has_a_smooth_outline_and_resolved_comb_ridges(skin) -> None:
    crop = head_hair.load_hair_crop()
    plain = replace(crop, comb_amplitude=0.0)  # the shape under the combed relief
    step, bounds = CHECKS["outline_step_rad"], CHECKS["outline_max_turn_deg"]
    for axis, view in enumerate(CHECKS["outline_views"]):
        fine = head_hair.build_short_crop(skin, "none", plain)
        turn = head_hair.outline_turns_deg(fine, axis, step, view).max()
        assert turn <= bounds[axis], (axis, turn)
        coarse = head_hair.build_short_crop(
            skin, "none", replace(plain, refine_splits=0)
        )
        assert head_hair.outline_turns_deg(coarse, axis, step, view).max() > 2.0 * turn
    ridge = head_hair.comb_ridge_vertices(skin, "left")
    assert ridge >= CHECKS["comb_min_ridge_vertices"], ridge
    with pytest.raises(ValueError):
        head_hair.outline_turns_deg(fine, 2, step, view)
    with pytest.raises(ValueError):
        head_hair.outline_turns_deg(fine, 0, step, (0.0, 0.5, 0.0, step))


def test_face_wears_the_body_skin_material_and_hair_is_smooth_shaded() -> None:
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python import full_body_mjcf
    from src.shared.python.model_appearance import defaults

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    spec["subject"]["body_visual_backend"] = "skinned_cc0"
    xml, _ = full_body_mjcf.export_full_body_mjcf(
        json.dumps(spec).encode(), appearance=defaults.default_appearance()
    )
    root = ET.fromstring(xml)  # noqa: S314 - XML built by this repository
    geoms = {g.get("name"): g for g in root.iter("geom")}
    body = next(s for s in root.iter("skin") if s.get("name") == "body_skin_skin")
    assert geoms["visual_head_skin_mesh"].get("material") == body.get("material")
    meshes = {m.get("name"): m for m in root.iter("mesh")}
    assert meshes["vmesh_head_hair_mesh"].get("smoothnormal") == "true"
    assert meshes["vmesh_head_skin_mesh"].get("smoothnormal") is None
