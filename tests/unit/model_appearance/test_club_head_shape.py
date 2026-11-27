"""Rounded driver body and blended hosel (#15150): geometry, pins, settings."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from src.shared.python.model_appearance import club_head_mesh as chm
from src.shared.python.model_appearance import club_head_shape as shape
from src.shared.python.model_appearance.club_assembly import (
    SHAFT_TIP_RADIUS_FRACTION,
    ClubAssembly,
    clubface_centre,
    clubface_vector,
)

pytestmark = pytest.mark.unit

TOL = 1e-6
# Measured on origin/main before this change (head frame, then club frame).
MAIN_FACE_CENTRE = [0.04967519026432704, -0.00042933104545348464, 2.8196e-06]
MAIN_FACE_NORMAL = [0.9832549075627033, 0.18223552549889938, 0.0]
MAIN_SOLE_POINT = [0.04989740528621987, -0.030949764722691862, -0.04712]
MAIN_HOSEL_TOP_M = 0.08800783786445467
MAIN_CLUB_VECTOR = [0.8382865995179801, -0.15108009768730235, 0.5238801209736601]
MAIN_CLUB_CENTRE = [0.03432207629844267, 0.06474816006907341, 0.01958150364412968]


def _unshaped(monkeypatch: pytest.MonkeyPatch) -> chm.ClubHeadMesh:
    """The driver as built on main: shaping switched off for this call only."""
    off = dataclasses.replace(shape.load_head_shape_settings(), shaped_club_types=())
    chm.load_club_head.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(chm, "load_head_shape_settings", lambda: off)
        raw = chm.load_club_head("driver")
    chm.load_club_head.cache_clear()
    return raw


def _sharpest_crease_deg(mesh, skip: np.ndarray) -> float:
    tri = mesh.vertices[mesh.faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-18)[:, None]
    owner: dict[tuple[int, int], list[int]] = {}
    for k, f in enumerate(mesh.faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            owner.setdefault((min(f[a], f[b]), max(f[a], f[b])), []).append(k)
    cos = [
        n[p[0]] @ n[p[1]] for p in owner.values() if len(p) == 2 and not skip[p].any()
    ]
    return float(np.degrees(np.arccos(np.clip(min(cos), -1.0, 1.0))))


def test_face_centre_normal_seat_and_impact_axes_match_main() -> None:
    head = chm.load_club_head("driver")
    assert np.allclose(chm.face_centre(head.head_mesh), MAIN_FACE_CENTRE, atol=TOL)
    nrm = chm.measured_face_normal(head.head_mesh)
    assert np.allclose(nrm, MAIN_FACE_NORMAL, atol=TOL)
    assert np.allclose(head.sole_point, MAIN_SOLE_POINT, atol=TOL)
    assert head.hosel_top_m == pytest.approx(MAIN_HOSEL_TOP_M, abs=TOL)
    club = ClubAssembly(
        "driver", 1.15, 0.265, 0.0065, face_roll_deg=-37.92, head_slide_m=0.0637
    )
    assert np.allclose(clubface_vector(club), MAIN_CLUB_VECTOR, atol=TOL)
    assert np.allclose(clubface_centre(club), MAIN_CLUB_CENTRE, atol=TOL)


def test_face_plate_is_pinned_and_other_club_types_are_untouched(monkeypatch) -> None:
    raw = _unshaped(monkeypatch)
    rounded = chm.load_club_head("driver")
    pins = shape._face_pin_vertices(
        raw.head_mesh,
        chm.measured_face_normal(raw.head_mesh),
        shape.load_head_shape_settings().face_pin_cos,
    )
    assert pins.sum() > 50
    assert np.array_equal(
        raw.head_mesh.vertices[pins], rounded.head_mesh.vertices[pins]
    )
    assert not np.array_equal(raw.head_mesh.vertices, rounded.head_mesh.vertices)
    iron = chm.load_club_head("iron7")
    plain = chm._weld(chm.triangles_from_tools(iron.library_name))
    assert np.array_equal(iron.head_mesh.vertices, plain.vertices)


def test_rounded_body_is_closed_outward_and_keeps_volume_and_ground(
    monkeypatch,
) -> None:
    raw = _unshaped(monkeypatch)
    head = chm.load_club_head("driver")
    assert head.head_mesh.volume() > 0.0
    assert head.head_mesh.volume() == pytest.approx(raw.head_mesh.volume(), rel=0.05)
    assert np.array_equal(head.head_mesh.faces, raw.head_mesh.faces)
    assert head.head_mesh.vertices[:, 1].min() >= head.sole_point[1] - 1e-9


def test_rounding_removes_the_creases_away_from_the_face(monkeypatch) -> None:
    raw = _unshaped(monkeypatch)
    head = chm.load_club_head("driver")
    nrm = chm.measured_face_normal(raw.head_mesh)
    skip = shape._face_pin_vertices(raw.head_mesh, nrm, 0.9)[raw.head_mesh.faces].any(1)
    before = _sharpest_crease_deg(raw.head_mesh, skip)
    after = _sharpest_crease_deg(head.head_mesh, skip)
    assert after < 0.6 * before and after < 40.0  # the rear cap edge is sharpest


def test_hosel_top_meets_the_shaft_end_at_the_shaft_tip_radius() -> None:
    head = chm.load_club_head("driver")
    s = shape.load_head_shape_settings()
    top = head.sole_point + head.hosel_top_m * head.shaft_direction
    verts = head.mesh.vertices[len(head.head_mesh.vertices) :]
    along = (verts - head.sole_point) @ head.shaft_direction
    radii = np.linalg.norm(
        verts[np.isclose(along, head.hosel_top_m, atol=1e-9)] - top, axis=1
    )
    assert len(radii) > s.hosel_sides
    assert np.allclose(radii[radii > 1e-9], s.hosel_top_radius_m, atol=1e-9)
    assert s.hosel_top_radius_m == pytest.approx(SHAFT_TIP_RADIUS_FRACTION * 0.0065)


def test_hosel_reaches_the_head_where_the_axis_leaves_it() -> None:
    head = chm.load_club_head("driver")
    s = shape.load_head_shape_settings()
    args = (head.head_mesh, head.sole_point, head.shaft_direction, head.hosel_top_m)
    contact = shape.axis_contact_param(*args, s.hosel_radius_m)
    assert 0.0 < contact < head.hosel_top_m
    at = head.sole_point + contact * head.shaft_direction
    assert (
        np.linalg.norm(head.head_mesh.vertices - at, axis=1).min() <= s.hosel_radius_m
    )
    with pytest.raises(ValueError, match="too short"):
        shape.hosel_stations(0.08, 0.088, s)


@pytest.mark.parametrize(
    "over",
    [
        {"smoothing_iterations": -1},
        {"smoothing_lambda": 1.0},
        {"smoothing_mu": -0.4},
        {"hosel_radius_m": 0.0},
        {"hosel_top_radius_m": 0.01},
    ],
)
def test_settings_validation(over) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(shape.load_head_shape_settings(), **over)


def test_settings_file_documents_every_key() -> None:
    raw = json.loads(shape.SETTINGS_PATH.read_text(encoding="utf-8"))
    names = sorted(f.name for f in dataclasses.fields(shape.HeadShapeSettings))
    assert sorted(k for k in raw if k != "sources") == names == sorted(raw["sources"])


def _club_dynamics(shaped: bool, monkeypatch) -> tuple[list[float], list[tuple]]:
    mujoco = pytest.importorskip("mujoco")
    from src.tools.native_viewer_export.backends.mujoco_worker import build_scene_xml

    spec = shape.SETTINGS_PATH.parents[4] / "docs/development/full_body_models"
    spec = spec / "full_body_spec_anthro_driver.json"
    cfg = shape.load_head_shape_settings()
    cfg = dataclasses.replace(cfg, shaped_club_types=("Driver",) if shaped else ())
    chm.load_club_head.cache_clear()
    monkeypatch.setattr(chm, "load_head_shape_settings", lambda: cfg)
    m = mujoco.MjModel.from_xml_string(build_scene_xml(spec.read_bytes())[0])
    chm.load_club_head.cache_clear()
    i = next(k for k in range(m.nbody) if m.body(k).name.endswith("Clubface Vector"))
    body = [m.body_mass[i], *m.body_inertia[i], *m.body_ipos[i], *m.body_iquat[i]]
    flags = zip(m.geom_bodyid, m.geom_contype, m.geom_conaffinity, strict=True)
    return [float(x) for x in body], [(c, a) for b, c, a in flags if b == i]


def test_shaped_head_is_visual_only_in_the_mujoco_model(monkeypatch) -> None:
    plain, plain_geoms = _club_dynamics(False, monkeypatch)
    shaped, shaped_geoms = _club_dynamics(True, monkeypatch)
    assert shaped == plain  # explicit <inertial>: mass, inertia, ipos, iquat
    assert shaped_geoms == plain_geoms and set(shaped_geoms) == {(0, 0)}  # no contact
