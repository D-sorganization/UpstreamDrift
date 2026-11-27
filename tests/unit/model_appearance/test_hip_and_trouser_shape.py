"""Pelvis width at the belt and the drape of the trouser legs (#15299, epic #15150).

Visual only. The belt markers of the driver capture sit on the skin, so the
drawn pelvis passes just inside them; trousers hang off the leg with ease.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from src.shared.python.model_appearance import geometry, library
from src.shared.python.model_appearance.body_meshes import body_meshes
from src.shared.python.model_appearance.cross_section import body_sections
from src.shared.python.model_appearance.skinned_body import spec_body_poses
from tests.unit.model_appearance.test_leg_shape import DOC, SPEC, _section, _world

pytestmark = pytest.mark.unit

# The golfer of the capture: a tour-average male (subject preset).
GOLFER_STATURE_M = 1.82
# data/C3D_TA_Driver.c3d, address frame: WaistLeft to WaistRight 344.9 mm
# (336-354 mm over the swing); the belt is rigid to 0.5 mm.
BELT_MARKER_HALF_SPREAD_M = 0.1725
# A marker centre sits at most this far outside the cloth (belt, base, radius).
MARKER_STANDOFF_MAX_M = 0.020
# Half width over stature of the CC0 athletic male body at the belt line
# (0.18-0.24 of hip joint to C7): 156 mm at 1.82 m.
BELT_HALF_WIDTH_OVER_STATURE = 0.156 / 1.82
BELT_WIDTH_TOLERANCE = 0.05
# The waistband against the shirt hem it meets, half widths: the shirt may
# overhang it by the first, the band may stand proud of the shirt by the second.
WAISTBAND_UNDER_MAX_M, WAISTBAND_PROUD_MAX_M = 0.005, 0.012
HEM_BAND_M = 0.04  # the shirt's lowest rings
# Trouser girth over skin girth where the cloth hangs free: at the knee and the
# calf, and at the ankle, where the leg of a trouser is far wider than the skin.
DRAPE_BAND = (1.18, 1.32)
HEM_BAND = (1.32, 1.50)
HEM_FLARE_MAX = 1.03  # hem girth over the girth at the ankle
THIGH_FIT_MAX = 1.10  # the upper thigh stays fitted: the thighs must clear


def _golfer() -> dict[str, Any]:
    spec = copy.deepcopy(SPEC)
    spec["subject"]["stature_m"] = GOLFER_STATURE_M
    return spec


def _edge_half_width(spec: dict[str, Any], key: str, label: str, top: bool) -> float:
    """Half width (m) of the top or bottom 2 mm of the ``key`` body's ``label``."""
    body = next(b for b in body_sections(spec) if b.endswith(key))
    section = body_sections(spec)[body]
    (mesh,) = [m for m in body_meshes(spec, DOC) if (m.body, m.label) == (body, label)]
    verts = mesh.mesh_in_body_frame.vertices
    height = verts @ np.cross(section.depth_axis, section.ml_axis)
    edge = height > height.max() - 2e-3 if top else height < height.min() + 2e-3
    return float(np.ptp(verts[edge] @ section.ml_axis)) / 2.0


@pytest.mark.parametrize("stature", [SPEC["subject"]["stature_m"], GOLFER_STATURE_M])
def test_pelvis_is_as_wide_as_the_measured_body_at_the_belt(stature: float) -> None:
    spec = _golfer()
    spec["subject"]["stature_m"] = stature
    got = _edge_half_width(spec, "LowerTorso", "skin0", top=True)
    want = BELT_HALF_WIDTH_OVER_STATURE * stature
    assert got == pytest.approx(want, rel=BELT_WIDTH_TOLERANCE)


def test_belt_markers_sit_just_outside_the_trousers() -> None:
    cloth = _edge_half_width(_golfer(), "LowerTorso", "garment0", top=True)
    standoff = BELT_MARKER_HALF_SPREAD_M - cloth
    assert 0.0 <= standoff <= MARKER_STANDOFF_MAX_M, f"{1e3 * standoff:.1f} mm"


def test_waistband_meets_the_shirt_hem() -> None:
    spec = SPEC  # the built subject: trunk and pelvis sized for one stature
    band = _edge_half_width(spec, "LowerTorso", "garment0", top=True)
    poses = spec_body_poses(spec)
    shirt = next(
        m
        for m in body_meshes(spec, DOC)
        if m.body.endswith("COMRod") and m.label == "garment0"
    )
    pose = poses[shirt.body]
    verts = shirt.mesh_in_body_frame.vertices @ pose[:3, :3].T + pose[:3, 3]
    hem = verts[:, 2] < verts[:, 2].min() + HEM_BAND_M
    proud = band - float(np.ptp(verts[hem, 1])) / 2.0
    assert -WAISTBAND_UNDER_MAX_M <= proud <= WAISTBAND_PROUD_MAX_M, proud


def _drape(body: str, frac: float) -> float:
    cloth, skin = (_section(body, frac, label) for label in ("garment0", "skin0"))
    return float(ConvexHull(cloth).area / ConvexHull(skin).area)


@pytest.mark.parametrize(
    ("body", "frac", "band"),
    [
        ("femur_r", 0.95, DRAPE_BAND),
        ("tibia_r", 0.28, DRAPE_BAND),
        ("tibia_r", 0.84, HEM_BAND),
    ],
)
def test_trouser_leg_hangs_with_ease_below_the_thigh(
    body: str, frac: float, band: tuple[float, float]
) -> None:
    assert band[0] <= _drape(body, frac) <= band[1]


def test_trouser_leg_narrows_from_the_calf_and_does_not_flare_at_the_hem() -> None:
    calf, mid, low, ankle, hem = (
        float(ConvexHull(_section("tibia_r", f, "garment0")).area)
        for f in (0.28, 0.5, 0.7, 0.84, 0.95)
    )
    assert calf > mid > low > ankle
    assert hem <= HEM_FLARE_MAX * ankle


def test_trousers_stay_fitted_on_the_upper_thigh() -> None:
    assert _drape("femur_r", 0.35) <= THIGH_FIT_MAX
    fits = [_drape("femur_r", f) for f in (0.35, 0.6, 0.8, 0.95)]
    assert fits == sorted(fits), fits  # the ease grows toward the knee, no step


def test_trouser_legs_do_not_interpenetrate_standing() -> None:
    legs = _world("garment0")
    for right, left in (("femur_r", "femur_l"), ("tibia_r", "tibia_l")):
        assert legs[left][0][:, 1].min() >= legs[right][0][:, 1].max(), right


def test_only_full_length_trousers_drape() -> None:
    assert library.garment_drape("golf_polo_trousers", "shin") != library.NO_DRAPE
    assert library.garment_drape("golf_polo_shorts", "thigh") == library.NO_DRAPE
    assert library.garment_drape("golf_polo_trousers", "torso") == library.NO_DRAPE
    with pytest.raises(ValueError, match="clothing"):
        library.garment_drape("kilt", "thigh")


def test_loft_drape_scales_the_radius_from_its_start_to_the_far_end() -> None:
    start, end = np.zeros(3), np.array([0.0, 0.0, 0.4])
    options = {"capsule_ends": True, "rings": 30}
    plain = geometry.lofted_segment(start, end, 0.05, geometry.LoftOptions(**options))
    loose = geometry.lofted_segment(
        start, end, 0.05, geometry.LoftOptions(drape=(1.0, 1.2, 0.5), **options)
    )
    for mesh, far in ((plain, 0.05), (loose, 0.06)):
        radius = np.linalg.norm(mesh.vertices[:, :2], axis=1)
        height = mesh.vertices[:, 2]
        assert radius[(height >= 0.0) & (height <= 0.2)] == pytest.approx(0.05)
        assert radius[height > 0.2].max() == pytest.approx(far, abs=5e-4)
    for bad in ((0.0, 1.2, 0.5), (1.0, 1.2, 1.0), (1.0, float("nan"), 0.0)):
        with pytest.raises(ValueError, match="drape"):
            geometry.lofted_segment(start, end, 0.05, geometry.LoftOptions(drape=bad))
