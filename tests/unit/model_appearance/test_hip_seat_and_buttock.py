"""Where the hip joint sits in the drawn thigh, and the buttock (#15299).

Visual only; the joint is not moved. Depths are measured in the plane square
to the hip-knee line, ahead of and behind that line, in a 20 mm strip through
the joint. The CC0 body's own leg pivot is a rig pivot, not a hip joint
centre, so only its whole depths are compared, never its front and back.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from src.shared.python.model_appearance import geometry, library
from src.shared.python.model_appearance.body_meshes import body_meshes
from src.shared.python.model_appearance.skinned_body import spec_body_poses
from tests.unit.model_appearance.test_leg_shape import ADDRESS, DOC, SPEC

pytestmark = pytest.mark.unit

STATURE_M = SPEC["subject"]["stature_m"]
STRIP_M = 0.020
# Front depth over whole depth of the thigh at the groin: the femoral head is
# near the middle of the thigh there (it was 0.60-0.62, the joint too far back).
SEAT_BAND = (0.45, 0.56)
GROIN_FRACS = (0.08, 0.16)
# Whole depth of the CC0 athletic male thigh over stature (1.82 m body).
THIGH_DEPTH_OVER_STATURE = {0.25: 0.192 / 1.82, 0.34: 0.189 / 1.82, 0.45: 0.187 / 1.82}
THIGH_DEPTH_TOLERANCE = 0.06
# Front of the upper thigh against the front of the pelvis at the hip joints,
# over stature. On the CC0 body the thigh is 5 mm behind it; it was 31 mm ahead.
THIGH_AHEAD_BAND = (-0.010 / 1.82, 0.018 / 1.82)
UPPER_FRACS = (0.08, 0.16, 0.25)
# The buttock before this change (skin, 1.71 m subject): how far it reaches
# behind the hip joints, and the girth of the pelvis at that level.
BUTTOCK_REACH_BEFORE_M = 0.0983
BUTTOCK_GIRTH_BEFORE_M = 0.8515
BUTTOCK_GAIN_BAND = (1.05, 1.10)  # owner, 2026-11-19: 5-10 % bigger
# Largest change of the outline between stations 35 mm apart across the hip,
# trousers on: the back standing and at address, the front standing (bent at
# the hip the belly is ahead of the thigh). The CC0 gluteal fold is 10 mm.
JOIN_STEP_MAX_M = 0.015
JOIN_STATIONS_M = (-0.035, 0.0, 0.035, 0.070, 0.105)
# A back belt marker of the driver capture on the 1.82 m golfer: behind and
# beside the hip joint midpoint and above it (subject preset marker seeds).
BACK_MARKER_OVER_STATURE = np.array([0.102, 0.0845, 0.110]) / 1.82


def _hip_line(q: dict[str, float] | None) -> tuple[np.ndarray, ...]:
    poses = spec_body_poses(SPEC, q)
    hip, knee = poses["femur_r"][:3, 3], poses["tibia_r"][:3, 3]
    pelvis = next(pose for body, pose in poses.items() if body.endswith("LowerTorso"))
    axis = (knee - hip) / np.linalg.norm(knee - hip)
    front = pelvis[:3, 0] - (pelvis[:3, 0] @ axis) * axis
    return hip, knee, axis, front / np.linalg.norm(front)


def _depths(
    along_m: float, bodies: tuple[str, ...], label: str, q: dict[str, float] | None
) -> tuple[float, float]:
    """(ahead, behind) of the hip-knee line, m, ``along_m`` from the hip joint."""
    hip, _, axis, front = _hip_line(q)
    poses = spec_body_poses(SPEC, q)
    found = []
    for mesh in body_meshes(SPEC, DOC):
        if mesh.label != label or not mesh.body.endswith(bodies):
            continue
        pose = poses[mesh.body]
        verts = mesh.mesh_in_body_frame.vertices @ pose[:3, :3].T + pose[:3, 3] - hip
        signed, faces = (
            verts @ axis - along_m,
            np.asarray(mesh.mesh_in_body_frame.faces),
        )
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = signed[faces[:, i]], signed[faces[:, j]]
            hit = a * b < 0
            t = (a[hit] / (a[hit] - b[hit]))[:, None]
            found.append(verts[faces[hit, i]] * (1 - t) + verts[faces[hit, j]] * t)
    points = np.vstack(found)
    strip = np.abs(points @ np.cross(axis, front)) < STRIP_M
    ahead = points[strip] @ front
    return float(ahead.max()), float(-ahead.min())


def _thigh(frac: float) -> tuple[float, float]:
    hip, knee, _, _ = _hip_line(None)
    return _depths(
        frac * float(np.linalg.norm(knee - hip)), ("femur_r",), "skin0", None
    )


@pytest.mark.parametrize("frac", GROIN_FRACS)
def test_hip_joint_sits_mid_depth_of_the_thigh_at_the_groin(frac: float) -> None:
    ahead, behind = _thigh(frac)
    assert SEAT_BAND[0] <= ahead / (ahead + behind) <= SEAT_BAND[1]


@pytest.mark.parametrize("frac", sorted(THIGH_DEPTH_OVER_STATURE))
def test_thigh_keeps_the_depth_of_the_measured_body(frac: float) -> None:
    want = THIGH_DEPTH_OVER_STATURE[frac] * STATURE_M
    assert sum(_thigh(frac)) == pytest.approx(want, rel=THIGH_DEPTH_TOLERANCE)


@pytest.mark.parametrize("frac", UPPER_FRACS)
def test_upper_thigh_does_not_stand_ahead_of_the_pelvis(frac: float) -> None:
    pelvis_front, _ = _depths(0.0, ("LowerTorso",), "skin0", None)
    ahead = (_thigh(frac)[0] - pelvis_front) / STATURE_M
    assert THIGH_AHEAD_BAND[0] <= ahead <= THIGH_AHEAD_BAND[1], ahead * STATURE_M


def _pelvis_at_the_hips() -> np.ndarray:
    """Outline of the pelvis skin at the hip joints: columns (forward, lateral), m."""
    body, pose = next(
        (b, p) for b, p in spec_body_poses(SPEC).items() if b.endswith("LowerTorso")
    )
    (mesh,) = [
        m for m in body_meshes(SPEC, DOC) if (m.body, m.label) == (body, "skin0")
    ]
    verts = mesh.mesh_in_body_frame.vertices @ pose[:3, :3].T + pose[:3, 3]
    hip = spec_body_poses(SPEC)["femur_r"][:3, 3]
    ring = verts[np.abs(verts[:, 2] - hip[2]) < 1e-4]
    return (ring - hip)[:, :2]


def test_buttock_is_five_to_ten_percent_bigger() -> None:
    ring = _pelvis_at_the_hips()
    reach = float(-ring[:, 0].min()) / BUTTOCK_REACH_BEFORE_M
    girth = float(ConvexHull(ring).area) / BUTTOCK_GIRTH_BEFORE_M
    assert BUTTOCK_GAIN_BAND[0] <= reach <= BUTTOCK_GAIN_BAND[1], reach
    assert BUTTOCK_GAIN_BAND[0] <= girth <= BUTTOCK_GAIN_BAND[1], girth
    assert float(ring[:, 0].max()) < float(-ring[:, 0].min())  # the mass is behind


def test_back_belt_markers_stay_outside_the_trousers() -> None:
    behind, beside, above = BACK_MARKER_OVER_STATURE * STATURE_M
    body, pose = next(
        (b, p) for b, p in spec_body_poses(SPEC).items() if b.endswith("LowerTorso")
    )
    (mesh,) = [
        m for m in body_meshes(SPEC, DOC) if (m.body, m.label) == (body, "garment0")
    ]
    verts = mesh.mesh_in_body_frame.vertices @ pose[:3, :3].T + pose[:3, 3]
    verts = verts - spec_body_poses(SPEC)["femur_r"][:3, 3] * [1.0, 0.0, 1.0]
    near = (np.abs(verts[:, 2] - above) < 0.015) & (np.abs(verts[:, 1]) > beside)
    assert -verts[near, 0].min() < behind, (-verts[near, 0].min(), behind)


@pytest.mark.parametrize("pose", [None, ADDRESS], ids=["standing", "address"])
def test_pelvis_meets_the_thigh_without_a_step(pose: dict[str, float] | None) -> None:
    bodies = ("LowerTorso", "femur_r")
    outline = np.array([_depths(s, bodies, "garment0", pose) for s in JOIN_STATIONS_M])
    steps = np.abs(np.diff(outline, axis=0))
    worst = steps[:, 1].max() if pose else steps.max()
    assert worst <= JOIN_STEP_MAX_M, np.round(1e3 * outline)


def test_pelvis_back_bulge_is_behind_only_and_checked() -> None:
    base = {
        "centre": np.zeros(3),
        "up": np.array([0.0, 0.0, 1.0]),
        "width": np.array([0.0, 1.0, 0.0]),
        "half_width": 0.18,
        "half_depth": 0.09,
        "half_up": 0.3,
        "half_down": 0.08,
        "top": 0.15,
    }
    back = {"back": np.array([-1.0, 0.0, 0.0]), "back_scale": 1.1}
    plain = geometry.split_ellipsoid_mesh(geometry.SplitEllipsoid(**base)).vertices
    full = geometry.split_ellipsoid_mesh(
        geometry.SplitEllipsoid(**base, **back)
    ).vertices
    assert full[:, 0].max() == pytest.approx(plain[:, 0].max())
    assert full[:, 0].min() == pytest.approx(1.1 * plain[:, 0].min(), rel=1e-3)
    cap = full[full[:, 2] > 0.15 - 1e-9]
    assert -cap[:, 0].min() == pytest.approx(
        cap[:, 0].max(), rel=1e-6
    )  # belt as before
    assert np.abs(full[:, 1]).max() == pytest.approx(0.18)
    for bad in (
        {"back_scale": 1.1},
        {**back, "back_scale": 0.0},
        {**back, "back_square": 1.0},
    ):
        with pytest.raises(ValueError, match="back"):
            geometry.split_ellipsoid_mesh(geometry.SplitEllipsoid(**base, **bad))
    assert library.PELVIS_BACK_SCALE > 1.0
