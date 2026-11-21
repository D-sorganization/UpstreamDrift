"""The impact card reads toe and high along the head's own axes (#15150).

The toe axis is the head's heel-to-toe line (head frame +z) projected into the
face plane; the high axis is ``toe x normal`` and points to the crown. Across
the shaft those lines lean 90 deg minus the lie (34 deg on the driver), which
turned the reported direction of a strike offset by as much.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.model_appearance.club_head_mesh import (
    face_roll_rotation,
    club_face_normal,
    head_to_club_rotation,
    load_club_head,
)
from src.tools.native_viewer_export import impact_card
from src.tools.native_viewer_export.ball import BallPlacement
from src.tools.native_viewer_export.club_pose import face_axes

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

TOL_M = 1e-4  # 0.1 mm
RADIUS_M = 0.02135
ROLL_DEG = 7.0
ROT_Z = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
)  # club body turned 90 deg about world z


def _head_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Club-body normal, toe line, crown line and the body 4x4 of a driver."""
    head = load_club_head("driver")
    roll = face_roll_rotation(ROLL_DEG)
    to_club = roll @ head_to_club_rotation(head)
    body = np.eye(4)
    body[:3, :3] = ROT_Z
    body[:3, 3] = (0.3, -0.2, 0.9)
    return (
        club_face_normal(head, ROLL_DEG),
        to_club @ np.array([0.0, 0.0, 1.0]),
        to_club @ np.array([0.0, 1.0, 0.0]),
        body,
    )


def test_face_axes_are_the_head_toe_line_and_crown_direction() -> None:
    normal_b, toe_b, crown_b, body = _head_axes()
    face = face_axes(body, np.zeros(3), normal_b, toe_b)
    rot = body[:3, :3]
    assert np.linalg.norm(face.toe) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(face.grip) == pytest.approx(1.0, abs=1e-12)
    assert face.toe @ face.normal == pytest.approx(0.0, abs=1e-12)
    assert face.grip @ face.normal == pytest.approx(0.0, abs=1e-12)
    assert face.toe @ face.grip == pytest.approx(0.0, abs=1e-12)
    # toe is the head toe line with its normal part removed
    want = rot @ toe_b - (rot @ toe_b @ face.normal) * face.normal
    assert face.toe == pytest.approx(want / np.linalg.norm(want), abs=1e-12)
    # high points to the crown, not across the shaft
    assert face.grip @ (rot @ crown_b) > 0.8
    assert np.cross(face.toe, face.normal) == pytest.approx(face.grip, abs=1e-12)


def test_card_reports_a_ball_toward_toe_and_crown_with_sign_and_size() -> None:
    normal_b, toe_b, _, body = _head_axes()
    face = face_axes(body, np.array([0.1, 0.0, 0.2]), normal_b, toe_b)
    toe_mm, high_mm = 12.0, 7.0
    contact = face.centre_m + 1e-3 * (
        toe_mm * face.toe + high_mm * np.cross(face.toe, face.normal)
    )
    n = 200
    t = np.arange(n) * 0.001
    from src.shared.python.impact_parameters import ClubheadSeries

    series = ClubheadSeries(
        times_s=t,
        face_center_m=face.centre_m + np.outer(40.0 * t, face.normal),
        velocity_mps=np.tile(40.0 * face.normal, (n, 1)),
        face_normal=np.tile(face.normal, (n, 1)),
        toe_axis=np.tile(face.toe, (n, 1)),
        grip_axis=np.tile(face.grip, (n, 1)),
    )
    idx = 149
    shift = series.face_center_m[idx] - face.centre_m
    ball = BallPlacement(
        tuple(contact + shift + RADIUS_M * face.normal), RADIUS_M, "strike"
    )
    card = impact_card.card_from_series(series, 0.15, ball)
    row = next(r for r in card.rows if r.key == "impact_location")
    assert row.value is not None, row.reason
    assert row.value == pytest.approx(toe_mm, abs=TOL_M * 1e3)
    assert float(row.note.split()[1]) == pytest.approx(high_mm, abs=TOL_M * 1e3)


def test_toe_axis_must_not_lie_along_the_normal() -> None:
    normal_b, _, _, body = _head_axes()
    with pytest.raises(ValueError, match="parallel"):
        face_axes(body, np.zeros(3), normal_b, normal_b)


def test_driver_head_toe_line_leans_off_the_shaft_cross_line() -> None:
    # documents the size of the old error: the shaft-derived line is not the toe
    normal_b, toe_b, _, _ = _head_axes()
    old = np.cross(normal_b, [0.0, -1.0, 0.0])
    old /= np.linalg.norm(old)
    toe = toe_b - (toe_b @ normal_b) * normal_b
    toe /= np.linalg.norm(toe)
    angle = math.degrees(math.acos(abs(float(old @ toe))))
    assert 30.0 < angle < 40.0


def test_model_face_toe_is_the_head_toe_line_for_the_spec_club() -> None:
    import json
    from pathlib import Path

    pytest.importorskip("mujoco")
    repo = Path(__file__).resolve().parents[4]
    spec_path = (
        repo / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    )
    fixture = repo / "tests/fixtures/club_face/swing_q_driver.npz"
    if not (spec_path.exists() and fixture.exists()):
        pytest.skip("specification or club-face fixture missing")
    from src.shared.python.model_appearance.club_assembly import assembly_from_spec
    from src.tools.native_viewer_export.club_pose import ClubPoseModel

    spec_bytes = spec_path.read_bytes()
    model = ClubPoseModel(spec_bytes)
    order = json.loads(spec_bytes)["coordinate_order"]
    q0 = np.load(fixture)["q"].astype(float)[0]
    coords = dict(zip(order, map(float, q0), strict=True))
    face = model.face_world(coords)
    rot = model.body_world(coords)[:3, :3]
    head = load_club_head(model.head_alias)
    club = assembly_from_spec(model.spec)
    assert club is not None
    to_club = face_roll_rotation(club.face_roll_deg) @ head_to_club_rotation(head)
    toe = rot @ to_club @ np.array([0.0, 0.0, 1.0])
    toe = toe - (toe @ face.normal) * face.normal
    assert face.toe == pytest.approx(toe / np.linalg.norm(toe), abs=1e-9)
    assert face.grip @ (rot @ to_club @ np.array([0.0, 1.0, 0.0])) > 0.7
