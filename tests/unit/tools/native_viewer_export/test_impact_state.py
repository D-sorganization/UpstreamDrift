"""Impact factors are read where the face reaches the ball (#15150)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.impact_parameters import ClubheadSeries
from src.tools.native_viewer_export import impact_card, impact_state
from src.tools.native_viewer_export.ball import BallPlacement

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REPO = Path(__file__).resolve().parents[4]
SPEC = REPO / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
FIXTURE = REPO / "tests/fixtures/club_face/swing_q_driver.npz"

RATE_HZ = 360.0
RADIUS_M = 0.021335
ARC_RADIUS_M = 1.2  # hub to face centre
TURN_RAD_S = 42.0  # 50.4 m/s at the face centre
FACE_OPEN_DEG = 6.0  # face normal ahead of the arc tangent, about the vertical
CONTACT_S = 0.5 + 0.4 / RATE_HZ  # 0.4 of a sample after sample 180
STRIKE_SLOWDOWN = 0.7  # head speed kept after the strike


def _arc_state(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face centre, velocity and normal of a head on a horizontal circle."""
    turn = TURN_RAD_S * (t - CONTACT_S)
    centre = ARC_RADIUS_M * np.array([math.sin(turn), -math.cos(turn), 0.0])
    velocity = ARC_RADIUS_M * TURN_RAD_S * np.array([math.cos(turn), math.sin(turn), 0])
    aim = turn + math.radians(FACE_OPEN_DEG)
    return centre, velocity, np.array([math.cos(aim), math.sin(aim), 0.0])


def _struck_series() -> tuple[ClubheadSeries, BallPlacement]:
    """Arc swing sampled at 360 Hz whose head slows after it meets the ball.

    The ball touches the face centre at ``CONTACT_S``; later samples lag as a
    struck head does, so any difference across the contact reads too slow.
    """
    times = np.arange(0.0, 1.0, 1.0 / RATE_HZ)
    lagged = np.where(
        times > CONTACT_S, CONTACT_S + STRIKE_SLOWDOWN * (times - CONTACT_S), times
    )
    states = [_arc_state(float(t)) for t in lagged]
    centre = np.array([s[0] for s in states])
    normal = np.array([s[2] for s in states])
    series = ClubheadSeries(
        times_s=times,
        face_center_m=centre,
        velocity_mps=np.gradient(centre, times, axis=0),
        face_normal=normal,
        toe_axis=np.cross(normal, [0.0, 0.0, 1.0]),
        grip_axis=np.tile([0.0, 0.0, 1.0], (len(times), 1)),
    )
    at, _, face = _arc_state(CONTACT_S)
    return series, BallPlacement(tuple(at + RADIUS_M * face), RADIUS_M, "strike")


def test_card_reads_the_state_at_contact_not_a_sample_early() -> None:
    series, ball = _struck_series()
    card = impact_card.card_from_series(series, 180 / RATE_HZ, ball)
    rows = {row.key: row for row in card.rows}
    # face to path needs no target line: the face is 6 deg ahead of the path
    assert rows["face_to_path_deg"].value == pytest.approx(-FACE_OPEN_DEG, abs=0.3)
    speed_mph = ARC_RADIUS_M * TURN_RAD_S * 2.2369362920544
    assert rows["clubhead_speed"].value == pytest.approx(speed_mph, rel=0.005)
    assert card.impact_time_s == pytest.approx(CONTACT_S, abs=2e-5)
    assert card.extras["impact_state"] == impact_card.STATE_AT_CONTACT
    # a ball carried back along a straight line reads the arc as a 19 mm miss
    place = rows["impact_location"]
    assert place.value == pytest.approx(0.0, abs=0.5), place.reason
    assert float(place.note.split()[1]) == pytest.approx(0.0, abs=0.5)


def test_face_and_path_are_those_of_the_contact_instant() -> None:
    series, ball = _struck_series()
    centre = np.asarray(ball.position_m)
    moved = impact_state.series_at_contact(series, centre, RADIUS_M, 180 / RATE_HZ)
    assert moved is not None
    at_contact, index = moved
    _, velocity, normal = _arc_state(CONTACT_S)
    close = 0.005 * ARC_RADIUS_M * TURN_RAD_S
    assert at_contact.velocity_mps[index] == pytest.approx(velocity, abs=close)
    face = at_contact.face_normal[index]
    # one sample early would be 42 rad/s / 360 Hz = 6.7 deg
    assert math.degrees(math.acos(float(face @ normal))) < 0.2
    # postconditions: the face plane touches the ball; earlier samples are kept
    gap = (centre - at_contact.face_center_m[index]) @ face - RADIUS_M
    assert abs(gap) < 1e-6
    assert np.array_equal(at_contact.face_center_m[:index], series.face_center_m[:181])


def test_no_face_crossing_in_the_window_gives_none() -> None:
    series, _ = _struck_series()
    hub = (0.0, 0.0, 0.0)  # the face plane keeps one distance from the hub
    assert impact_state.series_at_contact(series, hub, RADIUS_M, 0.1) is None


def test_series_at_contact_rejects_bad_input() -> None:
    series, ball = _struck_series()
    with pytest.raises(TypeError):
        impact_state.series_at_contact(object(), ball.position_m, RADIUS_M, 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="3-vector"):
        impact_state.series_at_contact(series, (0.0, 1.0), RADIUS_M, 0.5)
    with pytest.raises(ValueError, match="radius_m"):
        impact_state.series_at_contact(series, ball.position_m, 0.0, 0.5)
    with pytest.raises(ValueError, match="impact_time_s"):
        impact_state.series_at_contact(series, ball.position_m, RADIUS_M, math.nan)


@pytest.mark.parametrize(
    "change",
    [
        ("fit_order", 0),
        ("fit_order", 6),
        ("stencil_samples", 2.5),
        ("search_window_s", 0.0),
    ],
)
def test_settings_refuse_values_outside_their_range(change: tuple) -> None:
    good = impact_state.load_impact_state_settings()
    with pytest.raises(ValueError):
        dataclasses.replace(good, **dict([change]))


# --- the face of the model -----------------------------------------------------------


def _model_and_address(club_changes: dict | None = None):
    pytest.importorskip("mujoco")
    if not (SPEC.exists() and FIXTURE.exists()):
        pytest.skip("capture-A specification or club-face fixture missing")
    from src.tools.native_viewer_export.club_pose import ClubPoseModel

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    spec["club"] = {**(spec.get("club") or {}), **(club_changes or {})}
    q0 = np.load(FIXTURE)["q"].astype(float)[0]
    coords = dict(zip(spec["coordinate_order"], map(float, q0), strict=True))
    return ClubPoseModel(json.dumps(spec).encode("utf-8")), coords


def test_target_line_does_not_turn_with_the_head_roll_setting() -> None:
    plain, coords = _model_and_address()
    rolled, _ = _model_and_address({"face_roll_deg": -49.5})
    assert not np.allclose(
        plain.face_world(coords).normal, rolled.face_world(coords).normal, atol=0.05
    )
    assert rolled.address_square_normal_world(coords) == pytest.approx(
        plain.face_world(coords).normal, abs=1e-9
    )
    series = impact_card.ModelClubSeries(
        times_s=np.arange(3) / RATE_HZ,
        face_center_m=np.zeros((3, 3)),
        velocity_mps=np.zeros((3, 3)),
        face_normal=np.tile(rolled.face_world(coords).normal, (3, 1)),
        target_normal=tuple(plain.face_world(coords).normal),
    )
    flat = plain.face_world(coords).normal * [1.0, 1.0, 0.0]
    target = impact_card.address_target_frame(series).target_dir
    assert target == pytest.approx(flat / np.linalg.norm(flat), abs=1e-9)
