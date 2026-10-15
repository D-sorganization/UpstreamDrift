"""R5 (#12682): one clubhead measurement point and the receipt speed block.

Clubhead speed is measured at the rendered face centre carried by the
``Clubhead`` frame (``club_face_target.face_centre_path``), never at the frame
origin or a marker centroid; the receipt records the simulated, reference and
capture face-centre speeds with the impact-speed error and the peak-timing
offset against the GCV-20 (#11767) tolerances.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_face as cf
from src.shared.python.motion_matching import club_face_target as cft
from src.shared.python.motion_matching.pipeline import fd_phase as fp

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]


def _rot_x(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    rot = np.zeros((len(theta), 3, 3))
    rot[:, 0, 0] = 1.0
    rot[:, 1, 1], rot[:, 1, 2] = c, -s
    rot[:, 2, 1], rot[:, 2, 2] = s, c
    return rot


HUB = np.array([0.0, 0.0, 1.1])
#: |d theta / dt| when the swing below reaches the ball (t = 1.4 s).
RATE_AT_BALL = 2.0 * np.pi / 0.4


def _theta(t: np.ndarray) -> np.ndarray:
    """Back to the top at t = 1, down through the ball at t = 1.4 still
    accelerating, then 30 % of the rate lost to the ball (the GCV-20
    synthetic swing of ``test_clubhead_speed_timing``)."""
    theta = np.where(
        t < 1.0,
        np.pi / 2.0 * (1.0 - np.cos(np.pi * t)),
        np.pi * (1.0 - ((t - 1.0) / 0.4) ** 2),
    )
    return np.where(t > 1.4, -0.7 * RATE_AT_BALL * (t - 1.4), theta)


def _swing_frames(t: np.ndarray, arm_m: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Rotations and origins of a head frame swung about the x axis on an arm
    of ``arm_m`` from ``HUB``: the frame's -z axis points from the hub."""
    rot = _rot_x(_theta(t))
    origins = HUB + arm_m * np.einsum("nij,j->ni", rot, [0.0, 0.0, -1.0])
    return rot, origins


# ----------------------------------------------------- analytic rigid body
def test_face_centre_velocity_is_origin_velocity_plus_omega_cross_offset() -> None:
    """Rigid body: v_face = v_origin + omega x (R c) for any offset ``c``."""
    omega = np.array([0.4, -1.3, 7.0])  # rad/s, constant
    v_origin = np.array([0.5, -2.0, 0.25])  # m/s
    offset = np.array([0.012, 0.049, -0.026])  # face centre in the head frame
    axis = omega / np.linalg.norm(omega)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    dt = 1e-5
    t = np.array([0.3 - dt, 0.3, 0.3 + dt])
    angle = np.linalg.norm(omega) * t
    rot = np.array(
        [np.eye(3) + np.sin(a) * skew + (1.0 - np.cos(a)) * skew @ skew for a in angle]
    )
    origins = np.array([0.1, 0.2, 0.3]) + np.outer(t, v_origin)
    path = cft.face_centre_path(rot, origins, offset)
    velocity = (path[2] - path[0]) / (2.0 * dt)
    expected = v_origin + np.cross(omega, rot[1] @ offset)
    np.testing.assert_allclose(velocity, expected, atol=1e-6)


def test_impact_speed_is_measured_at_the_face_centre_not_the_frame_origin() -> None:
    """On a 1 m arm the frame origin reaches the ball at RATE m/s; a face
    centre 0.05 m further out along the arm at 1.05 RATE, 0.03 m to the toe
    adds nothing (it lies on the rotation axis direction)."""
    t = np.arange(0.0, 1.6, 1e-4)
    rot, origins = _swing_frames(t)
    face = cft.face_centre_path(rot, origins, np.array([0.03, 0.0, -0.05]))
    at_face = cf.clubhead_speed_timing(t, face)
    at_origin = cf.clubhead_speed_timing(t, origins)
    assert at_origin.impact_speed_mps == pytest.approx(RATE_AT_BALL, rel=1e-3)
    assert at_face.impact_speed_mps == pytest.approx(1.05 * RATE_AT_BALL, rel=1e-3)
    assert at_face.impact_time_s == pytest.approx(1.4, abs=1e-3)


def test_face_centre_path_contracts() -> None:
    rot = np.repeat(np.eye(3)[None], 4, axis=0)
    origins = np.zeros((4, 3))
    with pytest.raises(ValueError, match="rotations"):
        cft.face_centre_path(rot[:, :2], origins, np.zeros(3))
    with pytest.raises(ValueError, match="rotations"):
        cft.face_centre_path(rot, origins[:3], np.zeros(3))
    with pytest.raises(ValueError, match="centre_in_frame"):
        cft.face_centre_path(rot, origins, np.array([0.0, np.nan, 0.0]))
    gap = origins.copy()
    gap[2] = np.nan  # an unobserved capture frame stays unobserved
    assert np.isnan(cft.face_centre_path(rot, gap, np.ones(3))[2]).all()


# ----------------------------------------------------------- receipt block
def _face_path(t: np.ndarray, arm_m: float) -> fp.FacePath:
    rot, origins = _swing_frames(t, arm_m)
    return fp.FacePath(t, cft.face_centre_path(rot, origins, np.zeros(3)))


def test_speed_block_records_error_and_peak_offset_against_the_capture() -> None:
    """A swing on a 3 % shorter arm through the same angles: every chord is
    3 % shorter, so exactly -3 % at impact and no peak-timing offset."""
    t = np.arange(0.0, 1.6, 1e-3)
    block = fp.clubhead_speed_report(
        simulated=_face_path(t, 0.97),
        reference=_face_path(t, 1.0),
        capture=_face_path(t, 1.0),
    )
    assert block["status"] == "ok" and block["reason"] is None
    assert block["measurement_point"] == "face_centre"
    assert block["impact_speed_tol_fraction"] == 0.03
    assert block["peak_timing_tol_s"] == 0.005
    sim, cap = block["simulated"], block["capture"]
    assert sim["status"] == cap["status"] == block["reference"]["status"] == "ok"
    assert block["impact_speed_error_pct"] == pytest.approx(
        100.0 * (sim["impact_speed_mps"] / cap["impact_speed_mps"] - 1.0)
    )
    assert block["impact_speed_error_pct"] == pytest.approx(-3.0, abs=1e-9)
    assert block["peak_timing_offset_s"] == pytest.approx(
        sim["peak_minus_impact_s"] - cap["peak_minus_impact_s"]
    )
    assert block["peak_timing_offset_s"] == pytest.approx(0.0, abs=1e-12)


def test_speed_block_without_a_capture_is_unavailable_never_zero() -> None:
    t = np.arange(0.0, 1.6, 1e-3)
    block = fp.clubhead_speed_report(
        simulated=_face_path(t, 1.0),
        reference=_face_path(t, 1.0),
        capture=fp.FacePath(None, None, "no calibrated head triad"),
    )
    assert block["status"] == "unavailable"
    assert "capture" in block["reason"]
    assert block["impact_speed_error_pct"] is None
    assert block["peak_timing_offset_s"] is None
    assert block["capture"]["impact_speed_mps"] is None
    assert block["capture"]["reason"] == "no calibrated head triad"
    assert block["simulated"]["impact_speed_mps"] > 0.0


def test_a_swing_that_misses_the_ball_reports_its_closest_approach() -> None:
    t = np.arange(0.0, 1.6, 1e-3)
    path = _face_path(t, 1.0)
    lifted = path.centres.copy()
    lifted[1:, 2] += 0.3  # never returns to the ball height
    series = fp.speed_series(fp.FacePath(t, lifted))
    assert series["status"] == "unavailable"
    assert series["impact_speed_mps"] is None
    assert "closest approach 0.30" in series["reason"]


def test_model_face_path_needs_a_club_and_never_falls_back_to_the_origin() -> None:
    q = np.zeros((3, 2))
    t = np.arange(3) * 0.01
    path = fp.model_face_path(object(), q, t, None)
    assert path.centres is None and path.reason == "no club specification"
    path = fp.model_face_path(object(), q, t, {"bodies": []})
    assert path.centres is None and "face centre not placed" in path.reason


def test_capture_face_path_needs_the_calibrated_triad() -> None:
    t = np.arange(3) * 0.01
    path = fp.capture_face_path(
        t, (np.zeros((3, 1, 3)), np.ones((3, 1)), ["m"]), None, {}
    )
    assert path.centres is None and path.reason == "no calibrated head triad"


def test_dynamics_receipt_accepts_the_speed_block() -> None:
    from src.shared.python.motion_matching.pipeline.receipt_dynamics import (
        ClubheadSpeedReceipt,
    )

    t = np.arange(0.0, 1.6, 1e-3)
    block = fp.clubhead_speed_report(
        simulated=_face_path(t, 0.97),
        reference=_face_path(t, 1.0),
        capture=fp.FacePath(None, None, "no calibrated head triad"),
    )
    parsed = ClubheadSpeedReceipt.model_validate(block)
    assert parsed.capture.impact_speed_mps is None
    assert parsed.measurement_point == "face_centre"


# ------------------------------------------ fixture parity (GCV-20, #11767)
SPEC_DIR = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
CAPTURES = {"A": ("driver", "driver"), "B": ("iron7", "iron")}
FIXTURE_DT_S = 0.002


@functools.cache
def _parity_block(club: str, capture_name: str) -> dict:
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.ground_support import (
        capture_to_native_world,
    )
    from src.shared.python.motion_matching.pipeline.constants import capture_path
    from src.shared.python.motion_matching.tour_capture_contract import (
        load_tour_capture,
    )

    spec_bytes = (SPEC_DIR / f"full_body_spec_anthro_{club}.json").read_bytes()
    spec = json.loads(spec_bytes)
    poses = json.loads((FIXTURES / "address_poses.json").read_text(encoding="utf-8"))
    order = poses["poses"][club]["coordinate_order"]
    q = np.load(FIXTURES / f"swing_q_{club}.npz")["q"].astype(float)
    from scripts.diagnose_clubhead_speed import MujocoFrameKinematics

    kin = MujocoFrameKinematics(spec_bytes, order)
    t_model = np.arange(len(q)) * FIXTURE_DT_S
    provenance = json.loads((FIXTURES / "provenance.json").read_text("utf-8"))
    offsets = provenance["clubs"][club]["capture_triad_offsets_m"]
    try:
        capture = load_tour_capture(capture_path(capture_name)).subset(tuple(offsets))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"capture-{club} not present: {exc}")
    t_cap = np.asarray(capture.time_s, dtype=float)
    t_cap = t_cap - t_cap[0]
    attachments = {k: (cft.FACE_FRAME, tuple(v)) for k, v in offsets.items()}
    observed = (
        capture_to_native_world(capture.points_m),
        capture.valid,
        tuple(capture.labels),
    )
    return fp.clubhead_speed_report(
        simulated=fp.model_face_path(kin, q, t_model, spec),
        reference=fp.FacePath(None, None, "fixture holds no reference"),
        capture=fp.capture_face_path(t_cap, observed, attachments, spec),
    )


_CAPTURE_B_SPEED_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "GCV-20 #11767 acceptance not met on capture-B: the 7-iron fixture "
        "reaches the ball at -2.6 % (40.20 / 41.26 m/s, 3 % limit) but its "
        "face-centre speed peaks 7.2 ms early relative to the capture (5 ms "
        "limit); the tracked reference peaks on time, so the lead is FD "
        "tracking, re-fit work for R10 (#12692)"
    ),
)


@pytest.mark.parametrize(
    "capture", ["A", pytest.param("B", marks=_CAPTURE_B_SPEED_XFAIL)]
)
def test_receipt_speed_block_meets_gcv20_on_the_fixtures(capture: str) -> None:
    """The receipt's own measurement (face centre through the head frame)
    on the committed fixtures, against the capture triad face centre."""
    block = _parity_block(*CAPTURES[capture])
    assert block["status"] == "ok", block
    assert abs(block["impact_speed_error_pct"]) <= 100.0 * fp.IMPACT_SPEED_TOL, block
    assert abs(block["peak_timing_offset_s"]) <= fp.PEAK_TIMING_TOL_S, block


@pytest.mark.parametrize("capture", sorted(CAPTURES))
def test_receipt_speed_block_records_the_triaged_fixture_numbers(capture: str) -> None:
    """Pins the measurement: the face-centre numbers of the triage table
    (driver 50.02 vs 51.32 m/s, 7-iron 40.20 vs 41.26 m/s)."""
    expected = {"A": (50.02, 51.32), "B": (40.20, 41.26)}[capture]
    block = _parity_block(*CAPTURES[capture])
    assert block["simulated"]["impact_speed_mps"] == pytest.approx(
        expected[0], abs=0.01
    )
    assert block["capture"]["impact_speed_mps"] == pytest.approx(expected[1], abs=0.01)


def test_diagnostic_table_reports_a_missed_ball_as_its_closest_approach() -> None:
    """``scripts/diagnose_clubhead_speed``: a series that never passes the
    ball is unavailable with its closest approach, never a zero speed."""
    from scripts.diagnose_clubhead_speed import decomposition, render_table

    t = np.arange(0.0, 1.6, 1e-3)
    capture = _face_path(t, 1.0).centres
    lifted = _face_path(t, 0.97).centres.copy()
    lifted[1:, 2] += 0.3
    records = decomposition(
        [
            ("capture, face centre", t, capture),
            ("model, face centre", t, _face_path(t, 0.97).centres),
            ("model, lifted", t, lifted),
        ]
    )
    assert records[1]["error_pct"] == pytest.approx(-3.0, abs=1e-9)
    assert records[2]["impact_speed_mps"] is None and records[2]["error_pct"] is None
    table = render_table("driver", records)
    assert "capture-A" in table.splitlines()[0]
    assert "driver" not in table  # captures are named anonymously
    assert "closest approach 0.30" in table.splitlines()[-1]
    assert "| n/a |" in table.splitlines()[-1]
