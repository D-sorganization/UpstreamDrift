"""Coverage tests for ``starting_pose_matcher.core``.

Pure math + loader paths. Covers fallback skeleton building, RigidTransform
math, phase label utilities, the Simscape trajectory CSV loader and the
shaft-snap solver. Test-only; no production code changes (issue #4673).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.tools.starting_pose_matcher.core import (
    CM_TO_M,
    DEFAULT_EVENT_PRESET,
    DEFAULT_PHASE,
    EVENT_KEYS,
    EVENT_LABEL_PRESETS,
    FALLBACK_SEGMENTS,
    PHASE_BOUNDS,
    PHASE_KEYS,
    MocapEvents,
    PoseSlot,
    RigidTransform,
    SESSION_SCHEMA_VERSION,
    Skeleton,
    SkeletonTrajectory,
    _xyz_columns_for,
    clubtarget_from_multi,
    dispatch_cost_inputs,
    fallback_skeleton,
    load_simscape_trajectory_csv,
    load_skeleton,
    phase_display_label,
    phase_key_from_label,
    solve_shaft_rz_deg,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Module-level constants                                                      #
# --------------------------------------------------------------------------- #


def test_constants_have_expected_shape():
    assert CM_TO_M == 0.01
    assert SESSION_SCHEMA_VERSION >= 4
    assert EVENT_KEYS == ("A", "T", "I", "F")
    assert DEFAULT_EVENT_PRESET in EVENT_LABEL_PRESETS
    assert DEFAULT_PHASE in PHASE_KEYS
    for key in PHASE_KEYS:
        assert key in PHASE_BOUNDS
    # Fallback segment chain is non-empty and references compact joint names.
    assert ("hip", "spine") in FALLBACK_SEGMENTS


# --------------------------------------------------------------------------- #
# MocapEvents                                                                 #
# --------------------------------------------------------------------------- #


def test_mocap_events_frame_for_returns_zero_indexed():
    ev = MocapEvents(A_sample=1, T_sample=10, I_sample=20, F_sample=30, CHS_mph=85.0)
    assert ev.frame_for("A") == 0
    assert ev.frame_for("T") == 9
    assert ev.frame_for("I") == 19
    assert ev.frame_for("F") == 29


def test_mocap_events_frame_for_nan_returns_none():
    ev = MocapEvents()  # all NaN
    assert ev.frame_for("A") is None
    assert ev.frame_for("T") is None
    assert ev.frame_for("F") is None


def test_mocap_events_frame_for_clamps_to_zero():
    ev = MocapEvents(A_sample=0)  # 0 -> int(0) - 1 = -1, clamps to 0
    assert ev.frame_for("A") == 0


# --------------------------------------------------------------------------- #
# Phase label helpers                                                         #
# --------------------------------------------------------------------------- #


def test_phase_display_label_known_keys_use_event_labels():
    labels = {"A": "Address", "T": "Top", "I": "Impact", "F": "Finish"}
    assert phase_display_label("backswing", labels) == "Backswing (Address to Top)"
    assert phase_display_label("downswing", labels) == "Downswing (Top to Impact)"
    assert (
        phase_display_label("follow_through", labels)
        == "Follow-through (Impact to Finish)"
    )
    assert phase_display_label("full_swing", labels) == "Full swing (Address to Finish)"
    assert phase_display_label("manual", labels) == "Manual frame range"
    assert phase_display_label("none", labels).startswith("None")


def test_phase_display_label_uses_default_when_event_label_missing():
    # Empty dict -> falls back to canonical defaults.
    out = phase_display_label("backswing", {})
    assert "Address" in out and "Top of Backswing" in out


def test_phase_display_label_unknown_key_returned_verbatim():
    assert phase_display_label("zzz", {}) == "zzz"


def test_phase_key_from_label_legacy_and_current():
    assert phase_key_from_label("Backswing (A → T)") == "backswing"
    assert phase_key_from_label("backswing") == "backswing"
    # Leading-prefix lookup.
    assert phase_key_from_label("Backswing - whatever") == "backswing"
    assert phase_key_from_label("Manual:") == "manual"


def test_phase_key_from_label_unknown_returns_none():
    assert phase_key_from_label("xyzzy whatever") is None


# --------------------------------------------------------------------------- #
# RigidTransform                                                              #
# --------------------------------------------------------------------------- #


def test_rigid_transform_identity_is_no_op():
    rt = RigidTransform()
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = rt.apply(pts)
    np.testing.assert_allclose(out, pts)


def test_rigid_transform_pure_translation():
    rt = RigidTransform(tx=1.0, ty=2.0, tz=-3.0)
    pts = np.array([[0.0, 0.0, 0.0]])
    out = rt.apply(pts)
    np.testing.assert_allclose(out, [[1.0, 2.0, -3.0]])


def test_rigid_transform_rotation_z_90():
    rt = RigidTransform(rz=90.0)
    pts = np.array([[1.0, 0.0, 0.0]])
    out = rt.apply(pts)
    np.testing.assert_allclose(out, [[0.0, 1.0, 0.0]], atol=1e-12)


def test_rigid_transform_scale_with_pivot():
    rt = RigidTransform(scale=2.0, pivot=(1.0, 1.0, 0.0))
    pts = np.array([[2.0, 1.0, 0.0]])  # offset 1.0 in x from pivot
    out = rt.apply(pts)
    # 2*(2-1, 1-1, 0-0) + (1,1,0) + (0,0,0) = (3, 1, 0)
    np.testing.assert_allclose(out, [[3.0, 1.0, 0.0]])


def test_rigid_transform_matrix_returns_R_and_t():
    rt = RigidTransform(tx=5.0, scale=3.0)
    R, t = rt.matrix()
    assert R.shape == (3, 3)
    np.testing.assert_allclose(t, [5.0, 0.0, 0.0])
    # scale embedded in R
    np.testing.assert_allclose(R, np.eye(3) * 3.0)


# --------------------------------------------------------------------------- #
# SkeletonTrajectory + Skeleton                                               #
# --------------------------------------------------------------------------- #


def test_skeleton_trajectory_len_and_frame_at_time():
    times = np.array([0.0, 0.1, 0.2, 0.3])
    frames = [Skeleton(name=f"f{i}") for i in range(4)]
    traj = SkeletonTrajectory(times=times, frames=frames)
    assert len(traj) == 4
    assert traj.frame_at_time(0.0) == 0
    assert traj.frame_at_time(0.11) == 1
    # Out-of-range clamps.
    assert traj.frame_at_time(99.0) == 3
    assert traj.frame_at_time(-1.0) == 0


def test_skeleton_trajectory_empty_returns_zero():
    traj = SkeletonTrajectory()
    assert len(traj) == 0
    assert traj.frame_at_time(1.0) == 0


def test_pose_slot_dataclass_round_trip():
    skel = Skeleton(name="x")
    slot = PoseSlot(
        name="a",
        skeleton=skel,
        color="#fff",
        mocap_color="#000",
        target_event="A",
    )
    assert slot.visible is True
    assert slot.trajectory is None
    assert slot.trajectory_frame_index == 0


# --------------------------------------------------------------------------- #
# Fallback skeleton                                                           #
# --------------------------------------------------------------------------- #


def test_fallback_skeleton_address_has_required_joints():
    skel = fallback_skeleton("Impact")
    for name in ("hip", "spine", "torso", "hub", "ls", "rs"):
        assert name in skel.joints, f"missing {name}"
    assert skel.segments == FALLBACK_SEGMENTS


def test_fallback_skeleton_top_of_backswing_has_butt_alias():
    skel = fallback_skeleton("TopofBackswing")
    assert skel.name == "TopofBackswing"
    assert "butt" in skel.joints
    assert "mp" in skel.joints
    np.testing.assert_allclose(skel.joints["butt"], skel.joints["mp"])


def test_fallback_skeleton_top_case_insensitive():
    a = fallback_skeleton("top_of_thing")
    b = fallback_skeleton("Top")
    # both branch into TOB builder
    assert a.name == "TopofBackswing"
    assert b.name == "TopofBackswing"


# --------------------------------------------------------------------------- #
# load_skeleton                                                               #
# --------------------------------------------------------------------------- #


def test_load_skeleton_missing_file_falls_back(tmp_path: Path):
    skel = load_skeleton(tmp_path / "nope.json", fallback_pose="Impact")
    # FK-derived fallback returns a skeleton named "Impact".
    assert "hip" in skel.joints


def test_load_skeleton_reads_json_file(tmp_path: Path):
    blob = {
        "pose": "MyPose",
        "joints": {
            "hip": [0.0, 0.0, 0.0],
            "ch": [0.5, 0.0, 1.0],
        },
        "segments": [["hip", "ch"]],
    }
    path = tmp_path / "simscape_skeleton_MyPose.json"
    path.write_text(json.dumps(blob))
    skel = load_skeleton(path)
    assert skel.name == "MyPose"
    assert "hip" in skel.joints
    np.testing.assert_allclose(skel.joints["ch"], [0.5, 0.0, 1.0])
    assert skel.segments == [("hip", "ch")]


def test_load_skeleton_missing_segments_uses_fallback_segments(tmp_path: Path):
    blob = {"pose": "X", "joints": {"hip": [0.0, 0.0, 0.0]}}
    path = tmp_path / "skel.json"
    path.write_text(json.dumps(blob))
    skel = load_skeleton(path)
    assert skel.segments == FALLBACK_SEGMENTS


def test_load_skeleton_drops_malformed_segment_entries(tmp_path: Path):
    blob = {
        "pose": "X",
        "joints": {"hip": [0.0, 0.0, 0.0]},
        "segments": [["hip", "ch"], "not-a-list", ["only-one"]],
    }
    path = tmp_path / "skel.json"
    path.write_text(json.dumps(blob))
    skel = load_skeleton(path)
    assert skel.segments == [("hip", "ch")]


# --------------------------------------------------------------------------- #
# CSV column resolution                                                       #
# --------------------------------------------------------------------------- #


def test_xyz_columns_for_short_form():
    cols = ["club_head_X", "club_head_Y", "club_head_Z", "other"]
    assert _xyz_columns_for(cols, "club_head_X") == [
        "club_head_X",
        "club_head_Y",
        "club_head_Z",
    ]


def test_xyz_columns_for_lower_short():
    cols = ["club_head_x", "club_head_y", "club_head_z"]
    assert _xyz_columns_for(cols, "club_head_x") == [
        "club_head_x",
        "club_head_y",
        "club_head_z",
    ]


def test_xyz_columns_for_long_form():
    cols = [
        "ClubLogs_CHGlobalPosition_1",
        "ClubLogs_CHGlobalPosition_2",
        "ClubLogs_CHGlobalPosition_3",
    ]
    assert _xyz_columns_for(cols, "ClubLogs_CHGlobalPosition_1") == cols


def test_xyz_columns_for_dim_long_form():
    cols = [
        "HipLogs_HipGlobalPosition_dim1",
        "HipLogs_HipGlobalPosition_dim2",
        "HipLogs_HipGlobalPosition_dim3",
    ]
    assert _xyz_columns_for(cols, "HipLogs_HipGlobalPosition_dim1") == cols


def test_xyz_columns_for_unknown_pattern_returns_none():
    assert _xyz_columns_for(["a", "b"], "nope") is None


def test_xyz_columns_for_missing_y_returns_none():
    assert _xyz_columns_for(["club_X", "club_Z"], "club_X") is None


# --------------------------------------------------------------------------- #
# load_simscape_trajectory_csv                                                #
# --------------------------------------------------------------------------- #


def _write_short_form_csv(path: Path, n: int = 4) -> None:
    t = np.linspace(0.0, 0.3, n)
    df = pd.DataFrame(
        {
            "time": t,
            "left_hand_X": np.linspace(0.0, 0.1, n),
            "left_hand_Y": np.zeros(n),
            "left_hand_Z": np.zeros(n),
            "right_hand_X": np.linspace(0.0, -0.1, n),
            "right_hand_Y": np.zeros(n),
            "right_hand_Z": np.zeros(n),
            "club_head_X": np.linspace(0.0, 0.5, n),
            "club_head_Y": np.zeros(n),
            "club_head_Z": np.linspace(0.0, 0.2, n),
            "spine_X": np.zeros(n),
            "spine_Y": np.zeros(n),
            "spine_Z": np.full(n, 0.4),
            "hub_X": np.zeros(n),
            "hub_Y": np.zeros(n),
            "hub_Z": np.full(n, 0.6),
            "hip_X": np.zeros(n),
            "hip_Y": np.zeros(n),
            "hip_Z": np.zeros(n),
        }
    )
    df.to_csv(path, index=False)


def test_load_simscape_trajectory_csv_short_form(tmp_path: Path):
    p = tmp_path / "traj.csv"
    _write_short_form_csv(p, n=5)
    traj = load_simscape_trajectory_csv(p)
    assert len(traj) == 5
    assert traj.source_path == str(p)
    # mp synthesised from lw + rw
    assert "mp" in traj.frames[0].joints
    np.testing.assert_allclose(traj.frames[0].joints["mp"], [0.0, 0.0, 0.0])
    # torso synthesised from spine + hub
    assert "torso" in traj.frames[0].joints
    expected_torso_z = 0.4 + 0.2 * (0.6 - 0.4)
    np.testing.assert_allclose(traj.frames[0].joints["torso"][2], expected_torso_z)


def test_load_simscape_trajectory_csv_long_form(tmp_path: Path):
    n = 3
    df = pd.DataFrame(
        {
            "time": np.linspace(0.0, 0.2, n),
            "ClubLogs_CHGlobalPosition_1": np.zeros(n),
            "ClubLogs_CHGlobalPosition_2": np.zeros(n),
            "ClubLogs_CHGlobalPosition_3": np.zeros(n),
            "LWLogs_LHGlobalPosition_1": np.zeros(n),
            "LWLogs_LHGlobalPosition_2": np.zeros(n),
            "LWLogs_LHGlobalPosition_3": np.zeros(n),
            "RWLogs_RHGlobalPosition_1": np.zeros(n),
            "RWLogs_RHGlobalPosition_2": np.zeros(n),
            "RWLogs_RHGlobalPosition_3": np.zeros(n),
        }
    )
    p = tmp_path / "traj.csv"
    df.to_csv(p, index=False)
    traj = load_simscape_trajectory_csv(p)
    assert len(traj) == n
    assert "ch" in traj.frames[0].joints


def test_load_simscape_trajectory_csv_no_time_column_raises(tmp_path: Path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="time"):
        load_simscape_trajectory_csv(p)


def test_load_simscape_trajectory_csv_no_joints_raises(tmp_path: Path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"time": [0.0, 0.1], "irrelevant": [1, 2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="recognised joint columns"):
        load_simscape_trajectory_csv(p)


def test_load_simscape_trajectory_csv_drops_non_finite_rows(tmp_path: Path):
    p = tmp_path / "traj.csv"
    df = pd.DataFrame(
        {
            "time": [0.0, 0.1],
            "club_head_X": [0.0, np.nan],
            "club_head_Y": [0.0, 0.0],
            "club_head_Z": [0.0, 0.0],
        }
    )
    df.to_csv(p, index=False)
    traj = load_simscape_trajectory_csv(p)
    assert "ch" in traj.frames[0].joints
    assert "ch" not in traj.frames[1].joints  # NaN row excluded


# --------------------------------------------------------------------------- #
# solve_shaft_rz_deg                                                          #
# --------------------------------------------------------------------------- #


def test_solve_shaft_rz_zero_rotation():
    mp = np.zeros(3)
    ch = np.array([1.0, 0.0, 0.5])
    rz = solve_shaft_rz_deg(mp, ch, mp, ch)
    assert abs(rz) < 1e-9


def test_solve_shaft_rz_90_degrees():
    mp = np.zeros(3)
    ch_target = np.array([0.0, 1.0, 0.0])
    ch_skel = np.array([1.0, 0.0, 0.0])
    rz = solve_shaft_rz_deg(mp, ch_target, mp, ch_skel)
    assert abs(rz - 90.0) < 1e-9


def test_solve_shaft_rz_zero_magnitude_target_returns_zero():
    mp = np.zeros(3)
    ch = np.array([0.0, 0.0, 1.0])  # all magnitude on Z, projected XY is zero
    rz = solve_shaft_rz_deg(mp, ch, mp, np.array([1.0, 0.0, 0.0]))
    assert rz == 0.0


def test_solve_shaft_rz_zero_magnitude_skel_returns_zero():
    mp = np.zeros(3)
    rz = solve_shaft_rz_deg(mp, np.array([1.0, 0.0, 0.0]), mp, mp)
    assert rz == 0.0


def test_solve_shaft_rz_wraps_to_signed_range():
    rz = solve_shaft_rz_deg(
        np.zeros(3),
        np.array([-1.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
    )
    # 180 or -180 are both valid (wrapped to (-180, 180]).
    assert abs(abs(rz) - 180.0) < 1e-9


# --------------------------------------------------------------------------- #
# Multi-source dispatch helpers                                               #
# --------------------------------------------------------------------------- #


def _make_multi_target(*, club=None, body=None, ball=None, time=None):
    """Build a duck-typed MultiSourceTarget for unit tests."""
    obj = SimpleNamespace()
    obj.club = club
    obj.body = body
    obj.ball = ball
    obj.has_club = lambda: club is not None
    obj.has_body = lambda: body is not None
    obj.is_club_ball = lambda: ball is not None
    obj.shared_time = lambda: time if time is not None else np.array([])
    return obj


def test_dispatch_cost_inputs_with_only_body():
    body = SimpleNamespace(time=np.array([0.0, 0.1]))
    target = _make_multi_target(body=body, time=np.array([0.0, 0.1]))
    out = dispatch_cost_inputs(target)
    assert "time" in out
    assert "club" not in out
    assert out["body"] is body


def _make_clubtarget(n: int):
    from src.shared.python.motion_matching.target import ClubTarget, SourceProvenance

    src = SourceProvenance(
        filename="x", format="xlsx", subject_id="s", trial_id="t", sha256="0" * 64
    )
    return ClubTarget(
        time=np.linspace(0.0, 0.3, n),
        butt=np.zeros((n, 3)),
        clubhead=np.ones((n, 3)),
        club_quat=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        impact_idx=min(n - 1, 2),
        source=src,
    )


def test_dispatch_cost_inputs_with_club_and_ball_flags():
    club = _make_clubtarget(4)
    target = _make_multi_target(club=club, ball=object(), time=club.time)
    out = dispatch_cost_inputs(target)
    assert "club" in out
    assert out["has_ball"] is True


def test_clubtarget_from_multi_no_club_returns_none():
    target = _make_multi_target(time=np.array([0.0, 0.1]))
    assert clubtarget_from_multi(target) is None


def test_clubtarget_from_multi_returns_inner_clubtarget():
    club = _make_clubtarget(3)
    # Wrap the ClubTarget so .club exposes the inner one.
    wrapper = SimpleNamespace(club=club, time=club.time)
    target = _make_multi_target(club=wrapper, time=club.time)
    assert clubtarget_from_multi(target) is club


def test_clubtarget_from_multi_returns_clubtarget_directly():
    club = _make_clubtarget(3)
    target = _make_multi_target(club=club, time=club.time)
    assert clubtarget_from_multi(target) is club
