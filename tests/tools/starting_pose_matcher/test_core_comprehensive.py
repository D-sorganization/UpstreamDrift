"""Comprehensive unit tests for ``src.tools.starting_pose_matcher.core``.

Targets math, dataclasses, loaders, and dispatch helpers. Heavy modules
(Qt, matplotlib, motion_matching loaders) are accessed only via thin
mock seams — these tests run without any of those installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.tools.starting_pose_matcher import core


# ---------------------------------------------------------------------------
# Constants & display helpers
# ---------------------------------------------------------------------------


def test_constants_present_and_sane():
    assert core.CM_TO_M == 0.01
    assert core.DEFAULT_EVENT_PRESET in core.EVENT_LABEL_PRESETS
    assert set(core.EVENT_KEYS) == {"A", "T", "I", "F"}
    assert set(core.PHASE_BOUNDS.keys()) == set(core.PHASE_KEYS)
    assert core.DEFAULT_PHASE in core.PHASE_KEYS
    assert all(isinstance(s, tuple) and len(s) == 2 for s in core.FALLBACK_SEGMENTS)


def test_phase_display_label_default_labels():
    labels = core.EVENT_LABEL_PRESETS[core.DEFAULT_EVENT_PRESET]
    assert "Backswing" in core.phase_display_label("backswing", labels)
    assert core.phase_display_label("none", labels).startswith("None")
    assert core.phase_display_label("manual", labels) == "Manual frame range"


def test_phase_display_label_uses_event_labels():
    out = core.phase_display_label(
        "downswing", {"A": "X", "T": "Y", "I": "Z", "F": "W"}
    )
    assert "Y" in out and "Z" in out


def test_phase_display_label_unknown_returns_key():
    assert core.phase_display_label("nope", {}) == "nope"


def test_phase_display_label_missing_event_uses_default_string():
    # Empty dict — defaults baked into phase_display_label kick in.
    out = core.phase_display_label("full_swing", {})
    assert "Address" in out and "Finish" in out


def test_phase_key_from_label_legacy_table():
    assert core.phase_key_from_label("Backswing (A → T)") == "backswing"
    assert core.phase_key_from_label("None") == "none"


def test_phase_key_from_label_canonical_key():
    assert core.phase_key_from_label("backswing") == "backswing"


def test_phase_key_from_label_leading_word_match():
    # "Backswing (Address to Top of Backswing)" should match leading word
    assert core.phase_key_from_label("Backswing (anything)") == "backswing"


def test_phase_key_from_label_unknown_returns_none():
    assert core.phase_key_from_label("Mystery") is None


# ---------------------------------------------------------------------------
# MocapEvents
# ---------------------------------------------------------------------------


def test_mocap_events_default_all_nan():
    ev = core.MocapEvents()
    assert ev.frame_for("A") is None
    assert ev.frame_for("T") is None
    assert ev.frame_for("CHS") is None


def test_mocap_events_frame_for_is_zero_indexed():
    ev = core.MocapEvents(A_sample=1.0, T_sample=10.0)
    assert ev.frame_for("A") == 0
    assert ev.frame_for("T") == 9


def test_mocap_events_frame_for_clamps_at_zero():
    ev = core.MocapEvents(A_sample=0.0)
    assert ev.frame_for("A") == 0


# ---------------------------------------------------------------------------
# RigidTransform
# ---------------------------------------------------------------------------


def test_rigid_transform_identity_is_identity():
    rt = core.RigidTransform()
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = rt.apply(pts)
    np.testing.assert_allclose(out, pts)


def test_rigid_transform_translation():
    rt = core.RigidTransform(tx=1.0, ty=2.0, tz=3.0)
    out = rt.apply(np.zeros((1, 3)))
    np.testing.assert_allclose(out, [[1.0, 2.0, 3.0]])


def test_rigid_transform_z_rotation_90deg():
    rt = core.RigidTransform(rz=90.0)
    out = rt.apply(np.array([[1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.0, 1.0, 0.0]], atol=1e-9)


def test_rigid_transform_scale():
    rt = core.RigidTransform(scale=2.0)
    out = rt.apply(np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(out, [[2.0, 4.0, 6.0]])


def test_rigid_transform_pivot_translation_invariant_at_pivot():
    rt = core.RigidTransform(rz=45.0, pivot=(1.0, 1.0, 1.0))
    # Pivot point should be invariant under rotation about itself.
    out = rt.apply(np.array([[1.0, 1.0, 1.0]]))
    np.testing.assert_allclose(out, [[1.0, 1.0, 1.0]], atol=1e-9)


def test_rigid_transform_matrix_returns_R_t():
    rt = core.RigidTransform(tx=5.0, scale=2.0)
    R, t = rt.matrix()
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    # Scale folded into R.
    np.testing.assert_allclose(np.linalg.det(R), 8.0, atol=1e-9)
    np.testing.assert_allclose(t, [5.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Skeleton / SkeletonTrajectory / PoseSlot dataclasses
# ---------------------------------------------------------------------------


def test_skeleton_default_empty():
    s = core.Skeleton()
    assert s.name == "default"
    assert s.joints == {}
    assert s.segments == []


def test_skeleton_trajectory_len_and_frame_at_time():
    frames = [core.Skeleton(name=f"f{i}") for i in range(3)]
    traj = core.SkeletonTrajectory(
        times=np.array([0.0, 1.0, 2.0]), frames=frames, source_path="x"
    )
    assert len(traj) == 3
    assert traj.frame_at_time(0.0) == 0
    assert traj.frame_at_time(1.1) == 1
    # Clamping behaviour.
    assert traj.frame_at_time(99.0) == 2
    assert traj.frame_at_time(-99.0) == 0


def test_skeleton_trajectory_empty_returns_zero():
    traj = core.SkeletonTrajectory()
    assert traj.frame_at_time(0.0) == 0
    assert len(traj) == 0


def test_pose_slot_defaults():
    skel = core.Skeleton(name="x")
    p = core.PoseSlot(
        name="left",
        skeleton=skel,
        color="#f00",
        mocap_color="#0f0",
        target_event="A",
    )
    assert p.visible is True
    assert p.trajectory is None
    assert p.trajectory_frame_index == 0


# ---------------------------------------------------------------------------
# Shaft-snap math
# ---------------------------------------------------------------------------


def test_solve_shaft_rz_deg_aligned_is_zero():
    mp_t = np.array([0.0, 0.0, 0.0])
    ch_t = np.array([1.0, 0.0, 0.0])
    mp_s = np.array([0.0, 0.0, 5.0])
    ch_s = np.array([1.0, 0.0, 5.0])
    assert core.solve_shaft_rz_deg(mp_t, ch_t, mp_s, ch_s) == pytest.approx(0.0)


def test_solve_shaft_rz_deg_90deg_rotation():
    mp_t = np.array([0.0, 0.0, 0.0])
    ch_t = np.array([0.0, 1.0, 0.0])  # along +Y
    mp_s = np.array([0.0, 0.0, 0.0])
    ch_s = np.array([1.0, 0.0, 0.0])  # along +X
    assert core.solve_shaft_rz_deg(mp_t, ch_t, mp_s, ch_s) == pytest.approx(90.0)


def test_solve_shaft_rz_deg_wrap_to_180():
    mp_t = np.array([0.0, 0.0, 0.0])
    ch_t = np.array([-1.0, 0.0, 0.0])
    mp_s = np.array([0.0, 0.0, 0.0])
    ch_s = np.array([1.0, 0.0, 0.0])
    result = core.solve_shaft_rz_deg(mp_t, ch_t, mp_s, ch_s)
    assert abs(abs(result) - 180.0) < 1e-9


def test_solve_shaft_rz_deg_zero_magnitude_returns_zero():
    p = np.array([0.0, 0.0, 0.0])
    assert core.solve_shaft_rz_deg(p, p, p, p) == 0.0


# ---------------------------------------------------------------------------
# Fallback skeletons
# ---------------------------------------------------------------------------


def test_fallback_skeleton_impact_has_expected_joints():
    s = core.fallback_skeleton("Impact")
    assert isinstance(s, core.Skeleton)
    # FK-derived joints should include the major shorthand names.
    expected = {"hip", "spine", "hub", "ls", "rs", "le", "re", "lw", "rw"}
    assert expected.issubset(set(s.joints.keys()))
    assert s.segments == core.FALLBACK_SEGMENTS


def test_fallback_skeleton_topofbackswing_uses_handcrafted():
    s = core.fallback_skeleton("TopofBackswing")
    assert s.name == "TopofBackswing"
    # Handcrafted TOB has all matcher joints + butt alias for mp.
    assert "mp" in s.joints and "butt" in s.joints
    np.testing.assert_allclose(s.joints["butt"], s.joints["mp"])


def test_fallback_skeleton_default_branches_on_prefix():
    impact = core.fallback_skeleton("Impact")
    tob = core.fallback_skeleton("topofbackswing")  # lowercase, prefix "top"
    assert impact.name != "TopofBackswing"
    assert tob.name == "TopofBackswing"


# ---------------------------------------------------------------------------
# JSON skeleton loader
# ---------------------------------------------------------------------------


def test_load_skeleton_reads_json_file(tmp_path: Path):
    p = tmp_path / "skel.json"
    p.write_text(
        json.dumps(
            {
                "pose": "Test",
                "joints": {"hip": [0, 0, 0], "ls": [1.0, 2.0, 3.0]},
                "segments": [["hip", "ls"]],
            }
        )
    )
    s = core.load_skeleton(p)
    assert s.name == "Test"
    np.testing.assert_allclose(s.joints["ls"], [1.0, 2.0, 3.0])
    assert s.segments == [("hip", "ls")]


def test_load_skeleton_missing_file_uses_fallback(tmp_path: Path):
    missing = tmp_path / "nope.json"
    s = core.load_skeleton(missing, fallback_pose="Impact")
    assert isinstance(s, core.Skeleton)
    assert s.joints  # FK-derived non-empty


def test_load_skeleton_drops_malformed_segment_entries(tmp_path: Path):
    p = tmp_path / "skel.json"
    p.write_text(
        json.dumps(
            {
                "joints": {"hip": [0, 0, 0]},
                "segments": [["hip", "ls"], "not-a-list", [1, 2, 3]],
            }
        )
    )
    s = core.load_skeleton(p)
    assert s.segments == [("hip", "ls")]


def test_load_skeleton_no_segments_falls_back_to_fallback_segments(tmp_path: Path):
    p = tmp_path / "skel.json"
    p.write_text(json.dumps({"joints": {"hip": [0, 0, 0]}, "segments": []}))
    s = core.load_skeleton(p)
    assert s.segments == core.FALLBACK_SEGMENTS


# ---------------------------------------------------------------------------
# _safe helper (used by header parser; legacy path)
# ---------------------------------------------------------------------------


def test_safe_returns_default_for_missing_key():
    row = pd.Series({"x": 1.0})
    assert core._safe(row, 999, default=42.0) == 42.0


def test_safe_returns_default_for_nan():
    row = pd.Series({"x": float("nan")})
    assert core._safe(row, "x", default=7.0) == 7.0


def test_safe_returns_default_for_non_floatable():
    row = pd.Series({"x": "abc"})
    assert core._safe(row, "x", default=3.14) == 3.14


def test_safe_returns_float_when_present():
    row = pd.Series({"x": "1.5"})
    assert core._safe(row, "x") == 1.5


# ---------------------------------------------------------------------------
# Simscape trajectory CSV
# ---------------------------------------------------------------------------


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, index=False)
    return path


def test_load_simscape_trajectory_csv_short_form(tmp_path: Path):
    n = 3
    df = pd.DataFrame(
        {
            "time": np.linspace(0.0, 0.2, n),
            "club_head_X": [1.0, 1.1, 1.2],
            "club_head_Y": [0.0, 0.0, 0.0],
            "club_head_Z": [0.0, 0.0, 0.0],
            "left_hand_X": [0.5, 0.5, 0.5],
            "left_hand_Y": [0.1, 0.1, 0.1],
            "left_hand_Z": [0.2, 0.2, 0.2],
            "right_hand_X": [0.7, 0.7, 0.7],
            "right_hand_Y": [0.1, 0.1, 0.1],
            "right_hand_Z": [0.2, 0.2, 0.2],
        }
    )
    p = _write_csv(tmp_path / "t.csv", df)
    traj = core.load_simscape_trajectory_csv(p)
    assert len(traj) == n
    np.testing.assert_allclose(traj.times, df["time"].to_numpy())
    f0 = traj.frames[0]
    assert "ch" in f0.joints and "lw" in f0.joints and "rw" in f0.joints
    # mp synthesised from lw+rw
    np.testing.assert_allclose(f0.joints["mp"], (f0.joints["lw"] + f0.joints["rw"]) / 2)
    np.testing.assert_allclose(f0.joints["butt"], f0.joints["mp"])


def test_load_simscape_trajectory_csv_long_form(tmp_path: Path):
    df = pd.DataFrame(
        {
            "time": [0.0, 0.1],
            "ClubLogs_CHGlobalPosition_1": [0.0, 0.1],
            "ClubLogs_CHGlobalPosition_2": [0.0, 0.0],
            "ClubLogs_CHGlobalPosition_3": [0.0, 0.0],
        }
    )
    p = _write_csv(tmp_path / "t.csv", df)
    traj = core.load_simscape_trajectory_csv(p)
    assert len(traj) == 2
    assert "ch" in traj.frames[0].joints


def test_load_simscape_trajectory_csv_synthesizes_torso(tmp_path: Path):
    df = pd.DataFrame(
        {
            "time": [0.0],
            "spine_X": [0.0],
            "spine_Y": [0.0],
            "spine_Z": [0.0],
            "hub_X": [0.0],
            "hub_Y": [0.0],
            "hub_Z": [1.0],
        }
    )
    p = _write_csv(tmp_path / "t.csv", df)
    traj = core.load_simscape_trajectory_csv(p)
    j = traj.frames[0].joints
    assert "torso" in j
    np.testing.assert_allclose(j["torso"], [0.0, 0.0, 0.2], atol=1e-9)


def test_load_simscape_trajectory_csv_drops_non_finite_joints(tmp_path: Path):
    df = pd.DataFrame(
        {
            "time": [0.0],
            "club_head_X": [float("nan")],
            "club_head_Y": [0.0],
            "club_head_Z": [0.0],
        }
    )
    p = _write_csv(tmp_path / "t.csv", df)
    traj = core.load_simscape_trajectory_csv(p)
    assert "ch" not in traj.frames[0].joints


def test_load_simscape_trajectory_csv_missing_time_raises(tmp_path: Path):
    df = pd.DataFrame(
        {"club_head_X": [0.0], "club_head_Y": [0.0], "club_head_Z": [0.0]}
    )
    p = _write_csv(tmp_path / "t.csv", df)
    with pytest.raises(ValueError, match="time"):
        core.load_simscape_trajectory_csv(p)


def test_load_simscape_trajectory_csv_no_recognised_joints_raises(tmp_path: Path):
    df = pd.DataFrame({"time": [0.0], "junk": [1.0]})
    p = _write_csv(tmp_path / "t.csv", df)
    with pytest.raises(ValueError, match="no recognised joint columns"):
        core.load_simscape_trajectory_csv(p)


def test_xyz_columns_for_short_and_long_forms():
    cols = ["club_head_X", "club_head_Y", "club_head_Z", "foo_x", "foo_y", "foo_z"]
    assert core._xyz_columns_for(cols, "club_head_X") == [
        "club_head_X",
        "club_head_Y",
        "club_head_Z",
    ]
    assert core._xyz_columns_for(cols, "foo_x") == ["foo_x", "foo_y", "foo_z"]


def test_xyz_columns_for_returns_none_when_incomplete():
    cols = ["club_head_X"]
    assert core._xyz_columns_for(cols, "club_head_X") is None


def test_xyz_columns_for_long_form_dim_suffix():
    cols = ["a_1", "a_2", "a_3"]
    assert core._xyz_columns_for(cols, "a_1") == ["a_1", "a_2", "a_3"]


def test_xyz_columns_for_unknown_suffix_returns_none():
    assert core._xyz_columns_for(["x_Q"], "x_Q") is None


# ---------------------------------------------------------------------------
# _clubtarget_to_dataframe — mocked ClubTarget surface
# ---------------------------------------------------------------------------


def _fake_target(time, butt, clubhead, quat):
    return SimpleNamespace(
        time=np.asarray(time, dtype=float),
        butt=np.asarray(butt, dtype=float),
        clubhead=np.asarray(clubhead, dtype=float),
        club_quat=np.asarray(quat, dtype=float),
    )


def test_clubtarget_to_dataframe_identity_quaternion_rotmat():
    t = _fake_target(
        time=[0.0, 0.1],
        butt=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        clubhead=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        quat=[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    )
    df = core._clubtarget_to_dataframe(t)
    assert len(df) == 2
    # Identity quaternion -> identity matrix
    np.testing.assert_allclose(df["club_Xx"].to_numpy(), [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(df["club_Yy"].to_numpy(), [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(df["club_Zz"].to_numpy(), [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(df["mid_X"].to_numpy(), [0.0, 0.0])


def test_clubtarget_to_dataframe_rescales_when_shaft_too_long():
    # Large shaft (>1.4 m) triggers cm->in correction path.
    t = _fake_target(
        time=[0.0],
        butt=[[0.0, 0.0, 0.0]],
        clubhead=[[200.0, 0.0, 0.0]],  # 200 m apart -> way over 1.4
        quat=[[1.0, 0.0, 0.0, 0.0]],
    )
    df = core._clubtarget_to_dataframe(t)
    # After rescale, positions should be much smaller.
    assert float(df["club_X"].iloc[0]) < 200.0


# ---------------------------------------------------------------------------
# load_mocap_xlsx / read_event_header — patched loaders
# ---------------------------------------------------------------------------


def test_load_mocap_xlsx_delegates_to_canonical_loader():
    fake = _fake_target(
        time=[0.0, 0.1],
        butt=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        clubhead=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        quat=[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    )
    with patch.object(core, "load_club_target", return_value=fake) as m:
        df = core.load_mocap_xlsx("ignored.xlsx", "Sheet1")
    assert "time" in df.columns
    assert len(df) == 2
    m.assert_called_once()


def test_read_event_header_wraps_canonical_markers():
    fake = SimpleNamespace(
        A_sample=1.0, T_sample=10.0, I_sample=20.0, F_sample=30.0, CHS_mph=80.5
    )
    with patch.object(core, "read_excel_event_markers", return_value=fake):
        ev = core.read_event_header("ignored.xlsx", "Sheet1")
    assert ev.A_sample == 1.0
    assert ev.CHS_mph == 80.5


# ---------------------------------------------------------------------------
# Multi-source target dispatch
# ---------------------------------------------------------------------------


class _FakeMultiTarget:
    def __init__(
        self, has_club=False, has_body=False, is_club_ball=False, club=None, body=None
    ):
        self._has_club = has_club
        self._has_body = has_body
        self._is_club_ball = is_club_ball
        self.club = club
        self.body = body

    def has_club(self):
        return self._has_club

    def has_body(self):
        return self._has_body

    def is_club_ball(self):
        return self._is_club_ball

    def shared_time(self):
        return np.array([0.0, 0.1])


def test_dispatch_cost_inputs_with_no_sources():
    t = _FakeMultiTarget()
    out = core.dispatch_cost_inputs(t)
    assert "time" in out
    assert "club" not in out
    assert "body" not in out


def test_dispatch_cost_inputs_includes_body():
    body_obj = object()
    t = _FakeMultiTarget(has_body=True, body=body_obj)
    out = core.dispatch_cost_inputs(t)
    assert out["body"] is body_obj


def test_clubtarget_from_multi_returns_none_when_no_club():
    assert core.clubtarget_from_multi(_FakeMultiTarget()) is None


def test_clubtarget_from_multi_returns_inner_clubtarget_when_present():
    real_club = core.ClubTarget.__new__(core.ClubTarget)  # bypass __init__
    composed = SimpleNamespace(club=real_club)
    t = _FakeMultiTarget(has_club=True, club=composed)
    assert core.clubtarget_from_multi(t) is real_club


def test_clubtarget_from_multi_returns_slot_for_duck_typed_objects():
    duck = SimpleNamespace(butt=None, clubhead=None, club_quat=None)
    t = _FakeMultiTarget(has_club=True, club=duck)
    # Returns duck (not a ClubTarget instance) — the contract is duck-typed.
    assert core.clubtarget_from_multi(t) is duck


def test_dispatch_cost_inputs_with_club_and_ball_flag():
    real_club = core.ClubTarget.__new__(core.ClubTarget)
    composed = SimpleNamespace(club=real_club)
    t = _FakeMultiTarget(has_club=True, is_club_ball=True, club=composed)
    out = core.dispatch_cost_inputs(t)
    assert out["club"] is real_club
    assert out["has_ball"] is True
