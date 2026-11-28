"""Tests for Viewer Slice 2 visibility, trace lines, strobes, and torque curls (#15663)."""

from __future__ import annotations

import json
import math

import pytest

from src.shared.python.body_part_viz.viewer_strobe_controls import (
    BodyPartVisibilityState, ClubStrobeSample, CurledTorqueArrowSpec,
    FrozenClubStrobeSeries, TraceLineSeries, build_curled_torque_arrows,
    build_frozen_club_strobes, preset_club_and_arms_only, preset_full, preset_ghost_body,
)  # fmt: skip

pytestmark = pytest.mark.unit


def test_visibility_state_toggles_and_opacity_bounds() -> None:
    """Per-segment visibility toggles and [0.0, 1.0] opacity validation."""
    s = BodyPartVisibilityState({"torso": False, "club": True}, {"torso": 0.25, "club": 1.0})  # fmt: skip
    assert not s.is_visible("torso") and s.is_visible("club") and s.is_visible("head")
    assert s.opacity_for("torso") == pytest.approx(0.25) and s.effective_opacity("torso") == pytest.approx(0.0)  # fmt: skip
    assert s.with_segment("torso", visible=True, opacity=0.6).effective_opacity("torso") == pytest.approx(0.6)  # fmt: skip
    for bad in (-0.01, 1.01, math.nan, math.inf):
        with pytest.raises(ValueError):
            BodyPartVisibilityState(segment_opacity={"torso": bad})
    with pytest.raises(ValueError):
        BodyPartVisibilityState(default_opacity=1.5)


def test_visibility_presets() -> None:
    """Preset factories build full, ghost_body, and club_and_arms_only states."""
    segs = ("torso", "pelvis", "thigh_r", "upper_arm_l", "hand_r", "club_shaft")
    full = preset_full(segs)
    assert all(full.is_visible(x) and full.opacity_for(x) == pytest.approx(1.0) for x in segs)  # fmt: skip
    ghost = preset_ghost_body(segs, body_opacity=0.2, club_opacity=1.0)
    assert ghost.opacity_for("torso") == pytest.approx(0.2) and ghost.opacity_for("club_shaft") == pytest.approx(1.0)  # fmt: skip
    ac = preset_club_and_arms_only(segs)
    assert ac.is_visible("club_shaft") and ac.is_visible("upper_arm_l")
    assert not ac.is_visible("torso") and not ac.is_visible("thigh_r")


def test_trace_line_series_validation() -> None:
    """TraceLineSeries enforces strictly increasing times and finite color/width."""
    pts = ((0.0, 0.0, 0.1), (0.2, 0.1, 0.4))
    assert len(TraceLineSeries("club", pts, (0.0, 0.1), (0.1, 0.6, 0.9, 0.85), 2.5).points_m) == 2  # fmt: skip
    with pytest.raises(ValueError):
        TraceLineSeries("t", pts, (0.2, 0.2))
    with pytest.raises(ValueError):
        TraceLineSeries("c", pts, (0.0, 0.1), (0.5, math.nan, 0.2, 1.0))
    with pytest.raises(ValueError):
        TraceLineSeries("w", pts, (0.0, 0.1), width_px=0.0)


def test_build_frozen_club_strobes_stride_and_phase_indices() -> None:
    """Strobes extract non-overlapping poses with monotonic opacity decay."""
    t = [i * 0.05 for i in range(10)]
    g, h, d = [(0.1 * i, 0.0, 1.0) for i in range(10)], [(0.15 * i, 0.1, 0.3) for i in range(10)], [(0.16 * i, 0.12, 0.25) for i in range(10)]  # fmt: skip
    by_s = build_frozen_club_strobes(t, g, h, d, stride=3, opacity_range=(0.2, 1.0))
    ops = [s.opacity for s in by_s.samples]
    assert by_s.selection_mode == "stride" and [s.frame_idx for s in by_s.samples] == [0, 3, 6, 9]  # fmt: skip
    assert ops[0] == pytest.approx(0.2) and ops[-1] == pytest.approx(1.0)
    assert all(a < b for a, b in zip(ops[:-1], ops[1:], strict=True))
    g[4], h[4], d[4] = g[3], h[3], d[3]
    by_p = build_frozen_club_strobes(t, g, h, d, phase_indices={"address": 0, "top": 3, "dwell": 4, "impact": 9})  # fmt: skip
    assert [s.frame_idx for s in by_p.samples] == [0, 3, 9] and [s.phase_label for s in by_p.samples] == ["address", "top", "impact"]  # fmt: skip


def test_build_curled_torque_arrows_direction_bounds_and_cutoff() -> None:
    """Curled torque arrows enforce [0, 2*pi) sweep, CCW/CW sign, and floor cutoff."""
    arrows = build_curled_torque_arrows(
        ("pelvis_yaw", "shoulder_l", "wrist_r"),
        ((0.0, 0.0, 0.9), (0.0, 0.2, 1.4), (0.3, 0.1, 1.0)),  # fmt: skip
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        (50.0, -500.0, 0.2),
    )
    assert len(arrows) == 2
    ccw, cw = arrows
    assert ccw.curl_direction == "ccw" and 0.0 < ccw.sweep_angle_rad < 2.0 * math.pi
    assert cw.curl_direction == "cw" and 0.0 < cw.sweep_angle_rad < 2.0 * math.pi
    for arrow, sign in ((ccw, 1.0), (cw, -1.0)):
        rx, ry = arrow.head_tip_m[0] - arrow.center_m[0], arrow.head_tip_m[1] - arrow.center_m[1]  # fmt: skip
        tx, ty = arrow.head_tip_m[0] - arrow.head_base_m[0], arrow.head_tip_m[1] - arrow.head_base_m[1]  # fmt: skip
        assert (rx * ty - ry * tx) * sign > 0.0


def test_deterministic_json_round_trip() -> None:
    """All Slice 2 contracts serialize deterministically and round-trip cleanly."""
    vis = BodyPartVisibilityState({"z": True, "a": False}, {"z": 0.9, "a": 0.1})
    assert json.dumps(vis.to_dict()) == json.dumps(BodyPartVisibilityState.from_dict(vis.to_dict()).to_dict())  # fmt: skip
    tr = TraceLineSeries("hand", ((0.0, 0.0, 1.0), (0.2, 0.1, 1.1)), (0.0, 0.05))
    assert TraceLineSeries.from_dict(tr.to_dict()) == tr
    s0 = ClubStrobeSample(0, 0.0, (0.0, 0.0, 1.0), (0.0, 0.2, 0.2), (0.0, 0.25, 0.18), 0.5, "addr")  # fmt: skip
    st = FrozenClubStrobeSeries((s0,), "driver", "phase_indices")
    assert FrozenClubStrobeSeries.from_dict(st.to_dict()) == st
    ar = build_curled_torque_arrows(("pelvis",), ((0.0, 0.0, 0.9),), ((0.0, 0.0, 1.0),), (25.0,))[0]  # fmt: skip
    assert CurledTorqueArrowSpec.from_dict(ar.to_dict()) == ar
