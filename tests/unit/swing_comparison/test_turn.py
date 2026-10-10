"""Synthetic-fixture tests for the turn lines and X-factor (issue #12042).

A rigid line rotated by a known yaw about +Z must give exact turn values.
Frame: Z up, golfer faces -X, target -Y; backswing turn is positive.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.swing_comparison.events import SwingEvents
from src.shared.python.swing_comparison.turn import (
    MAX_FILL_GAP_S,
    TURN_BLOCK_SCHEMA,
    build_turn_block,
    fill_short_gaps,
    line_turn,
    marker_turn_lines,
    model_turn_lines,
    spec_model_points,
    validate_turn_block,
)

N = 101
T = np.arange(N) * 0.01  # 100 Hz
EVENTS = SwingEvents(
    address_idx=0,
    address_time=0.0,
    top_idx=50,
    top_time=0.5,
    impact_idx=80,
    impact_time=0.8,
    finish_idx=100,
    finish_time=1.0,
)


def _line(turn_deg: np.ndarray, width: float, z: float, address_deg: float = -90.0):
    """Left/right points whose right-to-left line sits at ``address - turn``.

    At address the left point is at -Y (target side) for a golfer facing -X, so
    the line yaw is -90 deg; a backswing turn rotates it clockwise (yaw falls).
    """
    yaw = np.radians(address_deg - turn_deg)
    half = width / 2.0
    left = np.column_stack([half * np.cos(yaw), half * np.sin(yaw), np.full(N, z)])
    return left, -left.copy() + np.array([0.0, 0.0, 2 * z])


def _markers(pelvis, trunk, shoulder):
    out = {}
    for (lname, rname), (turn, width, z) in {
        (("WaistLeft", "WaistRight")): (pelvis, 0.30, 0.9),
        (("BackLeft", "BackRight")): (trunk, 0.30, 1.3),
        (("LShoulderBack", "RShoulderBack")): (shoulder, 0.40, 1.4),
    }.items():
        out[lname], out[rname] = _line(turn, width, z)
    return out


RAMP = np.sin(np.pi * T)  # 0 -> 1 at t=0.5 -> 0 at t=1.0


@pytest.mark.unit
class TestKnownRotation:
    def test_exact_turn_values_and_xfactor(self) -> None:
        mk = _markers(40 * RAMP, 85 * RAMP, 100 * RAMP)
        res = marker_turn_lines(mk, T, EVENTS)
        assert res.pelvis.value_at(0.5) == pytest.approx(40.0, abs=1e-9)
        assert res.upper_trunk.value_at(0.5) == pytest.approx(85.0, abs=1e-9)
        assert res.shoulder_girdle.value_at(0.5) == pytest.approx(100.0, abs=1e-9)
        assert res.x_factor.value_at(0.5) == pytest.approx(45.0, abs=1e-9)
        assert res.x_factor_shoulder_girdle.value_at(0.5) == pytest.approx(
            60.0, abs=1e-9
        )
        assert res.pelvis.value_at(0.0) == 0.0
        assert res.pelvis.status == "ok"
        assert res.upper_trunk.points == ("BackLeft", "BackRight")

    def test_sign_positive_in_backswing(self) -> None:
        res = marker_turn_lines(_markers(10 * RAMP, 20 * RAMP, 30 * RAMP), T, EVENTS)
        assert np.nanmax(res.pelvis.turn_deg) > 0
        assert np.nanmin(res.pelvis.turn_deg) >= -1e-9

    def test_downswing_open_is_negative(self) -> None:
        ramp = np.where(T < 0.5, T / 0.5, 1 - 2 * (T - 0.5) / 0.5) * 50
        res = marker_turn_lines(_markers(ramp, ramp, ramp), T, EVENTS)
        assert res.pelvis.value_at(0.8) < 0

    def test_unwrap_across_180(self) -> None:
        # Address yaw 170 deg; a 40 deg backswing crosses the +-180 branch cut.
        left, right = _line(40 * RAMP, 0.3, 0.9, address_deg=170.0)
        lt = line_turn("pelvis", left, right, T, 0.0)
        assert lt.value_at(0.5) == pytest.approx(40.0, abs=1e-9)
        assert np.all(np.abs(np.diff(lt.turn_deg)) < 5.0)

    def test_independent_of_address_orientation(self) -> None:
        a = line_turn("p", *_line(30 * RAMP, 0.3, 0.9, address_deg=-90), T, 0.0)
        b = line_turn("p", *_line(30 * RAMP, 0.3, 0.9, address_deg=-80), T, 0.0)
        np.testing.assert_allclose(a.turn_deg, b.turn_deg, atol=1e-9)

    def test_max_backswing_window(self) -> None:
        res = marker_turn_lines(_markers(40 * RAMP, 40 * RAMP, 40 * RAMP), T, EVENTS)
        assert res.pelvis.max_backswing_deg(0.0, 0.8) == pytest.approx(40.0, abs=1e-6)


@pytest.mark.unit
class TestGapsAndAvailability:
    def test_short_gap_is_filled(self) -> None:
        mk = _markers(40 * RAMP, 40 * RAMP, 40 * RAMP)
        clean = marker_turn_lines(mk, T, EVENTS)
        mk["WaistLeft"] = mk["WaistLeft"].copy()
        mk["WaistLeft"][30:34] = np.nan  # 4 frames = 0.04 s
        res = marker_turn_lines(mk, T, EVENTS)
        assert res.pelvis.status == "ok"
        assert res.pelvis.filled_frames == 4
        np.testing.assert_allclose(res.pelvis.turn_deg, clean.pelvis.turn_deg, atol=0.1)

    def test_long_gap_stays_nan_with_reason(self) -> None:
        mk = _markers(40 * RAMP, 40 * RAMP, 40 * RAMP)
        n_gap = int(round(MAX_FILL_GAP_S / 0.01)) + 5
        mk["BackRight"] = mk["BackRight"].copy()
        mk["BackRight"][20 : 20 + n_gap] = np.nan
        res = marker_turn_lines(mk, T, EVENTS)
        assert res.upper_trunk.status == "partial"
        assert np.isnan(res.upper_trunk.turn_deg[20 : 20 + n_gap]).all()
        assert "unavailable" in (res.upper_trunk.reason or "")
        assert np.isnan(res.x_factor.turn_deg[20 : 20 + n_gap]).all()
        assert np.isfinite(res.x_factor.turn_deg[:15]).all()

    def test_mostly_missing_marker_is_unavailable_not_zero(self) -> None:
        mk = _markers(40 * RAMP, 40 * RAMP, 40 * RAMP)
        top = mk["LShoulderBack"].copy()
        mk["RShoulderBack"] = np.full((N, 3), np.nan)
        mk["LShoulderBack"] = top
        res = marker_turn_lines(mk, T, EVENTS)
        assert res.shoulder_girdle.status == "unavailable"
        assert np.isnan(res.shoulder_girdle.turn_deg).all()
        assert res.shoulder_girdle.reason
        assert np.isnan(res.x_factor_shoulder_girdle.turn_deg).all()
        assert res.x_factor.status == "ok"

    def test_missing_markers_give_nan_not_zero(self) -> None:
        res = marker_turn_lines({}, T, EVENTS)
        assert res.pelvis.status == "unavailable"
        assert res.pelvis.reason.startswith("missing_points")
        assert np.isnan(res.x_factor.turn_deg).all()

    def test_leading_gap_not_extrapolated(self) -> None:
        pts, _ = fill_short_gaps(
            np.vstack([np.full((2, 3), np.nan), np.ones((4, 3))]), 5
        )
        assert np.isnan(pts[:2]).all()

    def test_address_frame_missing(self) -> None:
        left, right = _line(10 * RAMP, 0.3, 0.9)
        left[:30] = np.nan
        lt = line_turn("pelvis", left, right, T, 0.0)
        assert lt.status == "unavailable"
        assert lt.reason == "address_frame_unavailable"

    def test_fallback_pair_when_preferred_mostly_missing(self) -> None:
        mk = _markers(10 * RAMP, 10 * RAMP, 10 * RAMP)
        mk["RShoulderBack"] = np.full((N, 3), np.nan)
        mk["LShoulderTop"], mk["RShoulderTop"] = _line(70 * RAMP, 0.4, 1.4)
        res = marker_turn_lines(mk, T, EVENTS)
        assert res.shoulder_girdle.points == ("LShoulderTop", "RShoulderTop")
        assert res.shoulder_girdle.value_at(0.5) == pytest.approx(70.0, abs=1e-9)


@pytest.mark.unit
class TestContracts:
    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            line_turn("p", np.zeros((N, 2)), np.zeros((N, 3)), T, 0.0)

    def test_rejects_non_array(self) -> None:
        with pytest.raises(TypeError):
            line_turn("p", [[0, 0, 0]] * N, np.zeros((N, 3)), T, 0.0)  # type: ignore[arg-type]

    def test_rejects_millimetres(self) -> None:
        left, right = _line(10 * RAMP, 0.3, 0.9)
        with pytest.raises(ValueError, match="metres"):
            line_turn("p", left * 1000.0, right * 1000.0, T, 0.0)

    def test_rejects_bad_time_and_gap(self) -> None:
        left, right = _line(10 * RAMP, 0.3, 0.9)
        with pytest.raises(ValueError):
            line_turn("p", left, right, T[::-1], 0.0)
        with pytest.raises(ValueError, match="max_gap_s"):
            line_turn("p", left, right, T, 0.0, max_gap_s=-1.0)

    def test_degenerate_vertical_line_is_unavailable(self) -> None:
        left = np.zeros((N, 3))
        right = np.zeros((N, 3))
        left[:, 2] = 0.5
        lt = line_turn("p", left, right, T, 0.0)
        assert lt.status == "unavailable"


@pytest.mark.unit
class TestModelAdapters:
    def test_model_points_use_same_definition(self) -> None:
        pts = {}
        pts["hip_l"], pts["hip_r"] = _line(45 * RAMP, 0.18, 0.9)
        pts["thorax_l"], pts["thorax_r"] = _line(95 * RAMP, 0.25, 1.3)
        pts["shoulder_l"], pts["shoulder_r"] = _line(105 * RAMP, 0.35, 1.4)
        res = model_turn_lines(pts, T, EVENTS)
        assert res.pelvis.value_at(0.5) == pytest.approx(45.0, abs=1e-9)
        assert res.upper_trunk.value_at(0.5) == pytest.approx(95.0, abs=1e-9)
        assert res.shoulder_girdle.value_at(0.5) == pytest.approx(105.0, abs=1e-9)
        assert res.x_factor.value_at(0.5) == pytest.approx(50.0, abs=1e-9)

    def test_model_waist_sites_fallback_for_pelvis(self) -> None:
        pts = {}
        pts["WaistLeft"], pts["WaistRight"] = _line(33 * RAMP, 0.3, 0.9)
        res = model_turn_lines(pts, T, EVENTS)
        assert res.pelvis.points == ("WaistLeft", "WaistRight")
        assert res.pelvis.value_at(0.5) == pytest.approx(33.0, abs=1e-9)

    def test_model_on_coarser_time_grid(self) -> None:
        # The model may be decimated; events are looked up by time.
        t4 = T[::5]
        pts = {}
        left, right = _line(40 * RAMP, 0.18, 0.9)
        pts["hip_l"], pts["hip_r"] = left[::5], right[::5]
        res = model_turn_lines(pts, t4, EVENTS)
        assert res.pelvis.value_at(0.5) == pytest.approx(40.0, abs=1e-9)

    def test_spec_model_points_rejects_bad_q(self) -> None:
        with pytest.raises(ValueError, match="q must be"):
            spec_model_points({}, np.zeros(5))


@pytest.mark.unit
class TestTurnBlock:
    def test_block_is_json_safe_and_valid(self) -> None:
        mk = marker_turn_lines(_markers(40 * RAMP, 80 * RAMP, 90 * RAMP), T, EVENTS)
        block = build_turn_block(EVENTS, markers=mk, model=mk, model_source="unit")
        assert block["schema"] == TURN_BLOCK_SCHEMA
        text = json.dumps(block, allow_nan=False)  # NaN would raise
        assert json.loads(text)["markers"]["pelvis"]["top_deg"] == pytest.approx(40.0)
        assert block["model"]["source"] == "unit"
        assert block["markers"]["x_factor"]["top_deg"] == pytest.approx(40.0)

    def test_unavailable_serialises_as_null_with_reason(self) -> None:
        block = build_turn_block(EVENTS, markers=marker_turn_lines({}, T, EVENTS))
        entry = block["markers"]["pelvis"]
        assert entry["status"] == "unavailable"
        assert entry["top_deg"] is None and entry["reason"]
        json.dumps(block, allow_nan=False)

    def test_validator_rejects_malformed_blocks(self) -> None:
        mk = marker_turn_lines(_markers(1 * RAMP, 1 * RAMP, 1 * RAMP), T, EVENTS)
        good = build_turn_block(EVENTS, markers=mk)
        validate_turn_block(good)
        bad = json.loads(json.dumps(good))
        del bad["markers"]["pelvis"]
        with pytest.raises(ValueError, match="pelvis"):
            validate_turn_block(bad)
        bad = json.loads(json.dumps(good))
        bad["markers"]["pelvis"]["top_deg"] = "40"
        with pytest.raises(ValueError, match="finite or null"):
            validate_turn_block(bad)
        bad = json.loads(json.dumps(good))
        bad["markers"]["pelvis"].update(status="unavailable", reason=None)
        with pytest.raises(ValueError, match="without reason"):
            validate_turn_block(bad)
        bad = json.loads(json.dumps(good))
        bad["markers"] = bad["model"] = None
        with pytest.raises(ValueError, match="at least one"):
            validate_turn_block(bad)
        with pytest.raises(ValueError, match="schema"):
            validate_turn_block({"schema": "x"})
