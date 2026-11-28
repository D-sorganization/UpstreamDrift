"""Tests for time interpolation, Butterworth smoothing, and hysteresis (#15788)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.force_overlay import contracts as fc
from src.shared.python.force_overlay import glyphs as fg
from src.tools.native_viewer_export import overlay_feed as of
from src.tools.native_viewer_export import overlay_interpolation as oi

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
Vec3 = tuple[float, float, float]
_ST = fg.ForceGlyphStyle(force_scale_m_per_n=0.005, magnitude_floor_n=20.0)
_ST = replace(_ST, magnitude_floor_nm=8.0)
_LOW = replace(_ST, magnitude_floor_n=0.1, magnitude_floor_nm=0.1)
_IW = oi.interpolate_wrenches_to_frame_times


def _fr(t: float, f: Vec3, m: Vec3 = (0, 0, 0), **kw: Any) -> fc.ForceTorqueFrame:
    """Build synthetic ForceTorqueFrame."""
    ow, wk, bo = fc.OverlayWrench, fc.WrenchKind, bool(kw.get("bo", False))
    ws = [ow(wk.CONTACT, "contact:grf_net", "s", (0, 0, 0), f, m, "p")]
    ws = ws if kw.get("on", True) else []
    if (act := kw.get("act")) is not None:
        ws.append(
            ow(wk.JOINT_ACTUATOR, "actuator:hip", "p", (0, 0, 0.9), None, act, "a")
        )
    ax = AxialLoadFrame(t, {"trunk": None} if bo else {"trunk": 450.0 + t}, "ax")
    md = {"axial_impact": "unavailable"} if bo else {}
    return fc.ForceTorqueFrame(t, "mujoco", tuple(ws), ax, metadata=md)


def _tog(flags: list[bool]) -> int:
    """Count visibility transitions."""
    return sum(1 for a, b in zip(flags[:-1], flags[1:], strict=True) if a != b)


def _stalls(feed: of.OverlayFeed, n: int) -> tuple[list[bool], int]:
    """Return presence flags and late-half stall count."""
    arr = [feed.glyphs_at(i).arrows for i in range(n)]
    zs = [a[0].tip_m[2] - a[0].tail_m[2] if a else None for a in arr]
    late = [v for v in zs[int(0.6 * n) :] if v is not None]
    st = sum(abs(a - b) < 1e-12 for a, b in zip(late[:-1], late[1:], strict=True))
    return [v is not None for v in zs], st


def test_nearest_neighbor_stalls_and_flickers_while_smooth_eliminates_both() -> None:
    """Slow-motion resampling stalls/flickers raw and is smooth when enabled."""
    rng = np.random.default_rng(15788)
    st, tt = np.arange(0.0, 0.80, 0.02), np.arange(0.02, 0.76, 0.25 / 60.0)
    osc = (st >= 0.08) & (st <= 0.24)
    fz = 12.0 + 60.0 * (st / 0.78) + rng.normal(0.0, 2.5, st.size)
    fz[osc] = 20.0 + np.where(np.arange(osc.sum()) % 2 == 0, 2.2, -2.2)
    tz = 4.0 + 12.0 * (st / 0.78) + rng.normal(0.0, 0.8, st.size)
    tz[osc] = 8.0 + np.where(np.arange(osc.sum()) % 2 == 0, 0.9, -0.9)
    u = np.maximum(1.0, fz)
    frames = [
        _fr(float(t), (0.15 * v, 0, v), (0, -0.1 * v, 0.5), act=(0, 0, m))
        for t, v, m in zip(st, u.tolist(), tz.tolist(), strict=True)
    ]
    near = [int(np.argmin(np.abs(st - t))) for t in tt]
    sw = SimpleNamespace(source_times_s=st.tolist(), sample_times_s=tt.tolist())
    base = of.OverlayFeed(lambda i: frames[i], style=_ST)
    raw, sm = (base.following(sw, near.__getitem__, smooth=b) for b in (False, True))
    rf, r_st = _stalls(raw, len(tt))
    sf, s_st = _stalls(sm, len(tt))
    assert r_st > len(tt) // 5 and s_st == 0 and _tog(rf) >= 4 and _tog(sf) == 1
    assert _tog([bool(raw.glyphs_at(i).torque_arcs) for i in range(len(tt))]) >= 4
    assert _tog([bool(sm.glyphs_at(i).torque_arcs) for i in range(len(tt))]) == 1


def test_cop_recomputed_and_butterworth_zero_phase_short_guards() -> None:
    """CoP recomputes above cop_fz_min_n; Butterworth has zero lag and short guard."""
    st, fzs = [0.0, 0.1, 0.2, 0.3], [4.0, 100.0, 100.0, 100.0]
    frames = [
        _fr(t, (0, 0, f), ((-0.05 + 0.2 * t) * f, -(0.1 + 0.5 * t) * f, 1.5))
        for t, f in zip(st, fzs, strict=True)
    ]
    cfg = oi.FlickerSuppressionConfig(cutoff_hz=100.0, cop_fz_min_n=10.0)
    out = _IW(frames, st, [0.0, 0.15, 0.25], config=cfg)
    exp = [(0.0, 0.0, 0.0), (0.175, -0.02, 0.0), (0.225, 0.0, 0.0)]
    assert np.allclose([f.wrenches[0].point_m for f in out], exp, atol=1e-6)
    fs, t_arr = 200.0, np.arange(0.0, 1.0, 1.0 / 200.0)
    filt = oi.zero_phase_butterworth_filter
    pulse = np.exp(-0.5 * ((t_arr - 0.5) / 0.05) ** 2)
    assert int(np.argmax(filt(pulse, fs, 15.0))) == int(np.argmax(pulse))
    assert np.std(filt(np.sin(2.0 * np.pi * 60.0 * t_arr), fs, 10.0)[20:-20]) < 0.05
    assert all(filt(np.ones(n), fs, 15.0).shape == (n,) for n in (0, 1, 2, 3, 5))
    assert np.allclose(filt(pulse, fs, 120.0), pulse)


def test_hysteresis_ramp_dwell_and_smooth_opacity_scale_near_floor() -> None:
    """Hysteresis stops flicker and smoothly ramps opacity/scale near floor."""
    with pytest.raises(ValueError):
        oi.FlickerSuppressionConfig(engage_n=15.0, release_n=20.0)
    cfg = oi.FlickerSuppressionConfig(engage_n=22.0, release_n=16.0, min_dwell_frames=3)
    states: dict[str, oi.HysteresisState] = {}
    mags = [18.0, 21.0, 19.0, 24.0, 12.0, 18.0, 22.0, 35.0, 14.0]
    hr, g0 = oi.apply_hysteresis_ramp, fg.build_glyphs
    fr_l = [_fr(k * 0.01, (0, 0, m)) for k, m in enumerate(mags)]
    outs = [hr(g0(fr, _LOW), cfg, _ST, states) for fr in fr_l]
    assert [bool(gs.arrows) for gs in outs] == [False] * 3 + [True] * 5 + [False]
    a5, a6, a7 = (outs[i].arrows[0].rgba[3] for i in (5, 6, 7))
    assert 0.0 < a5 < a6 < a7 <= 1.0


def test_liftoff_gap_blackout_and_raw_smooth_false_toggle() -> None:
    """Lift-off gaps and blackout windows are never bridged; smooth=False is raw."""
    st = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    act = [True, True, True, False, False, True, True, True, True]
    fz = [200.0, 200.0, 200.0, 0.0, 0.0, 400.0, 999.0, 100.0, 100.0]
    bo = [False] * 6 + [True, False, False]
    z = zip(st, fz, act, bo, strict=True)
    frames = [_fr(t, (0, 0, f), on=a, bo=b) for t, f, a, b in z]
    cfg = oi.FlickerSuppressionConfig(cutoff_hz=100.0)
    tq = [0.025, 0.175, 0.26, 0.3, 0.34]
    out = _IW(frames, st, tq, contact_active=act, config=cfg)
    assert out[1].wrenches == () and out[3].axial_loads is not None
    assert all(v is None for v in out[3].axial_loads.values_n.values())
    assert of.CONTACT_BLACKOUT_NOTE in of.OverlayFeed(lambda i: out[i]).hud_lines_at(3)
    assert (f2 := out[2].wrenches[0].force_n) and f2[2] == pytest.approx(400.0)
    assert (f4 := out[4].wrenches[0].force_n) and f4[2] == pytest.approx(100.0)
    style, near, t0 = of.default_glyph_style(), [0, 1, 1, 2], (0.01, 0.09, 0.11, 0.19)
    sw = SimpleNamespace(source_times_s=(0.0, 0.1, 0.2), sample_times_s=t0)
    rf = [_fr(t, (0, 0, v)) for t, v in ((0.0, 100.0), (0.1, 200.0), (0.2, 300.0))]
    fd = of.OverlayFeed(lambda i: rf[i], style=style)
    raw = fd.following(sw, near.__getitem__, smooth=False)
    for k, idx in enumerate(near):
        assert raw.glyphs_at(k) == fg.build_glyphs(rf[idx], style)
