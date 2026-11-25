"""Dispersion sample traces on the wide view of export clips (#15150 B1)."""

from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export import dispersion_traces as dt
from src.tools.native_viewer_export import flight_tracer as tracer
from src.tools.native_viewer_export import shot_dispersion as sd
from src.tools.native_viewer_export.core import ClipOverlays
from src.tools.native_viewer_export.display_units import DisplayUnits
from src.tools.native_viewer_export.overlay2d import pinhole_for_view
from tests.unit.tools.native_viewer_export import test_flight_tracer as base

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

WIDE = "down_the_line_flight"
GREEN = (0, 255, 0, 255)
SHIFTS_M = (-2.0, 2.0, 4.0)  # sample paths moved along world +X (left of target)


def _nominal() -> tracer.FlightPlayback:
    return base._playback(replace(base.DEFAULTS, ball_shading=None))


def _result(nominal: tracer.FlightPlayback) -> sd.DispersionResult:
    outcomes = []
    for k, shift in enumerate(SHIFTS_M):
        metrics = dict.fromkeys(sd.METRICS, 1.0)
        metrics.update(carry_m=150.0 + 10.0 * k, side_m=-shift)
        path = np.asarray(nominal.path_m) + (shift, 0.0, 0.0)
        trace = path - path[0]  # only the metrics and world path are read here
        outcomes.append(
            sd.SampleOutcome(
                sd.ShotSetup(0.1 * k, 0.0),
                metrics=metrics,
                trace_target_m=trace,
                path_m=path,
                times_s=np.asarray(nominal.times_s),
            )
        )
    outcomes.append(sd.SampleOutcome(sd.ShotSetup(0.0, 0.0), reason="no flight"))
    return sd.DispersionResult(tuple(outcomes))


def _settings(**changes) -> dt.DispersionTraceSettings:
    return replace(dt.load_trace_settings(), **changes)


def test_settings_load_and_every_value_is_explained() -> None:
    cfg = dt.load_trace_settings()
    raw = json.loads(dt.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert set(raw["sources"]) >= set(raw) - {"sources"}
    assert WIDE in cfg.views
    assert cfg.sample_width_px < base.DEFAULTS.tracer_width_px  # nominal thicker
    assert cfg.sample_color_rgba[3] < 255  # samples translucent


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"views": ()}, "views"),
        ({"views": ("no_such_view",)}, "unknown view"),
        ({"sample_width_px": 0}, "sample_width_px"),
        ({"sample_color_rgba": (0, 0, 300, 10)}, "0..255"),
        ({"label_format": "{nope}"}, "label_format"),
    ],
)
def test_bad_settings_raise(changes, match) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        _settings(**changes)


def test_label_states_mean_and_spread_of_carry_and_side_in_display_units() -> None:
    result = _result(_nominal())
    yards = DisplayUnits("yd")
    text = dt.spread_label(result, _settings(), yards)
    carry = np.array([150.0, 160.0, 170.0]) * yards.distance_scale
    side = -np.array(SHIFTS_M) * yards.distance_scale
    assert f"{carry.mean():.1f} +/- {carry.std():.1f} yd" in text
    assert f"{side.mean():+.1f} +/- {side.std():.1f} yd" in text
    assert "3 shots" in text
    with pytest.raises(ValueError, match="no shot"):
        dt.spread_label(sd.DispersionResult(()), _settings(), yards)


def test_overlay_keeps_available_shots_on_the_nominal_clock() -> None:
    nominal = _nominal()
    overlay = dt.overlay_for(_result(nominal), nominal, _settings())
    assert len(overlay.samples) == len(SHIFTS_M)
    assert all(s.impact_time_s == nominal.impact_time_s for s in overlay.samples)
    assert overlay.flown(base.IMPACT_S - 0.1) == []
    with pytest.raises(ValueError, match="flight path"):
        dt.overlay_for(_result(nominal), tracer.FlightPlayback(base.DEFAULTS, "x"))


def test_samples_drawn_on_the_wide_view_only_with_the_nominal_on_top() -> None:
    nominal = _nominal()
    overlay = dt.overlay_for(
        _result(nominal), nominal, _settings(sample_color_rgba=GREEN)
    )
    flight = replace(nominal, dispersion=overlay)
    tiles = {WIDE: base._tile(), "down_the_line": base._tile()}
    early = flight.draw(tiles, base.IMPACT_S - 0.1, base.LOOKAT, None)
    assert all((early[v] == 40).all() for v in tiles)
    t = base.IMPACT_S + 99.0  # landed: whole traces
    out = flight.draw(tiles, t, base.LOOKAT, None)
    green = (out[WIDE] == GREEN[:3]).all(axis=2)
    assert green.sum() > 30
    assert not (out["down_the_line"] == GREEN[:3]).all(axis=2).any()
    fov = math.radians(base.DEFAULTS.fov_y_deg)
    size = (base._tile().shape[1], base._tile().shape[0])
    camera = pinhole_for_view(WIDE, base.LOOKAT, None, fov, size)
    for path in overlay.flown(t):  # pixels changed along each projected trace
        mid = path[len(path) // 2 : len(path) // 2 + 1]
        pixels, _ = tracer.project(camera, mid, base.DEFAULTS.near_clip_m)
        u, v = (int(round(c)) for c in pixels[0])
        assert green[max(v - 2, 0) : v + 3, max(u - 2, 0) : u + 3].any()
    same = replace(overlay, samples=(nominal,))  # a sample on the nominal path
    over = replace(nominal, dispersion=same).draw(tiles, t, base.LOOKAT, None)[WIDE]
    pixels, _ = tracer.project(camera, nominal.path_m[1:2], base.DEFAULTS.near_clip_m)
    u, v = (int(round(c)) for c in pixels[0])
    assert tuple(over[v, u]) == base.DEFAULTS.tracer_color_rgba[:3]  # nominal on top


def test_hud_states_the_spread_after_impact() -> None:
    nominal = _nominal()
    overlay = dt.overlay_for(_result(nominal), nominal, _settings())
    flight = replace(nominal, dispersion=overlay)
    assert overlay.label not in flight.hud_lines(base.IMPACT_S - 0.1)
    assert flight.hud_lines(base.IMPACT_S + 1.0)[-1] == overlay.label


def test_with_dispersion_samples_around_the_shot_the_export_plays() -> None:
    nominal = _nominal()
    seen: list[sd.ShotSetup] = []
    picked = _result(nominal).outcomes

    def evaluate(setup: sd.ShotSetup) -> sd.SampleOutcome:
        seen.append(setup)
        return replace(picked[len(seen) % len(SHIFTS_M)], setup=setup)

    from src.tools.native_viewer_export.core import ClipOverlays, ExportSettings

    raw = ExportSettings(ball_flight=replace(base.DEFAULTS, dispersion_traces=None))
    overlays = ClipOverlays(flight=nominal)
    assert dt.with_dispersion(overlays, None, raw, raw, evaluate) is overlays
    on = replace(
        raw, ball_flight=replace(raw.ball_flight, dispersion_traces=_settings())
    )
    played = replace(on, club_roll_in_hands_deg=1.5)
    got = dt.with_dispersion(overlays, None, on, played, evaluate)
    cfg = sd.load_dispersion_settings()
    assert len(seen) == cfg.n_shots
    assert np.mean([s.club_roll_deg for s in seen]) == pytest.approx(1.5, abs=0.5)
    assert got.flight.dispersion is not None and got.flight.path_m is nominal.path_m


def test_cli_flag_turns_the_overlay_and_the_flight_on() -> None:
    parser = cli.build_parser()
    required = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]
    assert cli.build_settings(parser.parse_args(required)).ball_flight is None
    args = parser.parse_args([*required, "--dispersion-traces"])
    flight = cli.build_settings(args).ball_flight
    assert isinstance(flight, tracer.TracerSettings)
    assert isinstance(flight.dispersion_traces, dt.DispersionTraceSettings)


MAGENTA = (255, 0, 255, 255)


def _hole_flight(colour=MAGENTA) -> tracer.FlightPlayback:
    from src.tools.native_viewer_export.hole_view import load_hole_view_settings

    nominal = base._playback(
        replace(base.DEFAULTS, hole_view=load_hole_view_settings())
    )
    grow = np.linspace(0.0, 1.0, len(nominal.path_m))[:, None]  # fanning out
    fanned = [
        replace(o, path_m=nominal.path_m + grow * (shift, 0.0, 0.0))
        for o, shift in zip(_result(nominal).available, SHIFTS_M, strict=True)
    ]
    result = sd.DispersionResult(tuple(fanned))
    overlay = dt.overlay_for(result, nominal, _settings(sample_color_rgba=colour))
    return replace(nominal, dispersion=overlay)


def test_hole_settings_keep_samples_thin_and_beneath_the_nominal() -> None:
    from src.tools.golf_environment.shots import load_shots_settings

    cfg = _settings()
    assert cfg.hole_sample_width_pt < load_shots_settings()["figure"]["line_width"]
    assert cfg.hole_sample_zorder < cfg.hole_nominal_zorder
    style = cfg.hole_sample_style()
    assert style.colour == pytest.approx(tuple(c / 255 for c in cfg.sample_color_rgba))
    assert style.width_pt == cfg.hole_sample_width_pt
    with pytest.raises(ValueError, match="hole_sample_width_pt"):
        _settings(hole_sample_width_pt=0.0)
    with pytest.raises(ValueError, match="zorder"):
        _settings(hole_sample_zorder=cfg.hole_nominal_zorder)


def test_samples_use_the_single_shot_hole_projection() -> None:
    from src.tools.golf_environment.shots import hole_axes, hole_from_settings
    from src.tools.golf_environment.shots import load_shots_settings
    from src.tools.native_viewer_export import hole_view as hv

    flight = _hole_flight()
    cfg = flight.settings.hole_view
    hole = hole_from_settings(load_shots_settings(), cfg.preset)
    paths = hv.sample_paths(flight, hole)
    assert len(paths) == len(SHIFTS_M)
    same = replace(flight.dispersion, samples=(flight,))
    (own,) = hv.sample_paths(replace(flight, dispersion=same), hole)
    assert np.allclose(own, hv.hole_shot(flight, cfg).path_m)  # nominal projection
    tee, _, left = hole_axes(hole)
    nominal_left = float((np.asarray(own)[-1] - tee) @ left)
    for path, shift in zip(paths, SHIFTS_M, strict=True):
        assert np.allclose(path[0], tee)  # every sample starts on the tee
        lateral = float((np.asarray(path)[-1] - tee) @ left) - nominal_left
        assert np.sign(lateral) == np.sign(shift)  # world +X is left of target
    assert hv.sample_paths(replace(flight, dispersion=None), hole) == []


def test_hole_still_draws_samples_thin_beneath_the_nominal_shot(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt
    from PIL import Image

    from src.tools.native_viewer_export import hole_view as hv

    flight = _hole_flight()
    fig, plan, side = hv.build_still(flight)
    cfg = flight.dispersion.settings
    for ax in (plan, side):
        drawn = [ln for ln in ax.get_lines() if ln.get_gid() == "dispersion_sample"]
        assert len(drawn) == len(SHIFTS_M)
        shot = [
            ln
            for ln in ax.get_lines()
            if ln.get_color() == hv.load_hole_view_settings().colour_rgba
        ]
        assert shot and all(ln.get_zorder() > drawn[0].get_zorder() for ln in shot)
        assert all(ln.get_linewidth() == cfg.hole_sample_width_pt for ln in drawn)
    plt.close(fig)
    png = hv.write_still(ClipOverlays(flight=flight), tmp_path, "s", "e")["hole_view"]
    pixels = np.asarray(Image.open(png).convert("RGB")).astype(int)
    magenta = (pixels[..., 0] > 200) & (pixels[..., 1] < 60) & (pixels[..., 2] > 200)
    assert magenta.sum() > 30
    bare = hv.write_still(
        ClipOverlays(flight=replace(flight, dispersion=None)), tmp_path / "b", "s", "e"
    )["hole_view"]
    plain = np.asarray(Image.open(bare).convert("RGB")).astype(int)
    assert not (
        (plain[..., 0] > 200) & (plain[..., 1] < 60) & (plain[..., 2] > 200)
    ).any()


def test_warns_when_no_requested_view_shows_the_samples(caplog) -> None:
    from src.tools.native_viewer_export.core import ClipOverlays, ExportSettings

    nominal = _nominal()
    picked = _result(nominal).outcomes

    def evaluate(setup: sd.ShotSetup) -> sd.SampleOutcome:
        return replace(picked[0], setup=setup)

    flight = replace(base.DEFAULTS, dispersion_traces=_settings())
    views = ("face_on", "down_the_line")
    on = ExportSettings(ball_flight=flight, views=views, multiview=False)
    overlays = ClipOverlays(flight=nominal)
    with caplog.at_level("WARNING", logger=dt.__name__):
        dt.with_dispersion(overlays, None, on, on, evaluate)
    (record,) = [r for r in caplog.records if "none of the requested" in r.message]
    assert WIDE in record.message and "face_on" in record.message
    caplog.clear()
    wide = replace(on, views=("face_on", WIDE))
    with caplog.at_level("WARNING", logger=dt.__name__):
        dt.with_dispersion(overlays, None, wide, wide, evaluate)
    assert not [r for r in caplog.records if "none of the requested" in r.message]
