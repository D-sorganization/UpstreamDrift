"""Dispersion on the hole: shots, spread overlays, title, legend and CLI (#15150)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.tools.native_viewer_export import dispersion_hole as dh
from src.tools.native_viewer_export import shot_dispersion as sd
from src.tools.native_viewer_export.display_units import DisplayUnits
from src.tools.native_viewer_export.shot_dispersion import SampleOutcome, ShotSetup

pytestmark = pytest.mark.unit

YARDS = DisplayUnits("yd")
METRES = DisplayUnits("m")


def _trace(carry: float, side: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, 6)
    return np.column_stack([carry * t, side * t, 30.0 * t * (1.0 - t)])


def _linear(setup: ShotSetup) -> SampleOutcome:
    side, carry = 6.6 * setup.club_roll_deg, 250.0 - 20.0 * setup.ball_shift_m
    metrics = dict.fromkeys(sd.METRICS, 1.0)
    metrics.update(carry_m=carry, side_m=side)
    roll = (8.0, (8.0, 0.5)) if setup.club_roll_deg > 0 else (None, None)
    return SampleOutcome(
        setup, None, metrics, _trace(carry, side), roll_m=roll[0], roll_offset_m=roll[1]
    )


@pytest.fixture
def result() -> sd.DispersionResult:
    cfg = sd.load_dispersion_settings()
    return sd.run_dispersion(None, None, sd.sample_setups(cfg), evaluate=_linear)


def test_real_settings_load_and_every_number_is_explained():
    cfg = dh.load_dispersion_hole_settings()
    raw = json.loads(dh.SETTINGS_PATH.read_text())
    assert cfg.preset == "par4" and 0.0 < cfg.style["ellipse_fill_alpha"] <= 1.0
    assert set(raw["sources"]) >= set(raw) - {"sources"}


def test_bad_settings_raise():
    raw = json.loads(dh.SETTINGS_PATH.read_text())
    raw["ellipse_fill_alpha"] = 1.5
    with pytest.raises(ValueError, match="ellipse_fill_alpha"):
        dh.DispersionHoleSettings(raw.pop("preset"), raw)
    with pytest.raises(ValueError, match="preset"):
        dh.load_dispersion_hole_settings(preset="par9")


def test_shots_start_on_tee_keep_side_sign_and_roll(result):
    pytest.importorskip("matplotlib")
    from src.tools.golf_environment.shots import hole_axes, hole_from_settings
    from src.tools.golf_environment.shots import load_shots_settings

    cfg = dh.load_dispersion_hole_settings()
    hole = hole_from_settings(load_shots_settings(), cfg.preset)
    shots = dh.dispersion_hole_shots(result, hole, cfg)
    assert len(shots) == len(result.available)
    tee, _, left = hole_axes(hole)
    for shot, o in zip(shots, result.available, strict=True):
        assert np.allclose(shot.path_m[0], tee)
        end_left = float((np.asarray(shot.path_m)[-1] - tee) @ left)
        assert end_left == pytest.approx(-o.metrics["side_m"])
        assert (shot.roll_m is not None) == (o.roll_m is not None)
    with pytest.raises(ValueError, match="no shot"):
        dh.dispersion_hole_shots(sd.DispersionResult(()), hole, cfg)


def test_title_and_stats_line(result):
    cfg = dh.load_dispersion_hole_settings()
    note = dh.title_note(result, cfg)
    for text in ("Replay-level dispersion", "Aim: address line", "club roll"):
        assert text in note
    carries = np.array([o.metrics["carry_m"] for o in result.available])
    assert f"{carries.mean():.1f} +/- {carries.std():.1f} m" in dh.stats_line(
        result, cfg, METRES
    )
    yd = dh.stats_line(result, cfg, YARDS)
    assert f"{carries.mean() / 0.9144:.1f}" in yd and "yd" in yd


def test_figure_has_mean_ellipse_band_and_legend_stats(result):
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    cfg = dh.load_dispersion_hole_settings()
    fig, plan, side = dh.build_dispersion_hole_figure(
        result, sd.load_dispersion_settings(), cfg, METRES
    )
    gids = {a.get_gid() for a in plan.get_children() + side.get_children()}
    assert {"mean_landing", "spread_ellipse"} <= gids
    texts = [t.get_text() for t in plan.get_legend().get_texts()]
    assert any("Carry" in t and "+/-" in t for t in texts)
    plt.close(fig)


def test_write_png_and_refuse_empty(result, tmp_path):
    pytest.importorskip("matplotlib")
    sigma = sd.load_dispersion_settings()
    out = dh.write_dispersion_hole(result, sigma, tmp_path / "d" / "h.png")
    assert out.stat().st_size > 1000
    with pytest.raises(ValueError, match="no shot"):
        dh.write_dispersion_hole(sd.DispersionResult(()), sigma, tmp_path / "x.png")


def test_outcome_carries_the_flight_roll():
    flight = SimpleNamespace(
        path_m=np.zeros((2, 3)),
        axes_m=np.eye(3),
        carry_m=200.0,
        lateral_m=1.0,
        launch_deg=10.0,
        spin_rpm=2500.0,
        times_s=np.zeros(2),
        status="ok",
        rollout=SimpleNamespace(roll_m=5.0, end_offset_m=(4.0, 3.0)),
    )
    out = sd._outcome(ShotSetup(0.0, 0.0), flight, "n/a")
    assert out.roll_m == 5.0 and out.roll_offset_m == (4.0, 3.0)


def test_cli_flags():
    from src.tools.native_viewer_export.shot_dispersion_cli import build_parser

    base = ["--bundle", "b", "--swing", "s", "--out", "o"]
    args = build_parser().parse_args([*base, "--hole", "--hole-preset", "par5"])
    assert args.hole and args.hole_preset == "par5"
    assert not build_parser().parse_args(base).hole


def _nominal_lines(ax, gid="nominal_shot"):
    return [ln for ln in ax.get_lines() if ln.get_gid() == gid]


def test_nominal_shot_drawn_bold_on_top_of_thin_samples(result):
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    from src.tools.golf_environment.shots import load_shots_settings
    from src.tools.native_viewer_export.dispersion_traces import load_trace_settings
    from src.tools.native_viewer_export.hole_view import load_hole_view_settings

    traces, nominal = load_trace_settings(), _linear(ShotSetup(0.0, 0.0))
    fig, plan, side = dh.build_dispersion_hole_figure(
        result,
        sd.load_dispersion_settings(),
        dh.load_dispersion_hole_settings(),
        METRES,
        nominal,
    )
    sample = traces.hole_sample_style()
    width = load_shots_settings()["figure"]["line_width"]
    for ax in (plan, side):
        (bold,) = _nominal_lines(ax)
        assert bold.get_linewidth() == width > sample.width_pt
        assert bold.get_color() == load_hole_view_settings().colour_rgba
        assert bold.get_zorder() == traces.hole_nominal_zorder
        thin = [ln for ln in ax.get_lines() if ln.get_color() == sample.colour]
        assert len(thin) >= len(result.available)
        assert all(ln.get_linewidth() == sample.width_pt for ln in thin)
        assert all(ln.get_zorder() < bold.get_zorder() for ln in thin)
    texts = [t.get_text() for t in plan.get_legend().get_texts()]
    assert dh.load_dispersion_hole_settings().style["legend_format"]["nominal"] in texts
    plt.close(fig)
    fig, plan, _ = dh.build_dispersion_hole_figure(
        result, sd.load_dispersion_settings(), dh.load_dispersion_hole_settings()
    )
    assert not _nominal_lines(plan)  # no nominal given: samples only
    plt.close(fig)
    with pytest.raises(ValueError, match="nominal"):
        dh.build_dispersion_hole_figure(
            result,
            sd.load_dispersion_settings(),
            dh.load_dispersion_hole_settings(),
            METRES,
            SampleOutcome(ShotSetup(0.0, 0.0), reason="no flight"),
        )


def test_nominal_setup_is_the_base_of_the_samples():
    cfg = sd.load_dispersion_settings()
    from dataclasses import replace

    moved = replace(cfg, base_club_roll_deg=1.5, base_ball_shift_m=0.01)
    assert sd.nominal_setup(moved) == ShotSetup(1.5, 0.01)
