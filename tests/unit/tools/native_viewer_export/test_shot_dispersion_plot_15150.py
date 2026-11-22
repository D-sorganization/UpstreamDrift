"""Shot dispersion summary, ellipse, figure and CLI parser (#15150)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.tools.native_viewer_export import shot_dispersion as sd
from src.tools.native_viewer_export.shot_dispersion import SampleOutcome, ShotSetup

pytestmark = pytest.mark.unit


def _trace(carry: float, side: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, 5)
    return np.column_stack([carry * t, side * t, 30.0 * t * (1.0 - t)])


def _linear(setup: ShotSetup) -> SampleOutcome:
    side, carry = 6.6 * setup.club_roll_deg, 250.0 - 20.0 * setup.ball_shift_m
    metrics = dict.fromkeys(sd.METRICS, 1.0)
    metrics.update(carry_m=carry, side_m=side)
    return SampleOutcome(setup, None, metrics, _trace(carry, side))


def test_packaged_settings_and_clipping():
    cfg = sd.load_dispersion_settings()
    assert (cfg.n_shots, cfg.seed) == (10, 0)
    raw = json.loads(sd.SETTINGS_PATH.read_text())
    assert "not a measurement" in raw["sources"]["club_roll_sd_deg"]
    wild = sd.sample_setups(sd.DispersionSettings(50, 0, 1000.0, 0.005))
    assert max(abs(s.club_roll_deg) for s in wild) <= 20.0


def test_summary_matches_numpy():
    setups = sd.sample_setups(sd.load_dispersion_settings())
    summary = sd.run_dispersion(None, None, setups, evaluate=_linear).summary()
    sides = [6.6 * s.club_roll_deg for s in setups]
    assert summary["side_m"]["mean"] == pytest.approx(np.mean(sides))
    assert summary["side_m"]["std"] == pytest.approx(np.std(sides))


def test_ellipse_axes_match_covariance():
    pts = np.random.default_rng(3).standard_normal((4000, 2)) * [4.0, 1.0]
    _, width, height, angle = sd.landing_ellipse(pts, 1.0)
    assert width == pytest.approx(8.0, rel=0.1)
    assert height == pytest.approx(2.0, rel=0.1)
    assert abs(angle) < 5.0 or abs(abs(angle) - 180.0) < 5.0
    assert sd.landing_ellipse(pts[:2], 1.0) is None


def test_plot_writes_png_and_refuses_without_flight(tmp_path):
    pytest.importorskip("matplotlib")
    from src.tools.native_viewer_export.shot_dispersion_plot import draw_dispersion

    cfg = sd.load_dispersion_settings()
    result = sd.run_dispersion(None, None, sd.sample_setups(cfg), evaluate=_linear)
    assert draw_dispersion(result, cfg, tmp_path / "d.png").stat().st_size > 1000
    none = sd.run_dispersion(
        None, None, [ShotSetup(0.0, 0.0)], evaluate=lambda s: SampleOutcome(s, "x")
    )
    with pytest.raises(ValueError):
        draw_dispersion(none, cfg, tmp_path / "e.png")


def test_cli_parser_defaults_come_from_settings():
    from src.tools.native_viewer_export.shot_dispersion_cli import build_parser

    args = build_parser().parse_args(["--bundle", "b", "--swing", "s", "--out", "o"])
    assert (args.n, args.seed, args.roll_sd, args.ball_sd) == (10, 0, 0.5, 0.005)
