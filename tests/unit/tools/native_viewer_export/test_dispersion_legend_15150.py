"""Dispersion key, hole-bounds clipping and nominal fallback (#15150 B1)."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.tools.native_viewer_export import core
from src.tools.native_viewer_export import dispersion_hole as dh
from src.tools.native_viewer_export import dispersion_traces as dt
from src.tools.native_viewer_export import shot_dispersion as sd
from src.tools.native_viewer_export.display_units import DisplayUnits
from tests.unit.tools.native_viewer_export import test_dispersion_traces_15150 as t15

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _settings(**changes) -> dt.DispersionTraceSettings:
    return replace(dt.load_trace_settings(), **changes)


def test_new_settings_are_documented_and_validated() -> None:
    raw = json.loads(dt.SETTINGS_PATH.read_text(encoding="utf-8"))
    new = {"key_samples", "key_nominal", "hole_key_samples", "hole_clip_margin"}
    assert new <= set(raw) and new <= set(raw["sources"])
    with pytest.raises(ValueError, match="hole_clip_margin"):
        _settings(hole_clip_margin=-0.1)
    with pytest.raises(ValueError, match="key_samples"):
        _settings(key_samples="")
    with pytest.raises(ValueError, match="hole_key_samples"):
        _settings(hole_key_samples="{bad}")


def test_clip_key_names_samples_and_nominal_in_their_colours() -> None:
    flight = t15._hole_flight((10, 20, 30, 255))
    entries = flight.dispersion.key_entries(flight.settings.tracer_color_rgba)
    cfg = flight.dispersion.settings
    assert entries == (
        (cfg.key_samples, (10, 20, 30)),
        (cfg.key_nominal, tuple(flight.settings.tracer_color_rgba[:3])),
    )
    with_flight = core._key_entries(None, core.ExportSettings(), flight)
    assert with_flight == entries
    assert core._key_entries(None, core.ExportSettings(), None) == ()
    bare = replace(flight, dispersion=None)
    assert core._key_entries(None, core.ExportSettings(), bare) == ()


def test_hole_still_legend_names_the_samples() -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    from src.tools.native_viewer_export import hole_view as hv

    flight = t15._hole_flight()
    fig, plan, _ = hv.build_still(flight)
    texts = [t.get_text() for t in plan.get_legend().get_texts()]
    expect = flight.dispersion.settings.hole_key_samples.format(
        n=len(flight.dispersion.samples)
    )
    assert expect in texts and len(texts) >= 2
    plt.close(fig)


def test_clip_to_bounds_splits_a_path_at_the_edges() -> None:
    xy = np.array([[0.0, 0.0], [5.0, 0.0], [15.0, 0.0], [5.0, 0.0], [1.0, 0.0]])
    runs = dt.clip_to_bounds(xy, (0.0, 10.0), (-1.0, 1.0))
    assert len(runs) == 2
    assert runs[0][:, 0].max() == pytest.approx(10.0)  # cut at the edge
    assert runs[1][0, 0] == pytest.approx(10.0)
    inside = dt.clip_to_bounds(xy[:2], (0.0, 10.0), (-1.0, 1.0))
    assert len(inside) == 1 and np.allclose(inside[0], xy[:2])
    assert dt.clip_to_bounds(xy + 100.0, (0.0, 10.0), (-1.0, 1.0)) == []
    with pytest.raises(ValueError, match="bounds"):
        dt.clip_to_bounds(xy, (10.0, 0.0), (-1.0, 1.0))
    with pytest.raises(ValueError, match="points"):
        dt.clip_to_bounds(np.zeros((3, 3)), (0.0, 1.0), (0.0, 1.0))


def test_samples_leaving_the_hole_are_cut_at_its_bounds() -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    from src.tools.native_viewer_export import hole_view as hv

    flight = t15._hole_flight()
    first = flight.dispersion.samples[0]
    path = np.asarray(first.path_m)
    grow = np.linspace(0.0, 1.0, len(path))[:, None]
    wild = replace(first, path_m=path + grow * (4000.0, 0.0, 0.0))
    far = replace(flight.dispersion, samples=(wild,))
    fig, plan, side = hv.build_still(replace(flight, dispersion=far))
    for ax in (plan, side):
        lo, hi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        for ln in ax.get_lines():
            if ln.get_gid() == "dispersion_sample":
                x, y = ln.get_xdata(), ln.get_ydata()
                assert lo - 1e-6 <= x.min() and x.max() <= hi + 1e-6
                assert ylo - 1e-6 <= y.min() and y.max() <= yhi + 1e-6
    plt.close(fig)


def _nominal() -> sd.SampleOutcome:
    return t15_linear(sd.ShotSetup(0.0, 0.0))


def t15_linear(setup: sd.ShotSetup) -> sd.SampleOutcome:
    from tests.unit.tools.native_viewer_export.test_dispersion_hole_15150 import (
        _linear,
    )

    return _linear(setup)


def test_hole_still_without_samples_falls_back_to_the_nominal_shot(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    empty = sd.DispersionResult((sd.SampleOutcome(sd.ShotSetup(0.1, 0.0), reason="x"),))
    sigma, cfg = sd.load_dispersion_settings(), dh.load_dispersion_hole_settings()
    from src.tools.native_viewer_export.hole_view import load_hole_view_settings

    colour = load_hole_view_settings().colour_rgba
    fig, plan, side = dh.build_dispersion_hole_figure(
        empty, sigma, cfg, DisplayUnits("m"), _nominal()
    )
    for ax in (plan, side):
        assert [ln for ln in ax.get_lines() if ln.get_color() == colour]
    assert not [
        g
        for g in plan.get_children()
        if getattr(g, "get_gid", None)
        and g.get_gid() in {"mean_landing", "spread_ellipse"}
    ]
    assert cfg.style["title_format_nominal_only"].split("{")[0] in plan.get_title()
    plt.close(fig)
    out = dh.write_dispersion_hole(
        empty, sigma, tmp_path / "h.png", cfg, DisplayUnits("m"), _nominal()
    )
    assert out.stat().st_size > 1000
    with pytest.raises(ValueError, match="no shot"):
        dh.write_dispersion_hole(empty, sigma, tmp_path / "x.png")


def test_cli_writes_the_nominal_hole_still_when_no_sample_flies(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("matplotlib")
    from src.tools.native_viewer_export import shot_dispersion_cli as cli

    empty = sd.DispersionResult((sd.SampleOutcome(sd.ShotSetup(0.1, 0.0), reason="x"),))
    nominal = sd.DispersionResult((_nominal(),))
    runs = iter([empty, nominal])
    monkeypatch.setattr(core, "load_swing_input", lambda *a, **k: None)
    monkeypatch.setattr(sd, "run_dispersion", lambda *a, **k: next(runs))
    code = cli.main(["--bundle", "b", "--swing", "s", "--out", str(tmp_path), "--hole"])
    assert code == 0
    assert (tmp_path / "dispersion_hole.png").stat().st_size > 1000
    assert not (tmp_path / "dispersion.png").exists()
