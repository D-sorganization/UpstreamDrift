"""Matplotlib renderer for the ground-reaction plots (GCV-5, #11711)."""

from __future__ import annotations

from dataclasses import replace

from matplotlib.figure import Figure
import numpy as np
import pytest

from scripts.check_document_title_case import expected_title
from src.shared.python.biomechanics.ground_reaction import (
    ContactSet,
    GroundReactionSeries,
    analyze_ground_reaction,
)
from src.shared.python.biomechanics.ground_reaction_plot_model import (
    build_ground_reaction_plot_series,
    unavailable_ground_reaction_plot,
)
from src.shared.python.plotting.renderers.ground_reaction import (
    PANEL_TITLES,
    plot_ground_reaction,
)
from src.shared.python.plotting.renderers.time_panels import draw_time_panel

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _stance(fz_left: float, fz_right: float):
    contacts = {}
    for name, fz, y in (("left", fz_left, 0.15), ("right", fz_right, -0.15)):
        contacts[name] = (
            ContactSet(np.array([[0.02, 0.0, fz]]), np.array([[0.1, y, 0.0]]))
            if fz > 0.0
            else ContactSet.empty()
        )
    return analyze_ground_reaction(contacts, (0.0, 0.0, 0.95))


def _plot(**kwargs):
    series = GroundReactionSeries.from_breakdowns(
        [0.0, 0.1, 0.2], [_stance(400, 400), _stance(600, 200), _stance(800, 0)]
    )
    return build_ground_reaction_plot_series(series, **kwargs)


def test_six_panels_with_title_case_titles_and_event_markers() -> None:
    fig = Figure()
    plot_ground_reaction(fig, _plot(events={"impact": 0.1}))
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == list(PANEL_TITLES)
    assert all(expected_title(t) == t for t in titles)
    time_axes = [ax for ax in fig.axes if ax.get_title() != "Centre of Pressure Path"]
    for ax in time_axes:
        assert any(
            len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [0.1, 0.1])
            for line in ax.get_lines()
        )


def test_vertical_force_uses_body_weights_when_available() -> None:
    fig = Figure()
    plot_ground_reaction(fig, _plot(body_weight_n=800.0))
    ax = fig.axes[0]
    assert ax.get_ylabel() == "BW"
    net = next(line for line in ax.get_lines() if line.get_label() == "Net")
    np.testing.assert_allclose(net.get_ydata(), [1.0, 1.0, 1.0])


def test_cop_path_is_a_top_view_with_a_gap_for_the_unloaded_foot() -> None:
    fig = Figure()
    plot_ground_reaction(fig, _plot())
    ax = next(a for a in fig.axes if a.get_title() == "Centre of Pressure Path")
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("x (m)", "y (m)")
    right = next(line for line in ax.get_lines() if line.get_label() == "Right")
    assert np.isnan(right.get_xdata()[2])  # unloaded: a gap, not the origin


def test_unavailable_series_says_why_in_every_panel() -> None:
    fig = Figure()
    plot_ground_reaction(fig, unavailable_ground_reaction_plot("no GRF in Simscape"))
    for ax in fig.axes:
        texts = [t.get_text() for t in ax.texts]
        assert any("unavailable: no GRF in Simscape" in t for t in texts)


def test_time_panel_draws_gaps_and_unavailable_text() -> None:
    fig = Figure()
    ax = fig.subplots()
    y = np.array([1.0, np.nan, 3.0])
    draw_time_panel(
        ax, [0.0, 0.1, 0.2], [(y, "A", "#000000")], title="T", unit="N", events={}
    )
    assert np.isnan(ax.get_lines()[0].get_ydata()[1])
    assert not ax.texts
    empty = fig.add_subplot(2, 1, 2)
    draw_time_panel(
        empty,
        [0.0],
        [(np.array([np.nan]), "A", "#000000")],
        title="T",
        unit="N",
        events={},
        reason="engine",
    )
    assert empty.texts[0].get_text() == "unavailable: engine"
    with pytest.raises(ValueError, match="length"):
        draw_time_panel(
            ax, [0.0], [(y, "A", "#000000")], title="T", unit="N", events={}
        )


def test_measured_overlay_is_dashed_and_unqualified_banner_is_drawn() -> None:
    base = _plot()
    unobserved = {a: [None] * 3 for a in ("x", "y", "z", "magnitude")}
    cop = {a: [0.01, 0.02, 0.03] for a in ("x", "y", "z", "magnitude")}
    series = replace(
        base,
        traces={
            **base.traces,
            "measured_left_force_n": {**unobserved, "z": [400.0, 600.0, 800.0]},
            "measured_left_cop_m": cop,
        },
        measured_qualified=False,
    )
    fig = Figure()
    plot_ground_reaction(fig, series)
    dashed = [
        line
        for ax in fig.axes
        for line in ax.get_lines()
        if line.get_linestyle() == "--" and line.get_label().startswith("Measured")
    ]
    assert len(dashed) == 2
    assert "Unqualified measurement" in [t.get_text() for t in fig.texts]
    plot_ground_reaction(fig, replace(series, measured_qualified=True))
    assert "Unqualified measurement" not in [t.get_text() for t in fig.texts]


def _measured_series(**changes):
    base = _plot()
    unobserved = {a: [None] * 3 for a in ("x", "y", "z", "magnitude")}
    series = replace(
        base,
        traces={
            **base.traces,
            "measured_left_force_n": {**unobserved, "z": [400.0, 600.0, None]},
            "measured_left_force_sigma_n": {**unobserved, "z": [10.0, 20.0, None]},
        },
        measured_qualified=False,
        measured_uncertainty_source="datasheet rev 3",
    )
    return replace(series, **changes)


def _bands(fig):
    from matplotlib.collections import PolyCollection

    return [
        c
        for c in fig.axes[0].collections
        if isinstance(c, PolyCollection) and c.get_label().startswith("Measured")
    ]


def test_measured_load_carries_a_one_sigma_band() -> None:
    fig = Figure()
    plot_ground_reaction(fig, _measured_series())
    (band,) = _bands(fig)
    assert band.get_label() == "Measured Left ±1σ"
    ys = np.concatenate([p.vertices[:, 1] for p in band.get_paths()])
    assert ys.min() == pytest.approx(390.0)
    assert ys.max() == pytest.approx(620.0)


def test_the_band_names_its_declared_uncertainty_source() -> None:
    fig = Figure()
    plot_ground_reaction(fig, _measured_series())
    texts = [t.get_text() for t in fig.texts]
    assert any("datasheet rev 3" in t and "±1σ" in t for t in texts)


def test_no_band_or_caption_without_a_sigma_trace() -> None:
    series = _measured_series(measured_uncertainty_source=None)
    traces = dict(series.traces)
    traces.pop("measured_left_force_sigma_n")
    fig = Figure()
    plot_ground_reaction(fig, replace(series, traces=traces))
    assert _bands(fig) == []
    assert not any("±1σ" in t.get_text() for t in fig.texts)


def test_web_chart_uses_the_same_band_suffix_and_caption() -> None:
    from pathlib import Path

    from src.shared.python.plotting.renderers.ground_reaction import (
        SIGMA_BAND_SUFFIX,
    )

    chart = Path(__file__).resolve().parents[3] / (
        "ui/src/components/analysis/GroundReactionCharts.tsx"
    )
    source = chart.read_text(encoding="utf-8")
    escaped = "".join(
        c if c.isascii() else f"\\u{ord(c):04x}" for c in SIGMA_BAND_SUFFIX
    )
    assert f"const SIGMA_BAND_SUFFIX = '{escaped}';" in source
    assert "Shaded: measured load {SIGMA_BAND_SUFFIX} (declared uncertainty:" in source
