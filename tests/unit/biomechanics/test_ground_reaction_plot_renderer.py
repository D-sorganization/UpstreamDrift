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


def test_web_chart_uses_the_same_uncertainty_wording_and_cap() -> None:
    from pathlib import Path

    from src.shared.python.plotting.renderers import ground_reaction as gr

    module = Path(__file__).resolve().parents[3] / (
        "ui/src/components/analysis/groundReactionUncertainty.ts"
    )
    source = module.read_text(encoding="utf-8")
    suffix = gr.SIGMA_BAND_SUFFIX
    escaped = "".join(c if c.isascii() else f"\\u{ord(c):04x}" for c in suffix)
    assert f"export const SIGMA_BAND_SUFFIX = '{escaped}';" in source
    assert f"export const PER_AXIS = '{gr._PER_AXIS}';" in source
    assert f"export const MAX_COP_MARKERS = {gr.MAX_COP_MARKERS};" in source
    for phrase in (
        "shaded: measured load ${SIGMA_BAND_SUFFIX}",
        "circles: measured CoP ${SIGMA_BAND_SUFFIX} ${PER_AXIS}",
        "(declared uncertainty: ${source ?? 'not stated'})",
    ):
        assert phrase in source, phrase


def _cop_series(n: int, sigma: float | None = 0.01):
    """A measured left CoP path of ``n`` frames with a constant sigma."""
    base = _plot()
    times = tuple(i * 0.01 for i in range(n))
    nothing = {a: [None] * n for a in ("x", "y", "z", "magnitude")}
    cop = {**nothing, "x": [0.001 * i for i in range(n)], "y": [0.0] * n}
    traces = {"measured_left_cop_m": cop}
    if sigma is not None:
        traces["measured_left_cop_sigma_m"] = {**nothing, "z": [sigma] * n}
    return replace(
        base,
        time_s=times,
        traces=traces,
        load_share={},
        events={},
        measured_qualified=False,
        measured_uncertainty_source="datasheet rev 3",
    )


def _circles(fig):
    from matplotlib.patches import Circle

    return [p for p in fig.axes[3].patches if isinstance(p, Circle)]


def test_cop_markers_are_one_sigma_circles_on_the_path() -> None:
    from src.shared.python.plotting.renderers.ground_reaction import (
        MAX_COP_MARKERS,
    )

    fig = Figure()
    plot_ground_reaction(fig, _cop_series(50))
    circles = _circles(fig)
    assert 1 < len(circles) <= MAX_COP_MARKERS
    assert all(c.radius == pytest.approx(0.01) for c in circles)
    assert circles[0].center == (0.0, 0.0)
    legend = [t.get_text() for t in fig.axes[3].get_legend().get_texts()]
    assert legend.count("Measured Left \u00b11\u03c3 per axis") == 1


def test_cop_markers_skip_frames_without_a_cop_or_sigma() -> None:
    from src.shared.python.plotting.renderers.ground_reaction import (
        cop_marker_indices,
    )

    assert cop_marker_indices([0.1, None, 0.2], [0.01, 0.01, None], 12) == [0]
    assert cop_marker_indices([0.0] * 5, [0.01] * 5, 12) == [0, 1, 2, 3, 4]
    picked = cop_marker_indices([0.0] * 100, [0.01] * 100, 12)
    assert len(picked) <= 12 and picked[0] == 0
    assert cop_marker_indices([], [], 12) == []


def test_cop_markers_are_captioned_with_the_declared_source() -> None:
    fig = Figure()
    plot_ground_reaction(fig, _cop_series(5))
    texts = [t.get_text() for t in fig.texts]
    assert (
        "Circles: measured CoP ±1σ per axis (declared uncertainty: datasheet rev 3)"
    ) in texts


def test_no_cop_markers_without_a_cop_sigma_trace() -> None:
    fig = Figure()
    plot_ground_reaction(fig, _cop_series(5, sigma=None))
    assert _circles(fig) == []
    assert not any("Circles" in t.get_text() for t in fig.texts)
