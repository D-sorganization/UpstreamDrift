"""Flex comparison sheet on a synthetic sweep result (#15150 C5)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from src.shared.python.club_fitting.flex_comparison import (  # noqa: E402
    SETTINGS_PATH,
    load_comparison_settings,
    render_comparison,
    write_comparison,
)
from src.shared.python.club_fitting.flight_trace import FlightTrace  # noqa: E402
from src.shared.python.club_fitting.shaft_report import (  # noqa: E402
    SweepResult,
    load_sweep_result,
)

pytestmark = pytest.mark.unit
FLEXES = ("Ladies", "Senior", "Regular", "Stiff", "X-Stiff")


def _flight(carry_m: float, side_m: float) -> FlightTrace:
    s = np.linspace(0.0, 1.0, 20)
    return FlightTrace(carry_m * s, side_m * s**2, 30.0 * 4.0 * s * (1.0 - s))


@pytest.fixture
def sweep_json(tmp_path: Path) -> Path:
    """A synthetic ``flex_sweep.json`` with flight series (no simulation)."""
    rows = [
        {"flex": f, "impact_head_speed_mph": 100.0 + i, "shot_ball_speed_mph":
         148.0 + i, "shot_carry_yd": 240.0 + 2 * i, "shot_side_yd": 6.0 - 3 * i,
         "shot_face_angle_deg": 1.0 - 0.5 * i}
        for i, f in enumerate(FLEXES)
    ]  # fmt: skip
    flights = {
        f: _flight(220.0 + 2 * i, 5.0 - 3 * i).to_dict() for i, f in enumerate(FLEXES)
    }
    path = tmp_path / "flex_sweep.json"
    path.write_text(json.dumps({"rows": rows, "flights": flights}), encoding="utf-8")
    return path


def test_flight_trace_contracts_and_round_trip() -> None:
    trace = _flight(200.0, -4.0)
    assert FlightTrace.from_dict(trace.to_dict()).to_dict() == trace.to_dict()
    with pytest.raises(ValueError, match="same length"):
        FlightTrace(np.zeros(3), np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        FlightTrace(np.array([0.0, np.nan]), np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="missing"):
        FlightTrace.from_dict({"downrange_m": [0.0, 1.0]})


def test_sweep_result_reads_flights(sweep_json) -> None:
    result = load_sweep_result(sweep_json)
    assert set(result.flights) == set(FLEXES)
    assert result.flights["Stiff"].downrange_m[-1] == pytest.approx(226.0)


def test_traces_overlay_one_colour_per_flex(sweep_json) -> None:
    settings = load_comparison_settings()
    fig = render_comparison(load_sweep_result(sweep_json), settings)
    side, plan = fig.axes[:2]
    for ax in (side, plan):
        lines = ax.get_lines()
        assert [ln.get_label() for ln in lines] == list(FLEXES)
        assert [ln.get_color() for ln in lines] == list(settings.flex_colors[:5])
    stiff = side.get_lines()[3]
    assert stiff.get_xdata()[-1] == pytest.approx(226.0 * settings.distance_scale)
    assert plan.get_lines()[3].get_ydata()[-1] == pytest.approx(
        -4.0 * settings.distance_scale
    )


def test_sheet_and_table_are_written(sweep_json, tmp_path) -> None:
    settings = load_comparison_settings()
    paths = write_comparison(load_sweep_result(sweep_json), tmp_path / "c", settings)
    names = [p.name for p in paths]
    assert names == [f"{settings.sheet_stem}.png", f"{settings.sheet_stem}.md"]
    assert paths[0].stat().st_size > 0
    md = paths[1].read_text(encoding="utf-8").splitlines()
    assert len([ln for ln in md if ln.startswith("| ")]) == 1 + len(FLEXES)
    assert md[0].count("|") == len(settings.metrics) + 2
    assert "| Stiff | 103.0 | 151.0 | 246.0 | -3.0 | -0.5 |" in md


def test_sweep_without_flights_still_renders(sweep_json, tmp_path) -> None:
    rows = json.loads(sweep_json.read_text())["rows"]
    result = SweepResult(tuple(rows), {})
    fig = render_comparison(result, load_comparison_settings())
    assert fig.axes[0].get_lines() == []
    assert write_comparison(result, tmp_path)[0].stat().st_size > 0


def test_settings_contracts(tmp_path) -> None:
    good = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    bad = tmp_path / "s.json"
    bad.write_text(json.dumps({**good, "flex_colors": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="flex_colors"):
        load_comparison_settings(bad)
    extra = tmp_path / "x.json"
    extra.write_text(json.dumps({**good, "surprise": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="surprise"):
        load_comparison_settings(extra)
    with pytest.raises(ValueError, match="distance_scale"):
        dataclasses.replace(load_comparison_settings(), distance_scale=0.0)


def test_sweep_cli_has_a_comparison_option() -> None:
    from src.shared.python.club_fitting.__main__ import build_parser

    args = build_parser().parse_args(["--bundle", "b", "--out", "o", "--comparison"])
    assert args.comparison is True


def _shape_points(lead_m: float, droop_m: float) -> list[list[float]]:
    f = np.linspace(0.0, 1.0, 5)
    return np.column_stack([f, lead_m * f**2, droop_m * f**2]).tolist()


def test_shaft_shape_panel_draws_each_flex(sweep_json) -> None:
    data = json.loads(sweep_json.read_text())
    data["shapes"] = {
        f: {"points_m": _shape_points(0.01 * (5 - i), 0.004 * i)}
        for i, f in enumerate(FLEXES)
    }
    sweep_json.write_text(json.dumps(data))
    settings = load_comparison_settings()
    assert settings.shape_exaggeration != 1.0
    fig = render_comparison(load_sweep_result(sweep_json), settings)
    lead, droop = fig.axes[2:4]
    for ax in (lead, droop):
        assert [ln.get_label() for ln in ax.get_lines()] == list(FLEXES)
        assert [ln.get_color() for ln in ax.get_lines()] == list(
            settings.flex_colors[:5]
        )
        texts = [t.get_text() for t in ax.texts]
        assert (
            settings.shape_exaggeration_format.format(settings.shape_exaggeration)
            in texts
        )
    k = settings.shape_length_scale * settings.shape_exaggeration
    assert lead.get_lines()[1].get_ydata()[-1] == pytest.approx(0.04 * k)
    assert droop.get_lines()[1].get_ydata()[-1] == pytest.approx(0.004 * k)
    true_scale = dataclasses.replace(settings, shape_exaggeration=1.0)
    fig = render_comparison(load_sweep_result(sweep_json), true_scale)
    assert list(fig.axes[2].texts) == []


def test_shaft_shape_panel_says_when_shapes_are_not_recorded(sweep_json) -> None:
    settings = load_comparison_settings()
    fig = render_comparison(load_sweep_result(sweep_json), settings)
    for ax in fig.axes[2:4]:
        assert ax.get_lines() == []
        assert [t.get_text() for t in ax.texts] == [settings.no_shape_text]
