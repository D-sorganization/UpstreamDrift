"""Display units of the shot dispersion figure and record (#15443)."""

from __future__ import annotations

import numpy as np
import pytest

from src.tools.native_viewer_export import shot_dispersion as sd
from src.tools.native_viewer_export.display_units import METRIC, load_display_units
from src.tools.native_viewer_export.shot_dispersion import SampleOutcome, ShotSetup

pytestmark = pytest.mark.unit


def _evaluate(setup: ShotSetup) -> SampleOutcome:
    t = np.linspace(0.0, 1.0, 5)
    carry, side = 200.0, 5.0 * setup.club_roll_deg
    trace = np.column_stack([carry * t, side * t, 30.0 * t * (1.0 - t)])
    metrics = dict.fromkeys(sd.METRICS, 1.0)
    metrics.update(carry_m=carry, side_m=side)
    return SampleOutcome(setup, None, metrics, trace)


def _result():
    cfg = sd.load_dispersion_settings()
    return cfg, sd.run_dispersion(None, None, sd.sample_setups(cfg), evaluate=_evaluate)


def test_table_defaults_to_yards_and_m_restores_metres() -> None:
    pytest.importorskip("matplotlib")
    from src.tools.native_viewer_export import shot_dispersion_plot as plot

    _, result = _result()
    yd = plot._table_cells(result, load_display_units())
    assert yd[0][0] == "Carry (yd)"
    assert yd[0][1] == f"{200.0 / 0.9144:.1f}"
    metres = plot._table_cells(result, METRIC)
    assert metres[0][0] == "Carry (m)"
    assert metres[0][1] == "200.0"
    assert yd[2][0] == metres[2][0]  # degrees are not distances


def test_axis_labels_name_the_unit() -> None:
    pytest.importorskip("matplotlib")
    from src.tools.native_viewer_export import shot_dispersion_plot as plot

    assert "(yd)" in plot.forward_label(load_display_units())
    assert "(m)" in plot.forward_label(METRIC)


def test_figure_writes_png_in_both_units(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from src.tools.native_viewer_export.shot_dispersion_plot import draw_dispersion

    cfg, result = _result()
    for name, units in (("yd", load_display_units()), ("m", METRIC)):
        path = draw_dispersion(result, cfg, tmp_path / f"{name}.png", units)
        assert path.stat().st_size > 1000


def test_record_keeps_si_and_names_units() -> None:
    from src.tools.native_viewer_export.shot_dispersion_cli import record_for

    cfg, result = _result()
    record = record_for(cfg, result, load_display_units())
    assert record["shots"][0]["metrics"]["carry_m"] == 200.0
    assert record["units"]["distance_values"] == "m"
    assert record["units"]["display_distance_unit"] == "yd"


def test_cli_parser_has_distance_unit_choices() -> None:
    from src.tools.native_viewer_export.shot_dispersion_cli import build_parser

    base = ["--bundle", "b", "--swing", "s", "--out", "o"]
    assert build_parser().parse_args(base).distance_unit is None
    assert (
        build_parser().parse_args(base + ["--distance-unit", "m"]).distance_unit == "m"
    )
    with pytest.raises(SystemExit):
        build_parser().parse_args(base + ["--distance-unit", "ft"])
