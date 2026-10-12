"""Tests for Full/Custom plot selection (#12326, epic #11419)."""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.reporting.swing_report.contract import (
    SwingReport,
    report_from_dict,
)
from src.shared.python.reporting.swing_report.plots import (
    PLOT_REGISTRY,
    PlotSpec,
    make_registry,
)
from src.shared.python.reporting.swing_report.selection import (
    PlotSelection,
    UnknownPlotError,
    resolve_plots,
    select_plots,
)
from src.shared.python.reporting.swing_report.validator import validate_swing_report

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def report(valid_full: dict[str, Any]) -> SwingReport:
    return report_from_dict(valid_full)


def _ids(r: SwingReport) -> list[str]:
    return [p.plot_id for p in r.plots]


def _sel(report: SwingReport, ids: Any = None, **kw: Any) -> SwingReport:
    return select_plots(report, ids, **kw).report


def test_full_contains_every_registry_plot_in_registry_order(
    report: SwingReport,
) -> None:
    sel = select_plots(report, None)
    out = sel.report
    assert out.scope == "full"
    assert set(_ids(out)) | set(sel.unresolved) == set(PLOT_REGISTRY)
    assert _ids(out) == [p for p in PLOT_REGISTRY if p in set(_ids(out))]


def test_custom_withheld_plot_is_unavailable_with_reason_not_dropped(
    report: SwingReport,
) -> None:
    out = _sel(report, ["face_angle_ts", "clubhead_speed_ts"])
    assert out.scope == "custom"
    by_id = {p.plot_id: p for p in out.plots}
    assert set(by_id) == {"face_angle_ts", "clubhead_speed_ts"}
    assert by_id["face_angle_ts"].render == "unavailable"
    assert by_id["face_angle_ts"].unavailable_reason
    assert by_id["clubhead_speed_ts"].render == "values"


def test_custom_order_is_deterministic_registry_order(report: SwingReport) -> None:
    a = _sel(report, ["face_angle_ts", "clubhead_speed_ts"])
    b = _sel(report, ["clubhead_speed_ts", "face_angle_ts", "face_angle_ts"])
    assert _ids(a) == _ids(b)
    assert _ids(a) == [pid for pid in PLOT_REGISTRY if pid in set(_ids(a))]


def test_unknown_plot_id_raises(report: SwingReport) -> None:
    with pytest.raises(UnknownPlotError, match="nope"):
        select_plots(report, ["clubhead_speed_ts", "nope"])
    assert issubclass(UnknownPlotError, ValueError)


def test_empty_custom_selection_rejected(report: SwingReport) -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_plots(report, [])


def test_results_always_validate(report: SwingReport) -> None:
    for ids in (None, ["clubhead_speed_ts"], list(PLOT_REGISTRY)):
        out = _sel(report, ids)
        assert validate_swing_report(out.to_dict()).to_dict() == out.to_dict()


def test_sections_reference_only_selected_plots(report: SwingReport) -> None:
    out = _sel(report, ["clubhead_speed_ts", "joint_load_ts"])
    listed = [pid for s in out.sections for pid in s.plot_ids]
    assert sorted(listed) == sorted(_ids(out))


def test_input_report_not_mutated(report: SwingReport) -> None:
    before = report.to_dict()
    select_plots(report, ["clubhead_speed_ts"])
    assert report.to_dict() == before


def test_added_descriptor_needs_no_other_source_change(report: SwingReport) -> None:
    extra = PlotSpec("speed_bars", "Speed Bars", "clubhead_speed", "bar")
    registry = make_registry([*PLOT_REGISTRY.values(), extra])
    out = _sel(report, ["speed_bars"], registry=registry)
    assert _ids(out) == ["speed_bars"]
    assert out.plots[0].kind == "bar"
    assert "speed_bars" in _ids(_sel(report, None, registry=registry))


def test_withheld_plots_never_expose_values(report: SwingReport) -> None:
    for desc in _sel(report, None).plots:
        if desc.render == "unavailable":
            assert desc.units is None and desc.uncertainty_fields is None


def test_resolve_plots_lists_all_options_for_ui(report: SwingReport) -> None:
    descs = resolve_plots(report)
    assert [d.plot_id for d in descs] == list(PLOT_REGISTRY)
    assert any(d.render == "unavailable" and d.unavailable_reason for d in descs)


def test_plot_without_quantity_is_reported_not_silently_dropped(
    report: SwingReport,
) -> None:
    sel = select_plots(report, ["clubhead_speed_ts", "kinematic_sequence_timeline"])
    assert isinstance(sel, PlotSelection)
    assert _ids(sel.report) == ["clubhead_speed_ts"]
    assert "kinematic_sequence" in sel.unresolved["kinematic_sequence_timeline"]
    notes = " ".join(s.notes or "" for s in sel.report.sections)
    assert "kinematic_sequence_timeline" in notes


def test_no_unresolved_when_all_quantities_present(report: SwingReport) -> None:
    assert select_plots(report, ["face_angle_ts"]).unresolved == {}
