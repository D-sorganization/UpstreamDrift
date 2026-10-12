"""Tests for the swing report plot registry (#12326, epic #11419)."""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.reporting.swing_report.contract import report_from_dict
from src.shared.python.reporting.swing_report.plots import (
    PLOT_REGISTRY,
    PlotSpec,
    make_registry,
    resolve_descriptor,
)
from src.shared.python.reporting.swing_report.policy import classify, QuantityClass

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_registry_covers_advertised_and_withheld_quantities() -> None:
    classes = {classify(s.quantity_id) for s in PLOT_REGISTRY.values()}
    assert classes == set(QuantityClass)
    for pid in ("clubhead_speed_ts", "clubhead_path", "kinematic_sequence_timeline"):
        assert pid in PLOT_REGISTRY
    for spec_id, spec in PLOT_REGISTRY.items():
        assert spec.plot_id == spec_id


def test_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        PLOT_REGISTRY["x"] = PlotSpec("x", "X", "clubhead_speed", "time_series")  # type: ignore[index]


def test_make_registry_rejects_duplicates_and_empty() -> None:
    spec = PlotSpec("a_plot", "A", "clubhead_speed", "time_series")
    with pytest.raises(ValueError, match="duplicate"):
        make_registry([spec, spec])
    with pytest.raises(ValueError, match="at least one"):
        make_registry([])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"plot_id": "Bad-Id"},
        {"title": ""},
        {"kind": "pie"},
        {"quantity_id": ""},
    ],
)
def test_plot_spec_validates_contract(kwargs: dict[str, Any]) -> None:
    base = {
        "plot_id": "ok_plot",
        "title": "Ok",
        "quantity_id": "clubhead_speed",
        "kind": "time_series",
    }
    with pytest.raises(ValueError):
        PlotSpec(**{**base, "kind": "time_series", **kwargs})


def test_resolve_values_descriptor_matches_series(valid_full: dict[str, Any]) -> None:
    report = report_from_dict(valid_full)
    desc = resolve_descriptor(PLOT_REGISTRY["clubhead_speed_ts"], report)
    series = report.quantities[0].series
    assert desc.render == "values"
    assert desc.units == series[0].units
    assert desc.frame == series[0].frame
    assert desc.engines == tuple(s.engine for s in series)
    assert desc.unavailable_reason is None
    assert desc.uncertainty_fields == ("lower", "upper")


def test_resolve_withheld_is_unavailable_with_reason(
    valid_full: dict[str, Any],
) -> None:
    report = report_from_dict(valid_full)
    desc = resolve_descriptor(PLOT_REGISTRY["face_angle_ts"], report)
    assert desc.render == "unavailable"
    assert desc.unavailable_reason
    assert desc.units is None


def test_resolve_missing_quantity_reports_reason(valid_full: dict[str, Any]) -> None:
    # kinematic_sequence is advertised by policy but absent from this report
    report = report_from_dict(valid_full)
    desc = resolve_descriptor(PLOT_REGISTRY["kinematic_sequence_timeline"], report)
    assert desc.render == "unavailable"
    assert "kinematic_sequence" in (desc.unavailable_reason or "")


def test_resolve_mismatched_series_units_fail_closed(
    valid_full: dict[str, Any],
) -> None:
    valid_full["quantities"][0]["series"][1]["units"] = "mph"
    report = report_from_dict(valid_full)
    desc = resolve_descriptor(PLOT_REGISTRY["clubhead_speed_ts"], report)
    assert desc.render == "unavailable"
    assert "units" in (desc.unavailable_reason or "")
