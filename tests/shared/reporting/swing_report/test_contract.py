"""Tests for the frozen contract dataclasses (#12324)."""

from __future__ import annotations

import dataclasses

import pytest

from src.shared.python.reporting.swing_report.contract import (
    EngineRun,
    PlotDescriptor,
    Quantity,
    Series,
    Uncertainty,
)
from src.shared.python.reporting.swing_report.validator import validate_swing_report

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_dataclasses_are_frozen(valid_full: dict) -> None:
    report = validate_swing_report(valid_full)
    for obj in (
        report,
        report.runs[0],
        report.quantities[0],
        report.quantities[0].series[0],
        report.quantities[0].uncertainty,
        report.plots[0],
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.__setattr__("scope", "x")


def test_types_exposed(valid_full: dict) -> None:
    report = validate_swing_report(valid_full)
    assert isinstance(report.runs[0], EngineRun)
    assert isinstance(report.quantities[0], Quantity)
    assert isinstance(report.quantities[0].series[0], Series)
    assert isinstance(report.quantities[0].uncertainty, Uncertainty)
    assert isinstance(report.plots[0], PlotDescriptor)


def test_withheld_quantity_has_no_numeric_fields(valid_full: dict) -> None:
    report = validate_swing_report(valid_full)
    held = [q for q in report.quantities if q.status == "withheld"]
    assert held
    assert all(q.series == () and q.uncertainty is None for q in held)


def test_consumer_accepts_any_1x(valid_full: dict) -> None:
    valid_full["schema_version"] = "1.42.7"
    assert validate_swing_report(valid_full).schema_version == "1.42.7"
