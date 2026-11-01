"""Putt-class library, encounter weights, aggregation and JSON report tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.putting_analytics import aggregate as agg
from src.shared.python.putting_analytics import library as plib
from src.shared.python.putting_analytics import report as rpt
from src.shared.python.putting_analytics.expected_putts import (
    ExpectedPuttsCurve,
    PuttAction,
)
from tests.unit.putting_analytics.conftest import ConstantMakeModel, ToyBreakModel

pytestmark = pytest.mark.unit

ACTIONS = tuple(PuttAction(a, p) for a in (0, 2) for p in (1.0, 2.0))
GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0)
TOUR = ExpectedPuttsCurve(np.array(GRID), 1.0 + 0.05 * np.array(GRID))
CONFIG = agg.AggregateConfig(ACTIONS, TOUR, n=300, seed=7, grid_ft=GRID)
CLASSES = plib.default_library(
    (3.0, 10.0), (0.0, 2.0), ("uphill", "oblique"), (10.0,)
).classes


def factory(putt_class: plib.PuttClass) -> ToyBreakModel:
    return ToyBreakModel(break_deg=putt_class.slope_pct, speed_cv=0.02)


@pytest.fixture
def report() -> rpt.PuttingReport:
    skewed = plib.ClassLibrary.from_counts(CLASSES, [1, 2, 3, 4, 5, 6])
    return agg.aggregate(skewed, factory, CONFIG)


def test_default_library_is_uniform_and_flat_has_no_break():
    library = plib.default_library((3.0, 10.0), (0.0, 2.0), ("uphill", "oblique"))
    assert len(library.classes) == 12  # 2 distances x 2 stimps x (flat + 2 breaks)
    assert library.weights == pytest.approx([1 / 12] * 12)
    assert {c.orientation for c in CLASSES if c.slope_pct == 0.0} == {"flat"}


def test_contract_errors():
    for bad in [(1, 0, "flat", 10), (10, 5, "uphill", 10), (10, 2, "flat", 10)]:
        with pytest.raises(ValueError):
            plib.PuttClass(*bad)
    with pytest.raises(ValueError):
        agg.combine([rpt.Estimate(1.0, 0.1)], [0.0])
    with pytest.raises(ValueError):
        agg.AggregateConfig(ACTIONS, TOUR, n=10, seed=0, objective="bogus")


def test_frequency_table_loader(tmp_path):
    path = tmp_path / "putts.csv"
    path.write_text(
        "distance_ft,slope_pct,orientation,stimp,count\n"
        "5,0,flat,10,3\n8,2,uphill,10,1\n5,0,flat,10,4\n12,2,downhill,11,0\n"
    )
    table = plib.load_frequency_table(path)  # repeats add up; zero counts dropped
    assert table.classes == (
        plib.PuttClass(5, 0, "flat", 10),
        plib.PuttClass(8, 2, "uphill", 10),
    )
    assert table.weights == pytest.approx([7 / 8, 1 / 8])


def test_aggregate_is_weighted_sum_of_class_results(report: rpt.PuttingReport):
    w, holes = np.array([c.weight for c in report.classes]), report.holes_per_round
    for name in ("expected_putts", "sg_player", "sg_tour"):
        mean = np.array([getattr(c, name).mean for c in report.classes])
        se = np.array([getattr(c, name).se for c in report.classes])
        total = getattr(report.totals, f"{name}_per_round")
        assert total.mean == pytest.approx(holes * float(w @ mean))
        assert total.se == pytest.approx(holes * float(np.sqrt(np.sum((w * se) ** 2))))


def test_bands_pace_and_standard_errors(report: rpt.PuttingReport):
    assert [(b.lo_ft, b.hi_ft) for b in report.bands] == [(2.0, 5.0), (10.0, 20.0)]
    near = [c for c in report.classes if c.putt_class.distance_ft == 3.0]
    assert report.bands[0].weight == pytest.approx(sum(c.weight for c in near))
    mean = np.average(
        [c.make_rate.mean for c in near], weights=[c.weight for c in near]
    )
    assert report.bands[0].make_rate.mean == pytest.approx(mean)
    assert [(p.distance_ft, p.stimp) for p in report.pace] == [(3, 10), (10, 10)]
    estimates = [c.make_rate for c in report.classes] + list(
        vars(report.totals).values()
    )
    assert all(isinstance(e, rpt.Estimate) and e.se >= 0.0 for e in estimates)
    assert any(e.se > 0.0 for e in estimates)


def test_json_round_trip_and_schema_version(report: rpt.PuttingReport):
    assert rpt.PuttingReport.from_json(report.to_json()) == report
    data = json.loads(report.to_json())
    assert data["schema_version"] == rpt.SCHEMA_VERSION and data["seed"] == 7
    data["schema_version"] += 1
    with pytest.raises(ValueError):
        rpt.PuttingReport.from_json(json.dumps(data))


def test_reproducible_and_matches_closed_form():
    one = plib.default_library((3.0,), (0.0,), (), (10.0,))
    assert agg.aggregate(one, factory, CONFIG) == agg.aggregate(one, factory, CONFIG)
    makes = {10.0: 0.4, 12.0: 0.2}
    cfg = agg.AggregateConfig(ACTIONS[:1], TOUR, n=2000, seed=1, grid_ft=GRID)
    two = plib.default_library((10.0,), (0.0,), (), (10.0, 12.0))
    result = agg.aggregate(two, lambda c: ConstantMakeModel(makes[c.stimp]), cfg)
    assert result.totals.make_rate.mean == pytest.approx(0.3, abs=0.03)
