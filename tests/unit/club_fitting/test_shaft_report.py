"""Shaft report on a synthetic sweep result (#15150 C12)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from src.shared.python.club_fitting.bend_trace import BendTrace, bend_trace  # noqa: E402
from src.shared.python.club_fitting.grip_driven_shaft import (  # noqa: E402
    HeadGeometry,
    ShaftRun,
)
from src.shared.python.club_fitting.shaft_report import (  # noqa: E402
    SETTINGS_PATH,
    SweepResult,
    load_report_settings,
    load_sweep_result,
    summary_rows,
    write_report,
)

pytestmark = pytest.mark.unit
FLEXES = ("Regular", "Stiff")


def _run(lead_m: float, droop_m: float) -> ShaftRun:
    """Face normal club +z; club +x points down in the world, so it is droop."""
    times = np.linspace(-0.1, 0.02, 7)
    ramp = np.clip((times + 0.1) / 0.1, 0.0, None)
    defl = np.column_stack([droop_m * ramp, np.zeros(7), lead_m * ramp])
    rot = np.tile(np.eye(3), (7, 1, 1))
    rot[:] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    vec, zero = np.zeros((7, 3)), np.zeros(7)
    return ShaftRun(
        times, np.zeros((7, 8)), np.einsum("nij,nj->ni", rot, defl), vec, vec,
        vec, vec, rot, rot, zero, zero, zero, np.array([]),
    )  # fmt: skip


HEAD = HeadGeometry(np.zeros(3), np.array([0.0, 0.0, 1.0]))


def test_bend_trace_splits_lead_and_droop_and_stops_at_impact() -> None:
    trace = bend_trace(_run(0.05, 0.02), HEAD, 0.0)
    assert trace.time_s[-1] == pytest.approx(0.0)
    assert trace.lead_mm[-1] == pytest.approx(50.0)
    assert trace.droop_mm[-1] == pytest.approx(20.0)
    assert trace.at_impact() == pytest.approx((50.0, 20.0))


def test_bend_trace_contracts() -> None:
    with pytest.raises(ValueError, match="same length"):
        BendTrace(np.zeros(3), np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="before the run"):
        bend_trace(_run(0.05, 0.0), HEAD, -1.0)
    along_shaft = HeadGeometry(np.zeros(3), np.array([0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="shaft"):
        bend_trace(_run(0.05, 0.0), along_shaft, 0.0)
    trace = bend_trace(_run(0.05, 0.02), HEAD, 0.0)
    assert BendTrace.from_dict(trace.to_dict()).to_dict() == trace.to_dict()


@pytest.fixture
def sweep_json(tmp_path: Path) -> Path:
    """A synthetic ``flex_sweep.json`` with traces (no simulation)."""
    rows = [
        {"flex": f, "flex_cpm": 250.0 + 10 * i, "loft_delta_deg": 4.0 - i,
         "peak_tip_deflection_mm": 80.0 - 10 * i, "shot_carry_yd": 250.0 + i,
         "shot_dynamic_loft_deg": None}
        for i, f in enumerate(FLEXES)
    ]  # fmt: skip
    traces = {
        f: bend_trace(_run(0.05 - 0.01 * i, 0.02), HEAD, 0.0).to_dict()
        for i, f in enumerate(FLEXES)
    }
    path = tmp_path / "flex_sweep.json"
    path.write_text(json.dumps({"rows": rows, "traces": traces}), encoding="utf-8")
    return path


def test_report_writes_one_page_per_flex_and_a_summary(sweep_json, tmp_path) -> None:
    settings = load_report_settings()
    paths = write_report(load_sweep_result(sweep_json), tmp_path / "rep", settings)
    pages = [p for p in paths if p.suffix == ".png"]
    assert [p.name for p in pages] == [f"{settings.page_stem}_{f}.png" for f in FLEXES]
    assert all(p.stat().st_size > 0 for p in pages)
    summary = json.loads((tmp_path / "rep" / f"{settings.page_stem}.json").read_text())
    assert [r["flex"] for r in summary["rows"]] == list(FLEXES)
    assert summary["rows"][1]["lead_at_impact_mm"] == pytest.approx(40.0)
    assert summary["rows"][0]["shot_carry_yd"] == 250.0
    md = (tmp_path / "rep" / f"{settings.page_stem}.md").read_text()
    rows = [line for line in md.splitlines() if line.startswith("| ")]
    assert len(rows) == 1 + len(FLEXES) and "n/a" in md


def test_summary_without_traces_and_missing_metrics(sweep_json) -> None:
    data = json.loads(sweep_json.read_text())
    result = SweepResult(tuple(data["rows"]), {})
    row = summary_rows(result, load_report_settings())[0]
    assert row["lead_at_impact_mm"] is None and row["shot_spin_rpm"] is None


def test_settings_and_input_contracts(tmp_path) -> None:
    good = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    bad = tmp_path / "s.json"
    bad.write_text(json.dumps({**good, "dpi": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="dpi"):
        load_report_settings(bad)
    empty = tmp_path / "e.json"
    empty.write_text(json.dumps({"rows": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        load_sweep_result(empty)
    with pytest.raises(ValueError, match="page_size_in"):
        dataclasses.replace(load_report_settings(), page_size_in=(0.0, 11.0))


def test_sweep_cli_has_a_report_option() -> None:
    from src.shared.python.club_fitting.__main__ import build_parser

    args = build_parser().parse_args(["--bundle", "b", "--out", "o", "--report"])
    assert args.report is True
