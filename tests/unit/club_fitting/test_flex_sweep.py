"""Flex sweep and CLI output on a synthetic grip motion (#15684)."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("mujoco")

from src.shared.python.club_fitting.__main__ import build_parser, write_outputs  # noqa: E402
from src.shared.python.club_fitting.flex_sweep import run_flex_sweep  # noqa: E402
from src.shared.python.club_fitting.grip_driven_shaft import HeadGeometry  # noqa: E402
from src.shared.python.club_fitting.grip_motion import (  # noqa: E402
    ClubPoseTrack,
    load_settings,
)
from src.shared.python.motion_matching import club_models as cm  # noqa: E402

pytestmark = pytest.mark.unit

HEAD = HeadGeometry(np.array([0.0, 0.0, cm.HEAD_HEIGHT_M]), np.array([0.0, 0.0, 1.0]))
FAST = dataclasses.replace(
    load_settings(),
    lead_in_s=0.06,
    post_impact_s=0.005,
    integration_dt_s=2e-4,
    record_every=1,
)


def track() -> ClubPoseTrack:
    t = np.arange(-0.6, 0.05, 0.002)
    rot = Rotation.from_rotvec(np.outer(12.0 * t, [0.0, 0.0, 1.0]))
    rot = rot * Rotation.from_euler("z", -90.0, degrees=True)
    grip = np.stack([0 * t, 0 * t, 1.0 + 0 * t], axis=1)
    return ClubPoseTrack(t, grip, rot.as_matrix())


@pytest.fixture(scope="module")
def rows():
    stub = lambda run: {"ball_speed_mph": 100.0 if run is None else 99.0}  # noqa: E731
    return run_flex_sweep(
        track(), cm.DRIVER, HEAD, 0.0, ["Ladies", "X-Stiff"], FAST, shot=stub
    )


@pytest.mark.slow
def test_rows_follow_flex_order_and_deflection(rows) -> None:
    assert [r.flex for r in rows] == ["Ladies", "X-Stiff"]
    assert rows[0].peak_tip_deflection_mm > rows[1].peak_tip_deflection_mm > 0.0
    assert rows[0].flex_cpm < rows[1].flex_cpm
    assert rows[0].shot == {"ball_speed_mph": 99.0}
    for r in rows:
        assert r.peak_head_speed_mph >= r.impact_head_speed_mph * 0.99
        assert r.rigid_impact_head_speed_mph == pytest.approx(
            rows[0].rigid_impact_head_speed_mph
        )


@pytest.mark.slow
def test_rows_carry_the_shaft_shape_at_impact(rows) -> None:
    """Chain joints then the tip, matching the bend trace's tip at impact."""
    for r in rows:
        assert r.shape is not None and r.bend is not None
        assert r.shape.points_m.shape == (FAST.chain_links + 1, 3)
        lead_mm, droop_mm = r.bend.at_impact()
        assert r.shape.lead_m[-1] * 1e3 == pytest.approx(lead_mm, abs=0.5)
        assert r.shape.droop_m[-1] * 1e3 == pytest.approx(droop_mm, abs=0.5)
        assert r.shape.lead_m[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.slow
def test_outputs_round_trip(rows, tmp_path) -> None:
    json_path, csv_path = write_outputs(rows, tmp_path / "out")
    data = json.loads(json_path.read_text())["rows"]
    assert data[0]["flex"] == "Ladies" and data[0]["shot_ball_speed_mph"] == 99.0
    with csv_path.open() as handle:
        assert [r["flex"] for r in csv.DictReader(handle)] == ["Ladies", "X-Stiff"]


def test_flex_contracts() -> None:
    with pytest.raises(ValueError, match="subset"):
        run_flex_sweep(track(), cm.DRIVER, HEAD, 0.0, ["Soft-ish"], FAST)
    with pytest.raises(ValueError, match="subset"):
        run_flex_sweep(track(), cm.DRIVER, HEAD, 0.0, [], FAST)
    with pytest.raises(ValueError, match="no rows"):
        write_outputs([], Path("unused"))


def test_cli_parser_defaults() -> None:
    args = build_parser().parse_args(["--bundle", "b.npz", "--out", "o"])
    assert args.club == "driver" and args.flexes is None


def _run_with_tail(peak_after_impact_m: float):
    """A ShaftRun whose bend grows after impact (the extrapolated tail)."""
    from src.shared.python.club_fitting.grip_driven_shaft import ShaftRun

    times = np.linspace(-0.1, 0.025, 6)
    bend = np.where(times <= 0.0, 0.1, peak_after_impact_m)
    defl = np.column_stack([bend, np.zeros(6), np.zeros(6)])
    vec, rot = np.zeros((6, 3)), np.tile(np.eye(3), (6, 1, 1))
    zero = np.zeros(6)
    return ShaftRun(
        times, np.zeros((6, 8)), defl, vec, vec, vec, vec, rot, rot,
        zero, zero, zero, np.array([]),
    )  # fmt: skip


def test_peak_bend_is_measured_up_to_impact_not_in_the_extrapolated_tail() -> None:
    """After impact the grip motion is extrapolated and no ball load acts, so
    the bend there grows without limit; the reported peak ends at impact."""
    from src.shared.python.club_fitting.flex_sweep import _row

    run = _run_with_tail(0.3)
    assert run.peak_tip_deflection_until_m(0.0) == pytest.approx(0.1)
    assert _row("Stiff", 280.0, run, 0.0, None).peak_tip_deflection_mm == (
        pytest.approx(100.0)
    )
    with pytest.raises(ValueError, match="before the run"):
        run.peak_tip_deflection_until_m(-0.5)


def test_rows_carry_the_bend_trace_into_the_sweep_json(tmp_path) -> None:
    """The shaft report (#15150 C12) reads lead and droop from flex_sweep.json."""
    from src.shared.python.club_fitting.flex_sweep import _row

    row = _row("Stiff", 280.0, _run_with_tail(0.3), 0.0, None, HEAD)
    assert row.bend is not None and row.bend.time_s[-1] <= 0.0
    assert "bend" not in row.record()
    json_path, _ = write_outputs([row], tmp_path)
    assert set(json.loads(json_path.read_text())["traces"]) == {"Stiff"}


def test_rows_carry_the_flight_series_into_the_sweep_json(tmp_path) -> None:
    """The flex comparison (#15150 C5) overlays flights from flex_sweep.json."""
    from src.shared.python.club_fitting.flex_sweep import _row
    from src.shared.python.club_fitting.flight_trace import FLIGHT_KEY, FlightTrace

    flight = FlightTrace(np.array([0.0, 200.0]), np.zeros(2), np.zeros(2))
    stub = lambda run: {"carry_yd": 218.7, FLIGHT_KEY: flight}  # noqa: E731
    row = _row("Stiff", 280.0, _run_with_tail(0.3), 0.0, stub)
    assert row.flight is flight and row.shot == {"carry_yd": 218.7}
    assert "flight" not in row.record()
    json_path, _ = write_outputs([row], tmp_path)
    saved = json.loads(json_path.read_text())["flights"]["Stiff"]
    assert FlightTrace.from_dict(saved).to_dict() == flight.to_dict()
