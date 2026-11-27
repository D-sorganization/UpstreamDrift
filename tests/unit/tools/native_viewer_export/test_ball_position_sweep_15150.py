"""Ball position in the stance (#15150 epic: "Ball position in the stance").

Move the ball back and forward along the target line, keep contact on the
face centre, report how each impact factor changes, and give the roll that
keeps each position's shot straight. Replay only: no rendering, no refit.
"""

from __future__ import annotations

import json

import pytest

from src.tools.native_viewer_export import ball_position_sweep as bps
from src.tools.native_viewer_export.shot_dispersion import SampleOutcome, ShotSetup
from src.tools.native_viewer_export.shot_setup import load_shot_setup_settings

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

# A linear stand-in for the replay: side (m) = 6.6 roll + 300 shift + 13.
ROLL_M, SHIFT_M, BIAS_M = 6.6, 300.0, 13.0


def fake(setup: ShotSetup) -> SampleOutcome:
    if setup.ball_shift_m > 0.1:
        return SampleOutcome(setup, reason="no flight")
    side = ROLL_M * setup.club_roll_deg + SHIFT_M * setup.ball_shift_m + BIAS_M
    metrics = {
        "carry_m": 230.0 - 100.0 * setup.ball_shift_m,
        "side_m": side,
        "face_angle_deg": 0.73 * setup.club_roll_deg + 20.0 * setup.ball_shift_m,
        "club_path_deg": 2.0 + 40.0 * setup.ball_shift_m,
        "launch_deg": 12.0,
        "spin_rpm": 2600.0,
    }
    return SampleOutcome(setup, metrics=metrics)


def test_each_position_gets_its_factors_and_straight_roll() -> None:
    rows = bps.sweep(fake, (-0.04, 0.0, 0.02), load_shot_setup_settings())
    assert [r.shift_m for r in rows] == [-0.04, 0.0, 0.02]
    tolerance = load_shot_setup_settings().straight_tolerance_m
    for row in rows:
        assert row.reason is None
        assert row.metrics["club_path_deg"] == pytest.approx(2.0 + 40.0 * row.shift_m)
        expected = -(SHIFT_M * row.shift_m + BIAS_M) / ROLL_M
        assert row.straight_roll_deg == pytest.approx(expected, abs=tolerance / ROLL_M)
        assert abs(row.side_at_straight_m) <= tolerance


def test_a_position_without_a_flight_says_why_and_the_rest_still_run() -> None:
    rows = bps.sweep(fake, (0.0, 0.12), load_shot_setup_settings())
    assert rows[0].reason is None
    assert rows[1].reason == "no flight" and rows[1].straight_roll_deg is None


def test_a_position_with_no_straight_roll_in_the_bracket_keeps_its_factors() -> None:
    def far(setup: ShotSetup) -> SampleOutcome:
        outcome = fake(setup)
        metrics = {**outcome.metrics, "side_m": outcome.metrics["side_m"] + 500.0}
        return SampleOutcome(setup, metrics=metrics)

    (row,) = bps.sweep(far, (0.0,), load_shot_setup_settings())
    assert row.metrics["carry_m"] == pytest.approx(230.0)
    assert row.straight_roll_deg is None and "sign change" in row.reason


def test_positions_forward_of_the_valid_range_are_flagged() -> None:
    cfg = bps.load_sweep_settings()
    rows = bps.sweep(
        fake, (0.0, cfg.forward_valid_m + 0.01), load_shot_setup_settings()
    )
    assert rows[0].note == ""
    assert "slowed by the real ball" in rows[1].note


def test_the_record_has_slopes_per_centimetre_and_si_units(tmp_path) -> None:
    rows = bps.sweep(fake, (-0.04, -0.02, 0.0, 0.02), load_shot_setup_settings())
    record = bps.record_for(rows)
    json.dumps(record)  # serialisable
    slopes = record["slope_per_cm"]
    assert slopes["club_path_deg"] == pytest.approx(0.40)
    assert slopes["straight_roll_deg"] == pytest.approx(
        -SHIFT_M / ROLL_M / 100.0, rel=0.05
    )
    assert record["units"]["shift"] == "m"
    assert len(record["positions"]) == 4


def test_the_settings_stay_inside_the_shot_limits() -> None:
    cfg = bps.load_sweep_settings()
    limit = load_shot_setup_settings().max_abs_ball_shift_m
    assert cfg.shifts_m == tuple(sorted(set(cfg.shifts_m)))
    assert 0.0 in cfg.shifts_m and all(abs(s) <= limit for s in cfg.shifts_m)
    with pytest.raises(ValueError, match="shift"):
        bps.SweepSettings(shifts_m=(0.0, 0.5), forward_valid_m=0.03)
    with pytest.raises(ValueError, match="shift"):
        bps.SweepSettings(shifts_m=(), forward_valid_m=0.03)
    with pytest.raises(ValueError, match="halvings"):
        bps.SweepSettings(shifts_m=(0.0,), forward_valid_m=0.03, bracket_halvings=-1)


def test_the_command_line_takes_shifts_and_writes_nothing_without_a_bundle(
    tmp_path,
) -> None:
    parser = bps.build_parser()
    args = parser.parse_args(
        [
            "--bundle",
            "b.npz",
            "--swing",
            "s",
            "--out",
            "o.json",
            "--shifts",
            "-0.02",
            "0",
        ]
    )
    assert args.shifts == [-0.02, 0.0]
    assert (
        parser.parse_args(["--bundle", "b", "--swing", "s", "--out", "o"]).shifts
        is None
    )
    missing = tmp_path / "missing.npz"
    assert (
        bps.main(
            [
                "--bundle",
                str(missing),
                "--swing",
                "s",
                "--out",
                str(tmp_path / "o.json"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "o.json").exists()


def test_each_bracket_is_centred_on_the_neighbouring_straight_roll() -> None:
    """On the matched driver the straight roll drifts with the ball; a bracket
    fixed at +-6 deg about zero put its open end outside the flight model's
    spin envelope for back positions, and their solve failed."""

    def enveloped(setup: ShotSetup) -> SampleOutcome:
        # the envelope narrows as the ball moves back, as on the matched driver
        if setup.club_roll_deg > 6.0 + 50.0 * setup.ball_shift_m + 1e-9:
            return SampleOutcome(setup, reason="impact spin outside the envelope")
        return fake(setup)

    rows = bps.sweep(enveloped, (-0.06, -0.04, -0.02, 0.0), load_shot_setup_settings())
    assert all(r.straight_roll_deg is not None for r in rows), [r.reason for r in rows]
    assert rows[0].straight_roll_deg == pytest.approx(
        -(SHIFT_M * -0.06 + BIAS_M) / ROLL_M, abs=0.05
    )
