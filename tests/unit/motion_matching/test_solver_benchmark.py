"""Tests for the trajectory-solver benchmark scoring (#11058)."""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.motion_matching.solver_benchmark import (
    G1_TARGET_M,
    promotion_decision,
    render_report,
    row_from_receipt,
    unavailable_row,
)

pytestmark = pytest.mark.unit


def _weight(downswing_min: float) -> dict[str, Any]:
    return {"min": 0.0, "by_phase": {"downswing": {"min": downswing_min}}}


def _pipeline_receipt(rms: float, *, passes: int = 0) -> dict[str, Any]:
    history = [{"iteration": k} for k in range(passes + 1)] if passes else []
    return {
        "dynamics": {
            "marker_rms_m": rms,
            "segment_rms_m": {"club": 2 * rms, "trunk": rms / 2},
            "weight_fraction": _weight(0.4),
            "shooting_fit": {"iterations": history} if passes else None,
        }
    }


def _mjx_receipt(rms: float, *, stop: str = "max_iterations", wf: float = 0.4) -> dict:
    receipt = _pipeline_receipt(0.0845)
    receipt["trajectory_optimiser"] = {
        "iterations": 40,
        "stop_reason": stop,
        "best_replay_marker_rms_m": 0.03,  # MJX plant: must not be the score
        "shared_simulator_replay": {
            "replay_marker_rms_m": rms,
            "replay_segment_rms_m": {"head": rms, "club": 3 * rms},
            "weight_fraction": _weight(wf),
        },
    }
    return receipt


def test_baseline_row_scores_the_shared_simulator_replay() -> None:
    row = row_from_receipt(
        "driver", "none", _pipeline_receipt(0.0845), wall_clock_s=900
    )
    assert row["replay_marker_rms_m"] == 0.0845
    assert row["worst_segment"] == "club"
    assert row["iterations"] == 0 and row["downswing_weight_fraction_min"] == 0.4
    assert row["g1_met"] is False


def test_shooting_row_counts_passes() -> None:
    row = row_from_receipt(
        "driver", "shooting", _pipeline_receipt(0.07, passes=8), wall_clock_s=1
    )
    assert row["iterations"] == 8 and row["evaluations"] == 9
    assert row["stop_reason"] == "fixed_passes"


def test_mjx_row_uses_the_rescored_replay_not_the_mjx_plant() -> None:
    row = row_from_receipt(
        "driver",
        "mjx-adam",
        _mjx_receipt(0.06),
        wall_clock_s=1,
        mjx_receipt={"history": [{}] * 41},
    )
    assert row["replay_marker_rms_m"] == 0.06
    assert row["worst_segment"] == "club" and row["evaluations"] == 41


def test_mjx_row_without_rescoring_is_rejected() -> None:
    receipt = _mjx_receipt(0.06)
    del receipt["trajectory_optimiser"]["shared_simulator_replay"]
    with pytest.raises(KeyError):
        row_from_receipt("driver", "mjx-adam", receipt, wall_clock_s=1)


def test_g1_flag_uses_the_target() -> None:
    row = row_from_receipt(
        "driver", "none", _pipeline_receipt(G1_TARGET_M), wall_clock_s=1
    )
    assert row["g1_met"] is True


def test_unavailable_row_needs_a_reason_and_has_no_numbers() -> None:
    row = unavailable_row("driver", "ipopt", "no binding")
    assert "replay_marker_rms_m" not in row
    with pytest.raises(ValueError):
        unavailable_row("driver", "ipopt", "")


def _rows(adam: float, shooting: float, **mjx: Any) -> list[dict[str, Any]]:
    return [
        row_from_receipt(
            "driver", "shooting", _pipeline_receipt(shooting, passes=2), wall_clock_s=1
        ),
        row_from_receipt(
            "driver", "mjx-adam", _mjx_receipt(adam, **mjx), wall_clock_s=1
        ),
    ]


def test_promotion_requires_parity() -> None:
    assert promotion_decision(_rows(0.05, 0.06))["promote"] is True
    worse = promotion_decision(_rows(0.07, 0.06))
    assert (
        worse["promote"] is False
        and "70.0 mm > shooting 60.0 mm" in worse["reasons"][0]
    )


def test_promotion_refuses_non_finite_stop_and_lost_contact() -> None:
    assert not promotion_decision(_rows(0.05, 0.06, stop="non_finite_cost"))["promote"]
    assert not promotion_decision(_rows(0.05, 0.06, wf=0.0))["promote"]


def test_promotion_refuses_missing_solver() -> None:
    rows = _rows(0.05, 0.06)[:1] + [unavailable_row("driver", "mjx-adam", "no jax")]
    assert promotion_decision(rows)["promote"] is False


def test_report_lists_every_row_and_the_decision() -> None:
    rows = _rows(0.07, 0.06) + [unavailable_row("driver", "ipopt", "no binding")]
    text = render_report(rows, promotion_decision(rows), {"commit": "abc"})
    assert "| driver | mjx-adam | ok | 70.0 |" in text
    assert "unavailable: no binding" in text
    assert "**keep the current default**" in text
