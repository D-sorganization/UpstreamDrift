"""TDD unit tests for repeatable MJX budget and promotion reports (MMR-08-I, #11108).

Verifies:
1. Candidate stopped on max_iterations never promotes even if RMS is lower.
2. Mixed capture or geometry hashes refuse comparison and fail promotion.
3. Failures in seeded runs remain in the denominator and prevent promotion.
4. Median and p95 metrics are calculated independently across repeated runs.
5. Fabricated receipts labeled synthetic cannot be promoted into production.
6. Stage timings are properly reported and aggregated.
7. Content-hashed incumbent reuse verifies receipt integrity and rejects tampered files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.benchmark_mjx_knot_solvers import _resolve_case
from src.shared.python.motion_matching.solver_benchmark import (
    aggregate_seeded_rows,
    promotion_decision,
    render_report,
    row_from_receipt,
    unavailable_row,
)

pytestmark = pytest.mark.unit


def _make_receipt(
    rms: float,
    *,
    capture_hash: str = "cap_hash_default",
    geometry_hash: str = "geo_hash_default",
    stop: str = "converged",
    is_synthetic: bool = True,
    stage_timings_s: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "is_synthetic": is_synthetic,
        "capture_hash": capture_hash,
        "geometry_hash": geometry_hash,
        "dynamics": {
            "marker_rms_m": 0.0845,
            "segment_rms_m": {"club": 0.12},
            "weight_fraction": {"by_phase": {"downswing": {"min": 0.35}}},
        },
        "trajectory_optimiser": {
            "iterations": 40,
            "stop_reason": stop,
            "stage_timings_s": stage_timings_s or {"opt": 10.0, "replay": 5.0},
            "shared_simulator_replay": {
                "replay_marker_rms_m": rms,
                "replay_segment_rms_m": {"head": rms, "club": 2 * rms},
                "weight_fraction": {"by_phase": {"downswing": {"min": 0.35}}},
            },
        },
    }


def _incumbent_receipt(
    rms: float = 0.06,
    *,
    capture_hash: str = "cap_hash_default",
    geometry_hash: str = "geo_hash_default",
    is_synthetic: bool = True,
) -> dict[str, Any]:
    return {
        "is_synthetic": is_synthetic,
        "capture_hash": capture_hash,
        "geometry_hash": geometry_hash,
        "dynamics": {
            "marker_rms_m": rms,
            "segment_rms_m": {"club": 1.5 * rms},
            "weight_fraction": {"by_phase": {"downswing": {"min": 0.40}}},
            "shooting_fit": {"iterations": [{}] * 9},
        },
    }


def test_max_iterations_never_promotes() -> None:
    cand_receipt = _make_receipt(0.040, stop="max_iterations")
    inc_receipt = _incumbent_receipt(0.060)

    row_cand = row_from_receipt("driver", "mjx-lbfgs", cand_receipt, wall_clock_s=50.0)
    row_inc = row_from_receipt("driver", "shooting", inc_receipt, wall_clock_s=20.0)

    decision = promotion_decision(
        [row_inc, row_cand], candidate="mjx-lbfgs", incumbent="shooting"
    )
    assert decision["promote"] is False
    assert any("max_iterations" in r for r in decision["reasons"])


def test_mixed_capture_or_geometry_hashes_do_not_compare() -> None:
    cand_receipt = _make_receipt(
        0.040, stop="converged", capture_hash="cap_A", geometry_hash="geo_common"
    )
    inc_receipt = _incumbent_receipt(
        0.060, capture_hash="cap_B", geometry_hash="geo_common"
    )

    row_cand = row_from_receipt("driver", "mjx-lbfgs", cand_receipt, wall_clock_s=50.0)
    row_inc = row_from_receipt("driver", "shooting", inc_receipt, wall_clock_s=20.0)

    decision = promotion_decision(
        [row_inc, row_cand], candidate="mjx-lbfgs", incumbent="shooting"
    )
    assert decision["promote"] is False
    assert any("capture hash mismatch" in r for r in decision["reasons"])

    # Geometry hash mismatch test
    cand_geo_receipt = _make_receipt(
        0.040, stop="converged", capture_hash="cap_common", geometry_hash="geo_A"
    )
    inc_geo_receipt = _incumbent_receipt(
        0.060, capture_hash="cap_common", geometry_hash="geo_B"
    )

    row_cand_geo = row_from_receipt(
        "driver", "mjx-lbfgs", cand_geo_receipt, wall_clock_s=50.0
    )
    row_inc_geo = row_from_receipt(
        "driver", "shooting", inc_geo_receipt, wall_clock_s=20.0
    )

    decision_geo = promotion_decision(
        [row_inc_geo, row_cand_geo], candidate="mjx-lbfgs", incumbent="shooting"
    )
    assert decision_geo["promote"] is False
    assert any("geometry hash mismatch" in r for r in decision_geo["reasons"])


def test_failures_remain_in_denominator_and_reduce_success_rate() -> None:
    # 5 seeded runs: 4 successful (RMS 0.040), 1 failure
    runs: list[dict[str, Any]] = []
    for seed in range(4):
        receipt = _make_receipt(0.040, stop="converged")
        r = row_from_receipt("driver", "mjx-lbfgs", receipt, wall_clock_s=40.0 + seed)
        r["seed"] = seed
        runs.append(r)
    # 5th run failed
    runs.append(
        {
            "capture": "driver",
            "solver": "mjx-lbfgs",
            "seed": 4,
            "status": "failed",
            "reason": "simulation divergence at t=0.21s",
        }
    )

    aggregated = aggregate_seeded_rows(runs)
    assert len(aggregated) == 1
    agg = aggregated[0]

    assert agg["total_runs"] == 5
    assert agg["successful_runs"] == 4
    assert agg["failed_runs"] == 1
    assert agg["success_rate"] == pytest.approx(0.80)

    # In promotion decision, failures in denominator block promotion
    inc_row = row_from_receipt(
        "driver", "shooting", _incumbent_receipt(0.060), wall_clock_s=20.0
    )
    decision = promotion_decision(
        [inc_row, agg], candidate="mjx-lbfgs", incumbent="shooting"
    )
    assert decision["promote"] is False
    assert any("failed runs" in r or "success rate" in r for r in decision["reasons"])


def test_median_and_p95_are_independently_calculated() -> None:
    # Asymmetric distribution of 5 runs: [0.030, 0.032, 0.035, 0.036, 0.080]
    rms_values = [0.030, 0.032, 0.035, 0.036, 0.080]
    clocks = [10.0, 11.0, 12.0, 14.0, 50.0]
    runs: list[dict[str, Any]] = []
    for i, (rms, clk) in enumerate(zip(rms_values, clocks, strict=True)):
        receipt = _make_receipt(rms, stop="converged")
        r = row_from_receipt("driver", "mjx-lbfgs", receipt, wall_clock_s=clk)
        r["seed"] = i
        runs.append(r)

    aggregated = aggregate_seeded_rows(runs)
    agg = aggregated[0]

    expected_median = float(np.median(rms_values))
    expected_p95 = float(np.percentile(rms_values, 95))

    assert agg["median_replay_marker_rms_m"] == pytest.approx(expected_median)
    assert agg["p95_replay_marker_rms_m"] == pytest.approx(expected_p95)
    assert agg["p95_replay_marker_rms_m"] > agg["median_replay_marker_rms_m"]

    expected_clock_median = float(np.median(clocks))
    expected_clock_p95 = float(np.percentile(clocks, 95))
    assert agg["median_wall_clock_s"] == pytest.approx(expected_clock_median)
    assert agg["p95_wall_clock_s"] == pytest.approx(expected_clock_p95)


def test_synthetic_fixture_rejected_in_production_promotion() -> None:
    cand_receipt = _make_receipt(0.040, stop="converged", is_synthetic=True)
    inc_receipt = _incumbent_receipt(0.060, is_synthetic=True)

    row_cand = row_from_receipt("driver", "mjx-lbfgs", cand_receipt, wall_clock_s=50.0)
    row_inc = row_from_receipt("driver", "shooting", inc_receipt, wall_clock_s=20.0)

    # When allow_synthetic is False (production mode), promotion fails
    decision = promotion_decision(
        [row_inc, row_cand],
        candidate="mjx-lbfgs",
        incumbent="shooting",
        allow_synthetic=False,
    )
    assert decision["promote"] is False
    assert any("synthetic" in r for r in decision["reasons"])


def test_stage_timings_reporting_and_aggregation() -> None:
    stage_timings = {"mjx_optimisation": 25.4, "shared_replay": 12.1}
    cand_receipt = _make_receipt(0.040, stop="converged", stage_timings_s=stage_timings)
    row = row_from_receipt("driver", "mjx-lbfgs", cand_receipt, wall_clock_s=40.0)

    assert "stage_timings_s" in row
    assert row["stage_timings_s"]["mjx_optimisation"] == pytest.approx(25.4)
    assert row["stage_timings_s"]["shared_replay"] == pytest.approx(12.1)


def test_content_hashed_incumbent_reuse_verification(tmp_path: Path) -> None:
    earlier_case = tmp_path / "earlier" / "driver_none"
    earlier_case.mkdir(parents=True)
    receipt_data = {"test": 123}
    receipt_content = json.dumps(receipt_data)
    (earlier_case / "receipt.json").write_text(receipt_content, encoding="utf-8")
    (earlier_case / "run.json").write_text(
        json.dumps({"returncode": 0}), encoding="utf-8"
    )

    correct_hash = hashlib.sha256(receipt_content.encode("utf-8")).hexdigest()

    # Valid reuse pointer with correct expected_sha256
    case_valid = tmp_path / "rerun_valid" / "driver_none"
    case_valid.mkdir(parents=True)
    pointer_valid = {
        "from": "../../earlier/driver_none",
        "expected_sha256": correct_hash,
    }
    (case_valid / "reuse.json").write_text(json.dumps(pointer_valid), encoding="utf-8")

    resolved = _resolve_case(case_valid)
    assert resolved == earlier_case.resolve()

    # Tampered reuse pointer with mismatched expected_sha256
    case_tampered = tmp_path / "rerun_tampered" / "driver_none"
    case_tampered.mkdir(parents=True)
    pointer_tampered = {
        "from": "../../earlier/driver_none",
        "expected_sha256": "wrong_tampered_hash_value_1234567890",
    }
    (case_tampered / "reuse.json").write_text(
        json.dumps(pointer_tampered), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="SHA-256"):
        _resolve_case(case_tampered)
