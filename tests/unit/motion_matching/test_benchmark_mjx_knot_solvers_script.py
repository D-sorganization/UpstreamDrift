"""Tests for the benchmark runner's evidence loading (#11071)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_mjx_knot_solvers import load_row

pytestmark = pytest.mark.unit


def _write_case(case: Path, rms: float) -> None:
    case.mkdir(parents=True)
    receipt = {
        "dynamics": {
            "marker_rms_m": rms,
            "segment_rms_m": {"club": rms},
            "weight_fraction": {"by_phase": {"downswing": {"min": 0.4}}},
        }
    }
    (case / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    run = {"command": [], "returncode": 0, "wall_clock_s": 12.0}
    (case / "run.json").write_text(json.dumps(run), encoding="utf-8")


def test_load_row_follows_a_reuse_pointer(tmp_path: Path) -> None:
    _write_case(tmp_path / "earlier" / "driver_none", 0.0845)
    case = tmp_path / "rerun" / "driver_none"
    case.mkdir(parents=True)
    pointer = {"from": "../../earlier/driver_none"}
    (case / "reuse.json").write_text(json.dumps(pointer), encoding="utf-8")

    row = load_row("driver", "none", tmp_path / "rerun")

    assert row["replay_marker_rms_m"] == pytest.approx(0.0845)
    assert row["wall_clock_s"] == pytest.approx(12.0)


def test_load_row_rejects_a_dangling_reuse_pointer(tmp_path: Path) -> None:
    case = tmp_path / "rerun" / "driver_none"
    case.mkdir(parents=True)
    pointer = {"from": "../../missing/driver_none"}
    (case / "reuse.json").write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="reuse"):
        load_row("driver", "none", tmp_path / "rerun")


def _warm_start_receipt(replay_rms: float) -> dict:
    """A warm-start receipt whose decoy RMS fields disagree with the replay."""
    return {
        "dynamics": {"marker_rms_m": 0.0011, "segment_rms_m": {"club": 0.0011}},
        "trajectory_optimiser": {
            "iterations": 60,
            "stop_reason": "converged",
            "port_check_replay_marker_rms_m": 0.0012,  # MJX plant
            "best_replay_marker_rms_m": 0.0013,  # MJX plant
            "surrogate_predicted_rms_m": 0.0014,
            "shared_simulator_replay": {
                "replay_marker_rms_m": replay_rms,
                "replay_segment_rms_m": {"club": replay_rms},
                "weight_fraction": {"by_phase": {"downswing": {"min": 0.4}}},
            },
        },
    }


def test_warm_start_arm_reports_only_shared_simulator_replay_rms() -> None:
    from scripts.benchmark_mjx_knot_solvers import (
        WARM_START_SOLVER,
        warm_start_arm_budget,
        warm_start_row_from_receipt,
    )

    budget = warm_start_arm_budget("driver", mjx_iterations=60)
    row = warm_start_row_from_receipt(
        "driver", _warm_start_receipt(0.0408), wall_clock_s=100.0, budget=budget
    )

    assert row["solver"] == WARM_START_SOLVER == "mjx-warm-start"
    assert row["replay_marker_rms_m"] == pytest.approx(0.0408)
    assert row["worst_segment_rms_m"] == pytest.approx(0.0408)
    assert row["rms_source"] == "trajectory_optimiser.shared_simulator_replay"
    assert row["iterations"] == 60


def test_warm_start_arm_uses_equal_wallclock_and_iteration_budgets() -> None:
    from scripts.benchmark_mjx_knot_solvers import (
        LBFGS60_EVIDENCE,
        LBFGS60_MJX_ITERATIONS,
        warm_start_arm_budget,
        warm_start_solver_args,
    )

    reference = json.loads(
        (LBFGS60_EVIDENCE / "driver_mjx-lbfgs" / "run.json").read_text("utf-8")
    )
    budget = warm_start_arm_budget("driver", mjx_iterations=60)

    assert LBFGS60_MJX_ITERATIONS == 60
    assert budget == {
        "mjx_iterations": 60,
        "wall_clock_budget_s": pytest.approx(reference["wall_clock_s"]),
    }
    args = warm_start_solver_args("driver", mjx_iterations=60)
    assert args[args.index("--mjx-iterations") + 1] == "60"
    assert args[-1] == "--mjx-warm-start-proposal"
    with pytest.raises(ValueError, match="60"):
        warm_start_arm_budget("driver", mjx_iterations=40)


def test_warm_start_stub_case_is_unavailable_citing_blockers(tmp_path: Path) -> None:
    (tmp_path / "driver_mjx-warm-start").mkdir()

    row = load_row("driver", "mjx-warm-start", tmp_path)

    assert row["status"] == "unavailable"
    assert "replay_marker_rms_m" not in row
    for blocker in ("#13783", "#13784", "#13785"):
        assert blocker in row["reason"]
