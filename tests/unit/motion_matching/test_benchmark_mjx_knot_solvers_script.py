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
