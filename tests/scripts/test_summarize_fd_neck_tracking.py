"""Evidence summary of the forward-dynamics neck-tracking runs (OSV-3c, #11729)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_fd_neck_tracking import run_row, summarize

pytestmark = pytest.mark.unit


def _receipt(tmp_path: Path, name: str, capture: str, block: dict | None) -> Path:
    run = tmp_path / name
    run.mkdir()
    dynamics = {
        "marker_rms_m": 0.0712,
        "fd_phase": {"fd_rms_address_to_impact_m": 0.0333},
        "segment_rms_m": {"head": 0.087},
    }
    if block is not None:
        dynamics["head_gaze"] = block
    receipt = {
        "capture": capture,
        "ik": {"reference": {"marker_rms_m": 0.0282}},
        "dynamics": dynamics,
        "head_gaze": {"gaze_weight": 0.0, "address_to_impact": {"x": 1}},
    }
    (run / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return run


def test_row_converts_to_millimetres_and_keeps_the_block(tmp_path: Path) -> None:
    block = {"available": True, "neck_reference": "gaze_schedule"}
    row = run_row(_receipt(tmp_path, "iron_gaze_w0", "iron", block))
    assert row["fd_marker_rms_mm"] == pytest.approx(71.2)
    assert row["ik_marker_rms_mm"] == pytest.approx(28.2)
    assert row["fd_address_to_impact_rms_mm"] == pytest.approx(33.3)
    assert row["fd_segment_rms_mm"] == {"head": pytest.approx(87.0)}
    assert row["neck_reference"] == "gaze_schedule"
    assert row["fd_head_gaze"] == block


def test_summary_sorts_by_capture_and_rejects_missing_input(tmp_path: Path) -> None:
    block = {"neck_reference": "ik"}
    runs = [
        _receipt(tmp_path, "iron_ik_w0", "iron", block),
        _receipt(tmp_path, "driver_ik_w0", "driver", block),
    ]
    assert [r["capture"] for r in summarize(runs)["rows"]] == ["driver", "iron"]
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError):
        run_row(_receipt(tmp_path, "old", "driver", None))
