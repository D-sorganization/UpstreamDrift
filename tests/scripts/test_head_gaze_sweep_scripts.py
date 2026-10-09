"""Helpers of the OSV-3b sweep and engine-render scripts (no heavy runs)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.render_head_gaze_engines import head_glyph_arrows
from scripts.summarize_gaze_sweep import load_rows, summarize
from scripts.sweep_gaze_weight import _fixed_scales, row_from

pytestmark = pytest.mark.unit


def test_row_from_converts_metres_to_millimetres() -> None:
    ik = {
        "reference": {"marker_rms_m": 0.03, "closure_error_max_m": 0.004},
        "face_orientation": {"reference": {"rms_deg": 0.8}},
    }
    row = row_from(ik, {"gaze_weight": 3.0})
    assert row["marker_rms_mm"] == pytest.approx(30.0)
    assert row["closure_error_max_mm"] == pytest.approx(4.0)
    assert row["face_fit_deg"] == {"rms_deg": 0.8}


def test_row_from_without_face_block_reports_empty_not_zero() -> None:
    ik = {"reference": {"marker_rms_m": 0.03, "closure_error_max_m": 0.004}}
    assert row_from(ik, {})["face_fit_deg"] == {}


def test_fixed_scales_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        _fixed_scales(0.0, 1.0)


def test_head_glyphs_point_along_forward_and_at_the_ball() -> None:
    forward, sight = head_glyph_arrows([0, 0, 1.5], [1, 0, 0], [0.5, 0, 0.02])
    assert forward.label == "head_forward" and sight.label == "line_of_sight"
    assert np.allclose(forward.tail_m, [0, 0, 1.5])
    assert forward.tip_m[0] > 0.5 and forward.tip_m[2] == pytest.approx(1.5)
    assert np.allclose(sight.tip_m, [0.5, 0, 0.02])


def _row(capture: str, w: float, marker: float, gaze: float) -> dict:
    return {
        "capture": capture,
        "gaze_weight": w,
        "marker_rms_mm": marker,
        "closure_error_max_mm": 4.0,
        "face_fit_deg": {"rms_deg": 0.8},
        "head_gaze": {
            "address_to_impact": {
                "theta_gaze_rms_deg": gaze,
                "theta_gaze_max_deg": 2 * gaze,
                "eye_translation_range_mm": [50.0, 100.0, 25.0],
                "head_yaw_range_deg": 30.0,
                "head_pitch_range_deg": 5.0,
                "head_roll_range_deg": 20.0,
            },
            "neck_ik_schedule": {"frames_with_clamping": 3},
        },
    }


def _write_rows(root: Path, rows: list[dict]) -> None:
    for r in rows:
        d = root / f"{r['capture']}_{r['gaze_weight']:g}"
        d.mkdir(parents=True)
        (d / "sweep_row.json").write_text(json.dumps(r), encoding="utf-8")


def test_summary_reports_rows_knees_and_the_common_default(tmp_path: Path) -> None:
    _write_rows(
        tmp_path,
        [
            _row("driver", 0, 30.0, 20.0),
            _row("driver", 0.5, 30.5, 2.0),
            _row("driver", 3, 32.5, 1.4),
            _row("driver", 1, 31.0, 1.5),
            _row("iron", 0, 30.0, 20.0),
            _row("iron", 0.5, 31.0, 3.0),
            _row("iron", 1, 34.0, 1.0),
        ],
    )
    summary = summarize(load_rows(tmp_path))
    driver = summary["captures"]["driver"]
    assert [r["gaze_weight"] for r in driver["rows"]] == [0, 0.5, 1, 3]
    assert driver["label"] == "capture-A driver"
    assert summary["captures"]["iron"]["feasible_weights"] == [0, 0.5]
    assert summary["selected_weight"] == 0.5
    assert driver["rows"][0]["head_yaw_pitch_roll_range_deg"] == [30.0, 5.0, 20.0]
    json.dumps(summary)  # serialisable


EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "docs/development/full_body_models/evidence/head_gaze"
)


def test_committed_sweep_evidence_reproduces_the_reporting_weight() -> None:
    from src.shared.python.motion_matching import gaze_sweep as gs

    summary = json.loads((EVIDENCE / "gaze_weight_sweep.json").read_text())
    sweeps = {
        capture: [
            gs.SweepPoint(
                r["gaze_weight"],
                r["marker_rms_mm"],
                r["theta_gaze_rms_deg"],
                r["face_fit_rms_deg"],
            )
            for r in block["rows"]
        ]
        for capture, block in summary["captures"].items()
    }
    assert set(sweeps) == {"driver", "iron"}
    assert gs.select_default(sweeps) == summary["selected_weight"]
    assert summary["selected_weight"] == gs.REPORTING_GAZE_WEIGHT


def test_qualified_receipts_stay_marker_faithful_by_default() -> None:
    from src.shared.python.motion_matching.pipeline.cli import build_parser

    assert build_parser().get_default("gaze_weight") == 0.0


def test_load_rows_rejects_an_empty_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no sweep_row"):
        load_rows(tmp_path)


def test_head_glyphs_reject_degenerate_input() -> None:
    with pytest.raises(ValueError):
        head_glyph_arrows([0, 0, 1], [0, 0, 0], [1, 0, 0])
    with pytest.raises(ValueError):
        head_glyph_arrows([0, 0, 1], [1, 0, 0], [0, 0, 1])
