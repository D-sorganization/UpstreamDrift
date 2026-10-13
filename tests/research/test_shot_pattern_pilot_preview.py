"""The report-layout preview is visibly and structurally pilot-only."""

from __future__ import annotations

import pytest

from scripts.research.shot_pattern_analysis import build_pilot_preview

pytestmark = pytest.mark.unit


def _report(stage: str, n_shots: int = 32) -> dict:
    return {
        "schema": "matched-study-report/1",
        "stage": stage,
        "planned_cells": 1,
        "counts": {"completed": 0, "unavailable": 1, "not_run": 0},
        "cells": [
            {
                "cell": "driver_lean10_curve1_rotation1_shaft_only",
                "status": "unavailable",
                "reason": "nonairborne_nominal",
                "config": {
                    "club": {"club_id": "driver", "shaft_lean_deg": 10},
                    "noise": {
                        "standard_deviations": [1, 0, 0, 0, 0, 0, 0],
                        "correlation": None,
                    },
                    "curve_scale": 1,
                    "n_shots": n_shots,
                    "target_x_m": None if stage == "pilot" else 179,
                },
            }
        ],
        "plan_sha256": "a" * 64,
    }


def test_preview_rejects_wrong_stage_and_sample_count() -> None:
    with pytest.raises(ValueError, match="pilot"):
        build_pilot_preview.validate_pilot_matrices(_report("final"), _report("pilot"))
    with pytest.raises(ValueError, match="32"):
        build_pilot_preview.validate_pilot_matrices(
            _report("pilot", 10_000), _report("pilot")
        )


def test_preview_is_globally_watermarked_and_labels_attempted_count() -> None:
    primary = _report("pilot")
    control = _report("pilot")
    rendered = build_pilot_preview.render_pilot_preview(primary, control)
    assert rendered.startswith("# PILOT-ONLY:")
    assert "32 indexed errors attempted per physical pattern" in rendered
    assert "layout review only" in rendered.lower()
    assert "PILOT-ONLY" in rendered.split("##", 2)[1]
    assert "10,000" not in rendered
    assert "final conditional simulation report" not in rendered.lower()
    assert "nonairborne_nominal" in rendered
    assert "unavailable under the stated model/search bounds" in rendered
    assert "were infeasible" not in rendered
    assert (
        "presentation_preview_v8/details/driver_lean0_curve1_rotation1_shaft_only/overhead.png"
        in rendered
    )
    assert "[Full-Resolution Overhead PNG]" in rendered
