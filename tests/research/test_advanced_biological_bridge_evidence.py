"""Release-artifact checks for the advanced biological bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"


def test_advanced_bridge_record_retains_evidence_boundaries() -> None:
    record = json.loads((ARTICLE / "data/advanced_biological_bridge.json").read_text())

    assert record["schema_version"] == "1.0.0"
    assert record["epic"].endswith("/8505")
    assert record["frame_invariance"]["maximum_virtual_work_residual_w"] < 1e-11
    assert set(record["pose_adapter_round_trips"]) == {
        "mujoco",
        "pinocchio",
        "drake",
        "opensim",
        "myosuite",
    }
    assert record["engine_ladder"]["opensim"]["status"] != "human_validation"
    assert record["engine_ladder"]["myosuite"]["status"] != "human_validation"
    assert "does not identify" in record["claim_boundary"]


def test_advanced_bridge_trace_archive_reproduces_reported_metrics() -> None:
    record = json.loads((ARTICLE / "data/advanced_biological_bridge.json").read_text())
    with np.load(ARTICLE / "data/advanced_biological_bridge.npz") as arrays:
        np.testing.assert_allclose(
            arrays["redundancy__net_torque_nm"], 10.0, atol=1e-10
        )
        assert arrays["persistent_direction__time_s"][0] == pytest.approx(-0.18)
        assert 0.0 in arrays["persistent_direction__time_s"]
        assert (
            record["biological_programs"]["persistent_direction"][
                "post_transition_error_impulse_nms"
            ]
            < record["biological_programs"]["complete_role_reversal"][
                "post_transition_error_impulse_nms"
            ]
        )


def test_advanced_bridge_figures_exist_as_pdf_and_svg() -> None:
    for stem in (
        "fig_frame_power_invariance",
        "fig_biological_redundancy",
        "fig_biological_role_reversal",
        "fig_cross_engine_question_ladder",
        "fig_advanced_model_motion_plate",
    ):
        for suffix in (".pdf", ".svg"):
            path = ARTICLE / "figures" / f"{stem}{suffix}"
            assert path.is_file()
            assert path.stat().st_size > 1000
