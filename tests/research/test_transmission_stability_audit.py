"""Stability gates for the transmission robustness chapter."""

import json
from pathlib import Path

from scripts.research.proximal_distal_energy.audit_transmission_stability import (
    build_audit,
)

ROOT = Path(__file__).resolve().parents[2]
PATH = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/transmission_stability_audit.json"
)


def test_committed_transmission_stability_audit_is_fresh() -> None:
    assert json.loads(PATH.read_text()) == build_audit()


def test_pareto_and_local_linear_boundaries_are_explicit() -> None:
    audit = build_audit()
    assert set(audit["pareto_membership_counts"].values()) == {15}
    local = audit["local_task_map"]
    assert local["effective_rank_by_relative_threshold"] == {
        "0.01": 2,
        "0.05": 2,
        "0.1": 1,
        "0.2": 1,
    }
    assert local["held_out_linear_prediction_rmse"]["face_path_error_deg"] > 2.0
    assert "units and scaling" in local["boundary"]
