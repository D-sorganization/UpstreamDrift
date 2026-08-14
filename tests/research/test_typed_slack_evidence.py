from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/typed_slack_study.json"
)


def test_typed_slack_evidence_is_complete_and_neutral() -> None:
    record = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert set(record["classes"]) == {
        "contact_disengagement",
        "transmission_backlash",
        "structural_preload",
        "biological_series_compliance",
        "control_deadband",
    }
    assert set(record["cases"]) == set(record["classes"])
    assert record["claims"]["global_slack_benefit"] == "unsupported"
    assert record["claims"]["human_strategy"] == "untested"
    assert all(abs(case["energy_residual"]) < 2e-5 for case in record["cases"].values())
