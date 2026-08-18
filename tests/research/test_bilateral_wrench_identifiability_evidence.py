from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/bilateral_wrench_identifiability_study.json"
)


def test_bilateral_wrench_identifiability_evidence_is_complete_and_neutral() -> None:
    record = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert record["schema_version"] == "bilateral-wrench-identifiability-study/v1"
    assert record["point_force_map"]["rank"] == 5
    assert record["point_force_map"]["nullity"] == 1
    assert record["full_bilateral_wrench_map"]["rank"] == 6
    assert record["full_bilateral_wrench_map"]["nullity"] == 6
    assert record["augmented_point_force_map"]["rank"] == 6
    assert record["rotation_audit"]["maximum_singular_value_difference"] < 1e-12
    assert record["claims"]["individual_hand_allocation_from_net_wrench"] == (
        "structurally_unidentifiable"
    )
    assert record["claims"]["muscle_or_scapular_strategy"] == "not_identified"
    assert record["claims"]["human_validation"] == "untested"
    assert record["measurement_boundary"][
        "bilateral_six_axis_required_for_direct_allocation"
    ]
