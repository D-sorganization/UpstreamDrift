from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/"
    "bilateral_wrench_sensor_qualification.json"
)

pytestmark = pytest.mark.scientific


def test_sensor_qualification_evidence_is_complete_and_fail_closed() -> None:
    record = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert record["schema_version"] == "bilateral-wrench-sensor-qualification/v1"
    assert record["analysis_type"] == "synthetic_point_force_sensor_qualification"
    assert record["sample_count"] == 301
    assert record["trial_count"] == 32
    assert record["seed"] == 20260814
    assert len(record["cases"]) == 9
    assert record["cases"]["ideal_augmented"]["allocation_rmse_n"] < 1e-10
    assert record["cases"]["net_wrench_only"]["axial_mode_rmse_n"] > 5.0
    assert (
        record["cases"]["cross_talk_calibrated"]["allocation_rmse_n"]
        < record["cases"]["cross_talk_uncorrected"]["allocation_rmse_n"]
    )
    assert (
        record["cases"]["contact_migration_tracked"]["allocation_rmse_n"]
        < record["cases"]["contact_migration_fixed"]["allocation_rmse_n"]
    )
    assert record["qualification"]["net_wrench_only_allocation"] == "fails_by_structure"
    assert record["qualification"]["declared_synthetic_augmented_map"] == (
        "recoverable_with_error_conditioned_on_sensor_and_contact_assumptions"
    )
    assert record["boundaries"]["bilateral_full_wrench_allocation"] == "not_addressed"
    assert record["boundaries"]["human_validation"] == "untested"
    assert record["boundaries"]["muscle_or_scapular_strategy"] == "not_identified"
