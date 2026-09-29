from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data"
    / "articulated_ground_diagnostic.json"
)


import pytest

pytestmark = pytest.mark.scientific


def test_ground_diagnostic_evidence_is_complete_and_current() -> None:
    record = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert record["schema_version"] == "articulated-ground-diagnostic/v1"
    assert len(record["results"]) == 42
    assert len(record["parity"]) == 21
    assert len(record["refinement"]) == 14
    assert record["all_refinement_monotone"] is True
    assert record["all_active_sets_match"] is True
    assert all(item["remained_in_declared_domains"] for item in record["results"])
    assert all(item["trajectory_relative_error"] < 1e-9 for item in record["parity"])
    assert all(item["ground_force_relative_error"] < 1e-8 for item in record["parity"])
    assert all(
        0.45 < ratio < 0.55
        for item in record["refinement"]
        for ratio in item["successive_ratios"]
    )
    for relative, expected in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_ground_diagnostic_retains_initialization_sensitivity() -> None:
    record = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    initialization = record["initialization"]
    assert initialization["conditional_equilibrium"]["residual_norm"] < 1e-6
    assert initialization["conditional_equilibrium"]["active_station_count"] > 0

    rows = {
        item["branch"]: item
        for item in record["results"]
        if item["engine"] == "mujoco" and item["step_s"] == 0.000125
    }
    assert rows["coupled_gravity_only"]["peak_ground_force_n"] > 500.0
    assert rows["coupled_conditional"]["peak_ground_force_n"] > 450.0
    assert rows["coupled_natural_zero"]["peak_ground_force_n"] < 50.0
    assert (
        rows["coupled_gravity_only"]["final_club_translation_speed_m_s"]
        > 5.0 * rows["coupled_natural_zero"]["final_club_translation_speed_m_s"]
    )
