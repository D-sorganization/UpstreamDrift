"""Evidence gates for committed structural authority corner artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    load_scaled_authority,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
RECORD = DATA / "articulated_structural_authority_campaign.json"
pytestmark = pytest.mark.scientific


def test_structural_authority_campaign_is_complete_and_registered() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    assert record["schema_version"] == ("articulated-structural-authority-campaign/v1")
    assert record["status"] == "complete"
    assert [row["corner_id"] for row in record["corners"]] == [
        "nominal",
        "height_scale-low",
        "height_scale-high",
        "body_mass_scale-low",
        "body_mass_scale-high",
        "joint_limit_scale-low",
        "joint_limit_scale-high",
    ]
    results = record["results"]
    assert results["corner_count"] == 7
    assert (
        results["feasible_corner_count"]
        + results["infeasible_corner_count"]
        + results["failed_corner_count"]
        == 7
    )
    for relative, expected in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_every_generated_corner_loads_without_deleting_failures() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    for row in record["corners"]:
        assert row["status"] in {
            "feasible",
            "infeasible_retained",
            "failed_retained",
        }
        if row["status"] == "failed_retained":
            assert row["failure_class"]
            assert row["record_artifact"] is None
            assert row["array_artifact"] is None
            continue
        authority = load_scaled_authority(
            DATA / row["record_artifact"],
            DATA / row["array_artifact"],
        )
        selected = authority.selected_case_indices
        observed_failures = int((~authority.feasible[selected]).sum())
        assert observed_failures == row["failure_count"]
        assert len(authority.authority_sha256) == 64
