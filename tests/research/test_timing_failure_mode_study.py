from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/timing_failure_mode_study.json"
)


def test_factorial_is_complete_and_preserves_alternative_casting_definitions() -> None:
    record = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    grid = record["registered_grid"]
    expected = (
        len(grid["proximal_acceleration_onset_s"])
        * len(grid["proximal_braking_onset_s"])
        * len(grid["distal_release_onset_s"])
    )
    assert record["case_count"] == expected == 27
    assert len(record["rows"]) == expected
    assert set(record["casting_definitions"]) == {"angle", "rate", "agreement_window_s"}
    assert any(
        not row["casting_definitions_agree_within_20_ms"] for row in record["rows"]
    )


def test_study_rejects_universal_timing_and_retains_matching_boundary() -> None:
    record = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert record["claim_status"]["universal_casting_event"] == "unsupported"
    assert record["claim_status"]["universal_optimal_timing"] == "unsupported"
    assert record["claim_status"]["human_coaching_strategy"] == "untested"
    assert any("matched-work" in limitation for limitation in record["limitations"])
