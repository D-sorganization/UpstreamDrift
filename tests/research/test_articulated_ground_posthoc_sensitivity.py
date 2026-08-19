from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.scientific


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def test_ground_posthoc_sensitivity_preserves_primary_failure_and_mixed_signs() -> None:
    record = json.loads(
        (DATA / "articulated_ground_posthoc_sensitivity.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["schema_version"] == "articulated-ground-posthoc-sensitivity/v1"
    assert record["analysis_status"].startswith("post_hoc_sensitivity")
    assert record["primary_result"]["matched_cell_count"] == 0
    assert all(
        item["matched_cell_count"] == 0
        for item in record["primary_result"]["tolerance_sensitivity"][:-1]
    )
    alternative = record["nonground_dissipation_sensitivity"]["tolerance_sensitivity"][
        0
    ]
    assert alternative["matched_cell_count"] == 60
    assert alternative["speed_difference_m_s"]["positive_count"] > 0
    assert alternative["speed_difference_m_s"]["negative_count"] > 0
    for relative, expected in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
