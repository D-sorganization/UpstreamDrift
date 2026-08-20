"""Evidence gates for the committed nominal scaled-authority baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    load_scaled_authority,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
RECORD = DATA / "articulated_scaled_authority_nominal.json"
ARRAYS = DATA / "articulated_scaled_authority_nominal.npz"
pytestmark = pytest.mark.scientific


def test_nominal_scaled_authority_record_is_complete_and_reproducible() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    assert record["schema_version"] == "articulated-scaled-authority/v1"
    assert record["configuration"] == {
        "case_indices": [0, 8, 9, 17],
        "height_scale": 1.0,
        "body_mass_scale": 1.0,
        "joint_limit_scale": 1.0,
    }
    results = record["results"]
    assert results["selected_case_count"] == 4
    assert results["phase_sample_count_per_case"] == 13
    assert results["selected_feasible_sample_count"] == 52
    assert results["selected_total_sample_count"] == 52
    assert results["selected_failure_distribution"] == {"feasible": 52}
    assert results["maximum_nominal_state_error_rad"] <= 1.0e-8
    for relative, expected in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_nominal_scaled_authority_arrays_validate_without_silent_deletion() -> None:
    authority = load_scaled_authority(RECORD, ARRAYS)

    assert authority.selected_case_indices.tolist() == [0, 8, 9, 17]
    assert authority.selected_failure_class.shape == (4, 13)
    assert np.all(authority.selected_failure_class == "feasible")
    assert np.all(authority.feasible[authority.selected_case_indices])
