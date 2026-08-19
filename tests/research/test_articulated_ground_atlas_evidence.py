from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.scientific


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def test_committed_ground_atlas_is_complete_current_and_adverse() -> None:
    record = json.loads(
        (DATA / "articulated_ground_atlas.json").read_text(encoding="utf-8")
    )
    with np.load(DATA / "articulated_ground_atlas.npz") as source:
        assert source["primary_peak_grip_force"].shape == (12, 4, 2, 2, 2, 4)
        assert source["control_peak_grip_force"].shape == (12, 2, 2, 2, 2, 4)
        assert source["primary_trajectory_parity"].shape == (12, 4, 2, 2, 4)
        assert source["control_trajectory_parity"].shape == (12, 2, 2, 2, 4)
        assert np.all(source["primary_numerical"])
        assert np.all(source["control_numerical"])
        assert np.all(source["primary_parity"])
        assert np.all(source["control_parity"])
        assert np.max(source["primary_maximum_base_translation"]) < 0.05
        assert np.max(source["primary_maximum_base_pitch"]) < np.deg2rad(10.0)
    assert record["schema_version"] == "articulated-ground-atlas/v1"
    assert record["design"]["primary_trajectory_count"] == 384
    assert record["design"]["control_trajectory_count"] == 192
    assert record["results"]["all_registered_gates_passed"] is True
    assert record["results"]["time_refinement_passed"] is True
    assert record["results"]["initial_energy_match_passed"] is True
    assert record["results"]["matched_load_work_cell_count"] == 0
    assert record["results"]["matched_load_work_total_cell_count"] == 384
    assert record["limitations"]["calibration_status"] == (
        "synthetic_reference_not_human_or_force_plate_calibrated"
    )
    for relative, expected in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
