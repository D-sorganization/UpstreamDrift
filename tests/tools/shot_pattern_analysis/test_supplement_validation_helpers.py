"""Supplement validation preserves declared bounds and immutable input checks."""

import hashlib
from pathlib import Path

import pytest

from src.tools.shot_pattern_analysis import matched_supplement_report as report

pytestmark = pytest.mark.unit


def test_match_bounds_preserve_declared_units_and_validate_face_first():
    entry = {"face_bounds_deg": [-8.0, 8.0], "speed_bounds_mps": [20.0, 60.0]}
    config = {
        "match_face_bounds_deg": [-8.0, 8.0],
        "match_speed_factors": [0.5, 1.5],
        "club": {"club_speed_mps": 40.0},
    }
    assert report._validate_match_bounds("cell", entry, config) == (
        [-8.0, 8.0],
        [0.5, 1.5],
    )
    with pytest.raises(ValueError, match="supplement face bounds differ"):
        report._validate_match_bounds("cell", entry, {"match_face_bounds_deg": [1.0]})
    config["match_speed_factors"] = [0.5, 1.6]
    with pytest.raises(ValueError, match="supplement speed bounds differ"):
        report._validate_match_bounds("cell", entry, config)


def test_recheck_rejects_inventory_changed_after_aggregation(tmp_path):
    inventory = tmp_path / "inventory.json"
    inventory.write_bytes(b"original")
    digest = hashlib.sha256(inventory.read_bytes()).hexdigest()
    report._recheck_supplement_sources({}, [], inventory, digest)
    inventory.write_bytes(b"changed")
    with pytest.raises(
        ValueError, match="supplement inventory changed during aggregation"
    ):
        report._recheck_supplement_sources({}, [], inventory, digest)
