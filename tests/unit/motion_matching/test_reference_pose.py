"""Unit tests for the reference golfer pose and comparison helper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "shared" / "python"))

from src.shared.python.motion_matching.diagnostics.reference_pose import (  # noqa: E402
    ADDRESS_RANGES,
    REFERENCE_GOLFER_FIELDS,
    compare_to_reference,
    reference_golfer_setup,
)

pytestmark = pytest.mark.unit


def test_reference_pose_is_anatomically_plausible() -> None:
    """Knees / spine reflect real address: forward tilt, arms in front."""
    p = reference_golfer_setup()
    # Forward spine tilt is the load-bearing assertion.
    assert 15.0 <= p["SpineStartPositionX"] <= 45.0
    # Both elbows nearly straight at address.
    assert -5.0 <= p["LEStartPosition"] <= 30.0
    assert -5.0 <= p["REStartPosition"] <= 30.0
    # Lead wrist roughly flat.
    assert abs(p["LWStartPositionX"]) <= 30.0
    # Every reference field must be defined.
    for f in REFERENCE_GOLFER_FIELDS:
        assert f in p, f"reference_golfer_setup missing {f}"


def test_reference_pose_passes_its_own_ranges() -> None:
    """The reference pose must itself pass the address-range check."""
    flags = compare_to_reference(reference_golfer_setup())
    assert flags == [], f"reference pose flagged: {flags}"


def test_compare_to_reference_flags_outliers() -> None:
    """The actual values from 3DModelInputs_Impact.mat must be flagged."""
    impact_pose_from_csv = {
        "HipStartPositionZ": -45.0,
        "SpineStartPositionX": 0.0,  # zero forward tilt — the smoking gun
        "SpineStartPositionY": 0.0,
        "TorsoStartPosition": -45.0,
        "LSStartPositionZ": -135.72,  # top-of-backswing magnitude
        "RSStartPositionZ": 96.03,
        "LEStartPosition": 5.78,
        "REStartPosition": 100.69,  # right elbow flexed 100 deg
        "LWStartPositionX": -97.84,  # lead wrist hinged ~97 deg
        "RWStartPositionX": -80.02,
    }
    flags = compare_to_reference(impact_pose_from_csv)
    flagged = {f["field"] for f in flags}
    # The four 'must-flag' fields:
    assert "SpineStartPositionX" in flagged
    assert "REStartPosition" in flagged
    assert "LWStartPositionX" in flagged
    assert "RWStartPositionX" in flagged


def test_compare_to_reference_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        compare_to_reference([1.0, 2.0])  # type: ignore[arg-type]


def test_address_ranges_cover_known_problem_fields() -> None:
    """Sanity check the ranges-table has the fields we rely on flagging."""
    must_have = {
        "SpineStartPositionX",
        "REStartPosition",
        "LWStartPositionX",
        "RWStartPositionX",
        "LSStartPositionZ",
    }
    assert must_have.issubset(ADDRESS_RANGES.keys())
