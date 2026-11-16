"""NSX-04 anatomical lower-limb torque slots must be VERIFIED or ASSUMED with rationale (#12553)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_REG = _ROOT / "src/config/loop_nullspace_registry.json"
_ANATOMICAL_PREFIXES = (
    "hip_adduction_",
    "hip_rotation_",
    "subtalar_angle_",
    "mtp_angle_",
)


def test_anatomical_torque_capacity_slots_resolved() -> None:
    """#12553: all 16 anatomical-slot rows are VERIFIED or explicitly ASSUMED."""
    registry = json.loads(_REG.read_text(encoding="utf-8"))
    anatomical = [
        entry
        for entry in registry["torque_capacity"]
        if any(
            entry["joint_slot"].startswith(prefix) for prefix in _ANATOMICAL_PREFIXES
        )
    ]
    assert len(anatomical) == 16
    for entry in anatomical:
        label = f"{entry['joint_slot']}/{entry['direction']}"
        status = entry["verification_status"]
        if status == "VERIFIED":
            assert entry["assumed"] is False
            assert entry["source"].get("citation_id"), label
            assert entry["source"].get("location"), label
            continue
        assert status == "UNVERIFIED", label
        assert entry["assumed"] is True, label
        assert entry["source"]["note"].strip(), label
        assert entry["rationale"].strip(), label
        assert "No source read for this slot." not in entry["source"]["note"]
