from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data"
    / "torso_velocity_human_validation_registration.json"
)


def test_registration_is_participant_held_out_and_fail_closed() -> None:
    record = json.loads(REGISTRATION.read_text(encoding="utf-8"))

    assert record["registration_status"] == "frozen_before_governed_human_outcomes"
    assert record["dataset_status"] == "not_acquired"
    assert record["primary_split_unit"] == "participant"
    assert "bilateral_hand_wrenches" in record["required_modalities"]
    assert "synthetic dry runs do not satisfy" in record["completion_gate"]


def test_registration_contains_null_adverse_and_sensitivity_tests() -> None:
    record = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    tests = {item["test_id"]: item for item in record["preregistered_tests"]}

    assert set(tests) == {"TV-H1", "TV-H2", "TV-H3"}
    assert all(item["null"] and item["falsifier"] for item in tests.values())
    assert "inverse_dynamics_residual_threshold" in record["sensitivity_axes"]
    assert "synchronization_extremes" in record["sensitivity_axes"]
