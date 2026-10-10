"""Keep the actual native activation receipt bound to its reviewed sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_native_activation_receipt_is_source_bound_and_scoped() -> None:
    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "docs/development/feedback_controls/F05E_NATIVE_ACTIVATION_RECEIPT_MJ38.json"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "f05e-native-activation-diagnostic/1.0.0"
    assert receipt["runtime"] == {"mujoco": "3.8.0", "crocoddyl": "3.2.1"}
    assert receipt["dimensions"] == {
        "nq": 8,
        "nv": 7,
        "na": 1,
        "nu": 1,
        "physical_nx": 16,
        "tangent_ndx": 15,
        "integration_state_size": 50,
    }
    assert receipt["warmstart_clock_physical_delta"] == 0
    assert receipt["native_replay"]["max_independent_full_state_delta"] == 0
    assert receipt["optimization_acceptance"].startswith("not_run")
    assert "no golfer qualification" in receipt["scope"]
    errors = receipt["native_derivative"]["directional_errors"]
    assert len(errors) == 2 and max(errors.values()) < 2e-7
    assert abs(receipt["native_derivative"]["activation_to_hinge_velocity"]) > 1e-5
    assert receipt["native_derivative"]["command_to_activation"] > 0
    for relative, digest in receipt["source_sha256"].items():
        source = (root / relative).resolve(strict=True)
        assert source.is_relative_to(root)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
