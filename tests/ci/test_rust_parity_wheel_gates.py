from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.check_rust_parity_wheel_gates import REQUIRED_GATES, audit

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.gate


def test_rust_parity_wheel_gates_are_enforced() -> None:
    assert audit(REPO_ROOT) == []


def test_upstream_physics_has_explicit_crate_level_parity_test() -> None:
    physics_rows = [
        gate for gate in REQUIRED_GATES if gate.crate == "rust_core/upstream-physics"
    ]

    assert physics_rows
    assert {
        "src/shared/python/physics/rust_kernel.py",
        "src/shared/python/physics/ball_flight_physics.py",
    } <= {gate.facade for gate in physics_rows}
    assert all(
        gate.parity_test == "rust_core/upstream-physics/tests/parity_physics.rs"
        for gate in physics_rows
    )
