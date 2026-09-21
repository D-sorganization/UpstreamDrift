"""Unit tests for engine capability matrix generation and tile status derivation (MS-71, #10351)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from src.config.capability_state import CapabilityQualification
from src.shared.python.shadow_tracker.engine_matrix import (
    EngineReceipt,
    EngineQualificationResult,
    EngineCapabilityMatrix,
    audit_engine_conformance,
)

pytestmark = pytest.mark.unit


def test_audit_engine_conformance_unqualified() -> None:
    """An advertised engine with no receipt fails qualification closed."""
    res = audit_engine_conformance(
        engine_name="mujoco",
        receipt=None,
        is_advertised=True,
    )
    assert not res.is_qualified
    assert res.status == "qualification_failed"
    assert "missing_engine_receipt" in res.failure_reasons


def test_audit_engine_conformance_qualified() -> None:
    """An engine with an accepted receipt is advertised_and_qualified."""
    receipt = EngineReceipt(
        engine_name="mujoco",
        engine_version="3.2.0",
        model_name="anthro_driver",
        model_sha256="abc12345",
        state_convention="canonical_v2_quaternion",
        contact_model_type="elliptic_cone",
        is_physically_accepted=True,
        measured_closure_translation_m=0.001,
        measured_closure_rotation_rad=0.01,
    )
    res = audit_engine_conformance(
        engine_name="mujoco",
        receipt=receipt,
        required_model_hash="abc12345",
        is_advertised=True,
    )
    assert res.is_qualified
    assert res.status == "advertised_and_qualified"
    assert len(res.failure_reasons) == 0


def test_matrix_generation_from_fixture_ledger(tmp_path: Path) -> None:
    """Generate matrix from ledger data where one engine passes and others have no receipts."""
    from scripts.generate_engine_matrix import generate_matrix_data

    # Mock ledger with no accepted receipts
    ledger_rows: list[dict[str, Any]] = [
        {
            "engine": "mujoco",
            "receipt_path": "evidence/ground_support/anthro_driver/receipt.json",
            "acceptance": {
                "horizon": "G3",
                "is_physically_accepted": False,
                "status": "REJECTED",
            },
        }
    ]

    matrix = generate_matrix_data(ledger_rows=ledger_rows)
    assert "mujoco" in matrix["profiles"]
    assert not matrix["profiles"]["mujoco"]["is_qualified"]
    assert matrix["profiles"]["mujoco"]["status"] == "qualification_failed"
    assert "missing_engine_receipt" in matrix["profiles"]["mujoco"]["failure_reasons"]
    assert matrix["profiles"]["opensim"]["tier"] == "experimental"
    assert matrix["profiles"]["myosuite"]["tier"] == "experimental"
    assert matrix["profiles"]["jaxsim"]["tier"] == "experimental"


def test_tile_status_derivation_table() -> None:
    """Verify tile status derivation for engines and non-engines."""
    from scripts.generate_engine_matrix import derive_tile_status

    # 1. Advertised and qualified engine with installed runtime -> ready
    s1 = derive_tile_status(
        is_engine=True,
        declared_status="ready",
        qualification_status="advertised_and_qualified",
        runtime_available=True,
        tier="core",
    )
    assert s1 == "ready"

    # 2. Engine whose qualification failed -> experimental (never claims ready)
    s2 = derive_tile_status(
        is_engine=True,
        declared_status="ready",
        qualification_status="qualification_failed",
        runtime_available=True,
        tier="core",
    )
    assert s2 == "experimental"

    # 3. Engine in experimental tier -> experimental
    s3 = derive_tile_status(
        is_engine=True,
        declared_status="ready",
        qualification_status="qualification_failed",
        runtime_available=True,
        tier="experimental",
    )
    assert s3 == "experimental"

    # 4. Engine whose runtime is not installed -> runtime_unavailable
    s4 = derive_tile_status(
        is_engine=True,
        declared_status="ready",
        qualification_status="advertised_and_qualified",
        runtime_available=False,
        tier="core",
    )
    assert s4 == "runtime_unavailable"

    # 5. Non-engine tile -> preserves declared status
    s5 = derive_tile_status(
        is_engine=False,
        declared_status="ready",
        qualification_status="exempt",
        runtime_available=True,
        tier=None,
    )
    assert s5 == "ready"
