"""NM-02 first-wave adapter qualification tests."""

from __future__ import annotations

import pytest
from src.shared.python.data_io.dataset_generator.adapters import (
    FIRST_WAVE_MODEL_IDS,
    qualify_first_wave_adapters,
    qualify_mock_adapter,
    qualify_ode_double_pendulum_adapter,
)
from src.shared.python.data_io.dataset_generator.labels import LABEL_SCHEMA

pytestmark = pytest.mark.unit


def test_first_wave_model_ids_reuse_nm01_pilot_roster() -> None:
    from src.shared.python.neural_motion.roster import (
        RosterStage,
        build_neural_model_roster,
    )

    roster = build_neural_model_roster()
    pilot = {
        entry.model_id
        for entry in roster.entries
        if entry.pilot_stage is RosterStage.PILOT_ELIGIBLE
    }
    assert set(FIRST_WAVE_MODEL_IDS) == pilot
    assert "driven_double_pendulum" in FIRST_WAVE_MODEL_IDS


def test_mock_adapter_receipt_qualified() -> None:
    receipt = qualify_mock_adapter()
    assert receipt.schema == LABEL_SCHEMA
    assert receipt.qualified is True
    assert receipt.residual_norm is not None
    assert receipt.residual_norm < 1e-8
    assert "software-contract" in " ".join(receipt.limitations).lower()


def test_ode_double_pendulum_native_residual_receipt() -> None:
    receipt = qualify_ode_double_pendulum_adapter()
    assert receipt.model_id == "driven_double_pendulum"
    assert receipt.adapter == "ODEBackend"
    assert receipt.qualified is True
    assert receipt.residual_norm is not None
    assert receipt.residual_norm < 1e-6
    assert receipt.channel_summary["native_accelerations"] == "available"
    assert receipt.channel_summary["contact_forces"] == "not_requested"


def test_qualify_first_wave_returns_both_receipts() -> None:
    receipts = qualify_first_wave_adapters()
    assert len(receipts) == 2
    assert {r.model_id for r in receipts} == {
        "mock_software_contract",
        "driven_double_pendulum",
    }
