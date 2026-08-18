"""Tests for the closed-state forward-contact bridge evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.closed_state_forward_bridge import (
    ClosedStateBridgeConfig,
    canonical_state_from_vector,
    map_closed_contact_atlas,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "docs/research/proximal_distal_energy_transfer/data"
pytestmark = pytest.mark.scientific


def test_bridge_configuration_and_packed_state_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive"):
        ClosedStateBridgeConfig(velocity_closure_tolerance_m_s=0.0)
    with pytest.raises(ValueError, match="25-element"):
        canonical_state_from_vector(np.zeros(24))


def test_all_closed_states_map_with_declared_closure_gates() -> None:
    record, arrays = map_closed_contact_atlas()
    assert record["design"]["mapped_state_count"] == 234
    assert record["results"]["position_closure_gate_passed"] is True
    assert record["results"]["velocity_closure_gate_passed"] is True
    assert record["results"]["unique_initial_state_digest_count"] == 234
    assert arrays["canonical_state"].shape == (18, 13, 25)
    assert np.all(np.isfinite(arrays["canonical_state"]))


def test_committed_bridge_evidence_passes_constitutive_and_engine_gates() -> None:
    record = json.loads(
        (DATA_DIR / "closed_state_forward_bridge.json").read_text(encoding="utf-8")
    )
    assert record["schema_version"] == "closed-state-forward-bridge/v1"
    assert record["results"]["maximum_position_closure_error_m"] < 5.0e-4
    assert record["results"]["maximum_velocity_closure_error_m_s"] < 5.0e-3
    controls = record["results"]["constitutive_controls"]
    assert controls["maximum_zero_preload_force_n"] < 1.0e-8
    assert controls["action_reaction_residual_n"] < 1.0e-10
    assert controls["damping_passive"] is True
    forward = record["forward_subset"]
    assert forward["subset_case_count"] == 54
    assert forward["all_initial_state_digests_match"] is True
    assert forward["trajectory_gate_passed"] is True
    assert forward["wrench_gate_passed"] is True
    assert forward["energy_gate_passed"] is True
    for path, digest in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_committed_bridge_arrays_are_complete_and_finite() -> None:
    with np.load(DATA_DIR / "closed_state_forward_bridge.npz") as archive:
        assert archive["canonical_state"].shape == (18, 13, 25)
        assert archive["subset_club_position_m"].shape == (54, 2, 17, 3)
        assert archive["subset_contact_wrench"].shape == (54, 2, 17, 6)
        for name in archive.files:
            assert np.all(np.isfinite(archive[name])), name
