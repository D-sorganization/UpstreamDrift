from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.experimental_protocol import (
    ProtocolManifest,
    evaluate_dataset_readiness,
)
from scripts.research.proximal_distal_energy.run_experimental_protocol_dry_run import (
    build_readiness_record,
)

pytestmark = pytest.mark.unit


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _protocol() -> ProtocolManifest:
    return ProtocolManifest.from_json(DATA / "experimental_protocol_v1.json")


def test_registered_protocol_has_complete_measurement_and_analysis_contract() -> None:
    protocol = _protocol()
    assert protocol.registration_status == "frozen_before_human_outcomes"
    assert {item.name for item in protocol.modalities} == {
        "full_body_kinematics",
        "bilateral_hand_wrenches",
        "club_kinematics",
        "ground_reaction",
        "launch_monitor",
        "surface_emg",
    }
    assert protocol.held_out_fraction >= 0.2
    assert protocol.identity_policy == "pseudonym_only_no_identity_inference"
    assert {item.prediction_id for item in protocol.predictions} == {
        "EXP-H1",
        "EXP-H2",
        "EXP-H3",
        "EXP-H4",
    }


def test_synthetic_dry_run_cannot_advance_human_claims() -> None:
    protocol = _protocol()
    record = json.loads((DATA / "experimental_protocol_dry_run.json").read_text())
    readiness = evaluate_dataset_readiness(protocol, record)
    assert readiness.pipeline_ready
    assert not readiness.claims_evaluable
    assert readiness.status == "synthetic_dry_run_only"


def test_identity_bearing_fields_fail_closed() -> None:
    protocol = _protocol()
    record = json.loads((DATA / "experimental_protocol_dry_run.json").read_text())
    record["participants"][0]["name"] = "Not Allowed"
    with pytest.raises(ValueError, match="identity-bearing field"):
        evaluate_dataset_readiness(protocol, record)


def test_participant_split_overlap_fails_closed() -> None:
    protocol = _protocol()
    record = json.loads((DATA / "experimental_protocol_dry_run.json").read_text())
    record["participants"][1]["split"] = "training"
    record["participants"][1]["pseudonym"] = record["participants"][0]["pseudonym"]
    with pytest.raises(ValueError, match="pseudonyms must be unique"):
        evaluate_dataset_readiness(protocol, record)


def test_missing_modality_hash_or_wrong_units_fails_closed() -> None:
    protocol = _protocol()
    record = json.loads((DATA / "experimental_protocol_dry_run.json").read_text())
    record["modalities"]["bilateral_hand_wrenches"]["units"] = ["lbf"]
    with pytest.raises(ValueError, match="units do not match"):
        evaluate_dataset_readiness(protocol, record)


def test_governed_status_requires_ethics_consent_and_nonpublic_authority() -> None:
    protocol = _protocol()
    record = json.loads((DATA / "experimental_protocol_dry_run.json").read_text())
    record["dataset_status"] = "governed_human_data"
    with pytest.raises(ValueError, match="ethics_approval_reference"):
        evaluate_dataset_readiness(protocol, record)


def test_committed_readiness_matches_executable_dry_run() -> None:
    committed = json.loads((DATA / "experimental_protocol_readiness.json").read_text())
    assert committed == build_readiness_record()
    assert committed["human_data_evaluation"] == "not_executed"
    assert set(committed["claim_status"].values()) == {
        "untested_no_governed_human_data"
    }
