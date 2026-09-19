"""Unit tests for immutable candidate-session ingestion (MV-03 #10479).

Validates:
1. Immutable candidate-session ingestion from receipt-bound model package.
2. Verification of candidate content hash and model hash.
3. Rejection of wrong model hash, wrong candidate hash, missing sidecar, duplicate/permuted names.
4. Absent force/torque channels remain None, never fabricated as zeros.
5. Ingestion does not fabricate a GenericPhysicsRecorder.
6. Conspicuous inspection of rejected fits and capability discovery.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_session import (
    CandidateSession,
    ingest_candidate_session,
)

pytestmark = pytest.mark.unit


def _create_synthetic_model_spec(tmp_path: Path) -> tuple[Path, str, dict[str, Any]]:
    spec: dict[str, Any] = {
        "model_name": "test_full_body",
        "version": "1.0",
        "coordinate_order": ["pelvis_tx", "pelvis_ty", "pelvis_tz", "pelvis_tilt"],
        "velocity_order": ["pelvis_vx", "pelvis_vy", "pelvis_vz", "pelvis_tilt_rate"],
        "actuator_order": ["pelvis_tilt_actuator"],
        "joints": [],
    }
    raw = json.dumps(spec, sort_keys=True, indent=2).encode("utf-8")
    spec_path = tmp_path / "model_spec.json"
    spec_path.write_bytes(raw)
    spec_sha256 = hashlib.sha256(raw).hexdigest()
    return spec_path, spec_sha256, spec


def _create_synthetic_candidate_npz(
    tmp_path: Path,
    spec_sha256: str,
    *,
    is_dynamic: bool = False,
    non_monotone: bool = False,
    coordinate_order: list[str] | None = None,
) -> tuple[Path, str]:
    coords = coordinate_order or ["pelvis_tx", "pelvis_ty", "pelvis_tz", "pelvis_tilt"]
    n_frames = 10
    time_s = np.linspace(0.0, 0.5, n_frames)
    if non_monotone:
        time_s[3] = time_s[2]  # Duplicate timestamp

    q = np.ones((n_frames, len(coords)), dtype=np.float64) * 0.1
    v = np.ones((n_frames, 4), dtype=np.float64) * 0.2 if is_dynamic else None
    tau = np.ones((n_frames, 1), dtype=np.float64) * 5.0 if is_dynamic else None

    meta = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.DYNAMIC if is_dynamic else CandidateProfile.KINEMATIC,
        engine="pinocchio",
        model_name="test_full_body",
        model_sha256=spec_sha256,
        coordinate_names=tuple(coords),
        velocity_names=("pelvis_vx", "pelvis_vy", "pelvis_vz", "pelvis_tilt_rate")
        if is_dynamic
        else (),
        actuator_names=("pelvis_tilt_actuator",) if is_dynamic else (),
    )

    candidate_path = tmp_path / "candidate.npz"
    save_dict: dict[str, Any] = {
        "time_s": time_s,
        "q": q,
        "coordinate_order": np.array(coords),
        "metadata_json": json.dumps(meta.to_dict()),
    }
    if v is not None:
        save_dict["v"] = v
        save_dict["velocity_order"] = np.array(meta.velocity_names)
    if tau is not None:
        save_dict["tau"] = tau
        save_dict["actuator_order"] = np.array(meta.actuator_names)

    np.savez_compressed(candidate_path, **save_dict)
    candidate_bytes = candidate_path.read_bytes()
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    return candidate_path, candidate_sha256


def _create_receipt(
    tmp_path: Path,
    candidate_sha256: str,
    spec_sha256: str,
    *,
    is_accepted: bool = True,
    reason: str = "All parity criteria satisfied",
) -> Path:
    receipt = {
        "schema_version": "matched-swing-replay/1",
        "engine": "pinocchio",
        "engine_version": "4.1.0",
        "candidate_sha256": candidate_sha256,
        "document_sha256": spec_sha256,
        "spec_file": "model_spec.json",
        "acceptance": {
            "is_accepted": is_accepted,
            "status": "verified" if is_accepted else "rejected",
            "reason": reason,
            "marker_rms_mm": 3.2 if is_accepted else 123.5,
            "tolerance_mm": 5.0,
        },
        "shared_metrics": {
            "whole_marker_rmse_m": 0.0032 if is_accepted else 0.1235,
        },
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt_path


def test_ingest_candidate_session_happy_path(tmp_path: Path) -> None:
    spec_path, spec_sha256, _ = _create_synthetic_model_spec(tmp_path)
    cand_path, cand_sha256 = _create_synthetic_candidate_npz(tmp_path, spec_sha256)
    receipt_path = _create_receipt(tmp_path, cand_sha256, spec_sha256, is_accepted=True)

    session = ingest_candidate_session(
        cand_path, model_path=spec_path, receipt_path=receipt_path
    )

    assert isinstance(session, CandidateSession)
    assert session.candidate_sha256 == cand_sha256
    assert session.model_sha256 == spec_sha256
    assert session.is_accepted is True
    assert session.status == "verified"
    assert session.frame_count == 10
    assert session.time_s.shape == (10,)
    assert session.q.shape == (10, 4)
    # Absent forces remain None, never zeros!
    assert session.tau is None
    assert session.external_forces is None
    assert session.supports_forces is False
    assert session.supports_counterfactuals is False


def test_ingest_candidate_session_dynamic_profile_forces(tmp_path: Path) -> None:
    spec_path, spec_sha256, _ = _create_synthetic_model_spec(tmp_path)
    cand_path, cand_sha256 = _create_synthetic_candidate_npz(
        tmp_path, spec_sha256, is_dynamic=True
    )
    receipt_path = _create_receipt(tmp_path, cand_sha256, spec_sha256, is_accepted=True)

    session = ingest_candidate_session(
        cand_path, model_path=spec_path, receipt_path=receipt_path
    )

    assert session.tau is not None
    assert session.tau.shape == (10, 1)
    assert session.v is not None
    assert session.v.shape == (10, 4)
    assert session.supports_forces is True
    assert session.supports_counterfactuals is True


def test_ingest_candidate_session_rejects_wrong_model_hash(tmp_path: Path) -> None:
    spec_path, spec_sha256, _ = _create_synthetic_model_spec(tmp_path)
    cand_path, cand_sha256 = _create_synthetic_candidate_npz(tmp_path, spec_sha256)
    receipt_path = _create_receipt(tmp_path, cand_sha256, spec_sha256)

    # Tamper with model specification
    spec_path.write_text('{"tampered": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="Model hash mismatch"):
        ingest_candidate_session(
            cand_path, model_path=spec_path, receipt_path=receipt_path
        )


def test_ingest_candidate_session_rejects_wrong_candidate_hash(tmp_path: Path) -> None:
    spec_path, spec_sha256, _ = _create_synthetic_model_spec(tmp_path)
    cand_path, cand_sha256 = _create_synthetic_candidate_npz(tmp_path, spec_sha256)
    receipt_path = _create_receipt(tmp_path, "0" * 64, spec_sha256)  # Bad hash

    with pytest.raises(ValueError, match="Candidate content hash mismatch"):
        ingest_candidate_session(
            cand_path, model_path=spec_path, receipt_path=receipt_path
        )


def test_ingest_candidate_session_rejects_missing_sidecar_strict(
    tmp_path: Path,
) -> None:
    spec_path, spec_sha256, _ = _create_synthetic_model_spec(tmp_path)
    cand_path, _ = _create_synthetic_candidate_npz(tmp_path, spec_sha256)

    with pytest.raises(ValueError, match="Receipt sidecar missing"):
        ingest_candidate_session(
            cand_path, model_path=spec_path, receipt_path=None, strict=True
        )


def test_ingest_candidate_session_rejects_non_monotone_time(tmp_path: Path) -> None:
    spec_path, spec_sha256, _ = _create_synthetic_model_spec(tmp_path)
    cand_path, cand_sha256 = _create_synthetic_candidate_npz(
        tmp_path, spec_sha256, non_monotone=True
    )
    receipt_path = _create_receipt(tmp_path, cand_sha256, spec_sha256)

    with pytest.raises(ValueError, match="monotonically increasing"):
        ingest_candidate_session(
            cand_path, model_path=spec_path, receipt_path=receipt_path
        )


def test_ingest_candidate_session_rejects_duplicate_or_mismatched_coordinates(
    tmp_path: Path,
) -> None:
    spec_path, spec_sha256, _ = _create_synthetic_model_spec(tmp_path)
    cand_path, cand_sha256 = _create_synthetic_candidate_npz(
        tmp_path,
        spec_sha256,
        coordinate_order=["pelvis_tx", "pelvis_ty", "pelvis_tz", "unknown_joint"],
    )
    receipt_path = _create_receipt(tmp_path, cand_sha256, spec_sha256)

    with pytest.raises(ValueError, match="coordinate names"):
        ingest_candidate_session(
            cand_path, model_path=spec_path, receipt_path=receipt_path
        )


def test_ingest_candidate_session_reorders_permuted_coordinates(
    tmp_path: Path,
) -> None:
    spec_path, spec_sha256, _ = _create_synthetic_model_spec(tmp_path)
    # Permuted coordinate order:
    permuted = ["pelvis_tilt", "pelvis_tz", "pelvis_ty", "pelvis_tx"]
    cand_path, cand_sha256 = _create_synthetic_candidate_npz(
        tmp_path, spec_sha256, coordinate_order=permuted
    )
    receipt_path = _create_receipt(tmp_path, cand_sha256, spec_sha256)

    session = ingest_candidate_session(
        cand_path, model_path=spec_path, receipt_path=receipt_path
    )
    # The output session.coordinate_names must match model spec order
    assert session.coordinate_names == (
        "pelvis_tx",
        "pelvis_ty",
        "pelvis_tz",
        "pelvis_tilt",
    )


def test_ingest_candidate_session_rejected_fit_conspicuous(tmp_path: Path) -> None:
    spec_path, spec_sha256, _ = _create_synthetic_model_spec(tmp_path)
    cand_path, cand_sha256 = _create_synthetic_candidate_npz(tmp_path, spec_sha256)
    receipt_path = _create_receipt(
        tmp_path,
        cand_sha256,
        spec_sha256,
        is_accepted=False,
        reason="RMSE 123.5 mm exceeds tolerance 5.0 mm",
    )

    session = ingest_candidate_session(
        cand_path, model_path=spec_path, receipt_path=receipt_path
    )

    assert session.is_accepted is False
    assert session.status == "rejected"
    assert "123.5 mm" in session.rejection_reason
    assert session.supports_counterfactuals is False


def test_ingest_candidate_session_does_not_fabricate_simulation_recorder(
    tmp_path: Path,
) -> None:
    from src.shared.python.dashboard.recorder import GenericPhysicsRecorder

    spec_path, spec_sha256, _ = _create_synthetic_model_spec(tmp_path)
    cand_path, cand_sha256 = _create_synthetic_candidate_npz(tmp_path, spec_sha256)
    receipt_path = _create_receipt(tmp_path, cand_sha256, spec_sha256)

    session = ingest_candidate_session(
        cand_path, model_path=spec_path, receipt_path=receipt_path
    )

    assert not isinstance(session, GenericPhysicsRecorder)
    assert not hasattr(session, "record_step")
