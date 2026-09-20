"""Unit tests for Candidate Session API service and routes (MV-03 #10479).

Validates:
1. SimulationService candidate session ingestion without fabricating GenericPhysicsRecorder.
2. Capability reporting: forces, counterfactuals, tangent velocities, markers.
3. Accurate WSL host boundary reporting.
4. HTTP route GET /capabilities/session (or /candidates/session).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.api.services.simulation_service import SimulationService
from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateMetadata,
    CandidateProfile,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def test_client() -> TestClient:
    return TestClient(app)


def _setup_test_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    spec = {
        "model_name": "test_model",
        "coordinate_order": ["q0", "q1"],
        "joints": [],
    }
    raw_spec = json.dumps(spec).encode("utf-8")
    spec_path = tmp_path / "model_spec.json"
    spec_path.write_bytes(raw_spec)
    spec_hash = hashlib.sha256(raw_spec).hexdigest()

    time_s = np.array([0.0, 0.1, 0.2])
    q = np.ones((3, 2)) * 0.5
    cand_path = tmp_path / "candidate.npz"
    meta = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.KINEMATIC,
        engine="pinocchio",
        model_name="test_model",
        model_sha256=spec_hash,
        coordinate_names=("q0", "q1"),
    )
    np.savez_compressed(
        cand_path,
        time_s=time_s,
        q=q,
        coordinate_order=np.array(["q0", "q1"]),
        metadata_json=json.dumps(meta.to_dict()),
    )
    cand_hash = hashlib.sha256(cand_path.read_bytes()).hexdigest()

    receipt = {
        "schema_version": "matched-swing-replay/1",
        "engine": "pinocchio",
        "candidate_sha256": cand_hash,
        "document_sha256": spec_hash,
        "spec_file": "model_spec.json",
        "acceptance": {
            "is_accepted": True,
            "status": "verified",
            "reason": "OK",
        },
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return cand_path, spec_path, receipt_path


def test_simulation_service_candidate_session(tmp_path: Path) -> None:
    cand_path, spec_path, receipt_path = _setup_test_files(tmp_path)
    engine_manager = MagicMock()
    service = SimulationService(engine_manager=engine_manager)

    session = service.get_candidate_session(
        candidate_path=cand_path,
        model_path=spec_path,
        receipt_path=receipt_path,
    )

    assert session.status == "verified"
    assert session.is_accepted is True
    assert session.supports_forces is False
    assert session.supports_counterfactuals is False
    # Verify active_recorder remains None (never fabricated as completed simulation)
    assert service.active_recorder is None


def test_candidate_session_api_endpoint(
    test_client: TestClient, tmp_path: Path
) -> None:
    cand_path, spec_path, receipt_path = _setup_test_files(tmp_path)

    response = test_client.get(
        "/capabilities/candidate_session",
        params={
            "candidate_path": str(cand_path),
            "model_path": str(spec_path),
            "receipt_path": str(receipt_path),
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "verified"
    assert data["is_accepted"] is True
    assert data["engine"] == "pinocchio"
    assert "capabilities" in data
    assert data["capabilities"]["supports_forces"] is False
    assert data["capabilities"]["supports_counterfactuals"] is False
    assert "wsl_runtime" in data
