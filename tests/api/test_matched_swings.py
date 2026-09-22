"""API tests for matched-swing results routes (MS-85, #10358)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.routes.matched_swings import get_matched_swings_service
from src.api.server import app
from src.api.services.matched_swings_service import MatchedSwingsService
from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateMetadata,
    CandidateProfile,
)
from src.shared.python.motion_matching.ledger_schema import (
    ArtefactPaths,
    Ledger,
    LedgerRow,
    SharedMetrics,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def ledger_fixture(tmp_path: Path) -> tuple[Path, str]:
    """Build a minimal ledger with receipt, npz, gif, and parity artefacts."""
    evidence = tmp_path / "evidence" / "matched" / "driver_test"
    evidence.mkdir(parents=True)

    marker_names = ("pelvis", "thorax", "head")
    time_s = np.array([0.0, 0.1])
    markers = np.array(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.3, 1.2], [0.0, 0.5, 1.5]],
            [[0.01, 0.0, 1.0], [0.01, 0.3, 1.2], [0.01, 0.5, 1.5]],
        ],
        dtype=np.float64,
    )
    meta = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.KINEMATIC,
        engine="pinocchio",
        model_name="test_model",
        marker_names=marker_names,
    )
    npz_path = evidence / "candidate.npz"
    np.savez_compressed(
        npz_path,
        manifest_json=np.array(json.dumps(meta.to_dict())),
        time_s=time_s,
        q=np.zeros((2, 1)),
        model_markers_m=markers,
    )

    receipt = {
        "schema_version": "matched-swing-fit/test-v1",
        "engine": "pinocchio",
        "candidate_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
    }
    receipt_path = evidence / "receipt.json"
    receipt_bytes = json.dumps(receipt).encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()

    gif_path = evidence / "playback.gif"
    gif_path.write_bytes(
        b"GIF89a\x01\x00\x01\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    )

    parity = {
        "schema_version": "matched-swing-parity-report-v1",
        "candidate_sha256": receipt["candidate_sha256"],
        "status": "PARTIAL",
    }
    (evidence / "parity_vs_mujoco.json").write_text(
        json.dumps(parity), encoding="utf-8"
    )

    rel_receipt = receipt_path.relative_to(tmp_path).as_posix()
    row = LedgerRow(
        receipt_path=rel_receipt,
        sha256=receipt_sha,
        engine="pinocchio",
        lane="matched",
        capture="driver",
        candidate_sha=receipt["candidate_sha256"],
        horizon_s=0.85,
        metrics=SharedMetrics(whole_marker_rmse_m=0.022),
        acceptance={"status": "PASSED", "gates": [{"name": "G1", "passed": True}]},
        artefacts=ArtefactPaths(
            npz=npz_path.relative_to(tmp_path).as_posix(),
            gif=gif_path.relative_to(tmp_path).as_posix(),
        ),
    )
    ledger = Ledger(
        schema_version="1.0.0",
        generated_at="2026-09-21T00:00:00Z",
        total_receipts=1,
        rows=[row],
    )
    ledger_path = tmp_path / "reports" / "matched_swing_ledger.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(ledger.to_json(), encoding="utf-8")
    return ledger_path, receipt_sha


@pytest.fixture
def client(ledger_fixture: tuple[Path, str]) -> TestClient:
    ledger_path, _ = ledger_fixture
    service = MatchedSwingsService.from_ledger_file(
        ledger_path, repo_root=ledger_path.parent.parent
    )
    app.dependency_overrides[get_matched_swings_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_matched_swings_service, None)


def test_list_matched_swings(
    client: TestClient, ledger_fixture: tuple[Path, str]
) -> None:
    _, run_id = ledger_fixture
    response = client.get("/api/matched-swings")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["runs"][0]["id"] == run_id
    assert data["runs"][0]["verdict"] == "PASSED"
    assert data["runs"][0]["capabilities"]["has_candidate_npz"] is True
    assert "receipt_path" not in json.dumps(data)


def test_get_receipt(client: TestClient, ledger_fixture: tuple[Path, str]) -> None:
    _, run_id = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == run_id
    assert body["receipt"]["engine"] == "pinocchio"
    assert body["candidate_sha256"]


def test_get_candidate_npz(
    client: TestClient, ledger_fixture: tuple[Path, str]
) -> None:
    _, run_id = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id}/candidate")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")


def test_get_candidate_preview(
    client: TestClient, ledger_fixture: tuple[Path, str]
) -> None:
    _, run_id = ledger_fixture
    response = client.get(
        f"/api/matched-swings/{run_id}/candidate",
        params={"preview_frame": 0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["frame_count"] == 2
    assert len(body["joints"]) == 3
    assert body["joints"][0]["name"] == "pelvis"


def test_get_parity(client: TestClient, ledger_fixture: tuple[Path, str]) -> None:
    _, run_id = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id}/parity")
    assert response.status_code == 200
    assert response.json()["schema_version"] == "matched-swing-parity-report-v1"


def test_get_animation_gif(
    client: TestClient, ledger_fixture: tuple[Path, str]
) -> None:
    _, run_id = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id}/animation.gif")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/gif"


def test_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get("/api/matched-swings/not-a-real-id")
    assert response.status_code == 404


def test_remote_client_blocked(ledger_fixture: tuple[Path, str]) -> None:
    ledger_path, _ = ledger_fixture
    service = MatchedSwingsService.from_ledger_file(
        ledger_path, repo_root=ledger_path.parent.parent
    )
    app.dependency_overrides[get_matched_swings_service] = lambda: service
    try:
        remote_client = TestClient(app, client=("203.0.113.1", 1234))
        response = remote_client.get("/api/matched-swings")
        assert response.status_code == 403
    finally:
        app.dependency_overrides.pop(get_matched_swings_service, None)
