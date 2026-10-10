"""API tests for the matched-swing native-viewer handoff routes (#11987)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.routes import matched_swings_native_viewer as route_module
from src.api.routes.matched_swings import get_matched_swings_service
from src.api.server import app
from src.tools.matched_swing_browser import native_handoff
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
from src.shared.python.motion_matching.native_viewers import (
    ViewerLaunchResult,
    ViewerUnavailableError,
)
from src.shared.python.motion_matching.visualization.simulation_viewer import (
    SimulationData,
)
from src.api.services.matched_swings_service import MatchedSwingsService

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _local_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as the local desktop/Tauri API does: bearer auth disabled.

    Unlike ``/matched-swings`` itself, this router launches processes, so it is
    not in ``_PUBLIC_ROUTERS`` and keeps the global auth dependency.
    """
    monkeypatch.setenv("GOLF_AUTH_DISABLED", "true")


def _write_candidate_npz(evidence: Path) -> Path:
    time_s = np.array([0.0, 0.1])
    npz_path = evidence / "candidate.npz"
    meta = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.KINEMATIC,
        engine="pinocchio",
        model_name="test_model",
        marker_names=("pelvis",),
    )
    np.savez_compressed(
        npz_path,
        manifest_json=np.array(json.dumps(meta.to_dict())),
        time_s=time_s,
        q=np.zeros((2, 1)),
    )
    return npz_path


@pytest.fixture
def ledger_fixture(tmp_path: Path) -> tuple[Path, str, str]:
    """Build a ledger with one run that has an NPZ and one that does not."""
    evidence = tmp_path / "evidence" / "matched" / "driver_test"
    evidence.mkdir(parents=True)
    npz_path = _write_candidate_npz(evidence)

    receipt_with_npz = {"schema_version": "matched-swing-fit/test-v1"}
    receipt_path = evidence / "receipt.json"
    receipt_path.write_bytes(json.dumps(receipt_with_npz).encode("utf-8"))

    receipt_no_npz_path = evidence / "receipt_no_npz.json"
    receipt_no_npz_path.write_bytes(
        json.dumps({"schema_version": "v1"}).encode("utf-8")
    )

    row_with_npz = LedgerRow(
        receipt_path=receipt_path.relative_to(tmp_path).as_posix(),
        sha256="a" * 64,
        engine="pinocchio",
        lane="matched",
        capture="driver",
        metrics=SharedMetrics(whole_marker_rmse_m=0.02),
        artefacts=ArtefactPaths(npz=npz_path.relative_to(tmp_path).as_posix()),
    )
    row_without_npz = LedgerRow(
        receipt_path=receipt_no_npz_path.relative_to(tmp_path).as_posix(),
        sha256="b" * 64,
        engine="mujoco",
        lane="matched",
        capture="iron",
        metrics=SharedMetrics(whole_marker_rmse_m=0.03),
        artefacts=ArtefactPaths(),
    )
    ledger = Ledger(
        schema_version="1.0.0",
        generated_at="2026-09-21T00:00:00Z",
        total_receipts=2,
        rows=[row_with_npz, row_without_npz],
    )
    ledger_path = tmp_path / "reports" / "matched_swing_ledger.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(ledger.to_json(), encoding="utf-8")
    return ledger_path, row_with_npz.sha256, row_without_npz.sha256


@pytest.fixture
def client(ledger_fixture: tuple[Path, str, str]) -> TestClient:
    ledger_path, _, _ = ledger_fixture
    service = MatchedSwingsService.from_ledger_file(
        ledger_path, repo_root=ledger_path.parent.parent
    )
    app.dependency_overrides[get_matched_swings_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_matched_swings_service, None)


def test_list_native_viewers_has_trajectory_true(
    client: TestClient, ledger_fixture: tuple[Path, str, str]
) -> None:
    _, run_id, _ = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id}/native-viewers")
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "matched-swing-native-viewer/1"
    assert body["run_id"] == run_id
    assert body["has_trajectory"] is True
    assert [b["backend"] for b in body["backends"]] == [
        "mujoco",
        "meshcat",
        "gepetto",
        "opensim",
        "matlab",
    ]
    for entry in body["backends"]:
        assert set(entry) == {"backend", "available", "install_hint"}


def test_list_native_viewers_has_trajectory_false(
    client: TestClient, ledger_fixture: tuple[Path, str, str]
) -> None:
    _, _, run_id_without_npz = ledger_fixture
    response = client.get(f"/api/matched-swings/{run_id_without_npz}/native-viewers")
    assert response.status_code == 200
    assert response.json()["has_trajectory"] is False


def test_list_native_viewers_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get("/api/matched-swings/not-a-real-id/native-viewers")
    assert response.status_code == 404


def test_launch_native_viewer_success(
    client: TestClient,
    ledger_fixture: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, run_id, _ = ledger_fixture

    def fake_launch(npz_path: Path, backend: str, *, speed: float = 1.0) -> Any:
        return ViewerLaunchResult(
            success=True,
            backend="meshcat",
            url="http://127.0.0.1:7000/static/",
        )

    monkeypatch.setattr(route_module, "launch_native_viewer", fake_launch)

    response = client.post(
        f"/api/matched-swings/{run_id}/native-viewer",
        json={"backend": "meshcat", "speed": 1.0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "matched-swing-native-viewer/1"
    assert body["run_id"] == run_id
    assert body["success"] is True
    assert body["backend"] == "meshcat"
    assert body["url"] == "http://127.0.0.1:7000/static/"


def test_launch_native_viewer_unavailable_returns_503_with_install_hint(
    client: TestClient,
    ledger_fixture: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, run_id, _ = ledger_fixture

    def fake_launch(npz_path: Path, backend: str, *, speed: float = 1.0) -> Any:
        raise ViewerUnavailableError(
            "MuJoCo is not installed. Install with: pip install mujoco"
        )

    monkeypatch.setattr(route_module, "launch_native_viewer", fake_launch)

    response = client.post(
        f"/api/matched-swings/{run_id}/native-viewer",
        json={"backend": "mujoco"},
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error"]["code"] == "viewer_unavailable"
    assert "pip install mujoco" in detail["message"]


def test_launch_native_viewer_backend_failure_returns_502_without_path(
    client: TestClient,
    ledger_fixture: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, run_id, _ = ledger_fixture

    def fake_launch(npz_path: Path, backend: str, *, speed: float = 1.0) -> Any:
        raise RuntimeError(f"viewer crashed while reading {npz_path}")

    monkeypatch.setattr(route_module, "launch_native_viewer", fake_launch)

    response = client.post(
        f"/api/matched-swings/{run_id}/native-viewer",
        json={"backend": "meshcat"},
    )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["error"]["code"] == "viewer_launch_failed"
    assert "candidate.npz" not in response.text


def test_launch_native_viewer_unsupported_backend_returns_422(
    client: TestClient, ledger_fixture: tuple[Path, str, str]
) -> None:
    _, run_id, _ = ledger_fixture
    response = client.post(
        f"/api/matched-swings/{run_id}/native-viewer",
        json={"backend": "not-a-real-backend"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "unsupported_backend"


def test_launch_native_viewer_missing_trajectory_returns_404(
    client: TestClient, ledger_fixture: tuple[Path, str, str]
) -> None:
    _, _, run_id_without_npz = ledger_fixture
    response = client.post(
        f"/api/matched-swings/{run_id_without_npz}/native-viewer",
        json={"backend": "mujoco"},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "trajectory_unavailable"


def test_launch_native_viewer_malformed_npz_returns_422(
    client: TestClient,
    ledger_fixture: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, run_id, _ = ledger_fixture

    def fake_load_simulation_data(npz_path: Path) -> SimulationData:
        raise ValueError(f"NPZ file '{npz_path.name}' is missing 'time_s'")

    monkeypatch.setattr(
        native_handoff, "load_simulation_data", fake_load_simulation_data
    )

    response = client.post(
        f"/api/matched-swings/{run_id}/native-viewer",
        json={"backend": "mujoco"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "trajectory_invalid"


def test_remote_client_blocked(ledger_fixture: tuple[Path, str, str]) -> None:
    ledger_path, run_id, _ = ledger_fixture
    service = MatchedSwingsService.from_ledger_file(
        ledger_path, repo_root=ledger_path.parent.parent
    )
    app.dependency_overrides[get_matched_swings_service] = lambda: service
    try:
        remote_client = TestClient(app, client=("203.0.113.1", 1234))
        assert (
            remote_client.get(
                f"/api/matched-swings/{run_id}/native-viewers"
            ).status_code
            == 403
        )
        assert (
            remote_client.post(
                f"/api/matched-swings/{run_id}/native-viewer",
                json={"backend": "mujoco"},
            ).status_code
            == 403
        )
    finally:
        app.dependency_overrides.pop(get_matched_swings_service, None)


def test_no_response_leaks_tmp_path(
    client: TestClient, ledger_fixture: tuple[Path, str, str], tmp_path: Path
) -> None:
    _, run_id, run_id_without_npz = ledger_fixture
    responses = [
        client.get(f"/api/matched-swings/{run_id}/native-viewers"),
        client.get(f"/api/matched-swings/{run_id_without_npz}/native-viewers"),
        client.post(
            f"/api/matched-swings/{run_id_without_npz}/native-viewer",
            json={"backend": "mujoco"},
        ),
        client.post(
            f"/api/matched-swings/{run_id}/native-viewer",
            json={"backend": "not-a-real-backend"},
        ),
    ]
    for response in responses:
        assert str(tmp_path) not in response.text
        assert tmp_path.as_posix() not in response.text


def test_launch_requires_bearer_auth_outside_local_mode(
    client: TestClient,
    ledger_fixture: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, run_id, _ = ledger_fixture
    monkeypatch.setenv("GOLF_AUTH_DISABLED", "false")
    monkeypatch.delenv("GOLF_SUITE_MODE", raising=False)
    response = client.post(
        f"/api/matched-swings/{run_id}/native-viewer",
        json={"backend": "meshcat"},
    )
    assert response.status_code == 401
