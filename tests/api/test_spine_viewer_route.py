"""API tests for the web Spine Viewer routes (SPN-15c, #12815)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from tests.tools.spine_viewer.conftest import SOURCE, write_study

pytestmark = pytest.mark.unit

ROOT_ENV = "UPSTREAMDRIFT_SPINE_STUDY_ROOT"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GOLF_AUTH_DISABLED", "true")
    studies = tmp_path / "studies"
    studies.mkdir()
    monkeypatch.setenv(ROOT_ENV, str(studies))
    return studies


@pytest.fixture
def client(root: Path) -> TestClient:
    return TestClient(app)


def test_study_returns_verified_view_model(client: TestClient, root: Path) -> None:
    write_study(root / "a")
    resp = client.get(
        "/api/tools/spine-viewer/study", params={"receipt": "a/s1_receipt.json"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["scientific_status"] == "unqualified"
    assert body["distribution_source"] == SOURCE
    assert "UNQUALIFIED" in body["banner"]
    assert body["heatmaps"]["compression"]["values"][2] == [300.0, 20.0]


def test_qualified_study_is_reported_qualified(client: TestClient, root: Path) -> None:
    write_study(root / "q", status="pass", scientific_status="qualified")
    resp = client.get(
        "/api/tools/spine-viewer/study", params={"receipt": "q/s1_receipt.json"}
    )
    assert resp.json()["scientific_status"] == "qualified"


def test_tampered_artifact_is_rejected(client: TestClient, root: Path) -> None:
    write_study(root / "t")
    npz = root / "t" / "s1_loads.npz"
    npz.write_bytes(npz.read_bytes() + b"x")
    assert hashlib.sha256(npz.read_bytes()).hexdigest()
    resp = client.get(
        "/api/tools/spine-viewer/study", params={"receipt": "t/s1_receipt.json"}
    )
    assert resp.status_code == 422
    assert "sha256" in resp.json()["detail"]


@pytest.mark.parametrize(
    "bad", ["../outside.json", "/etc/passwd", "a/../../x.json", "", "a\\..\\b.json"]
)
def test_path_traversal_is_rejected(client: TestClient, root: Path, bad: str) -> None:
    resp = client.get("/api/tools/spine-viewer/study", params={"receipt": bad})
    assert resp.status_code in (400, 404, 422)


def test_symlink_escape_is_rejected(
    client: TestClient, root: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    write_study(outside)
    (root / "link").symlink_to(outside, target_is_directory=True)
    resp = client.get(
        "/api/tools/spine-viewer/study", params={"receipt": "link/s1_receipt.json"}
    )
    assert resp.status_code == 400


def test_missing_receipt_is_404(client: TestClient, root: Path) -> None:
    resp = client.get(
        "/api/tools/spine-viewer/study", params={"receipt": "nope/s1_receipt.json"}
    )
    assert resp.status_code == 404


def test_receipts_listing_is_relative_and_sorted(
    client: TestClient, root: Path
) -> None:
    write_study(root / "b")
    write_study(root / "a")
    resp = client.get("/api/tools/spine-viewer/receipts")
    assert resp.status_code == 200
    assert resp.json()["receipts"] == ["a/s1_receipt.json", "b/s1_receipt.json"]
