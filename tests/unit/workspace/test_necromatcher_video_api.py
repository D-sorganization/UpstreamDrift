"""Video job transport preserves ownership and guarded artifact download."""

from pathlib import Path
from typing import Any
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit


class VideoSession:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.ready = False
        self.busy = False
        self.cancelled = False
        self.force_layer: dict[str, Any] | None = None

    def submit(
        self, fit_id: str, force_layer: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.force_layer = force_layer
        if self.busy:
            raise RuntimeError("A video export is already running")
        if fit_id != "fit":
            raise KeyError(fit_id)
        self.busy = True
        return self.view("a" * 32)

    def view(self, run_id: str) -> dict[str, Any]:
        if run_id != "a" * 32:
            raise KeyError(run_id)
        return {
            "run_id": run_id,
            "source_fit_id": "fit",
            "status": "succeeded" if self.ready else "running",
            "acceptance": "rejected",
            "qualification": "monocular_research_hypothesis",
            "download_available": self.ready,
        }

    def cancel(self, run_id: str) -> dict[str, Any]:
        self.cancelled = True
        return self.view(run_id)

    def download(self, run_id: str) -> Path:
        self.view(run_id)
        if not self.ready:
            raise RuntimeError("Verified video output is not available")
        return self.path


@pytest.fixture
def video_api(tmp_path: Path):
    from src.api.routes import necromatcher as routes

    archive = tmp_path / "review.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("manifest.json", "{}")
    session = VideoSession(archive)
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    app.dependency_overrides[routes.get_video_exports] = lambda: session
    with TestClient(app) as client:
        yield client, session


def test_video_api_submits_polls_and_cancels_without_host_paths(video_api):
    client, session = video_api
    response = client.post("/api/necromatcher/fits/fit/video-exports")
    assert response.status_code == 202
    assert response.json()["source_fit_id"] == "fit"
    assert response.json()["acceptance"] == "rejected"
    run = response.json()["run_id"]
    assert client.get(f"/api/necromatcher/video-exports/{run}").status_code == 200
    assert (
        client.post(f"/api/necromatcher/video-exports/{run}/cancel").status_code == 200
    )
    assert session.cancelled
    assert str(session.path) not in response.text


def test_video_download_requires_verified_success(video_api):
    client, session = video_api
    url = "/api/necromatcher/video-exports/" + "a" * 32 + "/download"
    assert client.get(url).status_code == 409
    session.ready = True
    response = client.get(url)
    assert response.status_code == 200
    assert response.content == session.path.read_bytes()
    assert response.headers["content-type"] == "application/zip"
    assert "attachment" in response.headers["content-disposition"]


def test_video_api_rejects_unknown_identity_and_duplicate_admission(video_api):
    client, _ = video_api
    assert (
        client.post("/api/necromatcher/fits/missing/video-exports").status_code == 404
    )
    assert client.get("/api/necromatcher/video-exports/missing").status_code == 404
    assert client.post("/api/necromatcher/fits/fit/video-exports").status_code == 202
    assert client.post("/api/necromatcher/fits/fit/video-exports").status_code == 409


def test_video_routes_remain_local_only(video_api):
    from src.api.routes import necromatcher as routes

    _, session = video_api
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    app.dependency_overrides[routes.get_video_exports] = lambda: session
    with TestClient(app, client=("203.0.113.9", 40000)) as remote:
        assert (
            remote.post("/api/necromatcher/fits/fit/video-exports").status_code == 403
        )


def test_default_request_has_no_force_layer(video_api):
    client, session = video_api
    assert client.post("/api/necromatcher/fits/fit/video-exports").status_code == 202
    assert session.force_layer is None


def test_opt_in_force_layer_is_validated_and_forwarded(video_api):
    client, session = video_api
    url = "/api/necromatcher/fits/fit/video-exports"
    body = {
        "force_layer": {
            "enabled": True,
            "kinds": ["joint_reaction", "contact"],
            "scale": 2.0,
            "segment_shading": True,
        }
    }
    assert client.post(url, json=body).status_code == 202
    assert session.force_layer == body["force_layer"]


@pytest.mark.parametrize(
    "layer",
    [
        {"enabled": True, "kinds": ["nonsense"]},
        {"enabled": True, "scale": 0},
        {"enabled": True, "extra": 1},
    ],
)
def test_invalid_force_layer_is_rejected(video_api, layer):
    client, session = video_api
    response = client.post(
        "/api/necromatcher/fits/fit/video-exports", json={"force_layer": layer}
    )
    assert response.status_code == 422
    assert session.force_layer is None
