"""Local-only impact job API preserves typed assumptions and replay ownership."""

from pathlib import Path
from zipfile import ZipFile
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.unit.workspace.test_necromatcher_impact_contracts import geometry, selection

pytestmark = pytest.mark.unit


class Session:
    def __init__(self, path):
        self.path = path
        self.called = 0
        self.ready = False
        self.cancelled = False

    def submit(self, replay, g, s, budget):
        assert g == geometry() and s == selection() and budget == 30.0
        self.called += 1
        return self.view(replay, "a" * 32)

    def view(self, replay, run):
        if replay != "replay":
            raise ValueError("Foreign replay")
        if run != "a" * 32:
            raise KeyError(run)
        return {
            "run_id": run,
            "replay_id": replay,
            "status": "succeeded" if self.ready else "running",
            "acceptance": "rejected",
            "scientific_qualified": False,
            "physical_source_time_qualified": False,
            "download_available": self.ready,
        }

    def cancel(self, replay, run):
        self.view(replay, run)
        self.cancelled = True
        return self.view(replay, run)

    def download(self, replay, run):
        self.view(replay, run)
        if not self.ready:
            raise RuntimeError("Incomplete")
        return self.path


def body():
    return {
        "geometry": geometry().to_record(),
        "selection": selection().to_record(),
        "budget_wall_s": 30.0,
    }


@pytest.fixture
def api(tmp_path):
    from src.api.routes import necromatcher as routes

    p = tmp_path / "impact.zip"
    with ZipFile(p, "w") as archive:
        archive.writestr("impact-receipt.json", "{}")
    session = Session(p)
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    app.dependency_overrides[routes.get_impact_runs] = lambda: session
    with TestClient(app) as client:
        yield client, session


def test_submit_poll_cancel_preserves_local_assumptions(api):
    client, session = api
    url = "/api/necromatcher/replays/replay/impact-runs"
    result = client.post(url, json=body())
    assert result.status_code == 202
    assert result.json()["scientific_qualified"] is False
    assert client.get(url + "/" + "a" * 32).status_code == 200
    assert (
        client.post(url + "/" + "a" * 32 + "/cancel").status_code == 200
        and session.cancelled
    )
    assert str(session.path) not in result.text


@pytest.mark.parametrize(
    "change",
    [
        {"output_root": "C:/bad"},
        {"budget_wall_s": True},
        {"budget_wall_s": 601},
        {"geometry": {}},
        {"selection": {}},
        {"replay_id": "foreign"},
    ],
)
def test_invalid_record_rejected_before_session(api, change):
    client, session = api
    assert (
        client.post(
            "/api/necromatcher/replays/replay/impact-runs", json={**body(), **change}
        ).status_code
        == 422
    )
    assert session.called == 0


def test_download_incomplete_foreign_and_complete(api):
    client, session = api
    url = "/api/necromatcher/replays/replay/impact-runs/" + "a" * 32 + "/download"
    assert client.get(url).status_code == 409
    assert client.get(url.replace("/replay/", "/foreign/")).status_code == 422
    session.ready = True
    r = client.get(url)
    assert r.status_code == 200 and r.content == session.path.read_bytes()
    assert r.headers["content-type"] == "application/zip"


def test_remote_client_cannot_submit(tmp_path):
    from src.api.routes import necromatcher as routes

    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app, client=("203.0.113.9", 40000)) as client:
        assert (
            client.post(
                "/necromatcher/replays/replay/impact-runs", json=body()
            ).status_code
            == 403
        )
