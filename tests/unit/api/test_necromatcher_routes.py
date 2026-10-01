"""Necromatcher HTTP integration uses the actual persistent library."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from src.api.routes.necromatcher import router, get_library
from src.shared.python.workspace.necromatcher import NecromatcherLibrary

pytestmark = pytest.mark.unit


@pytest.fixture
def client(tmp_path):
    library = NecromatcherLibrary.create(tmp_path / "library")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_library] = lambda: library
    with TestClient(app) as connection:
        yield connection, library


def test_player_swing_and_asset_roundtrip(client, tmp_path):
    connection, library = client
    assert (
        connection.post(
            "/necromatcher/players", json={"id": "hogan", "name": "Ben Hogan"}
        ).status_code
        == 201
    )
    assert (
        connection.post(
            "/necromatcher/swings",
            json={"id": "practice", "player_id": "hogan", "name": "Practice"},
        ).status_code
        == 201
    )
    source = tmp_path / "model.xml"
    source.write_text("<mujoco/>", encoding="utf-8")
    response = connection.post(
        "/necromatcher/swings/practice/models",
        json={
            "id": "v1",
            "source_path": str(source),
            "engine": "mujoco",
            "dofs": ["hip"],
        },
    )
    assert response.status_code == 201
    assert "path" not in response.json()
    assert response.json()["metadata"]["qualification"] == "unqualified_candidate"
    assert (
        connection.get("/necromatcher/players").json()["players"][0]["display_name"]
        == "Ben Hogan"
    )
    assert (
        connection.get("/necromatcher/swings?player_id=hogan").json()["swings"][0][
            "session_id"
        ]
        == "practice"
    )
    exported = connection.get("/necromatcher/swings/practice/export")
    assert exported.status_code == 200
    assert exported.content[:2] == b"PK"
    fresh = NecromatcherLibrary(library.root)
    assert fresh.load_asset("v1").kind == "native_model"


def test_duplicate_and_unknown_subject_fail_cleanly(client):
    connection, _ = client
    payload = {"id": "hogan", "name": "Ben Hogan"}
    connection.post("/necromatcher/players", json=payload)
    assert connection.post("/necromatcher/players", json=payload).status_code == 409
    assert (
        connection.post(
            "/necromatcher/swings",
            json={"id": "practice", "player_id": "missing", "name": "Practice"},
        ).status_code
        == 404
    )
    assert connection.get("/necromatcher/swings/missing/assets").status_code == 404


def test_remote_client_cannot_read_or_modify_local_library(client):
    _, library = client
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_library] = lambda: library
    with TestClient(app, client=("203.0.113.8", 1000)) as connection:
        assert connection.get("/necromatcher/players").status_code == 403
        assert (
            connection.post(
                "/necromatcher/players", json={"id": "hogan", "name": "Ben Hogan"}
            ).status_code
            == 403
        )
    assert library.players() == []
