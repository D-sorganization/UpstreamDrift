"""Unit tests for the OpenCap API route (#11409)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.opencap import router
from tests.unit.motion_pipeline.sources.opencap_fixtures import (
    write_kinematics,
    write_opencap_session,
    write_scaled_model,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _create_test_session(
    tmp_path: Path, trials: tuple[str, ...] = ("neutral", "swing1")
) -> Path:
    session = write_opencap_session(tmp_path, trials)
    write_scaled_model(session)
    for trial in trials:
        write_kinematics(session, trial)
    return session


def test_inspect_session_success(client: TestClient, tmp_path: Path) -> None:
    session_dir = _create_test_session(tmp_path, ("neutral", "swing1", "swing2"))

    response = client.post(
        "/tools/opencap/inspect", json={"session_dir": str(session_dir)}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["session_dir"] == str(session_dir)
    assert data["trials"] == ["neutral", "swing1", "swing2"]
    assert data["subject"]["mass_kg"] == pytest.approx(79.5)
    assert data["model_file"] is not None
    assert "swing1" in data["kinematics_trials"]


def test_inspect_session_not_found(client: TestClient) -> None:
    response = client.post(
        "/tools/opencap/inspect",
        json={"session_dir": "/nonexistent/path/for/session"},
    )
    assert response.status_code == 404


def test_import_trial_success(client: TestClient, tmp_path: Path) -> None:
    session_dir = _create_test_session(tmp_path, ("neutral", "swing1"))

    response = client.post(
        "/tools/opencap/import",
        json={"session_dir": str(session_dir), "trial": "swing1"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["trial"] == "swing1"
    assert data["has_kinematics"] is True
    assert len(data["kinematics_columns"]) > 0
    assert data["model_file"] is not None


def test_import_trial_not_found(client: TestClient) -> None:
    response = client.post(
        "/tools/opencap/import",
        json={"session_dir": "/nonexistent/path/for/session", "trial": "swing1"},
    )
    assert response.status_code == 404
