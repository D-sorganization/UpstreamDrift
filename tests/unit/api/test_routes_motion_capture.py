"""Unit tests for the motion capture API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.motion_capture import router, _sessions, _recordings


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with the motion capture router."""
    test_app = FastAPI()
    test_app.include_router(router)
    # Clear state before each test
    _sessions.clear()
    _recordings.clear()
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_list_capture_sources(client: TestClient) -> None:
    """Test listing motion capture sources."""
    response = client.get("/tools/motion-capture/sources")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert any(s["id"] == "mediapipe" for s in data)


def test_get_skeleton_template(client: TestClient) -> None:
    """Test getting skeleton templates."""
    response = client.get("/tools/motion-capture/skeleton/mediapipe")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert data[0]["name"] == "nose"


def test_capture_session_flow(client: TestClient) -> None:
    """Test starting and stopping a capture session."""
    # Start session
    start_resp = client.post(
        "/tools/motion-capture/session/start",
        json={"source_type": "mediapipe", "frame_rate": 30.0},
    )
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]

    # Stop session
    stop_resp = client.post(f"/tools/motion-capture/session/{session_id}/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "stopped"

    # Check recordings list
    rec_resp = client.get("/tools/motion-capture/recordings")
    assert rec_resp.status_code == 200
    recs = rec_resp.json()
    assert len(recs) == 1
    rec_name = recs[0]["name"]

    # Get frame
    frame_resp = client.get(f"/tools/motion-capture/frame/{rec_name}/0")
    assert frame_resp.status_code == 200
    assert frame_resp.json()["frame_index"] == 0


def test_playback_control(client: TestClient) -> None:
    """Test playback control endpoints."""
    # Setup dummy recording
    _recordings["test_rec"] = {
        "source_type": "mediapipe",
        "frame_rate": 30.0,
        "frames": [],
    }

    response = client.post(
        "/tools/motion-capture/playback",
        json={"recording_name": "test_rec", "action": "play"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "playing"
