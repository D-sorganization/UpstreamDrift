"""Unit tests for the motion capture API route."""

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import motion_capture as mc
from src.api.routes.motion_capture import router, _sessions, _recordings

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCHER_MANIFEST = _REPO_ROOT / "src" / "config" / "launcher_manifest.json"
_HAS_EZC3D = importlib.util.find_spec("ezc3d") is not None


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


def test_capture_session_flow(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test starting and stopping a capture session."""
    # Source availability depends on the environment; force it for the flow.
    monkeypatch.setattr(mc, "_source_availability", lambda _sid: (True, None))
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


def test_sources_report_honest_availability(client: TestClient) -> None:
    """Each source's availability must match server-side importability (#7454)."""
    response = client.get("/tools/motion-capture/sources")
    assert response.status_code == 200
    sources = {s["id"]: s for s in response.json()}
    assert set(sources) == {"mediapipe", "openpose", "c3d"}

    for source_id, module_name in (
        ("mediapipe", "mediapipe"),
        ("openpose", "openpose"),
        ("c3d", "ezc3d"),
    ):
        expected = importlib.util.find_spec(module_name) is not None
        source = sources[source_id]
        assert source["available"] is expected, source_id
        if expected:
            assert source["reason"] is None
        else:
            assert isinstance(source["reason"], str) and source["reason"]


def test_session_start_rejects_unavailable_source(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Starting a session on an unavailable source must 409, not fall back (#7454)."""
    monkeypatch.setattr(
        mc, "_source_availability", lambda _sid: (False, "not installed")
    )
    response = client.post(
        "/tools/motion-capture/session/start",
        json={"source_type": "openpose", "frame_rate": 30.0},
    )
    assert response.status_code == 409
    assert "unavailable" in response.json()["detail"]


def test_c3d_skeleton_template_is_empty(client: TestClient) -> None:
    """C3D marker sets are file-defined; the template must not be MediaPipe (#7454)."""
    response = client.get("/tools/motion-capture/skeleton/c3d")
    assert response.status_code == 200
    assert response.json() == []


@pytest.fixture
def mm_c3d_file(tmp_path: Path) -> Path:
    """Generate a tiny C3D with mm units and known marker coordinates.

    Note: the repo's golden ``sample.c3d`` omits POINT:UNITS, which the
    canonical ``C3DDataReader`` cannot read (vendored bug), so the fixture
    is generated here with explicit units.
    """
    import ezc3d
    import numpy as np

    c3d = ezc3d.c3d()
    c3d["parameters"]["POINT"]["RATE"]["value"] = [60.0]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = ["HEAD", "BUTT"]
    c3d["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]
    points = np.zeros((4, 2, 12))
    points[0, 0, :] = 1000.0  # HEAD x = 1000 mm = 1.0 m
    points[1, 1, :] = 500.0  # BUTT y = 500 mm = 0.5 m
    points[3, :, :] = 0.0  # residuals (valid)
    c3d["data"]["points"] = points
    path = tmp_path / "swing.c3d"
    c3d.write(str(path))
    return path


@pytest.mark.skipif(not _HAS_EZC3D, reason="ezc3d not installed")
def test_c3d_upload_returns_metadata_and_playable_recording(
    client: TestClient, mm_c3d_file: Path
) -> None:
    """Uploading a C3D registers a recording with marker metadata (#7454)."""
    with mm_c3d_file.open("rb") as fh:
        response = client.post(
            "/tools/motion-capture/upload-c3d",
            files={"file": ("swing.c3d", fh, "application/octet-stream")},
        )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["marker_names"] == ["HEAD", "BUTT"]
    assert data["frame_rate"] == pytest.approx(60.0)
    assert data["total_frames"] == 12
    assert data["duration_seconds"] == pytest.approx(0.2)
    assert data["native_units"] == "mm"
    assert data["converted_units"] == "m"

    # Recording is listed with the file's marker names as joints
    recs = client.get("/tools/motion-capture/recordings").json()
    rec = next(r for r in recs if r["name"] == data["recording_name"])
    assert rec["source_type"] == "c3d"
    assert rec["joint_names"] == data["marker_names"]
    assert rec["total_frames"] == 12

    # Frames are playable through the existing frame endpoint, with mm->m
    # unit scaling applied (mirrors the desktop viewer; #7200 lineage).
    frame = client.get(f"/tools/motion-capture/frame/{data['recording_name']}/0").json()
    assert frame["frame_index"] == 0
    joints = {j["name"]: j for j in frame["joints"]}
    assert set(joints) == {"HEAD", "BUTT"}
    assert joints["HEAD"]["position"][0] == pytest.approx(1.0)
    assert joints["BUTT"]["position"][1] == pytest.approx(0.5)
    assert joints["HEAD"]["confidence"] == 1.0

    # Playback control works on the uploaded recording
    playback = client.post(
        "/tools/motion-capture/playback",
        json={"recording_name": data["recording_name"], "action": "play"},
    ).json()
    assert playback["status"] == "playing"
    assert playback["total_frames"] == 12


def test_c3d_upload_rejects_non_c3d_extension(client: TestClient) -> None:
    """Non-.c3d uploads are rejected before parsing (#7454)."""
    response = client.post(
        "/tools/motion-capture/upload-c3d",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_source_list_matches_desktop_launcher_capabilities(
    client: TestClient,
) -> None:
    """Parity: web source ids mirror the desktop Motion Capture tile (#7454).

    The desktop launcher manifest declares the three sub-tools as
    capabilities (c3d_viewer, openpose, mediapipe); the API source list is
    the shared enumeration both frontends consume.
    """
    manifest = json.loads(_LAUNCHER_MANIFEST.read_text(encoding="utf-8"))
    tile = next(m for m in manifest["tiles"] if m["id"] == "motion_capture")
    capability_to_source = {"c3d_viewer": "c3d"}
    desktop_tools = {
        capability_to_source.get(cap, cap)
        for cap in tile["capabilities"]
        if cap in ("c3d_viewer", "openpose", "mediapipe")
    }

    api_sources = {s["id"] for s in client.get("/tools/motion-capture/sources").json()}
    assert api_sources == desktop_tools


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
