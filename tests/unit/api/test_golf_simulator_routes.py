"""Unit tests for the Golf Simulator API routes.

Follows TDD, DbC, Law of Demeter, and DRY.
Acceptance criteria from Issue #10196 (GS-07).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.golf_simulator import router, reset_simulator_state

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    reset_simulator_state()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _sample_shot_payload(
    shot_id: str = "shot-test-01",
    session_id: str = "sess-test-01",
    source_kind: str = "manual",
) -> dict:
    return {
        "schema_version": 1,
        "shot_id": shot_id,
        "session_id": session_id,
        "source_kind": source_kind,
        "ball_velocity_m_s": [60.0, 0.0, 15.0],
        "ball_angular_velocity_rad_s": [0.0, -250.0, 0.0],
        "aim_context": {
            "source_to_target_rotation": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "revision": 1,
        },
        "qualification": {
            "contact": "qualified",
            "numerical": "converged",
            "scientific": "benchmarked",
            "evidence_refs": ["unit_test"],
        },
        "created_at_utc": "2026-09-15T12:00:00Z",
    }


def test_list_destinations(client: TestClient) -> None:
    response = client.get("/tools/golf-simulator/destinations")
    assert response.status_code == 200
    data = response.json()
    assert "destinations" in data
    dest_ids = [d["destination_id"] for d in data["destinations"]]
    assert "local" in dest_ids
    assert "gspro" in dest_ids


def test_create_and_get_session(client: TestClient) -> None:
    # 1. Create session
    resp = client.post(
        "/tools/golf-simulator/session",
        json={"destination_id": "local", "session_id": "sess-test-01"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess-test-01"
    assert data["destination_id"] == "local"
    assert data["state"] == "idle"

    # 2. Get session
    resp_get = client.get("/tools/golf-simulator/session")
    assert resp_get.status_code == 200
    data_get = resp_get.json()
    assert data_get["session_id"] == "sess-test-01"
    assert data_get["state"] == "idle"
    assert data_get["capabilities"]["shot_input"]["state"] == "supported"


def test_prepare_arm_submit_happy_path(client: TestClient) -> None:
    # Connect
    client.post(
        "/tools/golf-simulator/session",
        json={"destination_id": "local", "session_id": "sess-test-01"},
    )

    # 1. Prepare
    shot_payload = _sample_shot_payload()
    prep_resp = client.post(
        "/tools/golf-simulator/shot/prepare",
        json={"shot": shot_payload, "context_revision": 1},
    )
    assert prep_resp.status_code == 200
    prep_data = prep_resp.json()
    prep_id = prep_data["prepared_shot_id"]
    assert prep_id
    assert prep_data["is_armed"] is False

    # Check status
    sess_status = client.get("/tools/golf-simulator/session").json()
    assert sess_status["state"] == "prepared"

    # 2. Arm
    arm_resp = client.post(
        "/tools/golf-simulator/shot/arm",
        json={"prepared_shot_id": prep_id, "context_revision": 1},
    )
    assert arm_resp.status_code == 200
    arm_token = arm_resp.json()["arm_token"]
    assert arm_token.startswith("arm-")

    sess_status = client.get("/tools/golf-simulator/session").json()
    assert sess_status["state"] == "armed"

    # 3. Submit
    submit_resp = client.post(
        "/tools/golf-simulator/shot/submit",
        json={"prepared_shot_id": prep_id, "arm_token": arm_token},
    )
    assert submit_resp.status_code == 200
    receipt = submit_resp.json()
    assert receipt["shot_id"] == "shot-test-01"
    assert receipt["state"] == "confirmed_accepted"

    # Verify session returned to idle
    sess_status = client.get("/tools/golf-simulator/session").json()
    assert sess_status["state"] == "idle"

    # 4. Query delivery status
    status_resp = client.get("/tools/golf-simulator/shot/shot-test-01/status")
    assert status_resp.status_code == 200
    st = status_resp.json()
    assert st["shot_id"] == "shot-test-01"
    assert st["state"] == "confirmed_accepted"


def test_prepare_fails_if_session_id_mismatch(client: TestClient) -> None:
    client.post(
        "/tools/golf-simulator/session",
        json={"destination_id": "local", "session_id": "sess-test-01"},
    )
    # Shot has mismatched session_id
    shot_payload = _sample_shot_payload(session_id="wrong-session")
    resp = client.post(
        "/tools/golf-simulator/shot/prepare",
        json={"shot": shot_payload, "context_revision": 1},
    )
    assert resp.status_code == 400
    assert "mismatch" in resp.json()["detail"].lower()


def test_arm_conflict_on_context_revision_mismatch(client: TestClient) -> None:
    client.post(
        "/tools/golf-simulator/session",
        json={"destination_id": "local", "session_id": "sess-test-01"},
    )
    shot_payload = _sample_shot_payload()
    prep_resp = client.post(
        "/tools/golf-simulator/shot/prepare",
        json={"shot": shot_payload, "context_revision": 1},
    )
    prep_id = prep_resp.json()["prepared_shot_id"]

    # Attempt to arm with outdated context_revision 2
    arm_resp = client.post(
        "/tools/golf-simulator/shot/arm",
        json={"prepared_shot_id": prep_id, "context_revision": 2},
    )
    assert arm_resp.status_code == 409
    assert "revision" in arm_resp.json()["detail"].lower()


def test_disarm_and_cancel_workflows(client: TestClient) -> None:
    client.post(
        "/tools/golf-simulator/session",
        json={"destination_id": "local", "session_id": "sess-test-01"},
    )
    shot_payload = _sample_shot_payload()
    prep_resp = client.post(
        "/tools/golf-simulator/shot/prepare",
        json={"shot": shot_payload, "context_revision": 1},
    )
    prep_id = prep_resp.json()["prepared_shot_id"]

    # Arm
    client.post(
        "/tools/golf-simulator/shot/arm",
        json={"prepared_shot_id": prep_id, "context_revision": 1},
    )
    assert client.get("/tools/golf-simulator/session").json()["state"] == "armed"

    # Disarm -> should revert to PREPARED
    disarm_resp = client.post(
        "/tools/golf-simulator/shot/disarm",
        json={"prepared_shot_id": prep_id},
    )
    assert disarm_resp.status_code == 200
    assert client.get("/tools/golf-simulator/session").json()["state"] == "prepared"

    # Cancel -> should revert to IDLE
    cancel_resp = client.post(
        "/tools/golf-simulator/shot/cancel",
        json={"prepared_shot_id": prep_id},
    )
    assert cancel_resp.status_code == 200
    assert client.get("/tools/golf-simulator/session").json()["state"] == "idle"


def test_destination_switch_rejected_when_armed(client: TestClient) -> None:
    client.post(
        "/tools/golf-simulator/session",
        json={"destination_id": "local", "session_id": "sess-test-01"},
    )
    prep_resp = client.post(
        "/tools/golf-simulator/shot/prepare",
        json={"shot": _sample_shot_payload(), "context_revision": 1},
    )
    client.post(
        "/tools/golf-simulator/shot/arm",
        json={
            "prepared_shot_id": prep_resp.json()["prepared_shot_id"],
            "context_revision": 1,
        },
    )

    # Attempt switch destination
    switch_resp = client.post(
        "/tools/golf-simulator/session",
        json={"destination_id": "gspro", "session_id": "sess-test-01"},
    )
    assert switch_resp.status_code == 409


def test_replay_transport_controls(client: TestClient) -> None:
    client.post(
        "/tools/golf-simulator/session",
        json={"destination_id": "local", "session_id": "sess-test-01"},
    )

    # Play
    r1 = client.post("/tools/golf-simulator/replay/action", json={"action": "play"})
    assert r1.status_code == 200
    assert r1.json()["playback_state"] == "playing"

    # Pause
    r2 = client.post("/tools/golf-simulator/replay/action", json={"action": "pause"})
    assert r2.status_code == 200
    assert r2.json()["playback_state"] == "paused"

    # Seek
    r3 = client.post(
        "/tools/golf-simulator/replay/action",
        json={"action": "seek", "target_time_s": 0.5},
    )
    assert r3.status_code == 200
    assert r3.json()["current_time_s"] == 0.5

    # Rate
    r4 = client.post(
        "/tools/golf-simulator/replay/action",
        json={"action": "rate", "playback_rate": 0.5},
    )
    assert r4.status_code == 200
    assert r4.json()["playback_rate"] == 0.5

    # Stop
    r5 = client.post("/tools/golf-simulator/replay/action", json={"action": "stop"})
    assert r5.status_code == 200
    assert r5.json()["playback_state"] == "stopped"
    assert r5.json()["current_time_s"] == 0.0


def test_resolve_uncertain_submission(client: TestClient) -> None:
    client.post(
        "/tools/golf-simulator/session",
        json={"destination_id": "local", "session_id": "sess-test-01"},
    )
    prep_resp = client.post(
        "/tools/golf-simulator/shot/prepare",
        json={"shot": _sample_shot_payload(), "context_revision": 1},
    )
    prep_id = prep_resp.json()["prepared_shot_id"]
    arm_resp = client.post(
        "/tools/golf-simulator/shot/arm",
        json={"prepared_shot_id": prep_id, "context_revision": 1},
    )
    arm_token = arm_resp.json()["arm_token"]
    client.post(
        "/tools/golf-simulator/shot/submit",
        json={"prepared_shot_id": prep_id, "arm_token": arm_token},
    )

    # Put into uncertain state artificially for test
    from src.api.routes.golf_simulator import get_current_session_service
    from src.shared.python.golf_simulator.contracts import SessionState

    service = get_current_session_service()
    assert service is not None
    service._state = SessionState.UNCERTAIN

    # Resolve with operator confirmation
    resolve_resp = client.post(
        "/tools/golf-simulator/shot/shot-test-01/resolve",
        json={
            "operator_evidence": "Visually verified on simulator screen",
            "confirmed": True,
        },
    )
    assert resolve_resp.status_code == 200
    res_data = resolve_resp.json()
    assert res_data["state"] == "confirmed_accepted"
    assert "Visually verified" in res_data["detail"]

    # Session returned to idle
    assert client.get("/tools/golf-simulator/session").json()["state"] == "idle"
