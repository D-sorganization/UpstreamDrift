"""Research routes preserve source admission and local session ownership."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import golf_research, golf_simulator
from src.api.routes.matched_swings import require_local_client
from src.shared.python.golf_simulator import (
    AimContext,
    ContactStatus,
    NumericalStatus,
    ScientificStatus,
    ShotEnvelope,
    ShotQualification,
    SourceKind,
)

pytestmark = pytest.mark.unit


def test_reused_id_during_authentication_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect(client)
    entered, release, lock = Event(), Event(), Lock()
    calls = 0

    def authenticate(*args: object) -> SimpleNamespace:
        nonlocal calls
        with lock:
            calls += 1
            first = calls == 1
        if first:
            entered.set()
            assert release.wait(10)
        return admitted(args[-1])

    monkeypatch.setattr(golf_research, "load_research_impact_shot", authenticate)
    path = "/tools/golf-simulator/shot/prepare-research-impact"
    with ThreadPoolExecutor(max_workers=1) as pool:
        delayed = pool.submit(client.post, path, json=payload())
        try:
            assert entered.wait(10)
            first_completed = client.post(path, json=payload())
            assert first_completed.status_code == 200
            prepared_id = first_completed.json()["prepared_shot_id"]
            assert (
                client.post(
                    "/tools/golf-simulator/shot/cancel",
                    json={"prepared_shot_id": prepared_id},
                ).status_code
                == 200
            )
        finally:
            release.set()
        response = delayed.result(timeout=10)
    assert response.status_code == 409
    assert golf_simulator.get_current_session_service().current_state.value == "idle"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    golf_simulator.reset_simulator_state()
    app = FastAPI()
    app.include_router(golf_simulator.router)
    app.include_router(golf_research.router)
    app.dependency_overrides[require_local_client] = lambda: None
    app.dependency_overrides[golf_research.get_library] = lambda: object()
    with TestClient(app) as result:
        yield result
    golf_simulator.reset_simulator_state()


def payload() -> dict:
    return {
        "replay_id": "saved-replay",
        "run_id": "saved-impact",
        "shot_id": "research-shot",
        "session_id": "research-session",
        "created_at_utc": "2026-10-04T10:00:00Z",
        "aim_context": {
            "source_to_target_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "revision": 1,
        },
        "context_revision": 1,
    }


def admitted(metadata: object) -> SimpleNamespace:
    shot = ShotEnvelope(
        1,
        metadata.shot_id,
        metadata.session_id,
        SourceKind.MODEL_CONTACT,
        ShotQualification(
            ContactStatus.UNVERIFIED,
            NumericalStatus.UNVERIFIED,
            ScientificStatus.UNVERIFIED,
            ("authenticated-replay", "authored_clock"),
        ),
        (29, 2, 8),
        (3, -4, 5),
        metadata.aim_context,
        metadata.created_at_utc,
        model_run_id="saved-impact",
        trace_digest="sha256:" + "a" * 64,
        impact_id="saved-impact",
        impact_time_s=0.004,
    )
    return SimpleNamespace(
        shot=shot,
        to_record=lambda: {
            "replay_id": "saved-replay",
            "run_id": "saved-impact",
            "recorded_time_s": 0.004,
            "assumptions": {"selection_description": "Authored sample"},
            "qualification": {
                "contact": "unverified",
                "numerical": "unverified",
                "scientific": "unverified",
            },
        },
    )


def connect(client: TestClient, destination: str = "local") -> None:
    response = client.post(
        "/tools/golf-simulator/session",
        json={"session_id": "research-session", "destination_id": destination},
    )
    assert response.status_code == 200


def test_research_requires_existing_matching_local_session(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        golf_research,
        "load_research_impact_shot",
        lambda *args: calls.append(args),
    )
    path = "/tools/golf-simulator/shot/prepare-research-impact"
    assert client.post(path, json=payload()).status_code == 400
    connect(client, "fake")
    assert client.post(path, json=payload()).status_code == 400
    assert calls == []


def test_admitted_research_prepares_without_arming_or_promotion(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect(client)
    monkeypatch.setattr(
        golf_research, "load_research_impact_shot", lambda *args: admitted(args[-1])
    )
    response = client.post(
        "/tools/golf-simulator/shot/prepare-research-impact", json=payload()
    )
    assert response.status_code == 200
    record = response.json()
    assert record["is_armed"] is False
    assert record["research"]["qualification"]["scientific"] == "unverified"
    service = golf_simulator.get_current_session_service()
    assert service.current_state.value == "prepared"
    prepared = record["prepared_shot_id"]
    assert (
        client.post(
            "/tools/golf-simulator/shot/cancel", json={"prepared_shot_id": prepared}
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/tools/golf-simulator/shot/research-shot/local-trajectory"
        ).status_code
        == 404
    )


@pytest.mark.parametrize("error", [ValueError("Tampered"), RuntimeError("Incomplete")])
def test_failed_admission_keeps_session_idle(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    connect(client)

    def fail(*args: object) -> None:
        raise error

    monkeypatch.setattr(golf_research, "load_research_impact_shot", fail)
    response = client.post(
        "/tools/golf-simulator/shot/prepare-research-impact", json=payload()
    )
    assert response.status_code == 400
    assert golf_simulator.get_current_session_service().current_state.value == "idle"


def test_session_replacement_during_admission_refuses_preparation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect(client)

    def replace(*args: object) -> SimpleNamespace:
        golf_simulator.reset_simulator_state()
        return admitted(args[-1])

    monkeypatch.setattr(golf_research, "load_research_impact_shot", replace)
    response = client.post(
        "/tools/golf-simulator/shot/prepare-research-impact", json=payload()
    )
    assert response.status_code == 409


def test_promotion_field_is_rejected_before_admission(client: TestClient) -> None:
    connect(client)
    request = payload() | {"qualification": {"contact": "qualified"}}
    response = client.post(
        "/tools/golf-simulator/shot/prepare-research-impact", json=request
    )
    assert response.status_code == 422


def test_reused_shot_identity_cannot_replace_retained_research_context(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect(client)
    monkeypatch.setattr(
        golf_research, "load_research_impact_shot", lambda *args: admitted(args[-1])
    )
    path = "/tools/golf-simulator/shot/prepare-research-impact"
    first = client.post(path, json=payload())
    assert first.status_code == 200
    assert (
        client.post(
            "/tools/golf-simulator/shot/cancel",
            json={"prepared_shot_id": first.json()["prepared_shot_id"]},
        ).status_code
        == 200
    )
    assert client.post(path, json=payload()).status_code == 409


def test_foreign_admitted_context_never_prepares(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect(client)

    def foreign(*args: object) -> SimpleNamespace:
        record = admitted(args[-1])
        record.to_record = lambda: {"replay_id": "foreign", "run_id": "saved-impact"}
        return record

    monkeypatch.setattr(golf_research, "load_research_impact_shot", foreign)
    assert (
        client.post(
            "/tools/golf-simulator/shot/prepare-research-impact", json=payload()
        ).status_code
        == 400
    )
    assert golf_simulator.get_current_session_service().current_state.value == "idle"


def test_unknown_journal_shot_status_reports_not_found(client: TestClient) -> None:
    connect(client)
    assert (
        client.get("/tools/golf-simulator/shot/not-delivered/status").status_code == 404
    )
