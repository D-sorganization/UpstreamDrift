"""Flight visualization data survives the HTTP/save-record boundary (#15497)."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.bunker_workbench import record_digest, router
from src.tools.bunker_shot_gui.design import SandCondition, SwingSetup, WedgeDesign
from src.tools.bunker_shot_gui.flight import BunkerFlight
from src.tools.bunker_shot_gui.model import WorkbenchComparison, WorkbenchModel

pytestmark = [pytest.mark.contract, pytest.mark.integration]

FLIGHT = BunkerFlight(
    flight_m=((0.0, 0.0, 0.0), (3.0, 0.5, 2.0), (6.0, 1.0, 0.0)),
    rollout_m=((6.0, 1.0, 0.0), (7.0, 1.5, 0.0)),
    carry_m=6.082762530298219,
)


@pytest.fixture(scope="module")
def evaluation() -> Any:
    # Use a real refusal evaluation as the lightweight baseline; serialization
    # tests supply a known trace, without testing or substituting flight physics.
    return WorkbenchModel().evaluate(
        WedgeDesign(name="A", grind_preset="sm9_58_m"),
        SandCondition(preset="firm"),
        SwingSetup(dynamic_terms_active=False),
        include_playability=False,
    )


def _client(monkeypatch: pytest.MonkeyPatch, evaluation: Any) -> TestClient:
    monkeypatch.setattr(WorkbenchModel, "evaluate", lambda *a, **kw: evaluation)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(scope="module")
def traced_evaluation(evaluation: Any) -> Any:
    from bunkershot3d.solvers import EnvelopeStatus

    verdict = dataclasses.replace(
        evaluation.shot.verdict, status=EnvelopeStatus.BEYOND_VALIDATION
    )
    shot = dataclasses.replace(
        evaluation.shot,
        refused=False,
        verdict=verdict,
        flight=FLIGHT,
        carry_m=FLIGHT.carry_m,
        carry_verdict=verdict,
    )
    return dataclasses.replace(evaluation, shot=shot)


def test_evaluate_exposes_trace_and_digest_covers_all_points(
    monkeypatch: pytest.MonkeyPatch, traced_evaluation: Any
) -> None:
    client = _client(monkeypatch, traced_evaluation)
    response = client.post(
        "/tools/bunker-workbench/v1/evaluate",
        json={"handedness": "right", "design": {"name": "A"}},
    )
    assert response.status_code == 200, response.text
    record = response.json()
    trace = record["flight"]
    assert trace["flight_m"] == [list(p) for p in FLIGHT.flight_m]
    assert trace["rollout_m"] == [list(p) for p in FLIGHT.rollout_m]
    assert trace["carry_m"] == record["carry"]["carry_m"]
    assert trace["assumption"] == FLIGHT.assumption
    assert record["predictive"] is False
    assert record_digest(record) == record["digest"]
    assert (
        client.post("/tools/bunker-workbench/v1/verify", json=record).status_code == 200
    )
    trace["flight_m"][-1][0] += 1.0
    assert (
        client.post("/tools/bunker-workbench/v1/verify", json=record).status_code == 422
    )


@pytest.mark.parametrize("refused", [False, True])
def test_absent_trace_is_explicit_null(
    monkeypatch: pytest.MonkeyPatch, evaluation: Any, refused: bool
) -> None:
    shot = dataclasses.replace(evaluation.shot, refused=refused)
    client = _client(monkeypatch, dataclasses.replace(evaluation, shot=shot))
    response = client.post(
        "/tools/bunker-workbench/v1/evaluate",
        json={"handedness": "right", "design": {"name": "A"}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["flight"] is None


def test_comparison_preserves_each_sides_trace_and_digest(
    monkeypatch: pytest.MonkeyPatch, traced_evaluation: Any, evaluation: Any
) -> None:
    comparison = WorkbenchComparison(left=traced_evaluation, right=evaluation)
    monkeypatch.setattr(WorkbenchModel, "compare", lambda *a, **kw: comparison)
    client = _client(monkeypatch, evaluation)
    response = client.post(
        "/tools/bunker-workbench/v1/compare",
        json={
            "handedness": "right",
            "left": {"name": "A"},
            "right": {"name": "B"},
        },
    )
    assert response.status_code == 200, response.text
    record = response.json()
    assert record["left"]["flight"]["assumption"] == FLIGHT.assumption
    assert record["right"]["flight"] is None
    for child in (record["left"], record["right"], record):
        assert record_digest(child) == child["digest"]


def test_saved_v1_record_without_flight_remains_verifiable(
    monkeypatch: pytest.MonkeyPatch, evaluation: Any
) -> None:
    client = _client(monkeypatch, evaluation)
    record = client.post(
        "/tools/bunker-workbench/v1/evaluate",
        json={"handedness": "right", "design": {"name": "A"}},
    ).json()
    record.pop("flight")
    record["digest"] = record_digest(record)
    assert (
        client.post("/tools/bunker-workbench/v1/verify", json=record).status_code == 200
    )
