"""API contract tests for GET /analysis/grip-wrench (GCV-10, #11716)."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from src.api.dependencies import get_simulation_service
from src.api.routes.analysis import router
from src.shared.python.biomechanics.grip_extraction import (
    net_only_analysis,
    unavailable_analysis,
)
from src.shared.python.biomechanics.grip_wrench import HandWrench, analyze_grip

pytestmark = pytest.mark.unit

RL = (0.0, 0.0, 1.0)
RR = (0.0, 0.0, 0.8)


def _pair(k, method="efc_force"):
    return analyze_grip(
        HandWrench("L", RL, (10.0 * k, 0, 0), (0, 0, 1.0)),
        HandWrench("R", RR, (-10.0 * k, 0, 0), (0, 0, 1.0)),
        split_method=method,
    )


class _Service:
    def __init__(self, runs):
        self._runs = runs

    def get_run(self, run_id=None):
        return self._runs.get(run_id or "active")


def _run(data, engine=None):
    return SimpleNamespace(
        run_id="r1", engine_type="mujoco", simulation_data=data, engine=engine
    )


def _client(runs):
    app = FastAPI()
    app.include_router(router)
    svc = _Service(runs)
    app.dependency_overrides[get_simulation_service] = lambda: svc
    return TestClient(app)


def _record(n=3, **extra):
    return {
        "grip_analyses": {
            "times_s": [0.01 * i for i in range(n)],
            "analyses": [_pair(i) for i in range(n)],
            **extra,
        }
    }


def test_contract_for_recorded_run():
    c = _client({"r1": _run(_record())})
    body = c.get("/analysis/grip-wrench", params={"run_id": "r1"}).json()
    assert body["available"] is True and body["run_id"] == "r1"
    assert body["split_method"] == "efc_force"
    assert body["time_s"] == [0.0, 0.01, 0.02]
    left = body["traces"]["left_force_n"]
    assert left["x"] == [0.0, 10.0, 20.0]
    assert body["traces"]["net_force_n"]["magnitude"] == [0.0, 0.0, 0.0]
    assert body["traces"]["couple_nm"]["magnitude"][1] == pytest.approx(
        (2.0**2 + 2.0**2) ** 0.5
    )
    assert body["units"]["couple_nm"] == "N*m"


def test_unavailable_values_are_null_not_zero():
    net_only = net_only_analysis(
        point_m=(0, 0, 0.9),
        force_on_club_n=(5.0, 0, 0),
        torque_on_club_nm=(0, 1.0, 0),
        split_method="allocation",
        reason="allocation yields one net wrench",
    )
    data = {"grip_analyses": {"times_s": [0.0, 0.1], "analyses": [_pair(1), net_only]}}
    c = _client({"r1": _run(data)})
    body = c.get("/analysis/grip-wrench", params={"run_id": "r1"}).json()
    assert body["traces"]["left_force_n"]["x"][1] is None
    assert body["traces"]["net_force_n"]["x"][1] == 5.0
    assert body["split_method"] == "mixed"
    assert body["split_method_by_sample"] == ["efc_force", "allocation"]
    assert body["unavailable_reasons"][1]


def test_impact_event_passes_through():
    c = _client({"r1": _run(_record())})
    body = c.get(
        "/analysis/grip-wrench", params={"run_id": "r1", "impact_time_s": 0.02}
    ).json()
    assert body["events"] == {"impact": 0.02}


def test_recorded_impact_time_is_used():
    c = _client({"r1": _run({**_record(), "impact_time_s": 0.01})})
    body = c.get("/analysis/grip-wrench", params={"run_id": "r1"}).json()
    assert body["events"] == {"impact": 0.01}


def test_run_without_grip_data_is_unavailable_with_reason():
    c = _client({"r1": _run({})})
    r = c.get("/analysis/grip-wrench", params={"run_id": "r1"})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False and body["reason"]
    assert body["traces"] == {} and body["time_s"] == []


def test_all_unavailable_analyses():
    data = {
        "grip_analyses": {
            "times_s": [0.0, 0.1],
            "analyses": [unavailable_analysis("no grip welds")] * 2,
        }
    }
    c = _client({"r1": _run(data)})
    body = c.get("/analysis/grip-wrench", params={"run_id": "r1"}).json()
    assert body["available"] is False and "no grip welds" in body["reason"]
    assert body["split_method"] == "unavailable"


def test_engine_provider_is_used():
    times = [0.0, 0.01]
    engine = SimpleNamespace(get_grip_analyses=lambda: (times, [_pair(1), _pair(2)]))
    c = _client({"r1": _run({}, engine)})
    body = c.get("/analysis/grip-wrench", params={"run_id": "r1"}).json()
    assert body["available"] is True and len(body["time_s"]) == 2


def test_unknown_run_404_and_bad_input_400():
    c = _client({})
    assert c.get("/analysis/grip-wrench", params={"run_id": "zz"}).status_code == 404
    bad = {"grip_analyses": {"times_s": [0.0], "analyses": []}}
    c = _client({"r1": _run(bad)})
    assert c.get("/analysis/grip-wrench", params={"run_id": "r1"}).status_code == 400
    c = _client({"r1": _run(_record())})
    r = c.get("/analysis/grip-wrench", params={"run_id": "r1", "impact_time_s": "nan"})
    assert r.status_code in (400, 422)
