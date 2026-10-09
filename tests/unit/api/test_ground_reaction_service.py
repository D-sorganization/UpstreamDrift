"""Tests for the ground-reaction plot service and route (GCV-5, #11711)."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api.dependencies import get_simulation_service
from src.api.routes.analysis import router
from src.api.services.ground_reaction_service import (
    NO_GROUND_REACTION_REASON,
    compute_ground_reaction_plot,
    resolve_ground_reaction_series,
)
from src.shared.python.biomechanics.ground_reaction import (
    ContactSet,
    GroundReactionSeries,
    analyze_ground_reaction,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

G = 9.80665
ROUTE = "/analysis/ground-reaction"


def _series() -> GroundReactionSeries:
    def stance(fz: float):
        left = ContactSet(np.array([[0.0, 0.0, fz]]), np.array([[0.0, 0.15, 0.0]]))
        return analyze_ground_reaction(
            {"left": left, "right": ContactSet.empty()}, (0.0, 0.0, 0.95)
        )

    return GroundReactionSeries.from_breakdowns(
        [0.0, 0.1], [stance(784.0), stance(5.0)]
    )


def _run(simulation_data=None, engine=None, recorder=None):
    return SimpleNamespace(
        simulation_data=simulation_data or {}, engine=engine, recorder=recorder
    )


def test_recorded_series_is_preferred_over_the_engine() -> None:
    recorded = _series()
    engine = SimpleNamespace(get_ground_reaction_series=lambda: pytest.fail("used"))
    run = _run({"ground_reaction_series": recorded}, engine)
    assert resolve_ground_reaction_series(run) is recorded


def test_engine_provider_is_used_when_nothing_is_recorded() -> None:
    series = _series()
    engine = SimpleNamespace(get_ground_reaction_series=lambda: series)
    assert resolve_ground_reaction_series(_run(engine=engine)) is series


def test_recorder_provider_is_used_when_recorded_and_engine_have_nothing() -> None:
    series = _series()
    recorder = SimpleNamespace(get_ground_reaction_series=lambda: series)
    run = _run(engine=SimpleNamespace(), recorder=recorder)
    assert resolve_ground_reaction_series(run) is series


def test_recorder_provider_returning_none_is_still_unavailable() -> None:
    recorder = SimpleNamespace(get_ground_reaction_series=lambda: None)
    run = _run(engine=SimpleNamespace(), recorder=recorder)
    assert resolve_ground_reaction_series(run) is None


def test_engine_provider_is_preferred_over_the_recorder() -> None:
    series = _series()
    engine = SimpleNamespace(get_ground_reaction_series=lambda: series)
    recorder = SimpleNamespace(get_ground_reaction_series=lambda: pytest.fail("used"))
    run = _run(engine=engine, recorder=recorder)
    assert resolve_ground_reaction_series(run) is series


def test_missing_data_is_an_unavailable_payload_with_reason() -> None:
    payload = compute_ground_reaction_plot(_run(engine=SimpleNamespace()))
    assert payload["available"] is False
    assert payload["reason"] == NO_GROUND_REACTION_REASON
    assert payload["traces"] == {} and payload["time_s"] == []


def test_wrong_recorded_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match="GroundReactionSeries"):
        compute_ground_reaction_plot(_run({"ground_reaction_series": {"x": 1}}))


def test_unavailable_cop_is_null_in_the_json_payload() -> None:
    payload = compute_ground_reaction_plot(_run({"ground_reaction_series": _series()}))
    text = json.dumps(payload, allow_nan=False)  # no NaN leaks into the API
    assert payload["available"] is True
    assert payload["traces"]["left_cop_m"]["x"][1] is None  # 5 N < 10 N threshold
    assert payload["traces"]["right_cop_m"]["x"] == [None, None]
    assert "NaN" not in text


def test_body_weight_and_events_come_from_the_run_or_arguments() -> None:
    data = {
        "ground_reaction_series": _series(),
        "body_mass_kg": 80.0,
        "events": {"address": 0.0, "top": 0.05},
        "impact_time_s": 0.1,
    }
    payload = compute_ground_reaction_plot(_run(data))
    assert payload["traces"]["net_force_bw"]["z"][0] == pytest.approx(784.0 / (80 * G))
    assert payload["events"] == {"address": 0.0, "top": 0.05, "impact": 0.1}
    override = compute_ground_reaction_plot(
        _run(data), body_weight_n=784.0, impact_time_s=0.09
    )
    assert override["traces"]["net_force_bw"]["z"][0] == pytest.approx(1.0)
    assert override["events"]["impact"] == 0.09


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_impact_time_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="impact_time_s"):
        compute_ground_reaction_plot(
            _run({"ground_reaction_series": _series()}), impact_time_s=bad
        )


# ── GET /analysis/ground-reaction contract ──────────────────────────────────


class _Service:
    def __init__(self, runs: dict) -> None:
        self._runs = runs

    def get_run(self, run_id=None):
        return self._runs.get(run_id or "active")


def _client(runs: dict) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    service = _Service(runs)
    app.dependency_overrides[get_simulation_service] = lambda: service
    return TestClient(app)


def _api_run(simulation_data=None, engine=None):
    return SimpleNamespace(
        run_id="r1",
        engine_type="mujoco",
        simulation_data=simulation_data or {},
        engine=engine,
    )


def test_route_contract_for_recorded_run() -> None:
    data = {"ground_reaction_series": _series(), "body_weight_n": 784.0}
    c = _client({"r1": _api_run(data)})
    r = c.get(ROUTE, params={"run_id": "r1", "impact_time_s": 0.1})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True and body["run_id"] == "r1"
    assert body["engine"] == "mujoco"
    assert body["time_s"] == [0.0, 0.1] and body["feet"] == ["left", "right"]
    assert body["events"] == {"impact": 0.1}
    assert body["traces"]["net_force_bw"]["z"][0] == pytest.approx(1.0)
    assert body["traces"]["left_cop_m"]["x"][1] is None  # below CoP threshold
    assert body["load_share"]["left"][0] == pytest.approx(1.0)
    assert body["units"]["net_free_moment_nm"] == "N*m"
    assert body["labels"]["net_moment_com_nm"] == "Net Moment About CoM"


def test_route_run_without_ground_reaction_is_unavailable_with_reason() -> None:
    r = _client({"r1": _api_run()}).get(ROUTE, params={"run_id": "r1"})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == NO_GROUND_REACTION_REASON
    assert body["traces"] == {} and body["time_s"] == []


def test_route_active_run_is_the_default() -> None:
    c = _client({"active": _api_run({"ground_reaction_series": _series()})})
    assert c.get(ROUTE).json()["available"] is True


def test_route_unknown_run_404_and_bad_input_400() -> None:
    assert _client({}).get(ROUTE, params={"run_id": "zz"}).status_code == 404
    bad = _client({"r1": _api_run({"ground_reaction_series": {"x": 1}})})
    assert bad.get(ROUTE, params={"run_id": "r1"}).status_code == 400
    ok = _client({"r1": _api_run({"ground_reaction_series": _series()})})
    r = ok.get(ROUTE, params={"run_id": "r1", "impact_time_s": "nan"})
    assert r.status_code in (400, 422)
