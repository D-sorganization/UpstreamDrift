"""Server-side analysis statistics contract (issue #8941).

Covers the bounded metric history, the single-pass aggregation that runs off
the event loop, and the ``since``/``limit`` incremental-fetch parameters of
``GET /analysis/statistics``.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections import deque
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_engine_manager
from src.api.routes import analysis_tools
from src.api.routes.analysis_tools import (
    NEXT_SINCE_HEADER,
    _store_metric_snapshot,
    router,
)

pytestmark = pytest.mark.unit

MAX_HISTORY = 500


class _Engine:
    time = 2.5

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([0.1, 0.2]), np.array([0.01, 0.02])

    def compute_mass_matrix(self) -> np.ndarray:
        return np.eye(2)

    def compute_jacobian(self, name: str) -> dict[str, np.ndarray]:
        return {"linear": np.eye(2)}


class _Manager:
    def __init__(self) -> None:
        self.engine = _Engine()

    def get_active_physics_engine(self) -> _Engine:
        return self.engine


@pytest.fixture
def manager() -> _Manager:
    return _Manager()


@pytest.fixture
def client(manager: _Manager) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_engine_manager] = lambda: manager
    return TestClient(app)


def _store_values(manager: _Manager, count: int) -> None:
    for i in range(count):
        _store_metric_snapshot(manager, {"sim_time": i * 0.01, "value": float(i)})


def _brute_force(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Reference implementation: the pre-#8941 two-walk algorithm."""
    keys = sorted(
        {
            k
            for snap in history
            for k, v in snap.items()
            if isinstance(v, int | float) and not math.isnan(v)
        }
    )
    metrics, series = [], {}
    for key in keys:
        values = [
            float(snap[key])
            for snap in history
            if isinstance(snap.get(key), int | float) and not math.isnan(snap[key])
        ]
        arr = np.array(values)
        metrics.append(
            {
                "metric_name": key,
                "current": values[-1],
                "minimum": float(arr.min()),
                "maximum": float(arr.max()),
                "mean": float(arr.mean()),
                "std_dev": float(arr.std()),
            }
        )
        series[key] = values
    return {"metrics": metrics, "time_series": series}


# ── bounded history ─────────────────────────────────────────────


def test_history_is_bounded_deque_of_last_500(manager: _Manager) -> None:
    _store_values(manager, MAX_HISTORY + 137)

    history = manager._metric_history
    assert isinstance(history, deque)
    assert history.maxlen == MAX_HISTORY
    assert len(history) == MAX_HISTORY
    assert history[0]["value"] == 137.0
    assert history[-1]["value"] == float(MAX_HISTORY + 136)


def test_store_adopts_preexisting_list_history(manager: _Manager) -> None:
    manager._metric_history = [{"value": -1.0}]
    _store_values(manager, 2)

    assert isinstance(manager._metric_history, deque)
    assert [s["value"] for s in manager._metric_history] == [-1.0, 0.0, 1.0]


def test_store_rejects_missing_engine_manager() -> None:
    with pytest.raises(ValueError, match="engine_manager"):
        _store_metric_snapshot(None, {"value": 1.0})  # type: ignore[arg-type]


def test_metrics_endpoint_keeps_history_bounded(
    client: TestClient, manager: _Manager
) -> None:
    for _ in range(3):
        assert client.get("/analysis/metrics").status_code == 200
    assert isinstance(manager._metric_history, deque)
    assert len(manager._metric_history) == 3


# ── default response: correctness and unchanged shape ───────────


def test_default_response_shape_is_unchanged(
    client: TestClient, manager: _Manager
) -> None:
    manager._metric_history = [
        {"sim_time": 0.0, "value": 1.0, "joint_positions": [0.1]},
        {"sim_time": 1.0, "value": 3.0, "flag": float("nan")},
    ]

    response = client.get("/analysis/statistics")

    assert response.status_code == 200
    assert response.json() == {
        "sim_time": 2.5,
        "sample_count": 2,
        "metrics": [
            {
                "metric_name": "sim_time",
                "current": 1.0,
                "minimum": 0.0,
                "maximum": 1.0,
                "mean": 0.5,
                "std_dev": 0.5,
            },
            {
                "metric_name": "value",
                "current": 3.0,
                "minimum": 1.0,
                "maximum": 3.0,
                "mean": 2.0,
                "std_dev": 1.0,
            },
        ],
        "time_series": {"sim_time": [0.0, 1.0], "value": [1.0, 3.0]},
    }


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_statistics_match_brute_force_on_random_series(
    client: TestClient, manager: _Manager, seed: int
) -> None:
    rng = random.Random(seed)
    keys = ["a", "b", "c", "d"]
    history = []
    for _ in range(rng.randint(1, 700)):
        snap: dict[str, Any] = {}
        for key in keys:
            roll = rng.random()
            if roll < 0.7:
                snap[key] = rng.uniform(-1e3, 1e3)
            elif roll < 0.8:
                snap[key] = rng.randint(-50, 50)
            elif roll < 0.9:
                snap[key] = float("nan")
            # else: key absent from this snapshot
        snap["vector"] = [rng.random()]
        _store_metric_snapshot(manager, snap)
        history.append(snap)

    expected = _brute_force(history[-MAX_HISTORY:])
    data = client.get("/analysis/statistics").json()

    assert data["sample_count"] == min(len(history), MAX_HISTORY)
    assert data["time_series"] == expected["time_series"]
    assert [m["metric_name"] for m in data["metrics"]] == [
        m["metric_name"] for m in expected["metrics"]
    ]
    for got, want in zip(data["metrics"], expected["metrics"], strict=True):
        for field in ("current", "minimum", "maximum", "mean", "std_dev"):
            assert got[field] == pytest.approx(want[field], rel=1e-12, abs=1e-9)


def test_empty_history_returns_empty_statistics(client: TestClient) -> None:
    response = client.get("/analysis/statistics")

    assert response.status_code == 200
    assert response.json()["metrics"] == []
    assert response.json()["time_series"] == {}
    assert response.headers[NEXT_SINCE_HEADER] == "0"


def test_aggregation_runs_off_the_event_loop(
    client: TestClient, manager: _Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    _store_values(manager, 5)
    seen: list[bool] = []
    original = analysis_tools._compute_statistics

    def _spy(*args: Any) -> Any:
        try:
            asyncio.get_running_loop()
            seen.append(True)
        except RuntimeError:
            seen.append(False)
        return original(*args)

    monkeypatch.setattr(analysis_tools, "_compute_statistics", _spy)

    assert client.get("/analysis/statistics").status_code == 200
    assert seen == [False]


# ── since / limit ───────────────────────────────────────────────


def _series(client: TestClient, query: str) -> tuple[list[float], str]:
    response = client.get(f"/analysis/statistics?{query}")
    assert response.status_code == 200, response.text
    return response.json()["time_series"]["value"], response.headers[NEXT_SINCE_HEADER]


def test_since_returns_only_new_points(client: TestClient, manager: _Manager) -> None:
    _store_values(manager, 10)

    values, next_since = _series(client, "since=7")

    assert values == [7.0, 8.0, 9.0]
    assert next_since == "10"


def test_since_and_limit_do_not_change_summary_statistics(
    client: TestClient, manager: _Manager
) -> None:
    _store_values(manager, 10)

    full = client.get("/analysis/statistics").json()
    partial = client.get("/analysis/statistics?since=8&limit=1").json()

    assert partial["metrics"] == full["metrics"]
    assert partial["sample_count"] == full["sample_count"] == 10
    assert partial["time_series"]["value"] == [9.0]


def test_limit_returns_most_recent_points(
    client: TestClient, manager: _Manager
) -> None:
    _store_values(manager, 10)

    values, next_since = _series(client, "limit=2")

    assert values == [8.0, 9.0]
    assert next_since == "10"


def test_since_is_absolute_across_eviction(
    client: TestClient, manager: _Manager
) -> None:
    _store_values(manager, MAX_HISTORY + 10)

    tail, next_since = _series(client, f"since={MAX_HISTORY + 8}")
    clamped, _ = _series(client, "since=0")

    assert tail == [float(MAX_HISTORY + 8), float(MAX_HISTORY + 9)]
    assert next_since == str(MAX_HISTORY + 10)
    assert len(clamped) == MAX_HISTORY
    assert clamped[0] == 10.0


def test_since_past_the_end_returns_empty_series(
    client: TestClient, manager: _Manager
) -> None:
    _store_values(manager, 3)

    values, next_since = _series(client, "since=99")

    assert values == []
    assert next_since == "3"


def test_default_request_reports_next_since(
    client: TestClient, manager: _Manager
) -> None:
    _store_values(manager, 4)

    values, next_since = _series(client, "")

    assert values == [0.0, 1.0, 2.0, 3.0]
    assert next_since == "4"


@pytest.mark.parametrize(
    "query", ["since=-1", "limit=0", "limit=-3", f"limit={MAX_HISTORY + 1}"]
)
def test_out_of_range_parameters_are_rejected(
    client: TestClient, manager: _Manager, query: str
) -> None:
    _store_values(manager, 3)

    assert client.get(f"/analysis/statistics?{query}").status_code == 422


# ── export stays JSON-serialisable with a deque history ─────────


def test_json_export_after_stored_snapshots(
    client: TestClient, manager: _Manager
) -> None:
    _store_values(manager, 3)

    response = client.post("/analysis/export", json={"format": "json"})

    assert response.status_code == 200
    data = response.json()
    assert data["record_count"] == 3
    assert [row["value"] for row in data["data"]] == [0.0, 1.0, 2.0]
