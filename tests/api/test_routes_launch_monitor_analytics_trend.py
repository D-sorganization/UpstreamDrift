"""Tests for the ``POST /v2/trend`` launch-monitor analytics route.

Mirrors the PyQt Trends tab (``src/tools/launch_monitor_analytics/gui.py``
``_read_trend_params`` / ``_compute_trend``) so the API and desktop paths
share one implementation, :func:`analyze_trend`.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.launch_monitor_analytics import router

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _trend_records() -> list[dict[str, object]]:
    """Twelve daily, deterministic synthetic observations of ``club_speed``."""
    return [
        {
            "captured_at": f"2024-01-{index + 1:02d}T00:00:00Z",
            "club_speed": 90.0 + 0.4 * index,
        }
        for index in range(12)
    ]


def test_trend_v2_matches_direct_analyze_trend_call(client: TestClient) -> None:
    """The API path and the GUI's direct call must agree numerically."""
    import pandas as pd

    from src.tools.launch_monitor_model import analyze_trend

    records = _trend_records()
    response = client.post(
        "/tools/launch-monitor-analytics/v2/trend",
        json={
            "records": records,
            "metric": "club_speed",
            "time_column": "captured_at",
            "rolling_window": 3,
        },
    )
    assert response.status_code == 200
    payload = response.json()

    direct = analyze_trend(
        pd.DataFrame.from_records(records),
        metric="club_speed",
        time_column="captured_at",
        rolling_window=3,
    )

    assert payload["metric"] == direct.metric
    assert payload["sample_count"] == direct.sample_count
    assert payload["slope_per_day"] == pytest.approx(direct.slope_per_day)
    assert payload["robust_slope_per_day"] == pytest.approx(direct.robust_slope_per_day)
    assert payload["p_value"] == pytest.approx(direct.p_value)
    assert payload["earliest_mean"] == pytest.approx(direct.earliest_mean)
    assert payload["latest_mean"] == pytest.approx(direct.latest_mean)
    assert len(payload["rolling"]) == len(direct.rolling)
    assert len(payload["change_candidates"]) == len(direct.change_candidates)

    last_row = direct.rolling.iloc[-1]
    assert payload["rolling"][-1]["rolling_mean"] == pytest.approx(
        last_row["rolling_mean"]
    )
    assert payload["rolling"][-1]["ewma"] == pytest.approx(last_row["ewma"])

    if direct.change_candidates:
        expected = direct.change_candidates[0]
        actual = payload["change_candidates"][0]
        assert actual["row_index"] == expected.row_index
        assert actual["effect_size"] == pytest.approx(expected.effect_size)
        assert actual["captured_at"] == expected.captured_at.isoformat()


def test_trend_v2_rolling_leading_nan_serializes_to_null(client: TestClient) -> None:
    """Unavailable is never zero: pre-window rolling stats must be null."""
    response = client.post(
        "/tools/launch-monitor-analytics/v2/trend",
        json={
            "records": _trend_records(),
            "metric": "club_speed",
            "time_column": "captured_at",
            "rolling_window": 3,
        },
    )
    assert response.status_code == 200
    rolling = response.json()["rolling"]

    # rolling(window=3, min_periods=3) leaves the first two rows unfilled.
    assert rolling[0]["rolling_mean"] is None
    assert rolling[0]["rolling_std"] is None
    assert rolling[1]["rolling_mean"] is None
    assert rolling[2]["rolling_mean"] is not None
    assert rolling[2]["rolling_std"] is not None


def test_trend_v2_unknown_metric_is_contract_error(client: TestClient) -> None:
    """Domain validation maps to 400, like ``/v2/analyze`` (handle_api_errors)."""
    response = client.post(
        "/tools/launch-monitor-analytics/v2/trend",
        json={
            "records": _trend_records(),
            "metric": "not_a_column",
            "time_column": "captured_at",
            "rolling_window": 3,
        },
    )
    assert response.status_code == 400
    assert "not_a_column" in str(response.json())


def test_trend_v2_rolling_window_below_minimum_is_422(client: TestClient) -> None:
    """``rolling_window`` must stay within the GUI spinbox range [3, 500]."""
    response = client.post(
        "/tools/launch-monitor-analytics/v2/trend",
        json={
            "records": _trend_records(),
            "metric": "club_speed",
            "time_column": "captured_at",
            "rolling_window": 2,
        },
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("rolling_window" in tuple(error["loc"]) for error in errors)
