"""Tests for the ``POST /v2/comparison`` and ``/v2/model`` routes.

Mirrors the PyQt Monitor Comparison and Models tabs
(``src/tools/launch_monitor_analytics/gui.py``
``_read_comparison_params`` / ``_compute_comparison`` and
``_read_model_params`` / ``_compute_model``) so the API and desktop paths
share one implementation each: :func:`compare_monitors` for monitor
comparison, :func:`fit_predictive_model` for the predictive model.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.launch_monitor_analytics import router

pytestmark = pytest.mark.unit

_URL_COMPARISON = "/tools/launch-monitor-analytics/v2/comparison"
_URL_MODEL = "/tools/launch-monitor-analytics/v2/model"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _comparison_records() -> list[dict[str, object]]:
    """Twenty synthetic shots split evenly across two monitors, matched by shot id."""
    monitors = ["Trackman", "GCQuad"]
    return [
        {
            "monitor_vendor": monitors[index % 2],
            "shot_id": index // 2,
            "ball_speed": 140.0
            + index
            + (0.5 if monitors[index % 2] == "GCQuad" else 0.0),
        }
        for index in range(20)
    ]


def _unmatched_comparison_records() -> list[dict[str, object]]:
    """Twenty synthetic shots per monitor with no shared match identifier."""
    return [
        {"monitor_vendor": "Trackman", "ball_speed": 140.0 + index}
        for index in range(10)
    ] + [
        {"monitor_vendor": "GCQuad", "ball_speed": 150.0 + index} for index in range(10)
    ]


def _direct_comparison(records: list[dict[str, object]], **kwargs: Any) -> Any:
    import pandas as pd

    from src.tools.launch_monitor_model import compare_monitors

    return compare_monitors(pd.DataFrame.from_records(records), **kwargs)


def test_comparison_v2_matched_matches_direct_call(client: TestClient) -> None:
    """Matched-shot inputs reproduce the direct :func:`compare_monitors` call."""
    records = _comparison_records()
    response = client.post(
        _URL_COMPARISON,
        json={
            "records": records,
            "metric": "ball_speed",
            "match_column": "shot_id",
        },
    )
    assert response.status_code == 200
    payload = response.json()

    direct = _direct_comparison(records, metric="ball_speed", match_column="shot_id")
    assert payload["metric"] == direct.metric
    assert len(payload["summaries"]) == len(direct.summaries)
    for actual, expected in zip(payload["summaries"], direct.summaries, strict=True):
        assert actual["monitor"] == expected.monitor
        assert actual["sample_count"] == expected.sample_count
        assert actual["mean"] == pytest.approx(expected.mean)
        assert actual["standard_deviation"] == pytest.approx(
            expected.standard_deviation
        )
        assert actual["median"] == pytest.approx(expected.median)

    assert len(payload["pairwise"]) == len(direct.pairwise) == 1
    actual_pair = payload["pairwise"][0]
    expected_pair = direct.pairwise[0]
    assert actual_pair["reference"] == expected_pair.reference
    assert actual_pair["comparator"] == expected_pair.comparator
    assert actual_pair["matched"] is expected_pair.matched is True
    assert actual_pair["sample_count"] == expected_pair.sample_count
    assert actual_pair["mean_bias"] == pytest.approx(expected_pair.mean_bias)
    assert actual_pair["standard_deviation_bias"] == pytest.approx(
        expected_pair.standard_deviation_bias
    )
    assert actual_pair["lower_limit"] == pytest.approx(expected_pair.lower_limit)
    assert actual_pair["upper_limit"] == pytest.approx(expected_pair.upper_limit)
    assert actual_pair["slope"] == pytest.approx(expected_pair.slope)
    assert actual_pair["intercept"] == pytest.approx(expected_pair.intercept)
    assert actual_pair["correlation"] == pytest.approx(expected_pair.correlation)
    assert actual_pair["warning"] is expected_pair.warning is None


def test_comparison_v2_unmatched_matches_direct_call(client: TestClient) -> None:
    """No ``match_column`` reproduces the direct unmatched-mode call."""
    records = _unmatched_comparison_records()
    response = client.post(
        _URL_COMPARISON, json={"records": records, "metric": "ball_speed"}
    )
    assert response.status_code == 200
    payload = response.json()

    direct = _direct_comparison(records, metric="ball_speed")
    assert payload["match_column"] is None
    actual_pair = payload["pairwise"][0]
    expected_pair = direct.pairwise[0]
    assert actual_pair["matched"] is False
    assert actual_pair["mean_bias"] == pytest.approx(expected_pair.mean_bias)
    assert actual_pair["correlation"] == pytest.approx(expected_pair.correlation)
    assert actual_pair["warning"] == expected_pair.warning
    assert "descriptive" in actual_pair["warning"]
    assert math.isnan(expected_pair.lower_limit)
    assert actual_pair["lower_limit"] is None
    assert actual_pair["upper_limit"] is None
    assert actual_pair["slope"] is None
    assert actual_pair["intercept"] is None
    assert actual_pair["standard_deviation_bias"] is None


def test_comparison_v2_empty_reference_monitor_behaves_like_none(
    client: TestClient,
) -> None:
    """An empty-string ``reference_monitor`` behaves like the field being absent."""
    records = _unmatched_comparison_records()
    with_empty = client.post(
        _URL_COMPARISON,
        json={
            "records": records,
            "metric": "ball_speed",
            "reference_monitor": "",
        },
    )
    with_none = client.post(
        _URL_COMPARISON, json={"records": records, "metric": "ball_speed"}
    )
    assert with_empty.status_code == with_none.status_code == 200
    assert with_empty.json() == with_none.json()
    assert with_empty.json()["reference_monitor"] is None


def test_comparison_v2_single_shot_monitor_serializes_nan_as_null(
    client: TestClient,
) -> None:
    """A single-shot monitor's undefined standard deviation is JSON ``null``."""
    records = _unmatched_comparison_records() + [
        {"monitor_vendor": "FlightScope", "ball_speed": 155.0}
    ]
    response = client.post(
        _URL_COMPARISON,
        json={
            "records": records,
            "metric": "ball_speed",
            "reference_monitor": "FlightScope",
        },
    )
    assert response.status_code == 200
    payload = response.json()

    direct = _direct_comparison(
        records, metric="ball_speed", reference_monitor="FlightScope"
    )
    flightscope = next(
        item for item in direct.summaries if item.monitor == "FlightScope"
    )
    assert math.isnan(flightscope.standard_deviation)
    serialized = next(
        item for item in payload["summaries"] if item["monitor"] == "FlightScope"
    )
    assert serialized["standard_deviation"] is None
    assert serialized["sample_count"] == 1


def test_comparison_v2_missing_metric_column_is_400(client: TestClient) -> None:
    """A metric absent from the records maps to 400 via handle_api_errors."""
    response = client.post(
        _URL_COMPARISON,
        json={"records": _unmatched_comparison_records(), "metric": "launch_angle"},
    )
    assert response.status_code == 400
    assert "launch_angle" in str(response.json())


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"metric": ""}, "metric"),
        ({"records": [{"ball_speed": 1.0}] * 2}, "records"),
    ],
)
def test_comparison_v2_schema_violations_are_422(
    client: TestClient, override: dict[str, object], field: str
) -> None:
    """Inputs outside the desktop widgets' domain are schema-validation 422s."""
    body: dict[str, object] = {
        "records": _unmatched_comparison_records(),
        "metric": "ball_speed",
        **override,
    }
    response = client.post(_URL_COMPARISON, json=body)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(field in tuple(error["loc"]) for error in errors)


def _model_records() -> list[dict[str, object]]:
    """Twenty synthetic shots across two sessions with a predictable relationship."""
    sessions = ["session-a", "session-b"]
    return [
        {
            "session_id": sessions[index % 2],
            "club_speed": 90.0 + 0.5 * index,
            "ball_speed": 135.0 + 0.75 * index + (index % 3) * 0.1,
        }
        for index in range(20)
    ]


def _direct_model(records: list[dict[str, object]], **kwargs: Any) -> Any:
    import pandas as pd

    from src.tools.launch_monitor_model import fit_predictive_model

    return fit_predictive_model(pd.DataFrame.from_records(records), **kwargs)


def _assert_predictions_match(
    payload_predictions: Any, direct_predictions: Any
) -> None:
    direct_records = direct_predictions.to_dict(orient="records")
    assert len(payload_predictions) == len(direct_records)
    for actual, expected in zip(payload_predictions, direct_records, strict=True):
        assert actual["row_index"] == expected["row_index"]
        assert actual["actual"] == pytest.approx(expected["actual"])
        assert actual["predicted"] == pytest.approx(expected["predicted"])
        assert actual["residual"] == pytest.approx(expected["residual"])


def test_model_v2_random_split_matches_direct_call(client: TestClient) -> None:
    """Default (random-split) inputs reproduce the direct :func:`fit_predictive_model` call."""
    records = _model_records()
    response = client.post(
        _URL_MODEL,
        json={
            "records": records,
            "target": "ball_speed",
            "features": ["club_speed"],
        },
    )
    assert response.status_code == 200
    payload = response.json()

    direct = _direct_model(
        records, target="ball_speed", features=("club_speed",), model="linear"
    )
    assert payload["model"] == "linear" == direct.model
    assert payload["target"] == direct.target
    assert payload["features"] == list(direct.features)
    assert payload["random_seed"] == 42 == direct.random_seed
    assert payload["train_count"] == direct.train_count
    assert payload["test_count"] == direct.test_count
    for key, value in direct.metrics.items():
        assert payload["metrics"][key] == pytest.approx(value)
    assert payload["coefficients"].keys() == direct.coefficients.keys()
    for key, value in direct.coefficients.items():
        assert payload["coefficients"][key] == pytest.approx(value)
    _assert_predictions_match(payload["predictions"], direct.predictions)


def test_model_v2_group_column_matches_direct_call(client: TestClient) -> None:
    """A ``group_column`` reproduces the direct grouped-holdout call."""
    records = _model_records()
    response = client.post(
        _URL_MODEL,
        json={
            "records": records,
            "target": "ball_speed",
            "features": ["club_speed"],
            "model": "ridge",
            "group_column": "session_id",
            "random_seed": 7,
        },
    )
    assert response.status_code == 200
    payload = response.json()

    direct = _direct_model(
        records,
        target="ball_speed",
        features=("club_speed",),
        model="ridge",
        group_column="session_id",
        random_seed=7,
    )
    assert payload["model"] == "ridge"
    assert payload["random_seed"] == 7 == direct.random_seed
    assert payload["train_count"] == direct.train_count
    assert payload["test_count"] == direct.test_count
    _assert_predictions_match(payload["predictions"], direct.predictions)


def test_model_v2_missing_target_column_is_400(client: TestClient) -> None:
    """A target absent from the records maps to 400 via handle_api_errors."""
    response = client.post(
        _URL_MODEL,
        json={
            "records": _model_records(),
            "target": "launch_angle",
            "features": ["club_speed"],
        },
    )
    assert response.status_code == 400
    assert "launch_angle" in str(response.json())


def test_model_v2_missing_optional_dependency_is_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model whose optional dependency is missing is unavailable, not a 500."""

    def _raise_import_error(*_args: object, **_kwargs: object) -> None:
        raise ImportError("scikit-learn is required for mlp")

    monkeypatch.setattr(
        "src.api.routes.launch_monitor_analytics.fit_predictive_model",
        _raise_import_error,
    )
    response = client.post(
        _URL_MODEL,
        json={
            "records": _model_records(),
            "target": "ball_speed",
            "features": ["club_speed"],
            "model": "mlp",
        },
    )
    assert response.status_code == 503
    assert "mlp" in response.json()["detail"]


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"records": [{"ball_speed": 1.0}] * 2}, "records"),
        ({"features": []}, "features"),
        ({"features": ["club_speed", "club_speed"]}, "features"),
        ({"model": "logistic"}, "model"),
        ({"group_column": "player"}, "group_column"),
        ({"random_seed": -1}, "random_seed"),
    ],
)
def test_model_v2_schema_violations_are_422(
    client: TestClient, override: dict[str, object], field: str
) -> None:
    """Inputs outside the desktop widgets' domain are schema-validation 422s."""
    body: dict[str, object] = {
        "records": _model_records(),
        "target": "ball_speed",
        "features": ["club_speed"],
        **override,
    }
    response = client.post(_URL_MODEL, json=body)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(field in tuple(error["loc"]) for error in errors)
