"""Tests for the ``POST /v2/relationships`` and ``/v2/multivariate`` routes.

Mirrors the PyQt Relationships tab (``src/tools/launch_monitor_analytics/gui.py``
``_read_relationship_params`` / ``_compute_relationship`` /
``_compute_multivariate``) so the API and desktop paths share one
implementation each: :func:`compute_correlations` for relationships,
:func:`compute_pca` and :func:`compute_vif` for multivariate diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.launch_monitor_analytics import router

pytestmark = pytest.mark.integration

_URL = "/tools/launch-monitor-analytics/v2/relationships"
_URL_MULTIVARIATE = "/tools/launch-monitor-analytics/v2/multivariate"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _relationship_records() -> list[dict[str, object]]:
    """Twenty synthetic shots with one strong and one weak dependency."""
    return [
        {
            "ball_speed": 140.0 + index,
            "carry_distance": 200.0 + 2.0 * index + (index % 3),
            "spin_rate": 2500.0 + 37.0 * ((index * 7) % 11),
            "club_speed": 95.0 + 0.6 * index + (index % 4) * 0.3,
        }
        for index in range(20)
    ]


def _assert_matrix(rows: object, expected: Any) -> None:
    """Compare a serialized matrix with the direct result's DataFrame."""
    assert np.asarray(rows, dtype=float) == pytest.approx(
        expected.to_numpy(float), nan_ok=True
    )


def _direct(records: list[dict[str, object]], **kwargs: Any) -> Any:
    import pandas as pd

    from src.tools.launch_monitor_model import compute_correlations

    return compute_correlations(pd.DataFrame.from_records(records), **kwargs)


def test_relationships_v2_defaults_match_direct_call(client: TestClient) -> None:
    """Default method/threshold and no controls reproduce the direct result."""
    records = _relationship_records()
    metrics = ["ball_speed", "carry_distance", "spin_rate"]
    response = client.post(_URL, json={"records": records, "metrics": metrics})
    assert response.status_code == 200
    payload = response.json()

    direct = _direct(records, metrics=tuple(metrics))
    assert payload["method"] == "pearson"
    assert payload["metrics"] == metrics
    _assert_matrix(payload["coefficients"], direct.coefficients)
    _assert_matrix(payload["p_values"], direct.p_values)
    _assert_matrix(payload["adjusted_p_values"], direct.adjusted_p_values)
    assert payload["pair_counts"] == direct.pair_counts.to_numpy(int).tolist()
    assert payload["partial_coefficients"] is None
    assert payload["boolean_projected"] == []
    assert len(payload["edges"]) == len(direct.edges) >= 1
    first, expected = payload["edges"][0], direct.edges[0]
    assert (first["source"], first["target"]) == (expected.source, expected.target)
    assert first["coefficient"] == pytest.approx(expected.coefficient)
    assert first["adjusted_p_value"] == pytest.approx(expected.adjusted_p_value)
    assert first["sample_count"] == expected.sample_count
    assert first["includes_derived_metric"] is expected.includes_derived_metric


def test_relationships_v2_controls_drop_selected_metrics_like_desktop(
    client: TestClient,
) -> None:
    """Controls that are also metrics are dropped, as the desktop tab drops them."""
    records = _relationship_records()
    metrics = ["ball_speed", "carry_distance", "spin_rate"]
    response = client.post(
        _URL,
        json={
            "records": records,
            "metrics": metrics,
            "controls": ["club_speed", "spin_rate"],
            "method": "spearman",
            "edge_threshold": 0.5,
        },
    )
    assert response.status_code == 200
    payload = response.json()

    direct = _direct(
        records,
        metrics=tuple(metrics),
        method="spearman",
        controls=("club_speed",),
        edge_threshold=0.5,
    )
    assert payload["method"] == "spearman"
    _assert_matrix(payload["partial_coefficients"], direct.partial_coefficients)
    assert len(payload["edges"]) == len(direct.edges)


def test_relationships_v2_constant_metric_serializes_nan_as_null(
    client: TestClient,
) -> None:
    """An undefined coefficient is JSON ``null``, never ``0``."""
    records = [dict(row, flat=1.0) for row in _relationship_records()]
    response = client.post(
        _URL, json={"records": records, "metrics": ["ball_speed", "flat"]}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["coefficients"][0][1] is None
    assert payload["p_values"][1][0] is None
    assert payload["coefficients"][0][0] == pytest.approx(1.0)
    assert payload["edges"] == []


def test_relationships_v2_missing_column_is_400(client: TestClient) -> None:
    """A metric absent from the records maps to 400 via handle_api_errors."""
    response = client.post(
        _URL,
        json={
            "records": _relationship_records(),
            "metrics": ["ball_speed", "launch_angle"],
        },
    )
    assert response.status_code == 400
    assert "launch_angle" in str(response.json())


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"metrics": ["ball_speed"]}, "metrics"),
        ({"metrics": ["ball_speed", "ball_speed"]}, "metrics"),
        ({"edge_threshold": 1.5}, "edge_threshold"),
        ({"method": "cosine"}, "method"),
        ({"records": [{"ball_speed": 1.0}] * 2}, "records"),
    ],
)
def test_relationships_v2_schema_violations_are_422(
    client: TestClient, override: dict[str, object], field: str
) -> None:
    """Inputs outside the desktop widgets' domain are schema-validation 422s."""
    body: dict[str, object] = {
        "records": _relationship_records(),
        "metrics": ["ball_speed", "carry_distance"],
        **override,
    }
    response = client.post(_URL, json=body)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(field in tuple(error["loc"]) for error in errors)


def _direct_multivariate(
    records: list[dict[str, object]], metrics: tuple[str, ...]
) -> tuple[Any, Any]:
    import pandas as pd

    from src.tools.launch_monitor_model import compute_pca, compute_vif

    frame = pd.DataFrame.from_records(records)
    return compute_pca(frame, metrics=metrics), compute_vif(frame, metrics=metrics)


def test_multivariate_v2_matches_direct_call(client: TestClient) -> None:
    """PCA and VIF numbers reproduce the direct desktop-path call."""
    records = _relationship_records()
    metrics = ["ball_speed", "carry_distance", "spin_rate"]
    response = client.post(
        _URL_MULTIVARIATE, json={"records": records, "metrics": metrics}
    )
    assert response.status_code == 200
    payload = response.json()

    pca, vif = _direct_multivariate(records, tuple(metrics))
    assert payload["pca"]["metrics"] == metrics
    assert payload["pca"]["sample_count"] == pca.sample_count
    assert payload["pca"]["explained_variance_ratio"] == pytest.approx(
        pca.explained_variance_ratio.to_numpy(float).tolist()
    )
    _assert_matrix(payload["pca"]["loadings"], pca.loadings)
    _assert_matrix(payload["pca"]["scores"], pca.scores)

    assert payload["vif"]["sample_count"] == vif.sample_count
    assert payload["vif"]["warning_metrics"] == list(vif.warning_metrics)
    for metric in metrics:
        assert payload["vif"]["values"][metric] == pytest.approx(
            float(vif.values[metric])
        )


def test_multivariate_v2_collinear_vif_serializes_inf_as_null(
    client: TestClient,
) -> None:
    """A perfectly collinear pair's infinite VIF is JSON ``null``, never ``0``."""
    records = [
        {"ball_speed": 140.0 + index, "twice_ball_speed": 2.0 * (140.0 + index)}
        for index in range(20)
    ]
    response = client.post(
        _URL_MULTIVARIATE,
        json={"records": records, "metrics": ["ball_speed", "twice_ball_speed"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["vif"]["values"]["ball_speed"] is None
    assert payload["vif"]["values"]["twice_ball_speed"] is None
    assert set(payload["vif"]["warning_metrics"]) == {
        "ball_speed",
        "twice_ball_speed",
    }


def test_multivariate_v2_missing_column_is_400(client: TestClient) -> None:
    """A metric absent from the records maps to 400 via handle_api_errors."""
    response = client.post(
        _URL_MULTIVARIATE,
        json={
            "records": _relationship_records(),
            "metrics": ["ball_speed", "launch_angle"],
        },
    )
    assert response.status_code == 400
    assert "launch_angle" in str(response.json())


def test_multivariate_v2_single_metric_is_422(client: TestClient) -> None:
    """Fewer than two metrics is a schema-validation 422, like relationships."""
    response = client.post(
        _URL_MULTIVARIATE,
        json={"records": _relationship_records(), "metrics": ["ball_speed"]},
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("metrics" in tuple(error["loc"]) for error in errors)
