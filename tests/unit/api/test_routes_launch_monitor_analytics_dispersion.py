"""Tests for the ``POST /v2/dispersion`` launch-monitor analytics route.

Mirrors the PyQt Dispersion tab (``src/tools/launch_monitor_analytics/gui.py``
``_read_dispersion_params`` / ``_compute_dispersion``) so the API and desktop
paths share one implementation, :func:`analyze_dispersion`.
"""

from __future__ import annotations

import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.launch_monitor_analytics import (
    _dispersion_result_to_dict,
    router,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _dispersion_records() -> list[dict[str, object]]:
    """Twelve synthetic shots split evenly across two clubs."""
    clubs = ["driver", "7-iron"]
    return [
        {
            "club": clubs[index % 2],
            "carry_distance": 200.0 + index,
            "lateral_carry": -5.0 + 0.5 * index,
        }
        for index in range(12)
    ]


def test_dispersion_v2_default_columns_no_grouping_matches_direct_call(
    client: TestClient,
) -> None:
    """Default forward/lateral, no grouping: one "All Shots" group."""
    import pandas as pd

    from src.tools.launch_monitor_model import analyze_dispersion

    records = _dispersion_records()
    response = client.post(
        "/tools/launch-monitor-analytics/v2/dispersion",
        json={"records": records},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["forward"] == "carry_distance"
    assert payload["lateral"] == "lateral_carry"
    assert payload["group_column"] is None
    assert len(payload["groups"]) == 1
    group = payload["groups"][0]
    assert group["group"] == "All Shots"

    direct = analyze_dispersion(pd.DataFrame.from_records(records))
    assert group["sample_count"] == direct.sample_count
    assert group["center_forward"] == pytest.approx(direct.center_forward)
    assert group["center_lateral"] == pytest.approx(direct.center_lateral)
    assert group["mean_forward"] == pytest.approx(direct.mean_forward)
    assert group["mean_lateral"] == pytest.approx(direct.mean_lateral)
    assert group["ellipse_major"] == pytest.approx(direct.ellipse_major)
    assert group["ellipse_minor"] == pytest.approx(direct.ellipse_minor)
    assert group["ellipse_angle_rad"] == pytest.approx(direct.ellipse_angle_rad)
    assert group["area_95"] == pytest.approx(direct.area_95)
    assert group["radial_rmse"] == pytest.approx(direct.radial_rmse)
    assert group["radial_p50"] == pytest.approx(direct.radial_p50)
    assert group["radial_p90"] == pytest.approx(direct.radial_p90)


def test_dispersion_v2_group_by_club_matches_direct_per_group_call(
    client: TestClient,
) -> None:
    """``group_column="club"`` yields one entry per club, each matching direct analysis."""
    import pandas as pd

    from src.tools.launch_monitor_model import analyze_dispersion

    records = _dispersion_records()
    response = client.post(
        "/tools/launch-monitor-analytics/v2/dispersion",
        json={"records": records, "group_column": "club"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["group_column"] == "club"
    groups_by_name = {entry["group"]: entry for entry in payload["groups"]}
    assert set(groups_by_name) == {"driver", "7-iron"}
    for name, group in groups_by_name.items():
        assert isinstance(name, str)
        subset = pd.DataFrame.from_records([r for r in records if r["club"] == name])
        direct = analyze_dispersion(subset)
        assert group["sample_count"] == direct.sample_count
        assert group["center_forward"] == pytest.approx(direct.center_forward)
        assert group["radial_p90"] == pytest.approx(direct.radial_p90)


def test_dispersion_v2_group_column_absent_falls_back_to_all_shots(
    client: TestClient,
) -> None:
    """A group column absent from the records falls back to desktop parity."""
    records = _dispersion_records()
    response = client.post(
        "/tools/launch-monitor-analytics/v2/dispersion",
        json={"records": records, "group_column": "session_id"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["group_column"] == "session_id"
    assert len(payload["groups"]) == 1
    assert payload["groups"][0]["group"] == "All Shots"


def test_dispersion_v2_missing_forward_column_is_400(client: TestClient) -> None:
    """A missing forward/lateral column maps to 400 via handle_api_errors."""
    records = [{"lateral_carry": -5.0 + index} for index in range(5)]
    response = client.post(
        "/tools/launch-monitor-analytics/v2/dispersion",
        json={"records": records},
    )
    assert response.status_code == 400
    assert "carry_distance" in str(response.json())


def test_dispersion_v2_fewer_than_three_complete_shots_in_group_is_400(
    client: TestClient,
) -> None:
    """Fewer than three complete shots in a group is a domain ValueError -> 400."""
    records = [
        {"club": "driver", "carry_distance": 200.0, "lateral_carry": -1.0},
        {"club": "driver", "carry_distance": 201.0, "lateral_carry": -2.0},
        {"club": "7-iron", "carry_distance": 150.0, "lateral_carry": 1.0},
        {"club": "7-iron", "carry_distance": 151.0, "lateral_carry": 2.0},
        {"club": "7-iron", "carry_distance": 152.0, "lateral_carry": 3.0},
    ]
    response = client.post(
        "/tools/launch-monitor-analytics/v2/dispersion",
        json={"records": records, "group_column": "club"},
    )
    assert response.status_code == 400


def test_dispersion_result_to_dict_serializes_nan_as_null() -> None:
    """A non-finite field serializes as JSON ``null``, never ``0``."""
    from src.tools.launch_monitor_model import DispersionResult

    result = DispersionResult(
        sample_count=3,
        center_forward=200.0,
        center_lateral=-1.0,
        mean_forward=200.0,
        mean_lateral=-1.0,
        ellipse_major=float("nan"),
        ellipse_minor=1.0,
        ellipse_angle_rad=0.0,
        area_95=1.0,
        radial_rmse=1.0,
        radial_p50=1.0,
        radial_p90=1.0,
    )
    serialized = _dispersion_result_to_dict("All Shots", result)
    assert serialized["group"] == "All Shots"
    assert serialized["sample_count"] == 3
    assert serialized["ellipse_major"] is None
    assert serialized["ellipse_minor"] == pytest.approx(1.0)
    assert math.isnan(result.ellipse_major)


def test_dispersion_v2_invalid_group_column_is_422(client: TestClient) -> None:
    """An unknown ``group_column`` literal is a schema-validation 422."""
    response = client.post(
        "/tools/launch-monitor-analytics/v2/dispersion",
        json={"records": _dispersion_records(), "group_column": "player"},
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("group_column" in tuple(error["loc"]) for error in errors)


def test_dispersion_v2_fewer_than_three_records_is_422(client: TestClient) -> None:
    """``records`` must hold at least three rows, like the trend route."""
    records = _dispersion_records()[:2]
    response = client.post(
        "/tools/launch-monitor-analytics/v2/dispersion",
        json={"records": records},
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("records" in tuple(error["loc"]) for error in errors)
