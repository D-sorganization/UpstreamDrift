"""Tests for the ``POST /v2/treatment`` route.

Mirrors the PyQt Data Treatment tab (``src/tools/launch_monitor_analytics/gui.py``
``_read_treatment_config`` / ``_compute_treatment`` / ``_filter_rules``) so the
API and desktop paths share one implementation: :func:`apply_treatment`.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.launch_monitor_analytics import router

pytestmark = pytest.mark.unit

_URL_TREATMENT = "/tools/launch-monitor-analytics/v2/treatment"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _treatment_records() -> list[dict[str, object]]:
    """Ten synthetic shots: one missing metric, one speed outlier."""
    records: list[dict[str, object]] = [
        {
            "shot_id": index,
            "club": "Driver" if index % 2 == 0 else "Iron",
            "club_speed": 90.0 + index,
            "ball_speed": 135.0 + index,
        }
        for index in range(10)
    ]
    records[3]["club_speed"] = None
    records[7]["ball_speed"] = 400.0
    return records


def _direct_treatment(records: list[dict[str, object]], **kwargs: Any) -> Any:
    import pandas as pd

    from src.tools.launch_monitor_model import TreatmentConfig, apply_treatment

    return apply_treatment(
        pd.DataFrame.from_records(records), TreatmentConfig(**kwargs)
    )


def _assert_data_and_flags_match_direct(payload: dict[str, Any], direct: Any) -> None:
    direct_data = direct.data.to_dict(orient="records")
    assert len(payload["data"]) == len(direct_data) == payload["shot_count"]
    for actual, expected in zip(payload["data"], direct_data, strict=True):
        for key, value in expected.items():
            if isinstance(value, float):
                if math.isnan(value):
                    assert actual[key] is None
                else:
                    assert actual[key] == pytest.approx(value)
            else:
                assert actual[key] == value

    direct_flags = direct.flags.to_dict(orient="records")
    assert len(payload["flags"]) == len(direct_flags) == payload["flag_count"]
    for actual, expected in zip(payload["flags"], direct_flags, strict=True):
        assert actual["row_index"] == expected["row_index"]
        assert actual["flag_type"] == expected["flag_type"]
        assert actual["metric"] == expected["metric"]


def _assert_audit_log_matches_direct(
    payload_audit: list[dict[str, Any]], direct_audit: tuple[dict[str, object], ...]
) -> None:
    assert len(payload_audit) == len(direct_audit)
    for actual, expected in zip(payload_audit, direct_audit, strict=True):
        assert actual["action"] == expected["action"]
        for key, value in expected.items():
            if key == "row_index":
                assert int(actual[key]) == int(value)  # type: ignore[arg-type]
            elif isinstance(value, float):
                assert actual[key] == pytest.approx(value)
            else:
                assert actual[key] == value


def test_treatment_v2_required_metrics_matches_direct_call(client: TestClient) -> None:
    """Required-metrics-only inputs reproduce the direct :func:`apply_treatment` call."""
    records = _treatment_records()
    response = client.post(
        _URL_TREATMENT,
        json={"records": records, "required_metrics": ["club_speed", "ball_speed"]},
    )
    assert response.status_code == 200
    payload = response.json()

    direct = _direct_treatment(records, required_metrics=("club_speed", "ball_speed"))
    _assert_data_and_flags_match_direct(payload, direct)
    _assert_audit_log_matches_direct(payload["audit_log"], direct.audit_log)


@pytest.mark.parametrize("exclude_flagged", [False, True])
def test_treatment_v2_outlier_screening_matches_direct_call(
    client: TestClient, exclude_flagged: bool
) -> None:
    """Outlier screening, with and without exclusion, reproduces the direct call."""
    records = _treatment_records()
    response = client.post(
        _URL_TREATMENT,
        json={
            "records": records,
            "outlier_metrics": ["ball_speed"],
            "exclude_flagged": exclude_flagged,
        },
    )
    assert response.status_code == 200
    payload = response.json()

    direct = _direct_treatment(
        records, outlier_metrics=("ball_speed",), exclude_flagged=exclude_flagged
    )
    _assert_data_and_flags_match_direct(payload, direct)
    _assert_audit_log_matches_direct(payload["audit_log"], direct.audit_log)
    if exclude_flagged:
        assert payload["shot_count"] < len(records)
    else:
        assert payload["shot_count"] == len(records)


def test_treatment_v2_filter_rule_matches_direct_call(client: TestClient) -> None:
    """A structured filter rule reproduces the direct :func:`apply_treatment` call."""
    records = _treatment_records()
    response = client.post(
        _URL_TREATMENT,
        json={
            "records": records,
            "filters": [{"column": "club", "operator": "eq", "value": "Driver"}],
        },
    )
    assert response.status_code == 200
    payload = response.json()

    from src.tools.launch_monitor_model import FilterRule

    direct = _direct_treatment(records, filters=(FilterRule("club", "eq", "Driver"),))
    _assert_data_and_flags_match_direct(payload, direct)
    _assert_audit_log_matches_direct(payload["audit_log"], direct.audit_log)
    assert payload["shot_count"] == 5


def test_treatment_v2_blank_metric_entries_behave_like_desktop_comma_parsing(
    client: TestClient,
) -> None:
    """Whitespace/blank list entries are dropped, like the desktop's comma-split text."""
    records = _treatment_records()
    response = client.post(
        _URL_TREATMENT,
        json={"records": records, "required_metrics": [" club_speed ", ""]},
    )
    assert response.status_code == 200
    payload = response.json()

    direct = _direct_treatment(records, required_metrics=("club_speed",))
    assert payload["flag_count"] == len(direct.flags)
    assert payload["shot_count"] == len(direct.data)


def test_treatment_v2_nan_values_serialize_as_null(client: TestClient) -> None:
    """A missing metric cell is JSON ``null``, never ``0``, in the data view."""
    records = _treatment_records()
    response = client.post(_URL_TREATMENT, json={"records": records})
    assert response.status_code == 200
    payload = response.json()

    row = next(item for item in payload["data"] if item["shot_id"] == 3)
    assert row["club_speed"] is None


def test_frame_to_records_timestamps_are_iso_and_nat_is_null() -> None:
    """A timestamp is its ISO string; a missing timestamp (NaT) is ``null``."""
    import pandas as pd

    from src.api.routes.launch_monitor_analytics import _frame_to_records

    frame = pd.DataFrame({"captured_at": pd.to_datetime(["2026-01-02T03:04:05", None])})
    assert _frame_to_records(frame) == [
        {"captured_at": "2026-01-02T03:04:05"},
        {"captured_at": None},
    ]


def test_treatment_v2_unknown_required_metric_is_400(client: TestClient) -> None:
    """A required metric absent from the records maps to 400 via handle_api_errors."""
    response = client.post(
        _URL_TREATMENT,
        json={"records": _treatment_records(), "required_metrics": ["launch_angle"]},
    )
    assert response.status_code == 400
    assert "launch_angle" in str(response.json())


def test_treatment_v2_unknown_outlier_metric_is_400(client: TestClient) -> None:
    """An outlier metric absent from the records maps to 400 via handle_api_errors."""
    response = client.post(
        _URL_TREATMENT,
        json={"records": _treatment_records(), "outlier_metrics": ["launch_angle"]},
    )
    assert response.status_code == 400
    assert "launch_angle" in str(response.json())


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"records": [{"club_speed": 1.0}] * 2}, "records"),
        ({"robust_z_threshold": 0.5}, "robust_z_threshold"),
        ({"robust_z_threshold": 25}, "robust_z_threshold"),
        (
            {"filters": [{"column": "club", "operator": "regex", "value": "x"}]},
            "operator",
        ),
        ({"filters": [{"column": "", "operator": "eq", "value": "x"}]}, "column"),
    ],
)
def test_treatment_v2_schema_violations_are_422(
    client: TestClient, override: dict[str, object], field: str
) -> None:
    """Inputs outside the desktop widgets' domain are schema-validation 422s."""
    body: dict[str, object] = {"records": _treatment_records(), **override}
    response = client.post(_URL_TREATMENT, json=body)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(field in tuple(error["loc"]) for error in errors)


def test_apply_treatment_does_not_mutate_raw_records() -> None:
    """The core guarantees raw input records are never mutated (frame.copy(deep=True))."""
    import pandas as pd

    from src.tools.launch_monitor_model import TreatmentConfig, apply_treatment

    records = _treatment_records()
    snapshot = copy.deepcopy(records)
    frame = pd.DataFrame.from_records(records)
    apply_treatment(frame, TreatmentConfig(required_metrics=("club_speed",)))
    assert records == snapshot
