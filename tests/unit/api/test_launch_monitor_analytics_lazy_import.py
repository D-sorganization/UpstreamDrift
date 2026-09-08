"""Regression tests for issue #8943: launch-monitor analytics lazy pandas.

The route registry imports every routes module at API startup, so a
top-level ``import pandas`` made every API boot pay the pandas import
cost (~0.4-0.8 s, ~100 MB). Following the cv2/mediapipe deferral pattern
documented in ``route_registry.py`` (and locked in for ``video`` by
``test_video_route_lazy_import.py``), the module must import successfully
even when pandas is absent, deferring the import into its handlers.
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import pytest
from fastapi import APIRouter


def test_launch_monitor_analytics_imports_without_pandas() -> None:
    """Route module must not require pandas at import time (issue #8943)."""
    module_name = "src.api.routes.launch_monitor_analytics"
    previously_loaded = sys.modules.pop(module_name, None)
    try:
        with mock.patch.dict(sys.modules, {"pandas": None}):
            module = importlib.import_module(module_name)
            assert isinstance(module.router, APIRouter)
    finally:
        if previously_loaded is not None:
            sys.modules[module_name] = previously_loaded
        else:
            sys.modules.pop(module_name, None)
            importlib.import_module(module_name)


@pytest.mark.integration
def test_analyze_endpoint_still_serves_after_lazy_import() -> None:
    """Deferring the pandas import must not change the /analyze contract."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routes.launch_monitor_analytics import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    records = [
        {
            "shot_id": f"shot-{index}",
            "session_id": "api-session",
            "monitor_vendor": "FlightScope",
            "club_speed": float(index),
            "ball_speed": 1.5 * index + 2.0,
        }
        for index in range(1, 31)
    ]
    response = client.post(
        "/tools/launch-monitor-analytics/analyze",
        json={
            "records": records,
            "analysis": {"outcome": "ball_speed", "predictors": ["club_speed"]},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["contract_version"]
