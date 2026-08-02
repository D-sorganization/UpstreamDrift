"""Endpoint inventory regression tests for the calculation backend."""

from __future__ import annotations

import importlib

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_calc_backend_import_aliases_share_app_instance() -> None:
    shared_app_module = importlib.import_module("shared.python.calc_backend.app")
    top_level_app_module = importlib.import_module("calc_backend.app")

    assert top_level_app_module is shared_app_module
    assert top_level_app_module.app is shared_app_module.app


def test_list_endpoints_repairs_request_app_not_module_global() -> None:
    from calc_backend.app import _registered_calculator_route_signatures, list_endpoints

    isolated_app = FastAPI()
    isolated_app.add_api_route(
        "/api/calc/endpoints",
        list_endpoints,
        methods=["GET"],
    )
    isolated_client = TestClient(isolated_app)

    response = isolated_client.get("/api/calc/endpoints")

    assert response.status_code == 200
    assert "POST /api/calc/flare" in response.json()["calculators"]
    assert ("POST", "/api/calc/flare") in _registered_calculator_route_signatures(
        isolated_app
    )


def test_calculator_route_signatures_apply_router_prefix() -> None:
    from calc_backend.app import _calculator_route_signatures

    class PrefixlessRoute:
        path = ""
        methods = {"POST"}

    assert _calculator_route_signatures(
        [PrefixlessRoute()],
        prefix="/api/calc/flare",
    ) == {("POST", "/api/calc/flare")}

    class PrefixedRoute:
        path = "/api/calc/flare"
        methods = {"POST"}

    assert _calculator_route_signatures(
        [PrefixedRoute()],
        prefix="/api/calc/flare",
    ) == {("POST", "/api/calc/flare")}

    class PathFormatRoute:
        path_format = "/api/calc/flare"
        methods = {"POST"}

    assert _calculator_route_signatures(
        [PathFormatRoute()],
        prefix="/api/calc/flare",
    ) == {("POST", "/api/calc/flare")}
