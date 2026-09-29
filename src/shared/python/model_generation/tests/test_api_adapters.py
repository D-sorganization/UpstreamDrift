"""Regression tests for split model_generation API adapter modules."""

from __future__ import annotations

import asyncio
import inspect
import sys
import types
from typing import Any

import pytest


class _FakeUploadFile:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _FakeFlaskRequest:
    def __init__(self) -> None:
        self.method = "GET"
        self.path = "/api/v1/widgets/alpha"
        self.args: dict[str, str] = {}
        self.files: dict[str, _FakeUploadFile] = {}
        self.headers: dict[str, str] = {}
        self._json_body: dict[str, Any] | None = None

    def get_json(self, silent: bool = True) -> dict[str, Any] | None:
        return self._json_body


class _FakeFlaskResponse:
    def __init__(self, body: Any) -> None:
        self.body = body
        self.headers: dict[str, str] = {}
        self.status_code = 200
        self.content_type = "application/json"


class _FakeFlaskApp:
    def __init__(self) -> None:
        self.rules: list[dict[str, Any]] = []

    def add_url_rule(
        self,
        rule: str,
        endpoint: str,
        view_func: Any,
        methods: list[str],
    ) -> None:
        self.rules.append(
            {
                "rule": rule,
                "endpoint": endpoint,
                "view_func": view_func,
                "methods": methods,
            }
        )


class _FakeAPI:
    def __init__(
        self,
        *,
        route: Any,
        response_factory: Any,
    ) -> None:
        self._route = route
        self._response_factory = response_factory
        self.requests: list[Any] = []

    def get_routes(self) -> list[Any]:
        return [self._route]

    def handle_request(self, request: Any) -> Any:
        self.requests.append(request)
        return self._response_factory(request)


def test_split_modules_preserve_public_import_compatibility() -> None:
    """Split modules should preserve the public model_generation.api exports."""
    from model_generation.api import FastAPIAdapter, FlaskAdapter, ModelGenerationAPI
    from model_generation.api.rest_api import (
        FastAPIAdapter as ShimFastAPIAdapter,
    )
    from model_generation.api.rest_api import (
        FlaskAdapter as ShimFlaskAdapter,
    )
    from model_generation.api.rest_api import (
        ModelGenerationAPI as ShimModelGenerationAPI,
    )
    from model_generation.api.rest_api_core import ModelGenerationAPI as CoreAPI
    from model_generation.api.rest_api_fastapi import (
        FastAPIAdapter as CoreFastAPIAdapter,
    )
    from model_generation.api.rest_api_flask import FlaskAdapter as CoreFlaskAdapter

    assert ModelGenerationAPI is CoreAPI
    assert ShimModelGenerationAPI is CoreAPI
    assert FastAPIAdapter is CoreFastAPIAdapter
    assert ShimFastAPIAdapter is CoreFastAPIAdapter
    assert FlaskAdapter is CoreFlaskAdapter
    assert ShimFlaskAdapter is CoreFlaskAdapter


def test_flask_adapter_registers_routes_and_translates_requests(
    monkeypatch: Any,
) -> None:
    """FlaskAdapter should convert Flask requests into APIRequest objects."""
    from model_generation.api import APIResponse, HTTPMethod, Route

    fake_request = _FakeFlaskRequest()

    fake_flask = types.ModuleType("flask")
    fake_flask.request = fake_request  # type: ignore[attr-defined]
    fake_flask.jsonify = lambda body: body  # type: ignore[attr-defined]
    fake_flask.make_response = (  # type: ignore[attr-defined]
        lambda body: _FakeFlaskResponse(body)
    )
    monkeypatch.setitem(sys.modules, "flask", fake_flask)

    from model_generation.api.rest_api_flask import FlaskAdapter

    route = Route(
        method=HTTPMethod.POST,
        path="/api/v1/widgets/{widget_id}",
        handler=lambda request: APIResponse.ok({"unused": True}),
        description="Create widget",
        tags=["widgets"],
    )
    api = _FakeAPI(
        route=route,
        response_factory=lambda request: APIResponse.ok(
            {"widget": request.query_params["widget_id"]}
        ),
    )
    app = _FakeFlaskApp()

    adapter = FlaskAdapter(api)
    adapter.register(app)

    assert app.rules[0]["rule"] == "/api/v1/widgets/<widget_id>"
    assert app.rules[0]["methods"] == ["POST"]

    fake_request.method = "POST"
    fake_request.path = "/api/v1/widgets/alpha"
    fake_request.args = {"source": "query"}
    fake_request.files = {"upload": _FakeUploadFile(b"payload")}
    fake_request.headers = {"X-Test": "1"}
    fake_request._json_body = {"name": "alpha"}

    response = app.rules[0]["view_func"](widget_id="alpha")

    captured_request = api.requests[0]
    assert captured_request.method is HTTPMethod.POST
    assert captured_request.path == "/api/v1/widgets/alpha"
    assert captured_request.query_params == {"source": "query", "widget_id": "alpha"}
    assert captured_request.body == {"name": "alpha"}
    assert captured_request.files == {"upload": b"payload"}
    assert captured_request.headers == {"X-Test": "1"}
    assert response.body == {"widget": "alpha"}
    assert response.status_code == 200


pytest.importorskip("fastapi")


class _MultiRouteFakeAPI:
    """Records every translated APIRequest and answers via a response factory."""

    def __init__(self, *, routes: Any, response_factory: Any) -> None:
        self._routes: list[Any] = routes
        self._response_factory = response_factory
        self.requests: list[Any] = []

    def get_routes(self) -> list[Any]:
        return self._routes

    def handle_request(self, request: Any) -> Any:
        self.requests.append(request)
        return self._response_factory(request)


def _build_test_client(*, routes: Any, response_factory: Any) -> tuple[Any, Any]:
    """Register *routes* on a real FastAPI app and return (client, api)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from model_generation.api.rest_api_fastapi import FastAPIAdapter

    api = _MultiRouteFakeAPI(routes=routes, response_factory=response_factory)
    app = FastAPI()
    FastAPIAdapter(api).register(app)
    return TestClient(app), api


def test_fastapi_endpoint_is_callable_not_coroutine() -> None:
    """register() must hand FastAPI a callable async endpoint, not a coroutine."""
    from model_generation.api import APIResponse, HTTPMethod, Route

    captured: dict[str, Any] = {}

    class _CapturingApp:
        def add_api_route(self, path: str, endpoint: Any, **kwargs: Any) -> None:
            captured["endpoint"] = endpoint

    from model_generation.api.rest_api_fastapi import FastAPIAdapter

    route = Route(
        method=HTTPMethod.GET,
        path="/library/models/{model_id}",
        handler=lambda request: APIResponse.ok({}),
    )
    api = _MultiRouteFakeAPI(
        routes=[route],
        response_factory=lambda request: APIResponse.ok({}),
    )
    FastAPIAdapter(api).register(_CapturingApp())

    endpoint = captured["endpoint"]
    # Must be the inner async function itself, never an un-awaited coroutine.
    assert inspect.iscoroutinefunction(endpoint)
    assert not asyncio.iscoroutine(endpoint)


def test_fastapi_get_binds_path_param_via_testclient() -> None:
    """A real GET with a {path} param must bind and reach handle_request (no 422)."""
    from model_generation.api import APIResponse, HTTPMethod, Route

    route = Route(
        method=HTTPMethod.GET,
        path="/library/models/{model_id}",
        handler=lambda request: APIResponse.ok({}),
        description="Get model",
        tags=["library"],
    )
    client, api = _build_test_client(
        routes=[route],
        response_factory=lambda request: APIResponse.ok(
            {
                "model_id": request.query_params.get("model_id"),
                "source": request.query_params.get("source"),
            }
        ),
    )

    response = client.get("/library/models/alpha?source=query")

    assert response.status_code == 200
    assert response.json() == {"model_id": "alpha", "source": "query"}
    captured = api.requests[0]
    assert captured.method is HTTPMethod.GET
    assert captured.path == "/library/models/alpha"
    # The {model_id} path param reached APIRequest.query_params.
    assert captured.query_params["model_id"] == "alpha"
    assert captured.query_params["source"] == "query"


def test_fastapi_post_with_json_body_via_testclient() -> None:
    """A real POST must forward the JSON body and bind the path param."""
    from model_generation.api import APIResponse, HTTPMethod, Route

    route = Route(
        method=HTTPMethod.POST,
        path="/library/models/{model_id}",
        handler=lambda request: APIResponse.created({}),
        description="Add model",
        tags=["library"],
    )
    client, api = _build_test_client(
        routes=[route],
        response_factory=lambda request: APIResponse.created(
            {"model_id": request.query_params.get("model_id"), "body": request.body}
        ),
    )

    response = client.post("/library/models/beta", json={"name": "beta"})

    assert response.status_code == 201
    assert response.json() == {"model_id": "beta", "body": {"name": "beta"}}
    captured = api.requests[0]
    assert captured.method is HTTPMethod.POST
    assert captured.body == {"name": "beta"}
    assert captured.query_params["model_id"] == "beta"


def test_fastapi_file_download_returns_bytes_via_testclient() -> None:
    """A file-download route must return raw bytes with the declared content type."""
    from model_generation.api import APIResponse, HTTPMethod, Route

    route = Route(
        method=HTTPMethod.GET,
        path="/library/models/{model_id}/download",
        handler=lambda request: APIResponse.ok({}),
        description="Download model",
        tags=["library"],
    )
    client, api = _build_test_client(
        routes=[route],
        response_factory=lambda request: APIResponse.file(
            b"<robot />",
            "model.urdf",
            content_type="application/xml",
        ),
    )

    response = client.get("/library/models/gamma/download")

    assert response.status_code == 200
    assert response.content == b"<robot />"
    assert response.headers["content-type"] == "application/xml"
    assert (
        response.headers["content-disposition"] == 'attachment; filename="model.urdf"'
    )
    assert api.requests[0].query_params["model_id"] == "gamma"
