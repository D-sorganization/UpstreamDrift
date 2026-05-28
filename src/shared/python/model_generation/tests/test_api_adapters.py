"""Regression tests for split model_generation API adapter modules."""

from __future__ import annotations

import asyncio
import inspect
import sys
import types
from typing import Any


class _FakeUploadFile:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _AsyncFakeUploadFile:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def read(self) -> bytes:
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


class _FakeURL:
    def __init__(self, path: str) -> None:
        self.path = path


class _FakeFastAPIRequest:
    def __init__(
        self,
        *,
        method: str,
        path: str,
        query_params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        body: dict[str, Any] | Exception | None = None,
        form_data: dict[str, Any] | None = None,
    ) -> None:
        self.method = method
        self.url = _FakeURL(path)
        self.query_params = query_params or {}
        self.headers = headers or {}
        self._body = body
        self._form_data = form_data or {}

    async def json(self) -> dict[str, Any] | None:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    async def form(self) -> dict[str, Any]:
        return self._form_data


class _FakeFastAPIResponse:
    def __init__(
        self,
        *,
        content: Any,
        status_code: int,
        media_type: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.content = content
        self.status_code = status_code
        self.media_type = media_type
        self.headers = headers or {}


class _FakeJSONResponse(_FakeFastAPIResponse):
    def __init__(
        self,
        *,
        content: Any,
        status_code: int,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            content=content,
            status_code=status_code,
            media_type="application/json",
            headers=headers,
        )


class _FakeFastAPIApp:
    def __init__(self) -> None:
        self.routes: list[dict[str, Any]] = []

    def add_api_route(
        self,
        path: str,
        endpoint: Any,
        *,
        methods: list[str],
        tags: list[str],
        summary: str,
    ) -> None:
        self.routes.append(
            {
                "path": path,
                "endpoint": endpoint,
                "methods": methods,
                "tags": tags,
                "summary": summary,
            }
        )


class _FakeAPI:
    def __init__(
        self,
        *,
        route,
        response_factory,
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
    monkeypatch,
) -> None:
    """FlaskAdapter should convert Flask requests into APIRequest objects."""
    from model_generation.api import APIResponse, HTTPMethod, Route

    fake_request = _FakeFlaskRequest()

    fake_flask = types.ModuleType("flask")
    fake_flask.request = fake_request  # type: ignore[attr-defined]
    fake_flask.jsonify = lambda body: body  # type: ignore[attr-defined]
    fake_flask.make_response = lambda body: _FakeFlaskResponse(body)  # type: ignore[attr-defined]
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


def test_fastapi_adapter_registers_async_handlers(monkeypatch) -> None:
    """FastAPIAdapter should register callable async handlers, not coroutine objects."""
    from model_generation.api import APIResponse, HTTPMethod, Route

    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.Request = _FakeFastAPIRequest  # type: ignore[attr-defined]
    fake_fastapi.Response = _FakeFastAPIResponse  # type: ignore[attr-defined]
    fake_fastapi_responses = types.ModuleType("fastapi.responses")
    fake_fastapi_responses.JSONResponse = _FakeJSONResponse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.responses", fake_fastapi_responses)

    from model_generation.api.rest_api_fastapi import FastAPIAdapter

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
    app = _FakeFastAPIApp()

    adapter = FastAPIAdapter(api)
    adapter.register(app)

    registered_route = app.routes[0]
    assert registered_route["path"] == "/api/v1/widgets/{widget_id}"
    assert registered_route["methods"] == ["POST"]
    assert inspect.iscoroutinefunction(registered_route["endpoint"])

    request = _FakeFastAPIRequest(
        method="POST",
        path="/api/v1/widgets/alpha",
        query_params={"source": "query"},
        headers={"X-Test": "1"},
        body={"name": "alpha"},
        form_data={"upload": _AsyncFakeUploadFile(b"payload")},
    )

    response = asyncio.run(registered_route["endpoint"](request, widget_id="alpha"))

    captured_request = api.requests[0]
    assert captured_request.method is HTTPMethod.POST
    assert captured_request.path == "/api/v1/widgets/alpha"
    assert captured_request.query_params == {"source": "query", "widget_id": "alpha"}
    assert captured_request.body == {"name": "alpha"}
    assert captured_request.files == {"upload": b"payload"}
    assert captured_request.headers == {"X-Test": "1"}
    assert response.content == {"widget": "alpha"}
    assert response.status_code == 200
    assert response.media_type == "application/json"


def test_fastapi_adapter_uses_binary_response_for_file_payloads(monkeypatch) -> None:
    """FastAPIAdapter should preserve byte payloads via the framework response type."""
    from model_generation.api import APIResponse, HTTPMethod, Route

    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.Request = _FakeFastAPIRequest  # type: ignore[attr-defined]
    fake_fastapi.Response = _FakeFastAPIResponse  # type: ignore[attr-defined]
    fake_fastapi_responses = types.ModuleType("fastapi.responses")
    fake_fastapi_responses.JSONResponse = _FakeJSONResponse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.responses", fake_fastapi_responses)

    from model_generation.api.rest_api_fastapi import FastAPIAdapter

    route = Route(
        method=HTTPMethod.GET,
        path="/api/v1/widgets/{widget_id}/download",
        handler=lambda request: APIResponse.ok({"unused": True}),
        description="Download widget",
        tags=["widgets"],
    )
    api = _FakeAPI(
        route=route,
        response_factory=lambda request: APIResponse.file(
            b"<robot />",
            "widget.urdf",
            content_type="application/xml",
        ),
    )
    app = _FakeFastAPIApp()

    FastAPIAdapter(api).register(app)

    request = _FakeFastAPIRequest(
        method="GET",
        path="/api/v1/widgets/alpha/download",
    )
    response = asyncio.run(app.routes[0]["endpoint"](request, widget_id="alpha"))

    assert response.content == b"<robot />"
    assert response.status_code == 200
    assert response.media_type == "application/xml"
    assert (
        response.headers["Content-Disposition"] == 'attachment; filename="widget.urdf"'
    )
