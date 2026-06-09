"""TestClient-based tests for the FastAPI adapter request cycle.

Regression coverage for the bug where the inner handler's ``**kwargs``
catch-all made FastAPI treat ``request``/``kwargs`` as required query
parameters, so every request returned 422 instead of dispatching to
``ModelGenerationAPI.handle_request``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _client() -> object:
    """Build a TestClient with the FastAPI adapter routes registered."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from model_generation.api.rest_api_fastapi import FastAPIAdapter
    from model_generation.api.rest_api_routes import ModelGenerationAPI

    app = FastAPI()
    FastAPIAdapter(ModelGenerationAPI()).register(app)
    return TestClient(app)


def test_health_route_returns_200() -> None:
    """GET /api/v1/health dispatches and returns the healthy payload."""
    response = _client().get("/api/v1/health")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "model_generation"


def test_path_param_route_not_422() -> None:
    """GET on a ``{model_id}`` route dispatches instead of 422-ing.

    Before the fix, the handler's ``**kwargs`` catch-all made FastAPI demand
    ``request``/``kwargs`` as query fields, so this returned 422 ("Field
    required") before reaching the API. After the fix it dispatches into
    ``handle_request`` — proven here by the absence of the 422 and of the
    400 "Missing model_id" that an uncaptured path param would trigger.
    """
    client = _client()
    response = client.get("/api/v1/library/models/does-not-exist")

    assert response.status_code != 422, response.text
    # Reached the handler past its own "Missing model_id" precondition.
    assert response.status_code != 400, response.text


def test_path_param_value_is_captured() -> None:
    """The ``{model_id}`` segment is delivered to the handler as a path param.

    The DELETE handler returns 501 ("Remove not implemented") only once it
    has confirmed a non-empty ``model_id``; without the captured path param
    it would short-circuit to 400 ("Missing model_id"). The 501 therefore
    proves the path-param value flowed through the adapter.
    """
    response = _client().delete("/api/v1/library/models/some-model-id")

    assert response.status_code == 501, response.text
