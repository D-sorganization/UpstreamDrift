"""Route-handler tests for the model_generation REST API.

Covers issue #7002:
- health_check / get_api_info response shape
- conversion handlers (mjcf<->urdf) happy + error (422) paths
- validate / parse handlers happy + error paths
- inertia/calculate validation errors
- library handlers (missing id, not implemented, not found)
- route registration count + 404 for unknown route
- FastAPI adapter registration (guarded by importorskip)

Requests are driven through ``ModelGenerationAPI.handle_request`` which performs
the real route matching, handler dispatch, security-header injection, and
status-code mapping used by every framework adapter.
"""

from __future__ import annotations

from typing import Any

import pytest
from model_generation.api.rest_api_routes import ModelGenerationAPI
from model_generation.api.rest_api_types import APIRequest, APIResponse, HTTPMethod

SIMPLE_URDF = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""

SIMPLE_MJCF = """<mujoco model="m">
  <worldbody>
    <body name="base" pos="0 0 0">
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def api() -> ModelGenerationAPI:
    return ModelGenerationAPI()


def _request(
    method: HTTPMethod,
    path: str,
    body: dict[str, Any] | None = None,
    query: dict[str, str] | None = None,
) -> APIRequest:
    return APIRequest(
        method=method,
        path=f"/api/v1{path}",
        body=body,
        query_params=query or {},
    )


def _json(response: APIResponse) -> dict[str, Any]:
    assert isinstance(response.body, dict), f"expected dict, got {type(response.body)}"
    return response.body


class TestHealthAndInfo:
    def test_health_check_shape(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(_request(HTTPMethod.GET, "/health"))
        assert resp.status_code == 200
        body = _json(resp)
        assert body["status"] == "healthy"
        assert body["service"] == "model_generation"

    def test_info_shape(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(_request(HTTPMethod.GET, "/info"))
        assert resp.status_code == 200
        body = _json(resp)
        assert body["name"] == "Model Generation API"
        assert isinstance(body["endpoints"], list)
        assert len(body["endpoints"]) == len(api.get_routes())

    def test_security_headers_present(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(_request(HTTPMethod.GET, "/health"))
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"


class TestRouteRegistration:
    def test_routes_registered(self, api: ModelGenerationAPI) -> None:
        routes = api.get_routes()
        # Core + inertia/library + editor groups all register.
        assert len(routes) >= 18
        paths = {r.path for r in routes}
        assert "/api/v1/health" in paths
        assert "/api/v1/convert/mjcf-to-urdf" in paths
        assert "/api/v1/inertia/calculate" in paths

    def test_unknown_route_returns_404(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(_request(HTTPMethod.GET, "/nope"))
        assert resp.status_code == 404
        assert "error" in _json(resp)


class TestConversionHandlers:
    def test_mjcf_to_urdf_happy(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(
            _request(HTTPMethod.POST, "/convert/mjcf-to-urdf", {"content": SIMPLE_MJCF})
        )
        assert resp.status_code == 200
        assert "urdf" in _json(resp)

    def test_mjcf_to_urdf_missing_content(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(
            _request(HTTPMethod.POST, "/convert/mjcf-to-urdf", {})
        )
        assert resp.status_code == 400
        assert "error" in _json(resp)

    def test_mjcf_to_urdf_malformed_returns_422(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(
            _request(
                HTTPMethod.POST,
                "/convert/mjcf-to-urdf",
                {"content": "<mujoco><body></mujoco>"},
            )
        )
        assert resp.status_code == 422
        assert "error" in _json(resp)

    def test_urdf_to_mjcf_happy(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(
            _request(HTTPMethod.POST, "/convert/urdf-to-mjcf", {"content": SIMPLE_URDF})
        )
        assert resp.status_code == 200
        assert "mjcf" in _json(resp)

    def test_urdf_to_mjcf_missing_content(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(
            _request(HTTPMethod.POST, "/convert/urdf-to-mjcf", {})
        )
        assert resp.status_code == 400


class TestValidateAndParse:
    def test_validate_valid_urdf(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(
            _request(HTTPMethod.POST, "/validate", {"content": SIMPLE_URDF})
        )
        assert resp.status_code == 200
        body = _json(resp)
        assert "valid" in body
        assert "error_count" in body

    def test_validate_missing_content(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(_request(HTTPMethod.POST, "/validate", {}))
        assert resp.status_code == 400

    def test_parse_happy(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(
            _request(HTTPMethod.POST, "/parse", {"content": SIMPLE_URDF})
        )
        assert resp.status_code == 200
        body = _json(resp)
        assert body["name"] == "test_robot"
        assert body["root_link"] == "base_link"

    def test_parse_missing_content(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(_request(HTTPMethod.POST, "/parse", {}))
        assert resp.status_code == 400

    def test_parse_malformed_returns_422(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(
            _request(HTTPMethod.POST, "/parse", {"content": "<robot><link></robot>"})
        )
        assert resp.status_code == 422


class TestInertiaCalculate:
    def test_box_happy(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(
            _request(
                HTTPMethod.POST,
                "/inertia/calculate",
                {"shape": "box", "mass": 2.0, "dimensions": [0.1, 0.2, 0.3]},
            )
        )
        assert resp.status_code == 200
        body = _json(resp)
        assert body["shape"] == "box"
        assert body["inertia"]["ixx"] > 0

    def test_sphere_happy(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(
            _request(
                HTTPMethod.POST,
                "/inertia/calculate",
                {"shape": "sphere", "mass": 1.0, "dimensions": [0.5]},
            )
        )
        assert resp.status_code == 200

    def test_missing_shape(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(
            _request(HTTPMethod.POST, "/inertia/calculate", {"mass": 1.0})
        )
        assert resp.status_code == 400
        assert "shape" in _json(resp)["error"].lower()

    def test_wrong_dimension_count(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(
            _request(
                HTTPMethod.POST,
                "/inertia/calculate",
                {"shape": "box", "mass": 1.0, "dimensions": [0.1]},
            )
        )
        assert resp.status_code == 400

    def test_unknown_shape(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(
            _request(
                HTTPMethod.POST,
                "/inertia/calculate",
                {"shape": "tetrahedron", "mass": 1.0, "dimensions": [1]},
            )
        )
        assert resp.status_code == 400
        assert "Unknown shape" in _json(resp)["error"]


class TestLibraryHandlers:
    def test_get_model_missing_id(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(_request(HTTPMethod.GET, "/library/models/"))
        # Empty model_id path segment -> handler reports missing id (400).
        assert resp.status_code in (400, 404)

    def test_remove_not_implemented(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(
            _request(
                HTTPMethod.DELETE,
                "/library/models/some-id",
            )
        )
        assert resp.status_code == 501

    def test_diff_missing_content(self, api: ModelGenerationAPI) -> None:
        resp = api.handle_request(
            _request(HTTPMethod.POST, "/editor/diff", {"content_a": SIMPLE_URDF})
        )
        assert resp.status_code == 400


class TestRequestPreconditions:
    def test_none_request_raises(self, api: ModelGenerationAPI) -> None:
        with pytest.raises(ValueError, match="request must be provided"):
            api.handle_request(None)  # type: ignore[arg-type]

    def test_invalid_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="prefix must be provided"):
            ModelGenerationAPI(prefix=None)  # type: ignore[arg-type]


class TestFastAPIAdapter:
    """FastAPI adapter wiring (skipped if fastapi is unavailable)."""

    def test_register_adds_routes(self, api: ModelGenerationAPI) -> None:
        pytest.importorskip("fastapi")
        from fastapi import FastAPI
        from model_generation.api.rest_api_fastapi import FastAPIAdapter

        app = FastAPI()
        before = len(app.routes)
        FastAPIAdapter(api).register(app)
        # One FastAPI route added per registered API route.
        assert len(app.routes) >= before + len(api.get_routes())

    def test_adapter_requires_api(self) -> None:
        pytest.importorskip("fastapi")
        from model_generation.api.rest_api_fastapi import FastAPIAdapter

        with pytest.raises(ValueError, match="api must be provided"):
            FastAPIAdapter(None)  # type: ignore[arg-type]
