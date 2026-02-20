"""
Tests for the REST API module.
"""

from __future__ import annotations

from typing import Any

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


def _body(response: Any) -> dict[str, Any]:
    """Extract response body as dict, with runtime assertion."""
    body = response.body
    assert isinstance(body, dict), f"Expected dict body, got {type(body)}"
    return body


class TestAPIClasses:
    """Tests for API data classes."""

    def test_api_request_creation(self) -> None:
        """Test APIRequest creation."""
        from model_generation.api import APIRequest, HTTPMethod

        request = APIRequest(
            method=HTTPMethod.GET,
            path="/api/v1/health",
            query_params={"key": "value"},
        )

        assert request.method == HTTPMethod.GET
        assert request.path == "/api/v1/health"
        assert request.query_params["key"] == "value"

    def test_api_response_ok(self) -> None:
        """Test APIResponse.ok factory."""
        from model_generation.api import APIResponse

        response = APIResponse.ok({"status": "healthy"})

        assert response.status_code == 200
        assert _body(response)["status"] == "healthy"

    def test_api_response_error(self) -> None:
        """Test APIResponse.error factory."""
        from model_generation.api import APIResponse

        response = APIResponse.error("Something went wrong", 400)

        assert response.status_code == 400
        assert "error" in _body(response)

    def test_api_response_not_found(self) -> None:
        """Test APIResponse.not_found factory."""
        from model_generation.api import APIResponse

        response = APIResponse.not_found("Resource not found")

        assert response.status_code == 404

    def test_api_response_file(self) -> None:
        """Test APIResponse.file factory."""
        from model_generation.api import APIResponse

        response = APIResponse.file("<robot/>", "robot.urdf", "application/xml")

        assert response.status_code == 200
        assert response.content_type == "application/xml"
        assert "Content-Disposition" in response.headers


class TestModelGenerationAPI:
    """Tests for ModelGenerationAPI class."""

    def test_api_creation(self) -> None:
        """Test API instantiation."""
        from model_generation.api import ModelGenerationAPI

        api = ModelGenerationAPI()
        assert api is not None

    def test_api_routes_registered(self) -> None:
        """Test routes are registered."""
        from model_generation.api import ModelGenerationAPI

        api = ModelGenerationAPI()
        routes = api.get_routes()

        assert len(routes) > 0
        paths = [r.path for r in routes]
        assert any("/health" in p for p in paths)
        assert any("/generate" in p for p in paths)
        assert any("/validate" in p for p in paths)

    def test_health_endpoint(self) -> None:
        """Test health check endpoint."""
        from model_generation.api import APIRequest, HTTPMethod, ModelGenerationAPI

        api = ModelGenerationAPI()
        request = APIRequest(
            method=HTTPMethod.GET,
            path="/api/v1/health",
        )

        response = api.handle_request(request)

        assert response.status_code == 200
        assert _body(response)["status"] == "healthy"

    def test_info_endpoint(self) -> None:
        """Test API info endpoint."""
        from model_generation.api import APIRequest, HTTPMethod, ModelGenerationAPI

        api = ModelGenerationAPI()
        request = APIRequest(
            method=HTTPMethod.GET,
            path="/api/v1/info",
        )

        response = api.handle_request(request)

        assert response.status_code == 200
        assert "name" in _body(response)
        assert "endpoints" in _body(response)

    def test_generate_humanoid_endpoint(self) -> None:
        """Test humanoid generation endpoint."""
        from model_generation.api import APIRequest, HTTPMethod, ModelGenerationAPI

        api = ModelGenerationAPI()
        request = APIRequest(
            method=HTTPMethod.POST,
            path="/api/v1/generate/humanoid",
            body={
                "name": "test_humanoid",
                "height": 1.7,
                "mass": 70.0,
            },
        )

        response = api.handle_request(request)

        assert response.status_code == 200
        assert "urdf" in _body(response)
        assert "links" in _body(response)
        assert _body(response)["robot_name"] == "test_humanoid"

    def test_validate_endpoint_valid_urdf(self) -> None:
        """Test validation endpoint with valid URDF."""
        from model_generation.api import APIRequest, HTTPMethod, ModelGenerationAPI

        api = ModelGenerationAPI()
        request = APIRequest(
            method=HTTPMethod.POST,
            path="/api/v1/validate",
            body={"content": SIMPLE_URDF},
        )

        response = api.handle_request(request)

        assert response.status_code == 200
        assert _body(response)["valid"] is True

    def test_validate_endpoint_invalid_urdf(self) -> None:
        """Test validation endpoint with invalid URDF."""
        from model_generation.api import APIRequest, HTTPMethod, ModelGenerationAPI

        api = ModelGenerationAPI()
        request = APIRequest(
            method=HTTPMethod.POST,
            path="/api/v1/validate",
            body={"content": "<robot><invalid></robot>"},
        )

        response = api.handle_request(request)

        assert response.status_code == 200
        assert _body(response)["valid"] is False
        assert _body(response)["error_count"] > 0

    def test_parse_endpoint(self) -> None:
        """Test URDF parsing endpoint."""
        from model_generation.api import APIRequest, HTTPMethod, ModelGenerationAPI

        api = ModelGenerationAPI()
        request = APIRequest(
            method=HTTPMethod.POST,
            path="/api/v1/parse",
            body={"content": SIMPLE_URDF},
        )

        response = api.handle_request(request)

        assert response.status_code == 200
        assert _body(response)["name"] == "test_robot"
        assert "links" in _body(response)
        assert "joints" in _body(response)

    def test_inertia_calculation_endpoint(self) -> None:
        """Test inertia calculation endpoint."""
        from model_generation.api import APIRequest, HTTPMethod, ModelGenerationAPI

        api = ModelGenerationAPI()
        request = APIRequest(
            method=HTTPMethod.POST,
            path="/api/v1/inertia/calculate",
            body={
                "shape": "box",
                "mass": 1.0,
                "dimensions": [0.1, 0.2, 0.3],
            },
        )

        response = api.handle_request(request)

        assert response.status_code == 200
        assert "inertia" in _body(response)
        assert "ixx" in _body(response)["inertia"]
        assert _body(response)["is_positive_definite"] is True

    def test_inertia_sphere(self) -> None:
        """Test inertia calculation for sphere."""
        from model_generation.api import APIRequest, HTTPMethod, ModelGenerationAPI

        api = ModelGenerationAPI()
        request = APIRequest(
            method=HTTPMethod.POST,
            path="/api/v1/inertia/calculate",
            body={
                "shape": "sphere",
                "mass": 1.0,
                "dimensions": [0.1],
            },
        )

        response = api.handle_request(request)

        assert response.status_code == 200
        # Sphere should have equal ixx, iyy, izz
        inertia = _body(response)["inertia"]
        assert abs(inertia["ixx"] - inertia["iyy"]) < 1e-10
        assert abs(inertia["iyy"] - inertia["izz"]) < 1e-10

    def test_inertia_missing_shape(self) -> None:
        """Test inertia calculation with missing shape."""
        from model_generation.api import APIRequest, HTTPMethod, ModelGenerationAPI

        api = ModelGenerationAPI()
        request = APIRequest(
            method=HTTPMethod.POST,
            path="/api/v1/inertia/calculate",
            body={"mass": 1.0, "dimensions": [0.1]},
        )

        response = api.handle_request(request)

        assert response.status_code == 400
        assert "error" in _body(response)

    def test_library_list_endpoint(self) -> None:
        """Test library listing endpoint."""
        from model_generation.api import APIRequest, HTTPMethod, ModelGenerationAPI

        api = ModelGenerationAPI()
        request = APIRequest(
            method=HTTPMethod.GET,
            path="/api/v1/library/models",
        )

        response = api.handle_request(request)

        assert response.status_code == 200
        assert "models" in _body(response)
        assert "count" in _body(response)

    def test_diff_endpoint(self) -> None:
        """Test diff endpoint."""
        from model_generation.api import APIRequest, HTTPMethod, ModelGenerationAPI

        api = ModelGenerationAPI()

        urdf_v1 = SIMPLE_URDF
        urdf_v2 = SIMPLE_URDF.replace("test_robot", "modified_robot")

        request = APIRequest(
            method=HTTPMethod.POST,
            path="/api/v1/editor/diff",
            body={
                "content_a": urdf_v1,
                "content_b": urdf_v2,
            },
        )

        response = api.handle_request(request)

        assert response.status_code == 200
        assert _body(response)["has_changes"] is True
        assert "unified_diff" in _body(response)

    def test_not_found_route(self) -> None:
        """Test handling of non-existent route."""
        from model_generation.api import APIRequest, HTTPMethod, ModelGenerationAPI

        api = ModelGenerationAPI()
        request = APIRequest(
            method=HTTPMethod.GET,
            path="/api/v1/nonexistent",
        )

        response = api.handle_request(request)

        assert response.status_code == 404

    def test_generate_from_params_endpoint(self) -> None:
        """Test generation from detailed parameters."""
        from model_generation.api import APIRequest, HTTPMethod, ModelGenerationAPI

        api = ModelGenerationAPI()
        request = APIRequest(
            method=HTTPMethod.POST,
            path="/api/v1/generate/from-params",
            body={
                "name": "custom_robot",
                "links": [
                    {
                        "name": "base_link",
                        "inertia": {
                            "mass": 1.0,
                            "ixx": 0.1,
                            "iyy": 0.1,
                            "izz": 0.1,
                        },
                    },
                    {
                        "name": "link1",
                        "inertia": {
                            "mass": 0.5,
                            "ixx": 0.05,
                            "iyy": 0.05,
                            "izz": 0.05,
                        },
                    },
                ],
                "joints": [
                    {
                        "name": "joint1",
                        "type": "revolute",
                        "parent": "base_link",
                        "child": "link1",
                        "axis": [0, 0, 1],
                    }
                ],
            },
        )

        response = api.handle_request(request)

        assert response.status_code == 200
        assert _body(response)["robot_name"] == "custom_robot"
        assert _body(response)["links"] == 2
        assert _body(response)["joints"] == 1


class TestRoute:
    """Tests for Route class."""

    def test_route_creation(self) -> None:
        """Test Route creation."""
        from model_generation.api import HTTPMethod, Route

        def dummy_handler(request) -> None:
            return None

        route = Route(
            method=HTTPMethod.GET,
            path="/test",
            handler=dummy_handler,
            description="Test route",
            tags=["test"],
        )

        assert route.method == HTTPMethod.GET
        assert route.path == "/test"
        assert route.description == "Test route"
        assert "test" in route.tags


class TestHTTPMethod:
    """Tests for HTTPMethod enum."""

    def test_http_methods(self) -> None:
        """Test HTTP method values."""
        from model_generation.api import HTTPMethod

        assert HTTPMethod.GET.value == "GET"
        assert HTTPMethod.POST.value == "POST"
        assert HTTPMethod.PUT.value == "PUT"
        assert HTTPMethod.DELETE.value == "DELETE"
        assert HTTPMethod.PATCH.value == "PATCH"
