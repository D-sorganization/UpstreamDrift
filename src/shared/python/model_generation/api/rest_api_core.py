# ruff: noqa: E501
"""Core route registration and request dispatch for model_generation APIs."""

from __future__ import annotations

import logging
import os
import secrets
import time
from collections.abc import Callable

from src.shared.python.model_generation.api.rest_api_assets import (
    AssetLibraryEditorRoutesMixin,
)
from src.shared.python.model_generation.api.rest_api_contracts import (
    APIRequest,
    APIResponse,
    HTTPMethod,
    Route,
)
from src.shared.python.model_generation.api.rest_api_generation import (
    GenerationConversionRoutesMixin,
)


logger = logging.getLogger(__name__)


class ModelGenerationAPI(
    GenerationConversionRoutesMixin,
    AssetLibraryEditorRoutesMixin,
):
    """Framework-neutral REST API for model generation workflows."""

    def __init__(self, prefix: str = "/api/v1") -> None:
        """Initialize the route registry and security configuration.

        Security settings are read from the environment here rather than at
        request time so that a misconfigured ``MODEL_GEN_RATE_LIMIT`` is
        reported once, at construction, instead of silently disabling the
        limiter on every request.
        """
        if prefix is None:
            raise ValueError("prefix must be provided")
        self.prefix = prefix
        self._routes: list[Route] = []

        self._api_key: str | None = os.environ.get("MODEL_GEN_API_KEY")
        self._cors_origins: str = os.environ.get("MODEL_GEN_CORS_ORIGINS", "")
        self._rate_limit: int | None = None
        rate_limit_str = os.environ.get("MODEL_GEN_RATE_LIMIT")
        if rate_limit_str:
            try:
                self._rate_limit = int(rate_limit_str)
            except ValueError:
                logger.warning("Invalid MODEL_GEN_RATE_LIMIT value: %s", rate_limit_str)

        # Sliding window rate limiter: client IP -> request timestamps.
        self._rate_limit_windows: dict[str, list[float]] = {}

        self._register_routes()

    def _register_routes(self) -> None:
        """Register all supported route groups."""
        self._register_core_routes()
        self._register_inertia_and_library_routes()
        self._register_editor_routes()

    def _register_core_routes(self) -> None:
        """Register health, generation, conversion, validation, and parsing routes."""
        self.add_route(HTTPMethod.GET, "/health", self.health_check, "Health check")
        self.add_route(HTTPMethod.GET, "/info", self.get_api_info, "API information")
        self.add_route(
            HTTPMethod.POST,
            "/generate/humanoid",
            self.generate_humanoid,
            "Generate humanoid URDF",
            ["generation"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/generate/from-params",
            self.generate_from_params,
            "Generate URDF from parameters",
            ["generation"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/convert/simscape-to-urdf",
            self.convert_simscape_to_urdf,
            "Convert SimScape to URDF",
            ["conversion"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/convert/mjcf-to-urdf",
            self.convert_mjcf_to_urdf,
            "Convert MJCF to URDF",
            ["conversion"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/convert/urdf-to-mjcf",
            self.convert_urdf_to_mjcf,
            "Convert URDF to MJCF",
            ["conversion"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/validate",
            self.validate_urdf,
            "Validate URDF content",
            ["validation"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/parse",
            self.parse_urdf,
            "Parse URDF and return structure",
            ["parsing"],
        )

    def _register_inertia_and_library_routes(self) -> None:
        """Register inertia calculation and library-management routes."""
        self.add_route(
            HTTPMethod.POST,
            "/inertia/calculate",
            self.calculate_inertia,
            "Calculate inertia for shape",
            ["inertia"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/inertia/from-mesh",
            self.inertia_from_mesh,
            "Calculate inertia from mesh file",
            ["inertia"],
        )
        self.add_route(
            HTTPMethod.GET,
            "/library/models",
            self.library_list_models,
            "List available models",
            ["library"],
        )
        self.add_route(
            HTTPMethod.GET,
            "/library/models/{model_id}",
            self.library_get_model,
            "Get model details",
            ["library"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/library/models",
            self.library_add_model,
            "Add model to library",
            ["library"],
        )
        self.add_route(
            HTTPMethod.DELETE,
            "/library/models/{model_id}",
            self.library_remove_model,
            "Remove model from library",
            ["library"],
        )
        self.add_route(
            HTTPMethod.GET,
            "/library/models/{model_id}/download",
            self.library_download_model,
            "Download model URDF",
            ["library"],
        )

    def _register_editor_routes(self) -> None:
        """Register editor composition and diff routes."""
        self.add_route(
            HTTPMethod.POST,
            "/editor/compose",
            self.compose_models,
            "Compose model from multiple sources",
            ["editor"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/editor/diff",
            self.diff_urdfs,
            "Compare two URDF files",
            ["editor"],
        )

    def add_route(
        self,
        method: HTTPMethod,
        path: str,
        handler: Callable[[APIRequest], APIResponse],
        description: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """Append a route definition under the configured API prefix."""
        self._routes.append(
            Route(
                method=method,
                path=f"{self.prefix}{path}",
                handler=handler,
                description=description,
                tags=tags or [],
            )
        )

    def get_routes(self) -> list[Route]:
        """Return the registered route list."""
        return self._routes

    def handle_request(self, request: APIRequest) -> APIResponse:
        """Dispatch a framework-neutral request through the route registry.

        Authentication and rate limiting run before route matching, so an
        unauthenticated caller cannot learn which paths exist from the
        difference between a 401 and a 404.
        """
        if request is None:
            raise ValueError("request must be provided")

        auth_error = self._check_api_key(request)
        if auth_error is not None:
            return self._secure_response(auth_error)

        rate_error = self._check_rate_limit(request)
        if rate_error is not None:
            return self._secure_response(rate_error)

        for route in self._routes:
            params = self._match_route_params(route, request)
            if params is None:
                continue
            request.query_params.update(params)
            return self._execute_route(route, request)
        return self._secure_response(
            APIResponse.not_found(f"No route for {request.method.value} {request.path}")
        )

    def health_check(self, request: APIRequest) -> APIResponse:
        """Return a basic service-health payload."""
        return APIResponse.ok({"status": "healthy", "service": "model_generation"})

    def get_api_info(self, request: APIRequest) -> APIResponse:
        """Return API metadata and route documentation."""
        return APIResponse.ok(
            {
                "name": "Model Generation API",
                "version": "1.0.0",
                "description": "REST API for URDF generation, conversion, and manipulation",  # noqa: E501
                "endpoints": [
                    {
                        "method": route.method.value,
                        "path": route.path,
                        "description": route.description,
                        "tags": route.tags,
                    }
                    for route in self._routes
                ],
            }
        )

    def _match_route_params(
        self,
        route: Route,
        request: APIRequest,
    ) -> dict[str, str] | None:
        """Match request and route paths, returning extracted path parameters."""
        if route.method != request.method:
            return None
        route_parts = route.path.split("/")
        request_parts = request.path.split("/")
        if len(route_parts) != len(request_parts):
            return None

        params: dict[str, str] = {}
        for route_part, request_part in zip(route_parts, request_parts, strict=False):
            if route_part.startswith("{") and route_part.endswith("}"):
                params[route_part[1:-1]] = request_part
                continue
            if route_part != request_part:
                return None
        return params

    def _execute_route(self, route: Route, request: APIRequest) -> APIResponse:
        """Execute a matched route and attach standard response security headers."""
        return self._secure_response(route.handler(request))

    def _check_api_key(self, request: APIRequest) -> APIResponse | None:
        """Return a 401 when ``MODEL_GEN_API_KEY`` is set and unmatched.

        With no key configured the API is open, which is the documented
        development default. Comparison is constant-time so a wrong key
        cannot be recovered by timing the response.
        """
        if request is None:
            raise ValueError("request must be provided")
        if not self._api_key:
            return None

        provided_key = request.headers.get("X-API-Key")
        if not provided_key or not secrets.compare_digest(provided_key, self._api_key):
            return APIResponse.error("Unauthorized: invalid or missing API key", 401)
        return None

    def _check_rate_limit(self, request: APIRequest) -> APIResponse | None:
        """Return a 429 once a client exceeds ``MODEL_GEN_RATE_LIMIT`` per minute.

        The window is in-process and per client IP taken from
        ``X-Forwarded-For``. That makes it a guard against accidental
        hammering, not a defence against a distributed or spoofing attacker:
        a deployment behind an untrusted proxy must rate limit upstream too.
        """
        if request is None:
            raise ValueError("request must be provided")
        if self._rate_limit is None:
            return None

        client_ip = (
            request.headers.get("X-Forwarded-For", "unknown").split(",")[0].strip()
        )
        now = time.time()
        window_start = now - 60.0

        timestamps = self._rate_limit_windows.setdefault(client_ip, [])
        recent = [stamp for stamp in timestamps if stamp > window_start]
        self._rate_limit_windows[client_ip] = recent

        if len(recent) >= self._rate_limit:
            return APIResponse.error("Rate limit exceeded. Try again later.", 429)

        recent.append(now)
        return None

    def _add_cors_headers(self, response: APIResponse) -> None:
        """Attach CORS headers, defaulting to no cross-origin access.

        ``MODEL_GEN_CORS_ORIGINS`` is a comma-separated list and the first
        entry is echoed. Unset yields an empty origin rather than ``*``, so
        the default denies cross-origin reads instead of granting them.
        """
        if response is None:
            raise ValueError("response must be provided")
        origin = self._cors_origins.split(",")[0].strip() if self._cors_origins else ""
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PUT, DELETE, OPTIONS"
        )
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"

    def _secure_response(self, response: APIResponse) -> APIResponse:
        """Attach security and CORS headers to every response."""
        if response is None:
            raise ValueError("response must be provided")
        self._add_cors_headers(response)
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        return response
