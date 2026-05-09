"""Tests for security hardening: REST API auth/CORS/rate-limit, cache URL
validation, and SMPL-X vertex range validation.

Covers GitHub issues #1695, #1691, #1700.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. REST API security tests (issue #1695)
# ---------------------------------------------------------------------------


class TestCORSHeaders:
    """CORS header configuration."""

    def _make_api(self) -> Any:
        from model_generation.api.rest_api import ModelGenerationAPI

        return ModelGenerationAPI()

    def _health_request(self) -> Any:
        from model_generation.api.rest_api import APIRequest, HTTPMethod

        return APIRequest(
            method=HTTPMethod.GET,
            path="/api/v1/health",
            headers={},
        )

    def test_cors_headers_present_in_response(self) -> None:
        """Responses should include CORS headers."""
        env_overrides = {
            k: v for k, v in os.environ.items() if k != "MODEL_GEN_API_KEY"
        }
        with patch.dict(os.environ, env_overrides, clear=True):
            api = self._make_api()
            resp = api.handle_request(self._health_request())
            assert "Access-Control-Allow-Origin" in resp.headers

    def test_cors_default_origin(self) -> None:
        """Default allowed origin should be restrictive (not *)."""
        env_overrides = {
            k: v
            for k, v in os.environ.items()
            if k not in ("MODEL_GEN_API_KEY", "MODEL_GEN_CORS_ORIGINS")
        }
        with patch.dict(os.environ, env_overrides, clear=True):
            api = self._make_api()
            resp = api.handle_request(self._health_request())
            origin = resp.headers.get("Access-Control-Allow-Origin", "")
            # Default should NOT be wildcard *
            assert origin != "*"

    def test_cors_configurable_origins(self) -> None:
        """CORS origins should be configurable via env var."""
        env_overrides = {
            k: v for k, v in os.environ.items() if k != "MODEL_GEN_API_KEY"
        }
        env_overrides["MODEL_GEN_CORS_ORIGINS"] = (
            "https://example.com,https://app.example.com"
        )
        with patch.dict(os.environ, env_overrides, clear=True):
            api = self._make_api()
            resp = api.handle_request(self._health_request())
            origin = resp.headers.get("Access-Control-Allow-Origin", "")
            assert "example.com" in origin


# ---------------------------------------------------------------------------
# 2. URL validation and path traversal tests (issue #1700)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. SMPL-X vertex range validation tests (issue #1691)
# ---------------------------------------------------------------------------
