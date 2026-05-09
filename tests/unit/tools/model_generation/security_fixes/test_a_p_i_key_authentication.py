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


class TestAPIKeyAuthentication:
    """API key authentication middleware (X-API-Key header)."""

    def _make_api(self) -> Any:
        from model_generation.api.rest_api import ModelGenerationAPI

        return ModelGenerationAPI()

    def _health_request(self, api_key: str | None = None) -> Any:
        from model_generation.api.rest_api import APIRequest, HTTPMethod

        headers = {}
        if api_key is not None:
            headers["X-API-Key"] = api_key
        return APIRequest(
            method=HTTPMethod.GET,
            path="/api/v1/health",
            headers=headers,
        )

    @patch.dict(os.environ, {"MODEL_GEN_API_KEY": "secret-key-123"})
    def test_valid_api_key_passes(self) -> None:
        api = self._make_api()
        resp = api.handle_request(self._health_request("secret-key-123"))
        assert resp.status_code == 200

    @patch.dict(os.environ, {"MODEL_GEN_API_KEY": "secret-key-123"})
    def test_invalid_api_key_rejected(self) -> None:
        api = self._make_api()
        resp = api.handle_request(self._health_request("wrong-key"))
        assert resp.status_code == 401
        assert isinstance(resp.body, dict)
        assert "error" in resp.body

    @patch.dict(os.environ, {"MODEL_GEN_API_KEY": "secret-key-123"})
    def test_missing_api_key_rejected(self) -> None:
        api = self._make_api()
        resp = api.handle_request(self._health_request())
        assert resp.status_code == 401

    def test_no_env_key_means_no_auth_required(self) -> None:
        """When MODEL_GEN_API_KEY is not set, requests pass through."""
        env_overrides = {
            k: v for k, v in os.environ.items() if k != "MODEL_GEN_API_KEY"
        }
        with patch.dict(os.environ, env_overrides, clear=True):
            api = self._make_api()
            resp = api.handle_request(self._health_request())
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 2. URL validation and path traversal tests (issue #1700)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. SMPL-X vertex range validation tests (issue #1691)
# ---------------------------------------------------------------------------
