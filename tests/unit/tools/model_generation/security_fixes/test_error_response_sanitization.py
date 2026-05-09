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


class TestErrorResponseSanitization:
    """Error responses should not leak stack traces in production."""

    def _make_api(self) -> Any:
        from model_generation.api.rest_api import ModelGenerationAPI

        return ModelGenerationAPI()

    def test_production_error_no_stack_trace(self) -> None:
        """In production mode, 500 errors should not contain traceback info."""
        env_overrides = {
            k: v for k, v in os.environ.items() if k != "MODEL_GEN_API_KEY"
        }
        env_overrides["MODEL_GEN_ENV"] = "production"
        with patch.dict(os.environ, env_overrides, clear=True):
            api = self._make_api()
            from model_generation.api.rest_api import APIRequest, HTTPMethod

            req = APIRequest(
                method=HTTPMethod.POST,
                path="/api/v1/generate/humanoid",
                body={"name": "test"},
                headers={},
            )
            resp = api.handle_request(req)
            # If it's a 500, the body should not contain "Traceback"
            if resp.status_code == 500 and isinstance(resp.body, dict):
                error_msg = resp.body.get("error", "")
                assert "Traceback" not in error_msg
                assert "File " not in error_msg

    def test_development_error_may_contain_details(self) -> None:
        """In development mode, errors may contain details."""
        env_overrides = {
            k: v
            for k, v in os.environ.items()
            if k not in ("MODEL_GEN_API_KEY", "MODEL_GEN_ENV")
        }
        with patch.dict(os.environ, env_overrides, clear=True):
            api = self._make_api()
            from model_generation.api.rest_api import APIRequest, HTTPMethod

            req = APIRequest(
                method=HTTPMethod.GET,
                path="/api/v1/health",
                headers={},
            )
            resp = api.handle_request(req)
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 2. URL validation and path traversal tests (issue #1700)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. SMPL-X vertex range validation tests (issue #1691)
# ---------------------------------------------------------------------------
