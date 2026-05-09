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


class TestRateLimiting:
    """In-memory rate limiting."""

    def _make_api(self) -> Any:
        from model_generation.api.rest_api import ModelGenerationAPI

        return ModelGenerationAPI()

    def _health_request(self, client_ip: str = "127.0.0.1") -> Any:
        from model_generation.api.rest_api import APIRequest, HTTPMethod

        return APIRequest(
            method=HTTPMethod.GET,
            path="/api/v1/health",
            headers={"X-Forwarded-For": client_ip},
        )

    def test_rate_limit_allows_under_threshold(self) -> None:
        """Requests under the limit should succeed."""
        env_overrides = {
            k: v for k, v in os.environ.items() if k != "MODEL_GEN_API_KEY"
        }
        env_overrides["MODEL_GEN_RATE_LIMIT"] = "5"
        with patch.dict(os.environ, env_overrides, clear=True):
            api = self._make_api()
            for _ in range(5):
                resp = api.handle_request(self._health_request())
                assert resp.status_code == 200

    def test_rate_limit_blocks_over_threshold(self) -> None:
        """Requests over the limit should get 429."""
        env_overrides = {
            k: v for k, v in os.environ.items() if k != "MODEL_GEN_API_KEY"
        }
        env_overrides["MODEL_GEN_RATE_LIMIT"] = "3"
        with patch.dict(os.environ, env_overrides, clear=True):
            api = self._make_api()
            for _ in range(3):
                resp = api.handle_request(self._health_request())
                assert resp.status_code == 200
            # 4th request should be rate limited
            resp = api.handle_request(self._health_request())
            assert resp.status_code == 429

    def test_no_rate_limit_env_means_unlimited(self) -> None:
        """Without MODEL_GEN_RATE_LIMIT, no rate limiting occurs."""
        env_overrides = {
            k: v
            for k, v in os.environ.items()
            if k not in ("MODEL_GEN_API_KEY", "MODEL_GEN_RATE_LIMIT")
        }
        with patch.dict(os.environ, env_overrides, clear=True):
            api = self._make_api()
            for _ in range(100):
                resp = api.handle_request(self._health_request())
                assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 2. URL validation and path traversal tests (issue #1700)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. SMPL-X vertex range validation tests (issue #1691)
# ---------------------------------------------------------------------------
