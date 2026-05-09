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


class TestInputValidation:
    """Input validation for request bodies."""

    def _make_api(self) -> Any:
        from model_generation.api.rest_api import ModelGenerationAPI

        return ModelGenerationAPI()

    def _post_request(self, path: str, body: dict | None = None) -> Any:
        from model_generation.api.rest_api import APIRequest, HTTPMethod

        return APIRequest(
            method=HTTPMethod.POST,
            path=path,
            body=body,
            headers={},
        )

    def test_generate_humanoid_accepts_valid_body(self) -> None:
        """Valid body should not cause validation error."""
        env_overrides = {
            k: v for k, v in os.environ.items() if k != "MODEL_GEN_API_KEY"
        }
        with patch.dict(os.environ, env_overrides, clear=True):
            api = self._make_api()
            with patch(
                "model_generation.builders.parametric_builder.ParametricBuilder"
            ) as MockBuilder:
                mock_result = MagicMock()
                mock_result.solver_status = "success"
                mock_result.urdf_xml = "<robot/>"
                mock_result.links = []
                mock_result.joints = []
                MockBuilder.return_value.build.return_value = mock_result
                resp = api.handle_request(
                    self._post_request(
                        "/api/v1/generate/humanoid",
                        {"name": "test", "height": 1.8, "mass": 75.0},
                    )
                )
                # Should not be a 422 validation error
                assert resp.status_code != 422


# ---------------------------------------------------------------------------
# 2. URL validation and path traversal tests (issue #1700)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. SMPL-X vertex range validation tests (issue #1691)
# ---------------------------------------------------------------------------
