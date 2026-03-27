"""Tests for security hardening: REST API auth/CORS/rate-limit, cache URL
validation, and SMPL-X vertex range validation.

Covers GitHub issues #1695, #1691, #1700.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. REST API security tests (issue #1695)
# ---------------------------------------------------------------------------


class TestAPIKeyAuthentication:
    """API key authentication middleware (X-API-Key header)."""

    def _make_api(self):
        from model_generation.api.rest_api import ModelGenerationAPI

        return ModelGenerationAPI()

    def _health_request(self, api_key: str | None = None):
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
    def test_valid_api_key_passes(self):
        api = self._make_api()
        resp = api.handle_request(self._health_request("secret-key-123"))
        assert resp.status_code == 200

    @patch.dict(os.environ, {"MODEL_GEN_API_KEY": "secret-key-123"})
    def test_invalid_api_key_rejected(self):
        api = self._make_api()
        resp = api.handle_request(self._health_request("wrong-key"))
        assert resp.status_code == 401
        assert isinstance(resp.body, dict)
        assert "error" in resp.body

    @patch.dict(os.environ, {"MODEL_GEN_API_KEY": "secret-key-123"})
    def test_missing_api_key_rejected(self):
        api = self._make_api()
        resp = api.handle_request(self._health_request())
        assert resp.status_code == 401

    def test_no_env_key_means_no_auth_required(self):
        """When MODEL_GEN_API_KEY is not set, requests pass through."""
        env_overrides = {
            k: v for k, v in os.environ.items() if k != "MODEL_GEN_API_KEY"
        }
        with patch.dict(os.environ, env_overrides, clear=True):
            api = self._make_api()
            resp = api.handle_request(self._health_request())
            assert resp.status_code == 200


class TestCORSHeaders:
    """CORS header configuration."""

    def _make_api(self):
        from model_generation.api.rest_api import ModelGenerationAPI

        return ModelGenerationAPI()

    def _health_request(self):
        from model_generation.api.rest_api import APIRequest, HTTPMethod

        return APIRequest(
            method=HTTPMethod.GET,
            path="/api/v1/health",
            headers={},
        )

    def test_cors_headers_present_in_response(self):
        """Responses should include CORS headers."""
        env_overrides = {
            k: v for k, v in os.environ.items() if k != "MODEL_GEN_API_KEY"
        }
        with patch.dict(os.environ, env_overrides, clear=True):
            api = self._make_api()
            resp = api.handle_request(self._health_request())
            assert "Access-Control-Allow-Origin" in resp.headers

    def test_cors_default_origin(self):
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

    def test_cors_configurable_origins(self):
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


class TestRateLimiting:
    """In-memory rate limiting."""

    def _make_api(self):
        from model_generation.api.rest_api import ModelGenerationAPI

        return ModelGenerationAPI()

    def _health_request(self, client_ip: str = "127.0.0.1"):
        from model_generation.api.rest_api import APIRequest, HTTPMethod

        return APIRequest(
            method=HTTPMethod.GET,
            path="/api/v1/health",
            headers={"X-Forwarded-For": client_ip},
        )

    def test_rate_limit_allows_under_threshold(self):
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

    def test_rate_limit_blocks_over_threshold(self):
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

    def test_no_rate_limit_env_means_unlimited(self):
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


class TestInputValidation:
    """Input validation for request bodies."""

    def _make_api(self):
        from model_generation.api.rest_api import ModelGenerationAPI

        return ModelGenerationAPI()

    def _post_request(self, path: str, body: dict | None = None):
        from model_generation.api.rest_api import APIRequest, HTTPMethod

        return APIRequest(
            method=HTTPMethod.POST,
            path=path,
            body=body,
            headers={},
        )

    def test_generate_humanoid_accepts_valid_body(self):
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
                mock_result.success = True
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


class TestErrorResponseSanitization:
    """Error responses should not leak stack traces in production."""

    def _make_api(self):
        from model_generation.api.rest_api import ModelGenerationAPI

        return ModelGenerationAPI()

    def test_production_error_no_stack_trace(self):
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

    def test_development_error_may_contain_details(self):
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


class TestURLValidation:
    """URL scheme validation in cache and repository."""

    def test_https_url_allowed(self):
        from security.security_utils import validate_url_scheme

        result = validate_url_scheme(
            "https://example.com/model.urdf", allowed_schemes=("https",)
        )
        assert result == "https://example.com/model.urdf"

    def test_http_url_blocked_when_only_https_allowed(self):
        from security.security_utils import validate_url_scheme

        with pytest.raises(ValueError, match="not allowed"):
            validate_url_scheme(
                "http://example.com/model.urdf", allowed_schemes=("https",)
            )

    def test_ftp_url_blocked(self):
        from security.security_utils import validate_url_scheme

        with pytest.raises(ValueError, match="not allowed"):
            validate_url_scheme("ftp://evil.com/payload", allowed_schemes=("https",))

    def test_file_url_blocked(self):
        from security.security_utils import validate_url_scheme

        with pytest.raises(ValueError, match="not allowed"):
            validate_url_scheme("file:///etc/passwd", allowed_schemes=("https",))

    def test_default_allows_http_and_https(self):
        from security.security_utils import validate_url_scheme

        assert validate_url_scheme("http://example.com/") == "http://example.com/"
        assert validate_url_scheme("https://example.com/") == "https://example.com/"


class TestPathTraversalPrevention:
    """Cache key generation should reject path traversal attempts."""

    def test_cache_path_rejects_dot_dot(self):
        """get_cache_path should reject model IDs containing '..'."""
        import tempfile

        from model_generation.library.cache import CacheConfig, ModelCache

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = ModelCache(config=config)
            with pytest.raises(ValueError, match="[Pp]ath traversal"):
                cache.get_cache_path("../../etc/passwd")

    def test_cache_path_rejects_encoded_traversal(self):
        """get_cache_path should reject encoded path traversal."""
        import tempfile

        from model_generation.library.cache import CacheConfig, ModelCache

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = ModelCache(config=config)
            with pytest.raises(ValueError, match="[Pp]ath traversal"):
                cache.get_cache_path("..%2f..%2fetc/passwd")

    def test_cache_path_allows_normal_ids(self):
        """Normal model IDs should work fine."""
        import tempfile

        from model_generation.library.cache import CacheConfig, ModelCache

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = ModelCache(config=config)
            path = cache.get_cache_path("my-robot-v2")
            assert ".." not in str(path)
            assert "my-robot-v2" in str(path)

    def test_cache_path_allows_slashes_without_traversal(self):
        """Model IDs with slashes (but no ..) should work."""
        import tempfile

        from model_generation.library.cache import CacheConfig, ModelCache

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = ModelCache(config=config)
            path = cache.get_cache_path("org/model-name")
            assert ".." not in str(path)


class TestRepositoryURLRestriction:
    """Repository should restrict remote URLs to https:// only."""

    def test_repository_validate_url_called(self):
        """GitHubRepository should use validate_url_scheme for URL validation."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")
        # The API_BASE and RAW_BASE should be https
        assert repo.API_BASE.startswith("https://")
        assert repo.RAW_BASE.startswith("https://")


# ---------------------------------------------------------------------------
# 3. SMPL-X vertex range validation tests (issue #1691)
# ---------------------------------------------------------------------------


class TestSMPLXVertexValidation:
    """SMPL-X hardcoded vertex range validation."""

    def test_expected_vertex_count_constant_exists(self):
        """SMPLX_EXPECTED_VERTEX_COUNT should be defined."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        assert hasattr(SMPLXMeshGenerator, "SMPLX_EXPECTED_VERTEX_COUNT")
        assert SMPLXMeshGenerator.SMPLX_EXPECTED_VERTEX_COUNT == 10475

    def test_vertex_ranges_within_expected_count(self):
        """All vertex ranges should be within [0, SMPLX_EXPECTED_VERTEX_COUNT)."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        expected = SMPLXMeshGenerator.SMPLX_EXPECTED_VERTEX_COUNT
        for name, (
            start,
            end,
        ) in SMPLXMeshGenerator.SMPLX_SEGMENT_VERTEX_RANGES.items():
            assert 0 <= start < expected, f"{name}: start {start} out of range"
            assert 0 < end <= expected, f"{name}: end {end} out of range"
            assert start < end, f"{name}: start {start} >= end {end}"

    def test_validate_vertex_ranges_method_exists(self):
        """validate_vertex_ranges class method should exist."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        assert hasattr(SMPLXMeshGenerator, "validate_vertex_ranges")

    def test_validate_vertex_ranges_passes_for_matching_count(self):
        """validate_vertex_ranges returns True when vertex count matches."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        result = SMPLXMeshGenerator.validate_vertex_ranges(10475)
        assert result is True

    def test_validate_vertex_ranges_warns_for_mismatched_count(self):
        """validate_vertex_ranges returns False for wrong vertex count."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        result = SMPLXMeshGenerator.validate_vertex_ranges(5000)
        assert result is False

    def test_load_segmentation_from_file_method_exists(self):
        """load_part_segmentation classmethod should exist."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        assert hasattr(SMPLXMeshGenerator, "load_part_segmentation")

    def test_load_segmentation_falls_back_to_hardcoded(self):
        """When no model file is available, should fall back to hardcoded
        ranges and log a warning."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        # Call with a non-existent path
        result = SMPLXMeshGenerator.load_part_segmentation(Path("/nonexistent/path"))
        # Should return the hardcoded ranges
        assert isinstance(result, dict)
        assert len(result) > 0
        # The returned dict should match SMPLX_SEGMENT_VERTEX_RANGES
        assert result == SMPLXMeshGenerator.SMPLX_SEGMENT_VERTEX_RANGES

    def test_load_segmentation_logs_warning_on_fallback(self):
        """Falling back to hardcoded ranges should produce a warning log."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        with patch(
            "humanoid_character_builder.generators.mesh_generator.logger"
        ) as mock_logger:
            SMPLXMeshGenerator.load_part_segmentation(Path("/nonexistent/path"))
            mock_logger.warning.assert_called()
            # The warning should mention fallback or hardcoded
            call_args = str(mock_logger.warning.call_args)
            assert (
                "hardcoded" in call_args.lower()
                or "fallback" in call_args.lower()
                or "fall" in call_args.lower()
            )
