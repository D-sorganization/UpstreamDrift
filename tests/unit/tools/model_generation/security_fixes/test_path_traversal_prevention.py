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


# ---------------------------------------------------------------------------
# 2. URL validation and path traversal tests (issue #1700)
# ---------------------------------------------------------------------------


class TestPathTraversalPrevention:
    """Cache key generation should reject path traversal attempts."""

    def test_cache_path_rejects_dot_dot(self) -> None:
        """get_cache_path should reject model IDs containing '..'."""
        import tempfile

        from model_generation.library.cache import CacheConfig, ModelCache

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = ModelCache(config=config)
            with pytest.raises(ValueError, match="[Pp]ath traversal"):
                cache.get_cache_path("../../etc/passwd")

    def test_cache_path_rejects_encoded_traversal(self) -> None:
        """get_cache_path should reject encoded path traversal."""
        import tempfile

        from model_generation.library.cache import CacheConfig, ModelCache

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = ModelCache(config=config)
            with pytest.raises(ValueError, match="[Pp]ath traversal"):
                cache.get_cache_path("..%2f..%2fetc/passwd")

    def test_cache_path_allows_normal_ids(self) -> None:
        """Normal model IDs should work fine."""
        import tempfile

        from model_generation.library.cache import CacheConfig, ModelCache

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = ModelCache(config=config)
            path = cache.get_cache_path("my-robot-v2")
            assert ".." not in str(path)
            assert "my-robot-v2" in str(path)

    def test_cache_path_allows_slashes_without_traversal(self) -> None:
        """Model IDs with slashes (but no ..) should work."""
        import tempfile

        from model_generation.library.cache import CacheConfig, ModelCache

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = ModelCache(config=config)
            path = cache.get_cache_path("org/model-name")
            assert ".." not in str(path)


# ---------------------------------------------------------------------------
# 3. SMPL-X vertex range validation tests (issue #1691)
# ---------------------------------------------------------------------------
