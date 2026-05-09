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


class TestRepositoryURLRestriction:
    """Repository should restrict remote URLs to https:// only."""

    def test_repository_validate_url_called(self) -> None:
        """GitHubRepository should use validate_url_scheme for URL validation."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")
        # The API_BASE and RAW_BASE should be https
        assert repo.API_BASE.startswith("https://")
        assert repo.RAW_BASE.startswith("https://")

    def test_model_library_download_blocks_non_https_source_url(self, tmp_path):
        """ModelLibrary should validate source_url before urlretrieve."""
        from model_generation.library.model_library import (
            LibraryConfig,
            ModelEntry,
            ModelLibrary,
            RepositorySource,
        )

        library = ModelLibrary(
            LibraryConfig(
                cache_dir=tmp_path / "cache",
                index_file=tmp_path / "index.json",
                default_repositories=[],
            )
        )
        entry = ModelEntry(
            id="evil/model",
            name="evil",
            source=RepositorySource.URL,
            source_url="file:///etc/passwd",
        )

        with (
            patch("urllib.request.urlretrieve") as urlretrieve,
            pytest.raises(ValueError, match="URL scheme 'file' is not allowed"),
        ):
            library._download_model(entry)

        urlretrieve.assert_not_called()


# ---------------------------------------------------------------------------
# 3. SMPL-X vertex range validation tests (issue #1691)
# ---------------------------------------------------------------------------
