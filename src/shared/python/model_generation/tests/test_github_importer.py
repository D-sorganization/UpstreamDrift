from typing import Any

"""
Tests for GitHub Importer.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from src.shared.python.model_generation.library.github_importer import GitHubImporter


class TestGitHubImporter:
    """Tests for GitHubImporter."""

    @pytest.fixture
    def mock_library(self) -> Any:
        """Mock ModelLibrary."""
        return MagicMock()

    @pytest.fixture
    def importer(self, mock_library) -> Any:
        """Create importer with mock library."""
        return GitHubImporter(library=mock_library)

    @patch("urllib.request.urlopen")
    def test_import_from_search_dry_run(self, mock_urlopen, importer) -> Any:
        """Test search with dry_run."""
        # Mock GitHub response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "items": [
                    {
                        "name": "test_repo",
                        "owner": {"login": "test_owner"},
                        "html_url": "https://github.com/test_owner/test_repo",
                        "description": "Test Repo",
                        "stargazers_count": 50,
                        "default_branch": "main",
                    }
                ]
            }
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        results = importer.import_from_search(query="test", dry_run=True)

        assert len(results) == 1
        assert results[0].status == "found"
        assert results[0].name == "test_owner/test_repo"
        assert results[0].stars == 50

        # Should not have called add_repository
        importer.library.add_repository.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_import_from_search_import(self, mock_urlopen, importer) -> Any:
        """Test search and import."""
        # Mock GitHub response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "items": [
                    {
                        "name": "test_repo",
                        "owner": {"login": "test_owner"},
                        "html_url": "https://github.com/test_owner/test_repo",
                        "description": "Test Repo",
                        "stargazers_count": 50,
                        "default_branch": "main",
                    }
                ]
            }
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        # Mock library methods
        importer.library.refresh_repository.return_value = [MagicMock()]  # Found models

        results = importer.import_from_search(query="test", dry_run=False)

        assert len(results) == 1
        assert results[0].status == "success"

        importer.library.add_repository.assert_called_once()
        importer.library.refresh_repository.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_import_from_urls(self, mock_urlopen, importer) -> Any:
        """Test import from URLs."""
        # Mock repo metadata response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "name": "repo",
                "owner": {"login": "owner"},
                "default_branch": "master",
                "description": "Desc",
            }
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        importer.library.refresh_repository.return_value = [MagicMock(), MagicMock()]

        urls = ["https://github.com/owner/repo"]
        results = importer.import_from_urls(urls)

        assert len(results) == 1
        assert results[0].status == "success"
        assert "Imported 2 models" in results[0].description

        importer.library.add_repository.assert_called_with(
            name="github_owner_repo",
            repo_type="github",
            owner="owner",
            repo="repo",
            branch="master",
            description="Desc",
        )

    def test_import_from_urls_existing(self, importer) -> Any:
        """Test skipping existing repositories."""
        importer.library._repositories = {"github_owner_repo": {}}

        urls = ["https://github.com/owner/repo"]
        results = importer.import_from_urls(urls, skip_existing=True)

        assert len(results) == 1
        assert results[0].status == "exists"

        importer.library.add_repository.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_import_from_urls_rejects_non_https(self, mock_urlopen, importer) -> Any:
        """Reject non-HTTPS URLs for repository imports."""
        urls = ["file:///tmp/repo"]

        results = importer.import_from_urls(urls)

        assert len(results) == 1
        assert results[0].status == "failed"
        assert "Only HTTPS GitHub URLs are supported" in results[0].error
        mock_urlopen.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_import_from_urls_rejects_private_host(self, mock_urlopen, importer) -> Any:
        """Reject unsupported GitHub hosts to prevent SSRF."""
        urls = ["https://localhost:8080/owner/repo"]

        results = importer.import_from_urls(urls)

        assert len(results) == 1
        assert results[0].status == "failed"
        assert "Unsupported GitHub host" in results[0].error
        mock_urlopen.assert_not_called()
