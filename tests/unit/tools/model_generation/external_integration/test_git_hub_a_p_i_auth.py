"""
Tests for external integration improvements.

Covers:
- Xacro preprocessing support (URDFParser)
- ROS package:// URI resolution with ROS_PACKAGE_PATH
- GitHub API authentication headers and retry logic
- Model cache integrity verification
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, Mock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. Xacro preprocessing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. ROS package:// URI resolution
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. GitHub API authentication and retry logic
# ---------------------------------------------------------------------------
class TestGitHubAPIAuth:
    """Tests for GitHub API authentication and resilience."""

    def test_auth_header_from_env(self) -> None:
        """Should add Authorization header when GITHUB_TOKEN is set."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")
        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_testtoken123"}):
            req = repo._build_api_request(
                "https://api.github.com/repos/test/models/contents/"
            )
        assert req.get_header("Authorization") == "token ghp_testtoken123"
        assert req.get_header("Accept") == "application/vnd.github.v3+json"

    def test_no_auth_header_without_token(self) -> None:
        """Should not include Authorization header without GITHUB_TOKEN."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")
        with patch.dict("os.environ", {}, clear=True):
            req = repo._build_api_request(
                "https://api.github.com/repos/test/models/contents/"
            )
        assert req.get_header("Authorization") is None
        # Accept header should always be present
        assert req.get_header("Accept") == "application/vnd.github.v3+json"

    @patch("urllib.request.urlopen")
    def test_retry_on_transient_failure(self, mock_urlopen: Mock) -> None:
        """Should retry on HTTP 5xx errors with exponential backoff."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")

        # First two calls fail with 503, third succeeds
        error_response = urllib.error.HTTPError(
            url="https://api.github.com/test",
            code=503,
            msg="Service Unavailable",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=None,
        )
        success_response = MagicMock()
        success_response.read.return_value = b"[]"
        success_response.headers = MagicMock()
        success_response.headers.get.return_value = None
        success_response.__enter__ = Mock(return_value=success_response)
        success_response.__exit__ = Mock(return_value=False)

        mock_urlopen.side_effect = [
            error_response,
            error_response,
            success_response,
        ]

        with patch("time.sleep"):  # Don't actually sleep
            result = repo._api_request_with_retry("https://api.github.com/test")
        assert result == []
        assert mock_urlopen.call_count == 3

    @patch("urllib.request.urlopen")
    def test_retry_exhausted_raises(self, mock_urlopen: Mock) -> None:
        """Should raise after all retries exhausted."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")

        error = urllib.error.HTTPError(
            url="https://api.github.com/test",
            code=500,
            msg="Internal Server Error",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=None,
        )
        mock_urlopen.side_effect = error

        with patch("time.sleep"), pytest.raises(urllib.error.HTTPError):
            repo._api_request_with_retry("https://api.github.com/test")
        # 1 initial + 3 retries = 4 total attempts
        assert mock_urlopen.call_count == 4

    @patch("urllib.request.urlopen")
    def test_no_retry_on_4xx(self, mock_urlopen: Mock) -> None:
        """Should NOT retry on client errors (4xx)."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")

        error = urllib.error.HTTPError(
            url="https://api.github.com/test",
            code=404,
            msg="Not Found",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=None,
        )
        mock_urlopen.side_effect = error

        with pytest.raises(urllib.error.HTTPError):
            repo._api_request_with_retry("https://api.github.com/test")
        assert mock_urlopen.call_count == 1

    @patch("urllib.request.urlopen")
    def test_timeout_handling(self, mock_urlopen: Mock) -> None:
        """Should handle request timeouts as retryable errors."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")

        success_response = MagicMock()
        success_response.read.return_value = b"[]"
        success_response.headers = MagicMock()
        success_response.headers.get.return_value = None
        success_response.__enter__ = Mock(return_value=success_response)
        success_response.__exit__ = Mock(return_value=False)

        mock_urlopen.side_effect = [
            TimeoutError("Connection timed out"),
            success_response,
        ]

        with patch("time.sleep"):
            result = repo._api_request_with_retry("https://api.github.com/test")
        assert result == []
        assert mock_urlopen.call_count == 2

    @patch("urllib.request.urlopen")
    def test_pagination_follows_link_header(self, mock_urlopen: Mock) -> None:
        """Should follow Link headers for pagination."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")

        # First page with Link header pointing to page 2
        page1_response = MagicMock()
        page1_response.read.return_value = json.dumps(
            [{"name": "a.urdf", "type": "file", "path": "a.urdf"}]
        ).encode()
        page1_response.headers = MagicMock()
        page1_response.headers.get.return_value = (
            '<https://api.github.com/test?page=2>; rel="next"'
        )
        page1_response.__enter__ = Mock(return_value=page1_response)
        page1_response.__exit__ = Mock(return_value=False)

        # Second page with no Link header (last page)
        page2_response = MagicMock()
        page2_response.read.return_value = json.dumps(
            [{"name": "b.urdf", "type": "file", "path": "b.urdf"}]
        ).encode()
        page2_response.headers = MagicMock()
        page2_response.headers.get.return_value = None
        page2_response.__enter__ = Mock(return_value=page2_response)
        page2_response.__exit__ = Mock(return_value=False)

        mock_urlopen.side_effect = [page1_response, page2_response]

        with patch("time.sleep"):
            result = repo._api_request_with_retry(
                "https://api.github.com/test", paginate=True
            )
        assert len(result) == 2
        assert mock_urlopen.call_count == 2


# ---------------------------------------------------------------------------
# 4. Model cache integrity
# ---------------------------------------------------------------------------
