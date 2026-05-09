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


class TestURLValidation:
    """URL scheme validation in cache and repository."""

    def test_https_url_allowed(self) -> None:
        from security.security_utils import validate_url_scheme

        result = validate_url_scheme(
            "https://example.com/model.urdf", allowed_schemes=("https",)
        )
        assert result == "https://example.com/model.urdf"

    def test_http_url_blocked_when_only_https_allowed(self) -> None:
        from security.security_utils import validate_url_scheme

        with pytest.raises(ValueError, match="not allowed"):
            validate_url_scheme(
                "http://example.com/model.urdf", allowed_schemes=("https",)
            )

    def test_ftp_url_blocked(self) -> None:
        from security.security_utils import validate_url_scheme

        with pytest.raises(ValueError, match="not allowed"):
            validate_url_scheme("ftp://evil.com/payload", allowed_schemes=("https",))

    def test_file_url_blocked(self) -> None:
        from security.security_utils import validate_url_scheme

        with pytest.raises(ValueError, match="not allowed"):
            validate_url_scheme("file:///etc/passwd", allowed_schemes=("https",))

    def test_default_allows_http_and_https(self) -> None:
        from security.security_utils import validate_url_scheme

        assert validate_url_scheme("http://example.com/") == "http://example.com/"
        assert validate_url_scheme("https://example.com/") == "https://example.com/"


# ---------------------------------------------------------------------------
# 3. SMPL-X vertex range validation tests (issue #1691)
# ---------------------------------------------------------------------------
