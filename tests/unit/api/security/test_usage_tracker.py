"""Tests for security - Authentication and authorization utilities.

These tests verify the security module using Design by Contract principles.
"""

import os
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

# Configure async tests to use asyncio backend only
pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    """Use asyncio backend only (trio not installed)."""
    return "asyncio"


class TestUsageTracker:
    """Functional tests for UsageTracker."""

    def test_check_quota_within_limit(self) -> None:
        """Test check_quota when within limit."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.models import UserRole
            from src.api.auth.security import UsageTracker

            tracker = UsageTracker()
            user = MagicMock()
            user.role = UserRole.FREE.value
            user.api_calls_this_month = 100  # Free tier limit is 1000

            assert tracker.check_quota(user, "api_calls") is True

    def test_check_quota_exceeded(self) -> None:
        """Test check_quota when exceeded."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.models import UserRole
            from src.api.auth.security import UsageTracker

            tracker = UsageTracker()
            user = MagicMock()
            user.role = UserRole.FREE.value
            user.api_calls_this_month = 1001  # Exceeds free tier limit of 1000

            assert tracker.check_quota(user, "api_calls") is False

    def test_increment_usage(self) -> None:
        """Test incrementing usage counter."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import UsageTracker

            tracker = UsageTracker()
            user = MagicMock()
            user.api_calls_this_month = 10

            tracker.increment_usage(user, "api_calls")
            assert user.api_calls_this_month == 11

    def test_get_usage_summary(self) -> None:
        """Test getting usage summary."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.models import UserRole
            from src.api.auth.security import UsageTracker

            tracker = UsageTracker()
            user = MagicMock()
            user.role = UserRole.FREE.value
            user.api_calls_this_month = 100
            user.video_analyses_this_month = 2
            user.simulations_this_month = 5

            summary = tracker.get_usage_summary(user)

            assert summary["subscription_tier"] == "free"
            assert summary["api_calls"]["used"] == 100
            assert summary["api_calls"]["remaining"] == 900
            assert summary["video_analyses"]["used"] == 2
            assert summary["simulations"]["used"] == 5
