"""Tests for quota dependency defaults on protected API routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.api.auth.models import User

pytestmark = pytest.mark.unit


class TestQuotaDependencyDefaults:
    """Quota dependency behavior for protected resources."""

    def test_usage_quota_allows_available_quota(self) -> None:
        """Available quota returns the current user and records usage."""
        from src.api.auth.dependencies import check_usage_quota

        dependency = check_usage_quota("simulations")
        current_user = MagicMock(spec=User)
        db = MagicMock()

        with (
            patch(
                "src.api.auth.dependencies.usage_tracker.check_quota",
                return_value=True,
            ) as check_quota,
            patch(
                "src.api.auth.dependencies.usage_tracker.increment_usage"
            ) as increment_usage,
        ):
            result = dependency(current_user=current_user, db=db)

        assert result is current_user
        check_quota.assert_called_once_with(current_user, "simulations")
        increment_usage.assert_called_once_with(current_user, "simulations")
        db.commit.assert_called_once_with()

    def test_usage_quota_rejects_exhausted_quota(self) -> None:
        """Exhausted quota raises 429 and does not record usage."""
        from src.api.auth.dependencies import check_usage_quota

        dependency = check_usage_quota("simulations")
        current_user = MagicMock(spec=User)
        current_user.role = "free"
        current_user.simulations_this_month = 10
        current_user.api_calls_this_month = 0
        current_user.video_analyses_this_month = 0
        db = MagicMock()

        with (
            patch(
                "src.api.auth.dependencies.usage_tracker.check_quota",
                return_value=False,
            ),
            patch(
                "src.api.auth.dependencies.usage_tracker.increment_usage"
            ) as increment_usage,
            pytest.raises(HTTPException) as excinfo,
        ):
            dependency(current_user=current_user, db=db)

        assert excinfo.value.status_code == 429
        assert "Usage quota exceeded" in str(excinfo.value.detail)
        increment_usage.assert_not_called()
        db.commit.assert_not_called()
