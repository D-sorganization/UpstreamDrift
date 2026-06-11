"""Tests for quota dependency defaults on protected API routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.auth.models import Base, SubscriptionStatus, User, UserRole

pytestmark = pytest.mark.unit


def _create_sqlite_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, session_factory()


def _add_free_user(db, api_calls_this_month: int = 999) -> User:
    user = User(
        email="quota@example.test",
        hashed_password="hashed",
        role=UserRole.FREE.value,
        is_active=True,
        is_verified=True,
        subscription_status=SubscriptionStatus.ACTIVE.value,
        api_calls_this_month=api_calls_this_month,
        video_analyses_this_month=0,
        simulations_this_month=0,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


class TestQuotaDependencyDefaults:
    """Quota dependency behavior for protected resources."""

    def test_successful_protected_operation_consumes_quota_once(self) -> None:
        """Available quota yields the current user and records one successful use."""
        from src.api.auth.dependencies import check_usage_quota

        dependency = check_usage_quota("simulations")
        current_user = MagicMock(spec=User)
        db = MagicMock()

        with (
            patch(
                "src.api.auth.dependencies.usage_tracker.consume_quota",
                return_value=True,
            ) as consume_quota,
            patch(
                "src.api.auth.dependencies.usage_tracker.refund_quota",
            ) as refund_quota,
        ):
            quota_scope = dependency(current_user=current_user, db=db)
            result = next(quota_scope)

            with pytest.raises(StopIteration):
                next(quota_scope)

        assert result is current_user
        consume_quota.assert_called_once_with(db, current_user, "simulations")
        refund_quota.assert_not_called()
        db.commit.assert_not_called()

    def test_failed_protected_operation_does_not_consume_quota(self) -> None:
        """Endpoint failures propagate after refunding reserved quota."""
        from src.api.auth.dependencies import check_usage_quota

        dependency = check_usage_quota("simulations")
        current_user = MagicMock(spec=User)
        db = MagicMock()

        with (
            patch(
                "src.api.auth.dependencies.usage_tracker.consume_quota",
                return_value=True,
            ) as consume_quota,
            patch(
                "src.api.auth.dependencies.usage_tracker.refund_quota",
                return_value=True,
            ) as refund_quota,
        ):
            quota_scope = dependency(current_user=current_user, db=db)
            assert next(quota_scope) is current_user

            with pytest.raises(RuntimeError, match="protected request failed"):
                quota_scope.throw(RuntimeError("protected request failed"))

        consume_quota.assert_called_once_with(db, current_user, "simulations")
        refund_quota.assert_called_once_with(db, current_user, "simulations")
        db.commit.assert_not_called()

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
                "src.api.auth.dependencies.usage_tracker.consume_quota",
                return_value=False,
            ) as consume_quota,
            patch(
                "src.api.auth.dependencies.usage_tracker.refund_quota"
            ) as refund_quota,
            pytest.raises(HTTPException) as excinfo,
        ):
            quota_scope = dependency(current_user=current_user, db=db)
            next(quota_scope)

        assert excinfo.value.status_code == 429
        assert "Usage quota exceeded" in str(excinfo.value.detail)
        consume_quota.assert_called_once_with(db, current_user, "simulations")
        refund_quota.assert_not_called()
        db.commit.assert_not_called()


class TestUsageQuotaConsumption:
    """Database-backed contract tests for quota consumption."""

    def test_consume_quota_uses_bounded_database_update(self) -> None:
        """Quota consumption succeeds once at limit - 1 and refuses at limit."""
        from src.api.auth.security import UsageTracker

        engine, db = _create_sqlite_session()
        try:
            user = _add_free_user(db)
            tracker = UsageTracker()

            assert tracker.consume_quota(db, user, "api_calls") is True
            assert user.api_calls_this_month == 1000

            assert tracker.refund_quota(db, user, "api_calls") is True
            assert user.api_calls_this_month == 999

            assert tracker.consume_quota(db, user, "api_calls") is True
            assert user.api_calls_this_month == 1000

            assert tracker.consume_quota(db, user, "api_calls") is False
            db.refresh(user)
            assert user.api_calls_this_month == 1000
        finally:
            db.close()
            Base.metadata.drop_all(bind=engine)

    def test_failed_dependency_scope_refunds_reserved_quota(self) -> None:
        """A protected-operation failure leaves persisted usage unchanged."""
        from src.api.auth.dependencies import check_usage_quota

        engine, db = _create_sqlite_session()
        try:
            user = _add_free_user(db)
            dependency = check_usage_quota("api_calls")
            quota_scope = dependency(current_user=user, db=db)

            assert next(quota_scope) is user
            assert user.api_calls_this_month == 1000

            with pytest.raises(RuntimeError, match="protected request failed"):
                quota_scope.throw(RuntimeError("protected request failed"))

            db.refresh(user)
            assert user.api_calls_this_month == 999
        finally:
            db.close()
            Base.metadata.drop_all(bind=engine)
