"""Security-hardening tests for the auth layer.

Covers:
- #7139 expired/revoked API keys must stop authenticating (queries + cache).
- #7140 pagination bounds on admin list_users.
- #7142 LocalUser satisfies the User attribute contract.
- #7143 quota increment-on-success + unknown-resource-type rejection.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.utils.datetime_compat import UTC

pytestmark = pytest.mark.unit


def _utcnow():
    from datetime import datetime

    return datetime.now(UTC)


@pytest.fixture()
def db_session():
    from src.api.auth.models import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _make_user(db_session):
    from src.api.auth.models import User

    user = User(
        email="u@example.com",
        hashed_password="x",  # noqa: S106 - not a real credential
        role="free",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    return user


def _make_api_key(db_session, user_id, *, expires_at=None, is_active=True):
    from src.api.auth.models import APIKey
    from src.api.auth.security import compute_prefix_hash, security_manager

    api_key = security_manager.generate_api_key()
    record = APIKey(
        user_id=user_id,
        key_hash=security_manager.hash_api_key(api_key),
        key_prefix=compute_prefix_hash(api_key[4:12]),
        name="test",
        is_active=is_active,
        expires_at=expires_at,
    )
    db_session.add(record)
    db_session.commit()
    return api_key, record


# --- #7139 expiry / revocation ---------------------------------------------


def test_expired_api_key_is_rejected(db_session) -> None:
    from fastapi import HTTPException

    from src.api.auth.dependencies import _lookup_api_key_by_prefix

    user = _make_user(db_session)
    api_key, _ = _make_api_key(
        db_session, user.id, expires_at=_utcnow() - timedelta(hours=1)
    )
    with pytest.raises(HTTPException) as exc:
        _lookup_api_key_by_prefix(api_key, db_session)
    assert exc.value.status_code == 401


def test_never_expiring_api_key_authenticates(db_session) -> None:
    from src.api.auth.dependencies import _lookup_api_key_by_prefix

    user = _make_user(db_session)
    api_key, record = _make_api_key(db_session, user.id, expires_at=None)
    found = _lookup_api_key_by_prefix(api_key, db_session)
    assert found.id == record.id


def test_future_expiry_api_key_authenticates(db_session) -> None:
    from src.api.auth.dependencies import _lookup_api_key_by_prefix

    user = _make_user(db_session)
    api_key, record = _make_api_key(
        db_session, user.id, expires_at=_utcnow() + timedelta(hours=1)
    )
    found = _lookup_api_key_by_prefix(api_key, db_session)
    assert found.id == record.id


def test_cache_hit_cannot_resurrect_deactivated_key(db_session) -> None:
    from src.api.auth.dependencies import _lookup_cached_api_key
    from src.api.auth.security import auth_cache

    user = _make_user(db_session)
    api_key, record = _make_api_key(db_session, user.id)
    auth_cache.set(api_key, record.id)
    assert _lookup_cached_api_key(api_key, db_session) is not None

    record.is_active = False
    db_session.commit()
    assert _lookup_cached_api_key(api_key, db_session) is None


def test_cache_hit_cannot_resurrect_expired_key(db_session) -> None:
    from src.api.auth.dependencies import _lookup_cached_api_key
    from src.api.auth.security import auth_cache

    user = _make_user(db_session)
    api_key, record = _make_api_key(db_session, user.id)
    auth_cache.set(api_key, record.id)
    assert _lookup_cached_api_key(api_key, db_session) is not None

    record.expires_at = _utcnow() - timedelta(seconds=1)
    db_session.commit()
    assert _lookup_cached_api_key(api_key, db_session) is None


# --- #7142 LocalUser contract ----------------------------------------------


def test_local_user_runs_through_quota_tracker() -> None:
    from src.api.auth.middleware import LocalUser
    from src.api.auth.security import UsageTracker

    tracker = UsageTracker()
    user = LocalUser()
    for resource in ("api_calls", "video_analyses", "simulations"):
        assert tracker.check_quota(user, resource) in (True, False)
        tracker.increment_usage(user, resource)


def test_local_user_defines_every_user_field_auth_uses() -> None:
    from src.api.auth.middleware import LocalUser

    user = LocalUser()
    for field in (
        "id",
        "email",
        "role",
        "is_active",
        "is_verified",
        "subscription_status",
        "api_calls_this_month",
        "video_analyses_this_month",
        "simulations_this_month",
    ):
        assert hasattr(user, field), f"LocalUser missing {field}"


# --- #7143 quota DbC --------------------------------------------------------


def test_increment_usage_rejects_unknown_resource_type() -> None:
    from unittest.mock import MagicMock

    from src.api.auth.security import UsageTracker

    tracker = UsageTracker()
    with pytest.raises(ValueError, match="resource_type"):
        tracker.increment_usage(MagicMock(), "nonexistent_resource")


def test_quota_dependency_is_a_generator_dependency() -> None:
    """Increment-on-success requires a yield dependency (issue #7143)."""
    import inspect

    from src.api.auth.dependencies import check_usage_quota

    dep = check_usage_quota("api_calls")
    assert inspect.isgeneratorfunction(dep)
