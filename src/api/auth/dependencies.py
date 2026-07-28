"""Authentication dependencies for FastAPI endpoints."""

from collections.abc import Callable, Generator
from typing import TypeVar

from src.api.utils.datetime_compat import UTC

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.shared.python.config.environment import is_auth_disabled

from .models import APIKey, User, UserRole
from .security import (
    RoleChecker,
    compute_prefix_hash,
    security_manager,
    usage_tracker,
)

# Security scheme
security = HTTPBearer()


def _unauthorized(detail: str) -> HTTPException:
    """Return a 401 HTTPException with the standard Bearer WWW-Authenticate header."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


_AssertT = TypeVar("_AssertT")


def _assert_type(obj: object, cls: type[_AssertT], name: str = "object") -> _AssertT:
    """Return obj narrowed to cls, raising ValueError if it is not an instance.

    Provides runtime type safety AND mypy narrowing where SQLAlchemy query
    results are typed ambiguously, so callers need no cast().
    """
    if not isinstance(obj, cls):
        raise ValueError(
            f"Expected {cls.__name__}, got {type(obj).__name__} for {name}"
        )
    return obj


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Get current authenticated user from JWT token."""

    # Verify JWT token
    payload = security_manager.verify_token(credentials.credentials, "access")
    user_id = payload.get("sub")

    if user_id is None:
        raise _unauthorized("Could not validate credentials")

    # Get user from database
    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise _unauthorized("User not found")

    if not user.is_active:
        raise _unauthorized("Inactive user")

    return _assert_type(user, User, "current_user")


def _validate_api_key_format(api_key: str) -> None:
    if not api_key.startswith("gms_"):
        raise _unauthorized("Invalid API key format")


def _lookup_cached_api_key(api_key: str, db: Session) -> APIKey | None:
    from .security import auth_cache

    cached_key_id = auth_cache.get(api_key)
    if not cached_key_id:
        return None
    record = db.query(APIKey).filter(APIKey.id == cached_key_id).first()
    if not record or not record.is_active:
        return None

    return _assert_type(record, APIKey, "cached_api_key")


def _lookup_api_key_by_prefix(api_key: str, db: Session) -> APIKey:
    key_body = api_key[4:]
    if len(key_body) < 8:
        raise _unauthorized("Invalid API key format")

    prefix_for_index = key_body[:8]
    prefix_hash = compute_prefix_hash(prefix_for_index)

    try:
        active_keys = (
            db.query(APIKey)
            .filter(APIKey.is_active, APIKey.key_prefix == prefix_hash)
            .all()
        )
    except (OperationalError, ProgrammingError):
        # Fallback only for schema-missing errors (key_prefix column not yet migrated).
        # Broad exceptions like RuntimeError/OSError are NOT suppressed here to avoid
        # masking real DB failures and to prevent O(n) bcrypt DoS amplification.
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "key_prefix index unavailable, falling back to full key scan"
        )
        active_keys = db.query(APIKey).filter(APIKey.is_active).all()

    if not active_keys:
        raise _unauthorized("Invalid API key")

    for key_candidate in active_keys:
        if security_manager.verify_api_key(api_key, str(key_candidate.key_hash)):
            return _assert_type(key_candidate, APIKey, "api_key_candidate")

    raise _unauthorized("Invalid API key")


def _get_active_user_for_api_key(api_key_record: APIKey, db: Session) -> User:
    user = db.query(User).filter(User.id == api_key_record.user_id).first()
    if not user or not user.is_active:
        raise _unauthorized("User not found or inactive")
    return _assert_type(user, User, "api_key_user")


def _update_api_key_usage(api_key_record: APIKey, db: Session) -> None:
    if not (api_key_record is not None):
        raise ValueError("api_key_record must be provided")
    from datetime import datetime

    api_key_record.last_used = datetime.now(UTC)  # type: ignore[assignment]
    api_key_record.usage_count = int(api_key_record.usage_count) + 1  # type: ignore[assignment]
    db.commit()


async def get_current_user_from_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Get current user from API key.

    PERFORMANCE FIX: Uses prefix hash to filter candidates before bcrypt verification,
    reducing O(n) bcrypt calls to O(1) average case.
    """

    if not (credentials is not None):
        raise ValueError("credentials must be provided")
    api_key = credentials.credentials
    _validate_api_key_format(api_key)

    api_key_record = _lookup_cached_api_key(api_key, db)

    if not api_key_record:
        api_key_record = _lookup_api_key_by_prefix(api_key, db)
        from .security import auth_cache

        auth_cache.set(api_key, api_key_record.id)

    user = _get_active_user_for_api_key(api_key_record, db)
    _update_api_key_usage(api_key_record, db)

    return user


async def get_current_user_flexible(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Get current user from either JWT token or API key."""

    if not (credentials is not None):
        raise ValueError("credentials must be provided")
    token = credentials.credentials

    # Try API key first (if it starts with gms_)
    if token.startswith("gms_"):
        return await get_current_user_from_api_key(credentials, db)
    # Try JWT token
    return await get_current_user(credentials, db)


def require_role(required_role: UserRole) -> Callable[[User], User]:
    """Dependency factory for role-based access control."""

    def role_dependency(
        current_user: User = Depends(get_current_user_flexible),
    ) -> User:
        """Verify the current user has the required role."""
        role_checker = RoleChecker(required_role)

        if not role_checker(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role.value}",  # noqa: E501
            )

        return current_user

    return role_dependency


async def authenticate_bearer_request(request: Request, db: Session) -> User:
    """Resolve the caller from the raw ``Authorization: Bearer`` header.

    Used by dependencies that receive the raw :class:`~fastapi.Request` rather
    than FastAPI-parsed ``HTTPAuthorizationCredentials`` (router-level
    dependencies, quota dependencies, WebSocket-compatible dependencies).

    Preconditions:
        ``request`` and ``db`` must not be ``None``.

    Raises:
        HTTPException: 401 when the header is missing, malformed, or the
            credentials do not resolve to an active user.
    """
    if request is None:
        raise ValueError("request must be provided")
    if db is None:
        raise ValueError("db must be provided")

    authorization = request.headers.get("Authorization")
    if not authorization:
        raise _unauthorized("Not authenticated")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise _unauthorized("Invalid authentication credentials")

    credentials = HTTPAuthorizationCredentials(scheme=scheme, credentials=token)
    return await get_current_user_flexible(credentials=credentials, db=db)


async def require_cloud_auth(
    request: Request,
    db: Session = Depends(get_db),
) -> User | None:
    """Require a valid bearer token in cloud mode; no-op in local mode.

    Attach with ``dependencies=[Depends(require_cloud_auth)]``. Note the
    *call*: passing an un-instantiated class or an un-called function to
    ``Depends`` silently disables the check (issue #7987).

    Postcondition: in cloud mode this either returns an authenticated
    :class:`User` or raises 401 — it never returns ``None``.
    """
    if is_auth_disabled():
        return None
    return await authenticate_bearer_request(request, db)


def _usage_quota_exceeded(current_user: User, resource_type: str) -> HTTPException:
    quota_limit = usage_tracker.quota_limit(current_user, resource_type)
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=f"Usage quota exceeded for {resource_type}. "
        f"Limit: {quota_limit} per month. "
        f"Upgrade your subscription for higher limits.",
    )


def check_usage_quota(
    resource_type: str,
) -> Callable[[User, Session], Generator[User, None, None]]:
    """Dependency factory for usage quota checking."""

    def quota_dependency(
        current_user: User = Depends(get_current_user_flexible),
        db: Session = Depends(get_db),
    ) -> Generator[User, None, None]:
        """Enforce usage quota for the given resource type."""
        if not usage_tracker.consume_quota(db, current_user, resource_type):
            raise _usage_quota_exceeded(current_user, resource_type)

        try:
            yield current_user
        except Exception:  # noqa: BLE001 - refund any protected-operation failure.
            usage_tracker.refund_quota(db, current_user, resource_type)
            raise

    return quota_dependency


# Common dependency combinations
RequireAuth = Depends(get_current_user_flexible)
RequireProfessional = Depends(require_role(UserRole.PROFESSIONAL))
RequireEnterprise = Depends(require_role(UserRole.ENTERPRISE))
RequireAdmin = Depends(require_role(UserRole.ADMIN))

# Usage quota dependencies
CheckAPIQuota = Depends(check_usage_quota("api_calls"))
CheckVideoQuota = Depends(check_usage_quota("video_analyses"))
CheckSimulationQuota = Depends(check_usage_quota("simulations"))
