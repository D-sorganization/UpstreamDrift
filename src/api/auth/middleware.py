"""Authentication middleware that respects local mode."""

from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.shared.python.config.environment import is_auth_disabled


# Check deployment mode
def is_local_mode() -> bool:
    """Check if running in local mode (no auth required)."""
    return is_auth_disabled()


class LocalUser:
    """Mock user for local mode - full access, no restrictions.

    Must satisfy every ``User`` attribute that auth/quota code reads, otherwise
    local mode raises ``AttributeError`` on paths that work in cloud mode (a
    Liskov-substitution failure — issue #7142). The drift-guard test
    ``tests/api/test_local_user_contract.py`` ties this to the ``User`` model so
    a new column on ``User`` fails fast here.
    """

    def __init__(self) -> None:
        """Initialize local user with full permissions."""
        self.id: str = "local-user"
        self.email: str = "local@localhost"
        # Use the real UserRole *value* ("admin") so role checks and
        # UserRole(user.role) coercion in quota code succeed (issue #7142).
        self.role: str = "admin"  # Full access locally
        self.quota_remaining: float = float("inf")
        # Account-status fields consumed by auth code paths.
        self.is_active: bool = True
        self.is_verified: bool = True
        self.subscription_status: str = "active"
        # Quota counters consumed by usage_tracker.check_quota / increment_usage.
        # Reset each construction so local mode never hits a quota wall.
        self.api_calls_this_month: int = 0
        self.video_analyses_this_month: int = 0
        self.simulations_this_month: int = 0

    def has_permission(self, permission: str) -> bool:
        """Check whether the local user has the given permission."""
        return True  # Everything allowed locally


class OptionalAuth(HTTPBearer):
    """Bearer auth that's optional in local mode."""

    def __init__(self, auto_error: bool = True) -> None:
        super().__init__(auto_error=auto_error)

    async def __call__(  # type: ignore[override]
        self, request: Request
    ) -> HTTPAuthorizationCredentials | LocalUser | None:
        if is_local_mode():
            # Local mode: no auth required, return mock user
            return LocalUser()

        # Cloud mode: require real authentication
        return await super().__call__(request)
