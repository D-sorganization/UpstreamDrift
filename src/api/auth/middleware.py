"""Authentication middleware that respects local mode."""

import os

from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


# Check deployment mode
def is_local_mode() -> bool:
    """Check if running in local mode (no auth required)."""
    return (
        os.environ.get("GOLF_SUITE_MODE", "local") == "local"
        or os.environ.get("GOLF_AUTH_DISABLED", "false").lower() == "true"
    )


class LocalUser:
    """Mock user for local mode - full access, no restrictions."""

    def __init__(self) -> None:
        """Initialize local user with full permissions."""
        self.id: str = "local-user"
        self.email: str = "local@localhost"
        self.role: str = "ADMIN"  # Full access locally
        self.quota_remaining: float = float("inf")

    def has_permission(self, permission: str) -> bool:
        """Check whether the local user has the given permission."""
        return True  # Everything allowed locally


class OptionalAuth(HTTPBearer):
    """Bearer auth that's optional in local mode."""

    def __init__(self, auto_error: bool = True) -> None:
        super().__init__(auto_error=auto_error)

    async def __call__(
        self, request: Request
    ) -> HTTPAuthorizationCredentials | LocalUser | None:
        if is_local_mode():
            # Local mode: no auth required, return mock user
            return LocalUser()

        # Cloud mode: require real authentication
        return await super().__call__(request)
