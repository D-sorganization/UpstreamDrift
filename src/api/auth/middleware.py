"""Authentication middleware that respects local mode."""

from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.environment import is_production_environment, resolve_environment
from src.shared.python.config.environment import get_env_bool, is_auth_disabled

# Bind addresses that do not expose the process beyond the local machine.
_LOOPBACK_BIND_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


# Check deployment mode
def is_local_mode() -> bool:
    """Check if running in local mode (no auth required)."""
    return is_auth_disabled()


def assert_local_admin_bypass_allowed(host: str) -> None:
    """Fail closed when ``GOLF_AUTH_DISABLED`` would mint an ADMIN user.

    ``LocalUser`` grants an unrestricted ADMIN role and infinite quota.
    That bypass is only valid for a loopback bind outside production.

    Preconditions:
        ``host`` is the address the API process binds.

    Postconditions:
        Returns when the bypass flag is off, or when it is on, ``host`` is
        loopback, and the process is not production.

    Raises:
        TypeError: ``host`` is not a string.
        ValueError: ``host`` is blank.
        RuntimeError: the bypass is enabled in production or on a
            non-loopback bind.
    """
    if not isinstance(host, str):
        raise TypeError("host must be a string bind address")
    if not host.strip():
        raise ValueError("host must be a non-empty bind address")
    if not get_env_bool("GOLF_AUTH_DISABLED", default=False):
        return
    normalized = host.strip().lower()
    if is_production_environment() or normalized not in _LOOPBACK_BIND_HOSTS:
        raise RuntimeError(
            "GOLF_AUTH_DISABLED=true grants LocalUser unrestricted ADMIN "
            "and infinite quota. Refusing to start unless the process is "
            "outside production and bound to a loopback host. "
            f"environment={resolve_environment()!r} host={host.strip()!r}"
        )


class LocalUser:
    """Mock user for local mode - full access, no restrictions.

    Construction enforces :func:`assert_local_admin_bypass_allowed` against
    the configured API bind address so the ADMIN grant cannot be minted when
    ``GOLF_AUTH_DISABLED`` is set in production or on a public interface.
    """

    def __init__(self) -> None:
        """Initialize local user with full permissions.

        Preconditions:
            The configured server bind host is a non-empty string.

        Postconditions:
            ``role`` is ``ADMIN`` and ``quota_remaining`` is infinite when
            the auth-disabled bypass is allowed.

        Raises:
            RuntimeError: the bypass is enabled off loopback or in production.
        """
        from src.api.config import get_server_host

        assert_local_admin_bypass_allowed(get_server_host())
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

    async def __call__(  # type: ignore[override]
        self, request: Request
    ) -> HTTPAuthorizationCredentials | LocalUser | None:
        if is_local_mode():
            # Local mode: no auth required, return mock user
            return LocalUser()

        # Cloud mode: require real authentication
        return await super().__call__(request)
