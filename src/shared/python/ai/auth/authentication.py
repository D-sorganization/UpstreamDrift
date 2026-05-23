"""Authentication and Subscription Management for AI Features.

This module provides user authentication and subscription management
for AI-powered features in the UpstreamDrift application.

Features:
    - User authentication via API keys
    - Subscription tier management (Free, Pro, Enterprise)
    - Feature gating based on subscription level
    - Secure token storage and refresh

Note:
    OAuth and email/password authentication methods are NOT implemented.
    ``login_with_oauth`` and ``login_with_email_password`` raise
    ``NotImplementedError`` unconditionally (Phase 1 of issue #2757).
    Real OAuth (PKCE + token exchange + refresh tokens) is tracked in the
    Phase 2 follow-up issue.  Until then, ``is_authenticated`` will always
    return False unless a valid API key is supplied via ``login_with_api_key``.

Example:
    >>> from src.shared.python.ai.auth.authentication import AuthManager
    >>> auth = AuthManager()
    >>> if auth.is_subscribed("pro"):
    ...     # Enable pro features
"""

from __future__ import annotations

import contextlib
import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class SubscriptionTier(Enum):
    """Subscription tier levels."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class UserProfile:
    """User profile information.

    Attributes:
        user_id: Unique user identifier.
        email: User email address.
        subscription_tier: Current subscription level.
        subscription_expires: When the subscription expires.
        api_key: User's API key for authentication.
        created_at: Account creation timestamp.
        features_enabled: List of enabled features.
    """

    user_id: str
    email: str = ""
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    subscription_expires: datetime | None = None
    api_key: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    features_enabled: list[str] = field(default_factory=list)

    def is_active(self) -> bool:
        """Check if subscription is currently active.

        Returns:
            True if subscription is active, False otherwise.
        """
        if self.subscription_tier == SubscriptionTier.FREE:
            return True
        if self.subscription_expires is None:
            return True
        return datetime.now() < self.subscription_expires

    def has_feature(self, feature: str) -> bool:
        """Check if a feature is enabled for this user.

        Args:
            feature: Feature name to check.

        Returns:
            True if feature is enabled, False otherwise.
        """
        if not self.is_active():
            return False
        return feature in self.features_enabled or self._tier_includes_feature(feature)

    _TIER_FEATURES = {
        SubscriptionTier.FREE: {
            "ollama_chat",
            "basic_tools",
            "local_models",
        },
        SubscriptionTier.PRO: {
            "ollama_chat",
            "basic_tools",
            "local_models",
            "claude_code",
            "codex_cli",
            "cloud_models",
            "priority_support",
            "advanced_tools",
        },
        SubscriptionTier.ENTERPRISE: {
            "ollama_chat",
            "basic_tools",
            "local_models",
            "claude_code",
            "codex_cli",
            "cloud_models",
            "priority_support",
            "advanced_tools",
            "custom_integrations",
            "dedicated_support",
            "sso_auth",
            "audit_logs",
        },
    }

    def _tier_includes_feature(self, feature: str) -> bool:
        """Check if the subscription tier includes a feature.

        Args:
            feature: Feature name to check.

        Returns:
            True if tier includes feature, False otherwise.
        """
        return feature in self._TIER_FEATURES.get(self.subscription_tier, set())


@dataclass
class AuthToken:
    """Authentication token.

    Attributes:
        token: The token string.
        token_type: Type of token (access, refresh).
        expires_at: When the token expires.
        scope: Permissions granted by this token.
    """

    token: str
    token_type: str = "access"
    expires_at: datetime = field(
        default_factory=lambda: datetime.now() + timedelta(hours=1)
    )
    scope: list[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Check if token is still valid.

        Returns:
            True if token is valid, False if expired.
        """
        return datetime.now() < self.expires_at


class AuthManager:
    """Authentication and subscription manager.

    Manages user authentication, subscription status, and feature access.
    Stores credentials securely and handles token refresh.

    Example:
        >>> auth = AuthManager()
        >>> auth.login_with_api_key("your-api-key")
        >>> if auth.current_user and auth.current_user.has_feature("claude_code"):
        ...     # Enable Claude Code integration
    """

    CREDENTIALS_FILE = Path.home() / ".golf_modeling_suite" / "auth_credentials.json"

    def __init__(self) -> None:
        """Initialize authentication manager."""
        self._current_user: UserProfile | None = None
        self._access_token: AuthToken | None = None
        self._refresh_token: AuthToken | None = None
        self._load_credentials()

    def _load_credentials(self) -> None:
        """Load stored credentials from disk."""
        if not self.CREDENTIALS_FILE.exists():
            return

        try:
            import json

            data = json.loads(self.CREDENTIALS_FILE.read_text(encoding="utf-8"))

            if "user" in data:
                user_data = data["user"]
                self._current_user = UserProfile(
                    user_id=user_data.get("user_id", ""),
                    email=user_data.get("email", ""),
                    subscription_tier=SubscriptionTier(
                        user_data.get("subscription_tier", "free")
                    ),
                    subscription_expires=(
                        datetime.fromisoformat(user_data["subscription_expires"])
                        if user_data.get("subscription_expires")
                        else None
                    ),
                    api_key=user_data.get("api_key", ""),
                    created_at=(
                        datetime.fromisoformat(user_data["created_at"])
                        if user_data.get("created_at")
                        else datetime.now()
                    ),
                    features_enabled=user_data.get("features_enabled", []),
                )

            if "access_token" in data:
                token_data = data["access_token"]
                self._access_token = AuthToken(
                    token=token_data.get("token", ""),
                    token_type=token_data.get("token_type", "access"),
                    expires_at=(
                        datetime.fromisoformat(token_data["expires_at"])
                        if token_data.get("expires_at")
                        else datetime.now()
                    ),
                    scope=token_data.get("scope", []),
                )

            logger.info(
                "Loaded auth profile for user: %s",
                self._current_user.user_id if self._current_user else "none",
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to load auth profile: %s", e)

    def _save_credentials(self) -> None:
        """Save credentials to disk securely."""
        import json

        data: dict[str, Any] = {}

        if self._current_user:
            data["user"] = {
                "user_id": self._current_user.user_id,
                "email": self._current_user.email,
                "subscription_tier": self._current_user.subscription_tier.value,
                "subscription_expires": (
                    self._current_user.subscription_expires.isoformat()
                    if self._current_user.subscription_expires
                    else None
                ),
                "api_key": self._current_user.api_key,
                "created_at": self._current_user.created_at.isoformat(),
                "features_enabled": self._current_user.features_enabled,
            }

        if self._access_token:
            data["access_token"] = {
                "token": self._access_token.token,
                "token_type": self._access_token.token_type,
                "expires_at": self._access_token.expires_at.isoformat(),
                "scope": self._access_token.scope,
            }

        self.CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.CREDENTIALS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

        # Set restrictive file permissions on Unix-like systems
        with contextlib.suppress(OSError, RuntimeError, AttributeError):
            self.CREDENTIALS_FILE.chmod(0o600)

    def login_with_api_key(self, api_key: str) -> bool:
        """Login using an API key.

        Args:
            api_key: User's API key.

        Returns:
            True if login successful, False otherwise.
        """
        if not api_key:
            return False

        # Generate user ID from API key hash (simplified - in production, verify
        # with server)
        user_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        self._current_user = UserProfile(
            user_id=user_id,
            api_key=api_key,
            subscription_tier=SubscriptionTier.PRO,  # Default to PRO for API key users
            subscription_expires=datetime.now() + timedelta(days=365),
            features_enabled=[],
        )

        # Create access token
        self._access_token = AuthToken(
            token=secrets.token_urlsafe(32),
            token_type="access",
            expires_at=datetime.now() + timedelta(hours=24),
            scope=["chat", "tools", "models"],
        )

        self._save_credentials()
        logger.info("Logged in via key for user: %s", user_id)
        return True

    def login_with_oauth(self, provider: str, auth_code: str) -> bool:
        """Login using OAuth provider.

        Args:
            provider: OAuth provider name (google, github, etc.).
            auth_code: Authorization code from OAuth flow.

        Returns:
            Never returns — always raises ``NotImplementedError``.

        Raises:
            NotImplementedError: Always. Real OAuth (PKCE + token exchange +
                refresh-token handling) is deferred to Phase 2 (issue #5227).
                To use authenticated features, configure provider credentials
                directly via the keyring (chat/credentials.py) and supply an
                API key via ``login_with_api_key`` instead.

        Note:
            Phase 1 of issue #2757 removes the previously fabricated
            ``UserProfile`` that this method used to return so that callers
            can no longer be misled into trusting a fake identity.
        """
        raise NotImplementedError(
            f"OAuth login for provider {provider!r} is not implemented (issue #5227). "
            "To use authenticated features, configure provider credentials directly "
            "via the keyring (chat/credentials.py) and skip the OAuth flow."
        )

    def login_with_email_password(self, email: str, password: str) -> bool:
        """Login using email and password.

        Args:
            email: User email address.
            password: User password.

        Returns:
            Never returns — always raises ``NotImplementedError``.

        Raises:
            NotImplementedError: Always. Email/password authentication requires
                a backend service (e.g. Supabase, Auth0) that has not yet been
                selected or configured (Phase 2, issue #5227).
                To use authenticated features, supply an API key via
                ``login_with_api_key`` instead.

        Note:
            Phase 1 of issue #2757 removes the previously fabricated
            ``UserProfile`` that this method used to return so that callers
            can no longer be misled into trusting a fake identity.
        """
        raise NotImplementedError(
            f"Email/password login for {email!r} is not implemented (issue #5227). "
            "To use authenticated features, supply an API key via login_with_api_key. "
            "Email/password auth requires a backend service — see Phase 2 of #2757."
        )

    def logout(self) -> None:
        """Logout and clear credentials."""
        self._current_user = None
        self._access_token = None
        self._refresh_token = None

        if self.CREDENTIALS_FILE.exists():
            self.CREDENTIALS_FILE.unlink()

        logger.info("Logged out")

    @property
    def current_user(self) -> UserProfile | None:
        """Get current user profile."""
        return self._current_user

    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated.

        Returns:
            True only when a ``UserProfile`` is present and its subscription
            is active.  After a refused login attempt (``login_with_oauth`` or
            ``login_with_email_password`` raising ``NotImplementedError``) no
            profile is set, so this property returns False.  Callers must check
            this explicitly before consuming gated features.
        """
        return self._current_user is not None and self._current_user.is_active()

    @property
    def subscription_tier(self) -> SubscriptionTier:
        """Get current subscription tier."""
        if not self._current_user:
            return SubscriptionTier.FREE
        return self._current_user.subscription_tier

    def has_feature(self, feature: str) -> bool:
        """Check if a feature is available for current user.

        Args:
            feature: Feature name to check.

        Returns:
            True if feature is available, False otherwise.
        """
        if not self._current_user:
            return False
        return self._current_user.has_feature(feature)

    def upgrade_subscription(
        self, tier: SubscriptionTier, duration_days: int = 30
    ) -> None:
        """Upgrade subscription tier.

        Args:
            tier: New subscription tier.
            duration_days: Subscription duration in days.
        """
        if not self._current_user:
            raise ValueError("No user logged in")

        self._current_user.subscription_tier = tier
        self._current_user.subscription_expires = datetime.now() + timedelta(
            days=duration_days
        )

        # Update features based on tier
        self._current_user.features_enabled = list(
            UserProfile._TIER_FEATURES.get(tier, set())
        )

        self._save_credentials()
        logger.info(
            "Upgraded subscription to %s for %d days", tier.value, duration_days
        )

    def get_api_key(self) -> str | None:
        """Get current user's API key.

        Returns:
            API key if available, None otherwise.
        """
        if self._current_user:
            return self._current_user.api_key
        return None

    def refresh_token_if_needed(self) -> bool:
        """Refresh access token if expired.

        Returns:
            True if token is valid (or refreshed), False if refresh failed.
        """
        if not self._access_token or not self._access_token.is_valid():
            if self._refresh_token and self._refresh_token.is_valid():
                # Issue #5227: Exchange refresh token for new access token
                logger.info("Refreshing access token")
                return True
            return False
        return True


class FeatureGate:
    """Feature gate decorator for AI features.

    Use this decorator to restrict access to features based on
    subscription tier.

    Example:
        @FeatureGate.require("claude_code")
        def use_claude_code():
            # Only accessible to users with claude_code feature
    """

    _auth: AuthManager | None = None

    @classmethod
    def _get_auth(cls) -> AuthManager:
        """Get or create auth manager instance."""
        if cls._auth is None:
            cls._auth = AuthManager()
        return cls._auth

    @classmethod
    def require(cls, feature: str):
        """Decorator to require a feature for access.

        Args:
            feature: Required feature name.

        Returns:
            Decorator function.
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                auth = cls._get_auth()
                if not auth.has_feature(feature):
                    raise PermissionError(
                        f"Feature '{feature}' requires a higher subscription tier. "
                        f"Current tier: {auth.subscription_tier.value}"
                    )
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @classmethod
    def require_tier(cls, tier: SubscriptionTier):
        """Decorator to require a minimum subscription tier.

        Args:
            tier: Minimum required tier.

        Returns:
            Decorator function.
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                auth = cls._get_auth()
                tier_order = {
                    SubscriptionTier.FREE: 0,
                    SubscriptionTier.PRO: 1,
                    SubscriptionTier.ENTERPRISE: 2,
                }
                if tier_order.get(auth.subscription_tier, 0) < tier_order.get(tier, 0):
                    raise PermissionError(
                        f"This feature requires {tier.value} subscription or higher. "
                        f"Current tier: {auth.subscription_tier.value}"
                    )
                return func(*args, **kwargs)

            return wrapper

        return decorator


# Global auth manager instance
_auth_manager: AuthManager | None = None


def get_auth_manager() -> AuthManager:
    """Get the global authentication manager.

    Returns:
        AuthManager instance.
    """
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager
