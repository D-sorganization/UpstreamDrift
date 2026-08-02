"""Shared credential manager used by chat and AI packages."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = frozenset(
    {"ollama", "openai", "anthropic", "gemini", "cline", "codex"}
)

_ENV_KEY_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "codex": "OPENAI_API_KEY",
}

_keyring: Any = None
_keyring_available: bool | None = None


def _get_keyring() -> Any:
    """Lazy-load keyring, returning None if unavailable."""
    global _keyring, _keyring_available  # noqa: PLW0603
    if _keyring_available is None:
        try:
            import keyring as kr

            _keyring = kr
            _keyring_available = True
            logger.debug("keyring package available for credential storage")
        except ImportError:
            _keyring = None
            _keyring_available = False
            logger.warning(
                "keyring package not available - falling back to env vars. "
                "Install with: pip install keyring"
            )
    return _keyring


class CredentialManager:
    """Manages AI provider credentials with OS keyring storage."""

    def __init__(self, service_name: str = "shared_chat_ai") -> None:
        if not service_name or not service_name.strip():
            raise ValueError("service_name must be a non-empty string")
        self._service_name = service_name

    @property
    def service_name(self) -> str:
        """Return the keyring service namespace."""
        return self._service_name

    def _get_keyring(self) -> Any:
        """Return the keyring backend used by this manager instance."""
        return _get_keyring()

    def store_api_key(self, provider: str, api_key: str) -> bool:
        """Store an API key in the OS keyring."""
        if not provider or not provider.strip():
            raise ValueError("provider must be a non-empty string")
        if not api_key or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")

        provider = provider.lower().strip()
        kr = self._get_keyring()
        if kr is None:
            logger.warning(
                "Cannot store API key for %s - keyring not available",
                provider,
            )
            return False

        try:
            username = f"{self._service_name}_{provider}"
            kr.set_password(self._service_name, username, api_key)
            logger.info("Stored API key for provider: %s", provider)
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Failed to store API key for %s", provider)
            return False

    def get_api_key(self, provider: str) -> str | None:
        """Retrieve an API key, checking keyring first then env vars."""
        if not provider or not provider.strip():
            raise ValueError("provider must be a non-empty string")

        provider = provider.lower().strip()
        kr = self._get_keyring()
        if kr is not None:
            try:
                username = f"{self._service_name}_{provider}"
                key = kr.get_password(self._service_name, username)
                if isinstance(key, str):
                    return key
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Keyring lookup failed for %s, falling back to env",
                    provider,
                )

        env_var = _ENV_KEY_MAP.get(provider)
        if env_var:
            key = os.environ.get(env_var)
            if isinstance(key, str):
                logger.debug("Using env var %s for provider %s", env_var, provider)
                return key

        return None

    def delete_api_key(self, provider: str) -> bool:
        """Remove an API key from the keyring."""
        if not provider or not provider.strip():
            raise ValueError("provider must be a non-empty string")

        provider = provider.lower().strip()
        kr = self._get_keyring()
        if kr is None:
            return False

        try:
            username = f"{self._service_name}_{provider}"
            kr.delete_password(self._service_name, username)
            logger.info("Deleted API key for provider: %s", provider)
            return True
        except Exception:  # noqa: BLE001
            logger.debug("Failed to delete key for %s", provider)
            return False

    def list_configured_providers(self) -> list[str]:
        """List providers that have API keys configured."""
        configured: list[str] = []
        for provider in SUPPORTED_PROVIDERS:
            if self.get_api_key(provider) is not None:
                configured.append(provider)
        return sorted(configured)

    def has_credentials(self, provider: str) -> bool:
        """Check if credentials are available for a provider."""
        return self.get_api_key(provider) is not None
