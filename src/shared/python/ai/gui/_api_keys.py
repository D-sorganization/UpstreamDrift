"""API key helpers for the AI settings dialog.

Provides ``get_api_key``/``set_api_key``/``delete_api_key`` keyed by
:class:`AIProvider`. Persistence still uses the per-provider ``key_service``
strings declared in ``PROVIDER_INFO`` so previously-stored keys remain
readable. The lower-level keyring access goes through
:class:`src.shared.python.chat.credentials.CredentialManager` whenever
possible to keep keyring access centralised in one well-tested place.
"""

from __future__ import annotations

from src.shared.python.ai.gui._provider_registry_data import (
    PROVIDER_INFO,
    AIProvider,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


def _service_name(provider: AIProvider) -> str | None:
    info = PROVIDER_INFO.get(provider)
    if not info or not info.get("requires_key"):
        return None
    service_name = info.get("key_service", "")
    if not service_name or not isinstance(service_name, str):
        return None
    return service_name


def _credential_manager(service: str):  # noqa: ANN202 - lazy import for testability
    from src.shared.python.chat.credentials import CredentialManager

    return CredentialManager(service_name=service)


def get_api_key(provider: AIProvider) -> str | None:
    """Get API key from secure storage.

    Args:
        provider: Provider to get key for.

    Returns:
        API key if found, ``None`` otherwise.
    """
    service = _service_name(provider)
    if service is None:
        return None
    try:
        # Preserve historical keyring layout: ``service=service_name``,
        # ``username="api_key"``. CredentialManager namespaces by username
        # internally, so we go direct to ``keyring`` for read compatibility.
        import keyring

        result = keyring.get_password(service, "api_key")
        return result if isinstance(result, str) else None
    except ImportError:
        logger.warning("keyring package not installed for secure key storage")
        return None
    except (RuntimeError, TypeError, AttributeError) as e:
        logger.warning("Failed to get API key from keyring: %s", e)
        return None


def set_api_key(provider: AIProvider, key: str) -> bool:
    """Store API key in secure storage."""
    if provider is None:
        raise ValueError("provider must be provided")
    service = _service_name(provider)
    if service is None:
        return False
    try:
        import keyring

        keyring.set_password(service, "api_key", key)
        logger.info("Stored API key for %s", provider.name)
        return True
    except ImportError:
        logger.warning("keyring package not installed for secure key storage")
        return False
    except (RuntimeError, TypeError, AttributeError) as e:
        logger.warning("Failed to store API key: %s", e)
        return False


def delete_api_key(provider: AIProvider) -> bool:
    """Delete API key from secure storage."""
    service = _service_name(provider)
    if service is None:
        return False
    try:
        import keyring

        keyring.delete_password(service, "api_key")
        logger.info("Deleted API key for %s", provider.name)
        return True
    except ImportError:
        return False
    except (RuntimeError, TypeError, AttributeError) as e:
        logger.warning("Failed to delete API key: %s", e)
        return False


__all__ = ["delete_api_key", "get_api_key", "set_api_key"]
