"""Backward-compatible credential imports for the chat package."""

from __future__ import annotations

from typing import Any

import shared.python.chat_contracts.credentials as _contracts
from src.shared.python.chat_contracts.credentials import (
    SUPPORTED_PROVIDERS,
)


def _get_keyring() -> Any:
    """Legacy patch point forwarding to the shared implementation."""
    return _contracts._get_keyring()  # noqa: SLF001


class CredentialManager(_contracts.CredentialManager):
    """Compatibility wrapper preserving ``chat.credentials._get_keyring``."""

    def _get_keyring(self) -> Any:
        return _get_keyring()


__all__ = ["CredentialManager", "SUPPORTED_PROVIDERS", "_get_keyring"]
