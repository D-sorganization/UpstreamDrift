"""Dependency-free contracts shared by chat and AI packages."""

from .conversation import ArchivedConversationContext, ArchivedMessage
from .credentials import CredentialManager
from .models import (
    DEFAULT_RESPONSE_STYLE,
    RESPONSE_STYLE_PROMPTS,
    ResponseStyle,
    ThinkingCapabilities,
    ThinkingLevel,
    ThinkingLevelName,
    make_full_thinking_capabilities,
    make_none_only_capabilities,
    style_prompt,
)

__all__ = [
    "ArchivedConversationContext",
    "ArchivedMessage",
    "CredentialManager",
    "DEFAULT_RESPONSE_STYLE",
    "RESPONSE_STYLE_PROMPTS",
    "ResponseStyle",
    "ThinkingCapabilities",
    "ThinkingLevel",
    "ThinkingLevelName",
    "make_full_thinking_capabilities",
    "make_none_only_capabilities",
    "style_prompt",
]
