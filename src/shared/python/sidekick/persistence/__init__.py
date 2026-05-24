"""Shared persistence schemas for Sidekick state."""

from __future__ import annotations

from .schema import PROFILE_SCHEMA_VERSION, PROFILE_SCHEMA_VERSION_KEY, ProfilePayload

__all__ = [
    "PROFILE_SCHEMA_VERSION",
    "PROFILE_SCHEMA_VERSION_KEY",
    "ProfilePayload",
]
