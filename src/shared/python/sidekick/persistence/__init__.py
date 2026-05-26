"""Shared persistence schemas for Sidekick state."""

from __future__ import annotations

from .schema import PROFILE_SCHEMA_VERSION, PROFILE_SCHEMA_VERSION_KEY, ProfilePayload
from .state_profile import (
    LEGACY_SCHEMA_VERSION,
    SchemaMigration,
    current_schema_version,
    unwrap_payload,
    validate,
    wrap_state,
)

__all__ = [
    "LEGACY_SCHEMA_VERSION",
    "PROFILE_SCHEMA_VERSION",
    "PROFILE_SCHEMA_VERSION_KEY",
    "ProfilePayload",
    "SchemaMigration",
    "current_schema_version",
    "unwrap_payload",
    "validate",
    "wrap_state",
]
