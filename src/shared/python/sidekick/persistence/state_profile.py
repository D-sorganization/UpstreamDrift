"""Canonical state-profile payload helpers for round-trip between hosts.

This module is the single source of truth for the JSON shape of a Sidekick
state profile shared between the **embedded** sidebar
(``sidekick.ui.tools_sidebar.state_profiles.SidekickStateProfileStore``) and
the **standalone** session store
(``sidekick.standalone.session_store.StandaloneSessionStore``).

The contract is intentionally minimal: a profile is a JSON object whose
top-level keys are *the dataclass fields of* ``SidebarState`` (the embedded
shape) plus the well-known ``schema_version`` marker
(:data:`PROFILE_SCHEMA_VERSION_KEY`). Unknown top-level keys are preserved on
load so that profiles written by a newer Sidekick can still be loaded — and
resaved — by an older one without silent data loss.

Schema version bump rules
-------------------------

- ``LEGACY_SCHEMA_VERSION`` (= 0) means *no version key present*. Profiles
  written by embedded mode before this module existed are loaded as
  ``LEGACY_SCHEMA_VERSION`` and a :class:`SchemaMigration` warning is emitted
  exactly once per load.
- :data:`~sidekick.persistence.schema.PROFILE_SCHEMA_VERSION` (= 1) is the
  current shape — same key set as ``LEGACY_SCHEMA_VERSION`` plus the
  ``schema_version`` marker. No data migration required.
- Future schema bumps SHOULD ship a corresponding entry in
  :data:`MIGRATION_TABLE` so old profiles continue to load via a migrating
  callable. The migration must be **in-memory on load** — never rewrite the
  on-disk profile silently.

Preconditions
-------------

- ``wrap_state(state)`` requires ``state`` to be a ``dict[str, Any]``.
- ``unwrap_payload(payload)`` requires ``payload`` to be a ``dict[str, Any]``
  or a :class:`~sidekick.persistence.schema.ProfilePayload`.
- ``validate(payload)`` raises :class:`ValueError` whose message points at the
  JSON path of the first offending key.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

from .schema import (
    PROFILE_SCHEMA_VERSION,
    PROFILE_SCHEMA_VERSION_KEY,
    ProfilePayload,
)

__all__ = [
    "LEGACY_SCHEMA_VERSION",
    "MIGRATION_TABLE",
    "SchemaMigration",
    "current_schema_version",
    "unwrap_payload",
    "validate",
    "wrap_state",
]


LEGACY_SCHEMA_VERSION = 0
"""Sentinel for profiles written before ``schema_version`` was introduced."""


class SchemaMigration(UserWarning):
    """Emitted when a profile is migrated in-memory on load.

    The warning carries a human-readable description of what was migrated.
    Callers may assert on the warning with ``pytest.warns(SchemaMigration)``.
    """


def current_schema_version() -> int:
    """Return the schema version currently produced by :func:`wrap_state`."""
    return PROFILE_SCHEMA_VERSION


def _migrate_legacy_to_v1(raw: dict[str, Any]) -> dict[str, Any]:
    """Tag a legacy embedded payload with the current schema version.

    Preserves all unknown top-level keys for forward compatibility.
    """
    migrated = dict(raw)
    migrated[PROFILE_SCHEMA_VERSION_KEY] = PROFILE_SCHEMA_VERSION
    return migrated


MIGRATION_TABLE: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    LEGACY_SCHEMA_VERSION: _migrate_legacy_to_v1,
}
"""Maps ``from_version -> migrator(raw_payload) -> upgraded_payload``.

A migrator must be pure and idempotent: calling it twice on the same raw
payload must produce the same upgraded payload.
"""


def wrap_state(state: Mapping[str, Any]) -> ProfilePayload:
    """Wrap an embedded ``SidebarState.to_dict()`` mapping in a canonical payload.

    Preconditions:
        ``state`` is a mapping; any top-level ``schema_version`` value already
        in ``state`` is ignored and replaced by the current version.

    Postconditions:
        Returned payload satisfies :func:`validate` and round-trips through
        ``ProfilePayload.from_dict(payload.to_dict())``.
    """
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping (dict-like)")
    data = {k: v for k, v in state.items() if k != PROFILE_SCHEMA_VERSION_KEY}
    return ProfilePayload(data=dict(data), schema_version=PROFILE_SCHEMA_VERSION)


def unwrap_payload(
    payload: ProfilePayload | Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    """Return ``(state_dict, schema_version)`` from a canonical or legacy payload.

    Emits exactly one :class:`SchemaMigration` warning when migrating a
    payload that lacks ``schema_version`` (legacy embedded shape).

    Preconditions:
        ``payload`` is a :class:`ProfilePayload` or a JSON-object mapping.

    Postconditions:
        ``state_dict`` excludes the ``schema_version`` key. Unknown top-level
        keys are preserved verbatim. The returned ``schema_version`` is always
        :data:`~sidekick.persistence.schema.PROFILE_SCHEMA_VERSION` (migration
        is applied in-memory if needed).
    """
    if isinstance(payload, ProfilePayload):
        return dict(payload.data), payload.schema_version
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a ProfilePayload or mapping")

    raw = dict(payload)
    incoming_version = raw.get(PROFILE_SCHEMA_VERSION_KEY)
    if incoming_version is None:
        warnings.warn(
            "Loading legacy Sidekick state profile (no schema_version): "
            f"migrating in-memory to v{PROFILE_SCHEMA_VERSION}.",
            SchemaMigration,
            stacklevel=2,
        )
        raw = _migrate_legacy_to_v1(raw)
        incoming_version = LEGACY_SCHEMA_VERSION
    elif not isinstance(incoming_version, int) or incoming_version < 0:
        raise ValueError(f"$.{PROFILE_SCHEMA_VERSION_KEY}: must be a non-negative int")

    # Future-proof: walk the migration table for any non-current versions.
    version = int(incoming_version)
    while version < PROFILE_SCHEMA_VERSION:
        migrator = MIGRATION_TABLE.get(version)
        if migrator is None:
            raise ValueError(
                f"$.{PROFILE_SCHEMA_VERSION_KEY}: no migrator registered for "
                f"version {version}"
            )
        raw = migrator(raw)
        version += 1

    state_dict = {k: v for k, v in raw.items() if k != PROFILE_SCHEMA_VERSION_KEY}
    return state_dict, PROFILE_SCHEMA_VERSION


def validate(payload: Mapping[str, Any]) -> None:
    """Raise :class:`ValueError` if ``payload`` is not a canonical profile.

    The error message identifies the offending JSON path (e.g.
    ``$.schema_version``) so callers can surface actionable diagnostics.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("$: profile payload must be a JSON object")
    if PROFILE_SCHEMA_VERSION_KEY not in payload:
        raise ValueError(f"$.{PROFILE_SCHEMA_VERSION_KEY}: required key missing")
    version = payload[PROFILE_SCHEMA_VERSION_KEY]
    if not isinstance(version, int) or version < 0:
        raise ValueError(f"$.{PROFILE_SCHEMA_VERSION_KEY}: must be a non-negative int")
    if version > PROFILE_SCHEMA_VERSION:
        # Forward-compat: a newer-than-known version is loadable but we cannot
        # validate keys we do not know. Accept silently — unwrap_payload
        # preserves unknown keys.
        return
