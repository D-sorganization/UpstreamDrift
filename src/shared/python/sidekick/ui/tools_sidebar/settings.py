"""Backend contracts for per-tab Sidekick settings."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


class TabSettingsDefinition(Protocol):
    """Minimal tab definition contract needed by settings storage."""

    tab_id: str
    settings: SidebarTabSettingsDescriptor | None


@dataclass(frozen=True, slots=True)
class SidebarTabSettingsSchema:
    """Versioned JSON-safe settings schema for one Sidekick tab."""

    version: int = 1
    defaults: Mapping[str, Any] = field(default_factory=dict)
    allowed_keys: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("settings schema version must be explicit and positive")
        _validate_json_mapping(self.defaults)
        if self.allowed_keys is not None:
            unknown = set(self.defaults) - set(self.allowed_keys)
            if unknown:
                raise ValueError(
                    f"settings defaults include unsupported keys: {unknown}"
                )

    def materialize(self, values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return validated values merged over schema defaults."""
        merged = copy.deepcopy(dict(self.defaults))
        if values:
            self.validate(values)
            merged.update(copy.deepcopy(dict(values)))
        return merged

    def validate(self, values: Mapping[str, Any]) -> None:
        """Validate a settings mapping before persistence."""
        _validate_json_mapping(values)
        if self.allowed_keys is None:
            return
        unknown = set(values) - set(self.allowed_keys)
        if unknown:
            raise ValueError(f"settings payload includes unsupported keys: {unknown}")


@dataclass(frozen=True, slots=True)
class SidebarTabSettingsDescriptor:
    """Tab-level settings metadata and optional Qt panel factory."""

    schema: SidebarTabSettingsSchema = field(default_factory=SidebarTabSettingsSchema)
    widget_factory: Callable[[Any, str], Any] | None = None


class SidebarTabSettingsStore:
    """Single validation and persistence API for Sidekick tab settings."""

    def __init__(
        self,
        definitions: Iterable[TabSettingsDefinition],
        state: Any,
    ) -> None:
        self._definitions = {
            definition.tab_id: definition for definition in definitions
        }
        self._raw_settings = _sanitize_settings_payload(
            getattr(state, "tab_settings", {})
        )

    def raw_settings(self) -> dict[str, dict[str, Any]]:
        """Return sanitized raw settings, including stale preserved keys."""
        return copy.deepcopy(self._raw_settings)

    def materialized_settings(self) -> dict[str, dict[str, Any]]:
        """Return current-tab settings with defaults materialized."""
        return {
            tab_id: self.settings_for(tab_id)
            for tab_id in self._definitions
            if tab_id in self._definitions
        }

    def settings_for(self, tab_id: str) -> dict[str, Any]:
        """Return validated settings for ``tab_id``."""
        definition = self._definition(tab_id)
        schema = _schema_for(definition)
        entry = self._raw_settings.get(tab_id, {})
        values = entry.get("values", {}) if isinstance(entry, dict) else {}
        version = int(entry.get("schema_version", schema.version))
        if version != schema.version:
            values = {}
            version = schema.version
        return {
            "schema_version": version,
            "values": schema.materialize(values if isinstance(values, dict) else {}),
        }

    def update_settings(self, tab_id: str, values: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and persist settings for a known tab instance."""
        definition = self._definition(tab_id)
        schema = _schema_for(definition)
        schema.validate(values)
        materialized = {
            "schema_version": schema.version,
            "values": schema.materialize(values),
        }
        self._raw_settings[tab_id] = copy.deepcopy(materialized)
        return copy.deepcopy(materialized)

    def _definition(self, tab_id: str) -> TabSettingsDefinition:
        try:
            return self._definitions[tab_id]
        except KeyError as exc:
            raise KeyError(f"Unknown sidebar tab id: {tab_id}") from exc


def _schema_for(definition: TabSettingsDefinition) -> SidebarTabSettingsSchema:
    descriptor = definition.settings
    if descriptor is None:
        return SidebarTabSettingsSchema()
    return descriptor.schema


def _sanitize_settings_payload(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for raw_key, raw_entry in value.items():
        key = str(raw_key).strip()
        if not key or not isinstance(raw_entry, dict):
            continue
        values = raw_entry.get("values", {})
        if not isinstance(values, dict) or not _is_json_safe(values):
            continue
        try:
            version = int(raw_entry.get("schema_version", 1))
        except (TypeError, ValueError):
            version = 1
        result[key] = {
            "schema_version": max(1, version),
            "values": copy.deepcopy(values),
        }
    return result


def _validate_json_mapping(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("settings payload must be a mapping")
    if any(not isinstance(key, str) or not key.strip() for key in value):
        raise ValueError("settings keys must be non-empty strings")
    if not _is_json_safe(value):
        raise ValueError("settings payload must be JSON-safe")


def _is_json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True
