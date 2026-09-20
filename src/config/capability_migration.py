"""Capability baseline inventory and identity preservation contracts (Issue #10510 / ORG-01).

This module defines the canonical data structures and validation logic for
cataloging all capabilities across UpstreamDrift: launcher tiles, provider
models, feature parity contracts, excluded CLI packages, and representative
fixtures.

Contracts enforced:
1. Every entry has a stable, globally unique identifier.
2. Every entry has exactly one primary workspace domain and may declare secondary links.
3. Aliases are acyclic and resolve transitively to an active canonical target.
4. Provider absence alters runtime availability, preserving identity and ownership.
5. Saved layout configurations resolve valid models and reject unknown IDs.
6. Preservation fixtures retain their golden byte hashes and sizes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

ALLOWED_ENTITY_KINDS = frozenset(
    {"tile", "model", "feature", "cli_tool", "service", "library", "workflow_node"}
)

ALLOWED_PRIMARY_WORKSPACES = frozenset(
    {"simulation", "analysis", "capture", "putting", "training", "governance"}
)

ALLOWED_LIFECYCLES = frozenset(
    {
        "active_feature",
        "deprecated_alias",
        "intentionally_headless",
        "planned",
        "unavailable",
        "confirmed_prototype",
        "exempt",
    }
)

ALLOWED_PROVIDER_AUTHORITIES = frozenset(
    {
        "core",
        "tools",
        "mujoco",
        "drake",
        "pinocchio",
        "opensim",
        "movement_optimizer",
    }
)


@dataclass(frozen=True)
class MigrationEntry:
    """Represents the migration disposition and ownership metadata for a capability."""

    id: str
    name: str
    entity_kind: str
    primary_workspace: str
    secondary_links: list[str] = field(default_factory=list)
    provider_authority: str = "core"
    lifecycle: str = "active_feature"
    alias_target: str | None = None
    evidence: str = ""
    acceptance_owner: str = ""

    def __post_init__(self) -> None:
        """Validate invariant preconditions for capability metadata."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError(
                f"Capability ID must be a non-empty string, got: {self.id!r}"
            )
        if self.entity_kind not in ALLOWED_ENTITY_KINDS:
            raise ValueError(
                f"Unknown entity_kind '{self.entity_kind}' for '{self.id}'. "
                f"Allowed: {sorted(ALLOWED_ENTITY_KINDS)}"
            )
        if self.primary_workspace not in ALLOWED_PRIMARY_WORKSPACES:
            raise ValueError(
                f"Unknown primary_workspace '{self.primary_workspace}' for '{self.id}'. "
                f"Allowed: {sorted(ALLOWED_PRIMARY_WORKSPACES)}"
            )
        if self.lifecycle not in ALLOWED_LIFECYCLES:
            raise ValueError(
                f"Unknown lifecycle '{self.lifecycle}' for '{self.id}'. "
                f"Allowed: {sorted(ALLOWED_LIFECYCLES)}"
            )
        if self.provider_authority not in ALLOWED_PROVIDER_AUTHORITIES:
            raise ValueError(
                f"Unknown provider_authority '{self.provider_authority}' for '{self.id}'. "
                f"Allowed: {sorted(ALLOWED_PROVIDER_AUTHORITIES)}"
            )
        if self.lifecycle == "deprecated_alias" and not self.alias_target:
            raise ValueError(
                f"Deprecated alias '{self.id}' must specify a non-empty alias_target"
            )


@dataclass(frozen=True)
class FixtureEntry:
    """Represents a preserved test fixture with byte-exact golden integrity data."""

    path: str
    kind: str
    sha256: str
    size_bytes: int
    support_status: str = "supported"
    notes: str = ""


class CapabilityMigrationInventory:
    """Canonical registry holding capability dispositions, alias graphs, and fixtures."""

    def __init__(
        self,
        version: str = "1.0.0",
        description: str = "",
        workspaces: list[str] | None = None,
        entries: dict[str, MigrationEntry] | None = None,
        fixtures: dict[str, FixtureEntry] | None = None,
        saved_layouts: dict[str, list[str]] | None = None,
    ) -> None:
        self.version = version
        self.description = description
        self.workspaces = (
            list(workspaces)
            if workspaces is not None
            else sorted(ALLOWED_PRIMARY_WORKSPACES)
        )
        self.entries: dict[str, MigrationEntry] = (
            dict(entries) if entries is not None else {}
        )
        self.fixtures: dict[str, FixtureEntry] = (
            dict(fixtures) if fixtures is not None else {}
        )
        self.saved_layouts: dict[str, list[str]] = (
            dict(saved_layouts) if saved_layouts is not None else {}
        )

    def add_entry(self, entry: MigrationEntry) -> None:
        """Register a new capability entry, enforcing uniqueness."""
        if entry.id in self.entries:
            raise ValueError(f"Duplicate capability ID: '{entry.id}'")
        self.entries[entry.id] = entry

    def resolve_alias(self, entity_id: str) -> str:
        """Resolve an alias transitively to its canonical target, rejecting cycles."""
        current = entity_id
        visited: list[str] = [current]
        visited_set: set[str] = {current}

        while True:
            entry = self.entries.get(current)
            if (
                entry is None
                or entry.lifecycle != "deprecated_alias"
                or not entry.alias_target
            ):
                break
            target = entry.alias_target
            if target in visited_set:
                cycle_str = " -> ".join(visited + [target])
                raise ValueError(f"Alias cycle detected: {cycle_str}")
            visited.append(target)
            visited_set.add(target)
            current = target

        return current

    def validate_local_coverage(self, models_path: Path) -> None:
        """Ensure all local desktop models defined in models.yaml are classified."""
        if not models_path.is_file():
            raise FileNotFoundError(f"models.yaml not found: {models_path}")
        raw_data = yaml.safe_load(models_path.read_text(encoding="utf-8"))
        models = raw_data.get("models", [])
        for model in models:
            mid = model.get("id")
            if not mid or mid not in self.entries:
                raise ValueError(f"Unclassified local tile in models.yaml: '{mid}'")

    def validate_provider_model_id(self, provider_model_id: str) -> None:
        """Ensure a provider model ID has an assigned migration entry."""
        if provider_model_id not in self.entries:
            raise ValueError(
                f"Unclassified provider model ID: '{provider_model_id}'. "
                f"Assign a migration disposition in capability_migration.json."
            )

    def validate_manifest_tile(self, tile_data: dict[str, Any]) -> None:
        """Ensure a manifest entry has an assigned migration entry."""
        tile_id = tile_data.get("id")
        if not tile_id or tile_id not in self.entries:
            raise ValueError(
                f"Unclassified manifest entry: '{tile_id}'. "
                f"Assign a migration disposition in capability_migration.json."
            )

    def validate_excluded_cli_package(self, exclusion_data: dict[str, Any]) -> None:
        """Ensure an excluded CLI package has an assigned migration entry."""
        pkg = exclusion_data.get("package")
        if not pkg:
            raise ValueError("Exclusion entry missing package field")
        found = (
            pkg in self.entries
            or f"cli_{pkg}" in self.entries
            or f"tools_{pkg}" in self.entries
        )
        if not found:
            raise ValueError(
                f"Unclassified excluded CLI package: '{pkg}'. "
                f"Assign a migration disposition in capability_migration.json."
            )

    def validate_entry_provenance(self, entry: MigrationEntry, root: Path) -> None:
        """Validate that an entry has non-empty, reachable provenance evidence."""
        if not entry.evidence:
            raise ValueError(f"Empty provenance evidence for capability '{entry.id}'")
        evidence_path = root / entry.evidence
        if not evidence_path.exists():
            raise ValueError(
                f"Missing provenance evidence: '{entry.evidence}' for '{entry.id}'"
            )

    def validate_saved_layout(self, layout_model_ids: list[str]) -> list[str]:
        """Validate and resolve model IDs in a saved layout to canonical targets."""
        resolved: list[str] = []
        for mid in layout_model_ids:
            if mid not in self.entries:
                raise ValueError(
                    f"Unregistered or removed layout model ID '{mid}' cannot be resolved"
                )
            target = self.resolve_alias(mid)
            resolved.append(target)
        return resolved

    def evaluate_availability(
        self, entity_id: str, provider_available: bool
    ) -> dict[str, Any]:
        """Evaluate capability status when a provider is absent or present."""
        if entity_id not in self.entries:
            raise KeyError(f"Capability '{entity_id}' not found in inventory")
        entry = self.entries[entity_id]
        return {
            "id": entry.id,
            "name": entry.name,
            "entity_kind": entry.entity_kind,
            "primary_workspace": entry.primary_workspace,
            "provider_authority": entry.provider_authority,
            "available": provider_available,
            "status": "ready" if provider_available else "offline",
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize inventory to a dictionary."""
        return {
            "version": self.version,
            "description": self.description,
            "workspaces": self.workspaces,
            "entries": {eid: asdict(e) for eid, e in self.entries.items()},
            "fixtures": {fid: asdict(f) for fid, f in self.fixtures.items()},
            "saved_layouts": self.saved_layouts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityMigrationInventory:
        """Deserialize inventory from a dictionary."""
        entries: dict[str, MigrationEntry] = {}
        for eid, edata in data.get("entries", {}).items():
            entries[eid] = MigrationEntry(
                id=edata["id"],
                name=edata["name"],
                entity_kind=edata["entity_kind"],
                primary_workspace=edata["primary_workspace"],
                secondary_links=edata.get("secondary_links", []),
                provider_authority=edata.get("provider_authority", "core"),
                lifecycle=edata.get("lifecycle", "active_feature"),
                alias_target=edata.get("alias_target"),
                evidence=edata.get("evidence", ""),
                acceptance_owner=edata.get("acceptance_owner", ""),
            )

        fixtures: dict[str, FixtureEntry] = {}
        for fid, fdata in data.get("fixtures", {}).items():
            fixtures[fid] = FixtureEntry(
                path=fdata["path"],
                kind=fdata["kind"],
                sha256=fdata["sha256"],
                size_bytes=fdata["size_bytes"],
                support_status=fdata.get("support_status", "supported"),
                notes=fdata.get("notes", ""),
            )

        return cls(
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            workspaces=data.get("workspaces"),
            entries=entries,
            fixtures=fixtures,
            saved_layouts=data.get("saved_layouts"),
        )


def load_migration_inventory(
    path: Path | None = None,
) -> CapabilityMigrationInventory:
    """Load the canonical capability migration baseline inventory from JSON."""
    if path is None:
        path = Path(__file__).parent / "capability_migration.json"
    if not path.is_file():
        raise FileNotFoundError(f"Capability migration JSON file not found: {path}")
    raw_text = path.read_text(encoding="utf-8")
    data = json.loads(raw_text)
    return CapabilityMigrationInventory.from_dict(data)
