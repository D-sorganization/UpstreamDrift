"""Versioned model-pack manifest contracts for shared launcher integration.

This module provides the canonical manifest representation for discoverable
model packs. It is designed to support both the current local registry flow
and future external provider repositories without changing launcher callers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from src.shared.python.contracts import ContractChecker, ensure, require


def _normalize_string_sequence(
    values: list[Any] | tuple[Any, ...] | None,
) -> tuple[str, ...]:
    """Normalize a sequence of strings into a unique, deterministic tuple."""
    if values is None:
        return ()

    require(isinstance(values, list | tuple), "sequence fields must be lists or tuples")

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        require(isinstance(value, str), "sequence entries must be strings", value)
        item = value.strip().lower()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _normalize_path_sequence(
    values: list[Any] | tuple[Any, ...] | None,
) -> tuple[str, ...]:
    """Normalize path-like strings into a unique, deterministic tuple."""
    if values is None:
        return ()

    require(isinstance(values, list | tuple), "path fields must be lists or tuples")

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        require(isinstance(value, str), "path entries must be strings", value)
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


@dataclass(frozen=True)
class ModelPackEntry:
    """A single launchable model entry inside a model pack."""

    id: str
    name: str
    description: str
    type: str
    path: str
    engine_type: str | None = None
    capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    provider: str | None = None
    source_root: str | None = None
    working_dir: str | None = None
    python_paths: tuple[str, ...] = ()
    order: int = 99

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelPackEntry:
        """Build a validated entry from raw manifest data."""
        require(isinstance(data, dict), "model entry must be a mapping", data)

        required = {"id", "name", "description", "type", "path"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"Model pack entry missing required fields: {sorted(missing)}"
            )

        for field_name in required:
            value = data[field_name]
            require(isinstance(value, str), f"{field_name} must be a string", value)
            require(value.strip() != "", f"{field_name} must be non-empty", value)

        engine_type = data.get("engine_type")
        if engine_type is not None:
            require(
                isinstance(engine_type, str) and engine_type.strip() != "",
                "engine_type must be a non-empty string when provided",
                engine_type,
            )

        provider = data.get("provider")
        if provider is not None:
            require(
                isinstance(provider, str) and provider.strip() != "",
                "provider must be a non-empty string when provided",
                provider,
            )

        source_root = data.get("source_root")
        if source_root is not None:
            require(
                isinstance(source_root, str) and source_root.strip() != "",
                "source_root must be a non-empty string when provided",
                source_root,
            )

        working_dir = data.get("working_dir")
        if working_dir is not None:
            require(
                isinstance(working_dir, str) and working_dir.strip() != "",
                "working_dir must be a non-empty string when provided",
                working_dir,
            )

        order = data.get("order", 99)
        require(isinstance(order, int), "order must be an integer", order)
        require(order >= 0, "order must be non-negative", order)

        return cls(
            id=data["id"].strip(),
            name=data["name"].strip(),
            description=data["description"].strip(),
            type=data["type"].strip(),
            path=data["path"].strip(),
            engine_type=engine_type.strip() if isinstance(engine_type, str) else None,
            capabilities=_normalize_string_sequence(data.get("capabilities")),
            tags=_normalize_string_sequence(data.get("tags")),
            provider=provider.strip() if isinstance(provider, str) else None,
            source_root=source_root.strip() if isinstance(source_root, str) else None,
            working_dir=working_dir.strip() if isinstance(working_dir, str) else None,
            python_paths=_normalize_path_sequence(data.get("python_paths")),
            order=order,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entry to a JSON/YAML-friendly dictionary."""
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "path": self.path,
            "capabilities": list(self.capabilities),
            "tags": list(self.tags),
            "order": self.order,
        }
        if self.engine_type:
            data["engine_type"] = self.engine_type
        if self.provider:
            data["provider"] = self.provider
        if self.source_root:
            data["source_root"] = self.source_root
        if self.working_dir:
            data["working_dir"] = self.working_dir
        if self.python_paths:
            data["python_paths"] = list(self.python_paths)
        return data


@dataclass(frozen=True)
class ModelPackManifest(ContractChecker):
    """Versioned manifest describing a provider-compatible model pack."""

    manifest_version: str
    pack_id: str
    pack_name: str
    provider: str
    models: tuple[ModelPackEntry, ...]
    description: str = ""
    source_repo: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelPackManifest:
        """Create a strict model-pack manifest from a dictionary."""
        require(isinstance(data, dict), "manifest must be a mapping", data)

        required = {"manifest_version", "pack_id", "pack_name", "provider", "models"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"Model pack manifest missing required fields: {sorted(missing)}"
            )

        models_raw = data["models"]
        require(isinstance(models_raw, list), "models must be a list", models_raw)

        entries = tuple(
            sorted(
                (ModelPackEntry.from_dict(item) for item in models_raw),
                key=lambda entry: (entry.order, entry.id),
            )
        )

        manifest = cls(
            manifest_version=str(data["manifest_version"]).strip(),
            pack_id=str(data["pack_id"]).strip(),
            pack_name=str(data["pack_name"]).strip(),
            provider=str(data["provider"]).strip(),
            models=entries,
            description=str(data.get("description", "")).strip(),
            source_repo=(
                str(data["source_repo"]).strip()
                if data.get("source_repo") is not None
                else None
            ),
        )
        manifest.verify_invariants()

        ids = manifest.model_ids
        duplicates = [model_id for model_id in ids if ids.count(model_id) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate model IDs in manifest: {sorted(set(duplicates))}"
            )

        ensure(
            manifest.models == entries, "models must remain sorted deterministically"
        )
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> ModelPackManifest:
        """Load a strict versioned manifest from disk."""
        require(path is not None, "path must be provided")
        manifest_path = Path(path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Model pack manifest not found: {manifest_path}")

        with open(manifest_path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        require(isinstance(raw, dict), "manifest root must be a mapping", raw)
        return cls.from_dict(raw)

    @classmethod
    def from_legacy_registry(
        cls,
        data: dict[str, Any],
        *,
        pack_id: str = "legacy-local-pack",
        pack_name: str = "Legacy Local Pack",
        provider: str = "local",
        source_repo: str | None = None,
    ) -> ModelPackManifest:
        """Wrap a legacy ``models.yaml`` registry in the versioned manifest shape."""
        require(isinstance(data, dict), "legacy registry must be a mapping", data)
        models_raw = data.get("models", [])
        require(
            isinstance(models_raw, list), "legacy models must be a list", models_raw
        )
        return cls.from_dict(
            {
                "manifest_version": "1.0.0",
                "pack_id": pack_id,
                "pack_name": pack_name,
                "provider": provider,
                "source_repo": source_repo,
                "models": models_raw,
            }
        )

    @property
    def model_ids(self) -> list[str]:
        """Return model identifiers in deterministic order."""
        return [model.id for model in self.models]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest to a dictionary."""
        data: dict[str, Any] = {
            "manifest_version": self.manifest_version,
            "pack_id": self.pack_id,
            "pack_name": self.pack_name,
            "provider": self.provider,
            "description": self.description,
            "models": [model.to_dict() for model in self.models],
        }
        if self.source_repo:
            data["source_repo"] = self.source_repo
        return data

    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        """Define class invariants for strict model-pack manifests."""
        return [
            (
                lambda: (
                    isinstance(self.manifest_version, str)
                    and self.manifest_version.strip() != ""
                ),
                "manifest_version must be a non-empty string",
            ),
            (
                lambda: isinstance(self.pack_id, str) and self.pack_id.strip() != "",
                "pack_id must be a non-empty string",
            ),
            (
                lambda: (
                    isinstance(self.pack_name, str) and self.pack_name.strip() != ""
                ),
                "pack_name must be a non-empty string",
            ),
            (
                lambda: isinstance(self.provider, str) and self.provider.strip() != "",
                "provider must be a non-empty string",
            ),
            (
                lambda: isinstance(self.models, tuple),
                "models must be stored as an immutable tuple",
            ),
        ]
