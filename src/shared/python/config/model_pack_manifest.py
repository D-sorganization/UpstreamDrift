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

_CAPABILITY_ALIASES = {
    "inverse-kinematics": "ik",
    "inverse_kinematics": "ik",
    "inverse kinematics": "ik",
    "forward-dynamics": "dynamics",
    "forward_dynamics": "dynamics",
    "forward dynamics": "dynamics",
    "balance-control": "balance",
    "balance_control": "balance",
    "balance control": "balance",
}


def _slugify(value: str) -> str:
    """Normalize human-readable metadata to a stable lowercase slug."""
    require(isinstance(value, str), "metadata values must be strings", value)
    return "-".join(value.strip().replace("_", " ").replace(".", " ").lower().split())


def _normalize_canonical_id(value: str) -> str:
    """Normalize canonical IDs while preserving dotted conceptual namespaces."""
    require(isinstance(value, str), "canonical_id must be a string", value)
    segments = [segment for segment in value.strip().split(".") if segment.strip()]
    require(len(segments) > 0, "canonical_id must contain at least one segment", value)
    return ".".join(_slugify(segment) for segment in segments)


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
        item = _CAPABILITY_ALIASES.get(item, item)
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
class CrossEngineIdentity:
    """Canonical identity shared across semantically equivalent engine packs."""

    canonical_id: str
    motion_family: str
    exercise: str
    humanoid: str
    dataset: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossEngineIdentity:
        """Create normalized cross-engine identity metadata."""
        require(isinstance(data, dict), "identity must be a mapping", data)

        required = {"canonical_id", "motion_family", "exercise", "humanoid"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"Identity metadata missing required fields: {sorted(missing)}"
            )

        dataset = data.get("dataset")
        if dataset is not None:
            require(
                isinstance(dataset, str) and dataset.strip() != "",
                "dataset must be a non-empty string when provided",
                dataset,
            )

        return cls(
            canonical_id=_normalize_canonical_id(str(data["canonical_id"])),
            motion_family=_slugify(str(data["motion_family"])),
            exercise=_slugify(str(data["exercise"])),
            humanoid=_slugify(str(data["humanoid"])),
            dataset=_slugify(str(dataset)) if dataset is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize canonical identity metadata."""
        data: dict[str, Any] = {
            "canonical_id": self.canonical_id,
            "motion_family": self.motion_family,
            "exercise": self.exercise,
            "humanoid": self.humanoid,
        }
        if self.dataset:
            data["dataset"] = self.dataset
        return data


@dataclass(frozen=True)
class ExchangeArtifact:
    """Metadata for a shared interchange artifact."""

    format: str
    path: str
    role: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExchangeArtifact:
        """Create validated exchange-artifact metadata."""
        require(isinstance(data, dict), "exchange artifact must be a mapping", data)
        required = {"format", "path", "role"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"Exchange artifact metadata missing required fields: {sorted(missing)}"
            )

        format_value = str(data["format"]).strip().lower()
        role_value = str(data["role"]).strip().lower()
        path_value = str(data["path"]).strip()
        require(format_value != "", "exchange artifact format must be non-empty")
        require(path_value != "", "exchange artifact path must be non-empty")
        require(
            role_value in {"source", "derived"},
            "exchange artifact role must be source or derived",
            role_value,
        )

        return cls(format=format_value, path=path_value, role=role_value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize exchange-artifact metadata."""
        return {"format": self.format, "path": self.path, "role": self.role}


@dataclass(frozen=True)
class ProvenanceMetadata:
    """Version and derivation metadata for generated assets."""

    source_format: str
    source_path: str
    version: str
    derived_from: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceMetadata:
        """Create validated provenance metadata."""
        require(isinstance(data, dict), "provenance must be a mapping", data)
        required = {"source_format", "source_path", "version"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"Provenance metadata missing required fields: {sorted(missing)}"
            )

        for field_name in required:
            value = data[field_name]
            require(
                isinstance(value, str) and value.strip() != "",
                f"{field_name} must be a non-empty string",
                value,
            )

        return cls(
            source_format=str(data["source_format"]).strip().lower(),
            source_path=str(data["source_path"]).strip(),
            version=str(data["version"]).strip(),
            derived_from=_normalize_path_sequence(data.get("derived_from")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize provenance metadata."""
        data: dict[str, Any] = {
            "source_format": self.source_format,
            "source_path": self.source_path,
            "version": self.version,
        }
        if self.derived_from:
            data["derived_from"] = list(self.derived_from)
        return data


@dataclass(frozen=True)
class LauncherPresentationMetadata:
    """UI-facing launcher presentation metadata for a model-pack entry."""

    category: str
    logo: str
    status: str
    web_route: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LauncherPresentationMetadata:
        """Create validated launcher presentation metadata."""
        require(
            isinstance(data, dict),
            "launcher metadata must be a mapping",
            data,
        )
        required = {"category", "logo", "status"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"Launcher metadata missing required fields: {sorted(missing)}"
            )

        category = str(data["category"]).strip()
        require(
            category
            in {
                "physics_engine",
                "simulation",
                "motion_matching",
                "motion_capture",
                "tool",
                "documentation",
                "biomechanics",
                "external",
            },
            "launcher category must be one of: physics_engine, simulation, motion_matching, motion_capture, tool, documentation, biomechanics, external",
            category,
        )

        logo = str(data["logo"]).strip()
        status = str(data["status"]).strip()
        require(logo != "", "launcher logo must be non-empty", logo)
        require(status != "", "launcher status must be non-empty", status)

        web_route_raw = data.get("web_route")
        if web_route_raw is not None:
            require(
                isinstance(web_route_raw, str) and web_route_raw.strip() != "",
                "launcher web_route must be a non-empty string when provided",
                web_route_raw,
            )

        return cls(
            category=category,
            logo=logo,
            status=status,
            web_route=(
                web_route_raw.strip() if isinstance(web_route_raw, str) else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize launcher presentation metadata."""
        data: dict[str, Any] = {
            "category": self.category,
            "logo": self.logo,
            "status": self.status,
        }
        if self.web_route:
            data["web_route"] = self.web_route
        return data


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
    identity: CrossEngineIdentity | None = None
    exchange_artifacts: tuple[ExchangeArtifact, ...] = ()
    provenance: ProvenanceMetadata | None = None
    launcher: LauncherPresentationMetadata | None = None
    order: int = 99
    hidden: bool = False

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

        hidden_raw = data.get("hidden", False)
        require(
            isinstance(hidden_raw, bool),
            "hidden must be a boolean when provided",
            hidden_raw,
        )

        identity_raw = data.get("identity")
        identity = (
            CrossEngineIdentity.from_dict(identity_raw)
            if identity_raw is not None
            else None
        )

        exchange_artifacts_raw = data.get("exchange_artifacts", [])
        require(
            isinstance(exchange_artifacts_raw, list),
            "exchange_artifacts must be a list when provided",
            exchange_artifacts_raw,
        )
        exchange_artifacts = tuple(
            ExchangeArtifact.from_dict(item) for item in exchange_artifacts_raw
        )

        provenance_raw = data.get("provenance")
        provenance = (
            ProvenanceMetadata.from_dict(provenance_raw)
            if provenance_raw is not None
            else None
        )
        launcher_raw = data.get("launcher")
        launcher = (
            LauncherPresentationMetadata.from_dict(launcher_raw)
            if launcher_raw is not None
            else None
        )

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
            identity=identity,
            exchange_artifacts=exchange_artifacts,
            provenance=provenance,
            launcher=launcher,
            order=order,
            hidden=bool(hidden_raw),
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
        if self.identity:
            data["identity"] = self.identity.to_dict()
        if self.exchange_artifacts:
            data["exchange_artifacts"] = [
                artifact.to_dict() for artifact in self.exchange_artifacts
            ]
        if self.provenance:
            data["provenance"] = self.provenance.to_dict()
        if self.launcher:
            data["launcher"] = self.launcher.to_dict()
        if self.hidden:
            data["hidden"] = True
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


def group_entries_by_canonical_id(
    entries: tuple[ModelPackEntry, ...] | list[ModelPackEntry],
) -> dict[str, tuple[ModelPackEntry, ...]]:
    """Group semantically equivalent entries by canonical cross-engine identity."""
    grouped: dict[str, list[ModelPackEntry]] = {}

    for entry in entries:
        if entry.identity is None:
            continue
        grouped.setdefault(entry.identity.canonical_id, []).append(entry)

    return {
        canonical_id: tuple(
            sorted(items, key=lambda entry: ((entry.engine_type or ""), entry.id))
        )
        for canonical_id, items in sorted(grouped.items())
    }
