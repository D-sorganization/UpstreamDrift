"""Build the deterministic UpstreamDrift provider catalog for AffineDrift.

The catalog deliberately exports software facts only. Calculation definitions,
scientific qualification, and engineering approval remain in their governed
authorities and are represented here only by explicit, conservative status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from scripts import companion_workflows

# This module must stay importable with only ``jsonschema`` and ``pyyaml``
# installed: the release workflow's companion job pip-installs exactly those
# two packages, and importing anything under ``src.*`` drags the runtime
# package tree (numpy, Qt-adjacent helpers) into a publication job (#9416).

SCHEMA_ID = "https://upstreamdrift.dev/schemas/upstreamdrift-companion-v1.schema.json"
SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "1.1.0"
CAPABILITIES_SCHEMA_ID = (
    "https://upstreamdrift.dev/schemas/"
    "upstreamdrift-companion-capabilities-v1.schema.json"
)
CAPABILITIES_MANIFEST_ID = "upstreamdrift-companion-capabilities"
SCREENSHOTS_SCHEMA_ID = (
    "https://upstreamdrift.dev/schemas/"
    "upstreamdrift-companion-screenshots-v1.schema.json"
)
SCREENSHOTS_MANIFEST_ID = "upstreamdrift-companion-screenshots"
SCREENSHOT_PENDING_REASON = "No governed capture exists at this commit (#9191)."
DEFAULT_OUTPUT = Path("dist/companion/upstreamdrift-companion.v1.json")
INPUT_PATHS = (
    Path("pyproject.toml"),
    companion_workflows.REGISTRY_PATH,
    Path("src/config/feature_parity.json"),
    Path("src/config/launcher_manifest.json"),
    Path("src/config/models.yaml"),
)
_HEX_40 = frozenset("0123456789abcdef")
_ENGINE_TIERS = {
    "mujoco": "supported",
    "drake": "extended",
    "pinocchio": "extended",
    "opensim": "experimental",
    "myosuite": "experimental",
}
_ENGINE_DISPLAY_NAMES = {
    "mujoco": "MuJoCo",
    "drake": "Drake",
    "pinocchio": "Pinocchio",
    "opensim": "OpenSim",
    "myosuite": "MyoSuite",
}
_STABLE_LEGACY_STATUSES = frozenset(
    {
        "engine_ready",
        "gui_ready",
        "provider_ready",
        "ready",
        "simulator",
        "utility",
    }
)


class CatalogAuthorityError(RuntimeError):
    """Raised when an export cannot prove authoritative, immutable inputs."""


# ---------------------------------------------------------------------------
# Standalone, import-free reader for ``src/config/models.yaml``.
#
# The runtime registry (``src/shared/python/config/model_registry.py``) is the
# launcher's authority for *runtime* semantics. The catalog only needs the ten
# presentation facts consumed by ``_model_payload`` below, so it re-reads the
# YAML with ``yaml.safe_load`` and replicates the handful of legacy
# normalisations that change those facts. Discovery is explicitly local-only:
# no provider manifests, no sibling repositories, no environment variables.
# ---------------------------------------------------------------------------

# Duplicated from ``src/shared/python/config/model_registry.py``
# (``_METADATA_ONLY_TYPES``): entries of these built-in types are launched via
# an engine/mode rather than a concrete file, so the registry synthesises a
# ``virtual/<type>/<id>`` path when ``path`` is absent (#6882). Importing the
# registry would pull numpy into the publication job (#9416), so the tuple is
# copied here; keep the two in sync.
_METADATA_ONLY_TYPES = ("biomech_exercise", "physics_informed")

# Duplicated from ``src/shared/python/config/model_pack_manifest.py``
# (``_CAPABILITY_ALIASES``) for the same import-isolation reason; the runtime
# registry folds these spellings into one capability id and the consumer
# catalog must report the same ids the launcher does.
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
_LAUNCHER_CATEGORIES = frozenset(
    {
        "physics_engine",
        "simulation",
        "motion_matching",
        "motion_capture",
        "tool",
        "documentation",
        "biomechanics",
        "external",
    }
)
_REQUIRED_MODEL_FIELDS = ("id", "name", "description", "type", "path")


@dataclass(frozen=True)
class LocalLauncherRecord:
    """Launcher presentation facts the catalog reads from a models.yaml entry."""

    category: str
    status: str


@dataclass(frozen=True)
class LocalModelRecord:
    """The models.yaml facts consumed by the catalog; nothing else is read."""

    id: str
    name: str
    description: str
    type: str
    path: str
    engine_type: str | None
    provider: str | None
    hidden: bool
    capabilities: tuple[str, ...]
    launcher: LocalLauncherRecord | None


def _optional_string(raw: Mapping[str, Any], key: str, *, entry: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CatalogAuthorityError(
            f"models.yaml entry {entry!r}: {key} must be a non-empty string"
        )
    return value.strip()


def normalize_capability_ids(values: object) -> tuple[str, ...]:
    """Fold capability spellings the way the runtime registry does."""
    if values is None:
        return ()
    if not isinstance(values, list | tuple):
        raise CatalogAuthorityError("capabilities must be a list of strings")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise CatalogAuthorityError("capabilities must be a list of strings")
        item = value.strip().lower()
        item = _CAPABILITY_ALIASES.get(item, item)
        if item and item not in normalized:
            normalized.append(item)
    return tuple(normalized)


def _launcher_record(raw: object, *, entry: str) -> LocalLauncherRecord | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise CatalogAuthorityError(
            f"models.yaml entry {entry!r}: launcher must be a mapping"
        )
    values: dict[str, str] = {}
    for key in ("category", "logo", "status"):
        if key not in raw:
            raise CatalogAuthorityError(
                f"models.yaml entry {entry!r}: launcher is missing {key}"
            )
        text = str(raw[key]).strip()
        if not text:
            raise CatalogAuthorityError(
                f"models.yaml entry {entry!r}: launcher {key} must be non-empty"
            )
        values[key] = text
    if values["category"] not in _LAUNCHER_CATEGORIES:
        raise CatalogAuthorityError(
            f"models.yaml entry {entry!r}: unknown launcher category "
            f"{values['category']!r}"
        )
    return LocalLauncherRecord(category=values["category"], status=values["status"])


def _local_model_record(raw: object, *, index: int) -> LocalModelRecord:
    """Mirror ``_normalize_legacy_model_entry`` + ``ModelPackEntry.from_dict``.

    Only the validations that affect the exported facts are replicated. Keys
    the catalog does not consume are ignored on purpose so this builder
    tolerates registry additions.
    """
    # TODO(#9412): the one-tile-registry PR adds optional top-level keys
    # (``maturity``, ``surfaces``, ``help``, ``feature_id``, ``web``) to
    # models.yaml entries. They are intentionally not read here; once that PR
    # lands, decide whether the catalog should export any of them.
    if not isinstance(raw, Mapping):
        raise CatalogAuthorityError(
            f"models.yaml entry [{index}]: legacy model entries must be mappings"
        )
    data = dict(raw)
    entry = str(data.get("id", f"entry[{index}]"))
    description = data.get("description")
    if isinstance(description, str) and description.strip() == "":
        fallback = data.get("name")
        if isinstance(fallback, str) and fallback.strip():
            data["description"] = fallback.strip()
    entry_type = data.get("type")
    entry_id = data.get("id")
    has_path = isinstance(data.get("path"), str) and bool(data["path"].strip())
    if (
        not has_path
        and entry_type in _METADATA_ONLY_TYPES
        and isinstance(entry_id, str)
        and entry_id.strip()
    ):
        data["path"] = f"virtual/{entry_type}/{entry_id.strip()}"
    missing = [field for field in _REQUIRED_MODEL_FIELDS if field not in data]
    if missing:
        raise CatalogAuthorityError(
            f"models.yaml entry {entry!r}: missing required fields {missing}"
        )
    for field in _REQUIRED_MODEL_FIELDS:
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise CatalogAuthorityError(
                f"models.yaml entry {entry!r}: {field} must be a non-empty string"
            )
    hidden = data.get("hidden", False)
    if not isinstance(hidden, bool):
        raise CatalogAuthorityError(
            f"models.yaml entry {entry!r}: hidden must be a boolean"
        )
    return LocalModelRecord(
        id=data["id"].strip(),
        name=data["name"].strip(),
        description=data["description"].strip(),
        type=data["type"].strip(),
        path=data["path"].strip(),
        engine_type=_optional_string(data, "engine_type", entry=entry),
        provider=_optional_string(data, "provider", entry=entry),
        hidden=hidden,
        capabilities=normalize_capability_ids(data.get("capabilities")),
        launcher=_launcher_record(data.get("launcher"), entry=entry),
    )


def read_local_models(payload: bytes) -> list[LocalModelRecord]:
    """Parse committed models.yaml bytes with local-only discovery.

    Postconditions:
        Later duplicate ids overwrite earlier ones (the registry's
        ``overwrite_existing=True`` legacy policy) and first-seen order is
        preserved, exactly like ``ModelRegistry.get_all_models``.
    """
    try:
        document = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CatalogAuthorityError(f"models.yaml is not valid YAML: {exc}") from exc
    if not isinstance(document, Mapping) or "models" not in document:
        raise CatalogAuthorityError("models.yaml root must be a mapping with 'models'")
    models_raw = document["models"]
    if not isinstance(models_raw, list):
        raise CatalogAuthorityError("models.yaml 'models' must be a list")
    records: dict[str, LocalModelRecord] = {}
    for index, raw in enumerate(models_raw):
        record = _local_model_record(raw, index=index)
        records[record.id] = record
    return list(records.values())


def validate_repo_relative(value: Path) -> PurePosixPath:
    """Validate and normalize a repository-relative contract path."""
    text = value.as_posix()
    candidate = PurePosixPath(text)
    if (
        value.is_absolute()
        or candidate.is_absolute()
        or PureWindowsPath(text).is_absolute()
        or ".." in candidate.parts
    ):
        raise ValueError(f"path must be repo-relative and contained: {value}")
    if not candidate.parts or candidate.parts == (".",):
        raise ValueError(f"path must be repo-relative and non-empty: {value}")
    return candidate


def _run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CatalogAuthorityError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _git_commit(repo_root: Path) -> str:
    commit = _run_git(repo_root, "rev-parse", "HEAD").lower()
    if len(commit) != 40 or any(char not in _HEX_40 for char in commit):
        raise CatalogAuthorityError(f"HEAD is not an exact commit: {commit!r}")
    expected = os.environ.get("GITHUB_SHA")
    if expected and expected.lower() != commit:
        raise CatalogAuthorityError(
            f"GITHUB_SHA {expected!r} does not match checked-out commit {commit}"
        )
    return commit


def _dirty_paths(repo_root: Path) -> tuple[str, ...]:
    raw = _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    dirty: list[str] = []
    for line in raw.splitlines():
        if not line:
            continue
        path_text = line[3:]
        if " -> " in path_text:
            path_text = path_text.rsplit(" -> ", 1)[1]
        normalized = path_text.strip('"').replace("\\", "/")
        if normalized == "dist/companion" or normalized.startswith("dist/companion/"):
            continue
        dirty.append(normalized)
    return tuple(sorted(dirty))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_input(repo_root: Path, relative: Path, *, require_committed: bool) -> bytes:
    normalized = validate_repo_relative(relative)
    path = repo_root.joinpath(*normalized.parts)
    if not path.is_file():
        raise CatalogAuthorityError(f"required input is missing: {normalized}")
    diff = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            normalized.as_posix(),
        ],
        check=False,
        capture_output=True,
        shell=False,
    )
    if diff.returncode != 0 and require_committed:
        raise CatalogAuthorityError(f"input differs from HEAD: {normalized}")
    committed = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"HEAD:{normalized.as_posix()}"],
        check=False,
        capture_output=True,
        shell=False,
    )
    if committed.returncode != 0 and require_committed:
        raise CatalogAuthorityError(f"input is not tracked at HEAD: {normalized}")
    if committed.returncode != 0:
        return path.read_bytes()
    if diff.returncode == 0:
        # Hash and parse committed blob bytes so checkout line-ending policy
        # cannot change the manifest on Windows versus Linux.
        return committed.stdout
    # Non-authoritative in-process builds exist only for RED/GREEN tests while
    # their own input/dependency edits are uncommitted. The public CLI never
    # selects this path.
    return path.read_bytes()


def _tools_gitlink(repo_root: Path) -> str:
    line = _run_git(repo_root, "ls-tree", "HEAD", "--", "vendor/ud-tools")
    fields = line.split()
    if len(fields) < 4 or fields[0] != "160000" or fields[1] != "commit":
        raise CatalogAuthorityError("vendor/ud-tools is not an immutable gitlink")
    commit = fields[2].lower()
    if len(commit) != 40 or any(char not in _HEX_40 for char in commit):
        raise CatalogAuthorityError("vendor/ud-tools gitlink is not an exact commit")
    return commit


def _maturity(statuses: Iterable[str]) -> str:
    normalized = {status.lower() for status in statuses}
    if "deprecated" in normalized:
        return "deprecated"
    if "experimental" in normalized:
        return "experimental"
    if "beta" in normalized:
        return "beta"
    if normalized & _STABLE_LEGACY_STATUSES:
        return "stable"
    return "unclassified"


def _availability(
    *, hidden: bool, provider_id: str, entry_point: str | None
) -> dict[str, str | None]:
    if hidden:
        return {"state": "unavailable", "reason": "Hidden by provider registry"}
    if provider_id != "upstreamdrift":
        return {
            "state": "conditional",
            "reason": "Requires the provider pinned by this manifest",
        }
    if entry_point is None:
        return {
            "state": "conditional",
            "reason": "No direct entry point; use a declared shell surface",
        }
    if entry_point.startswith("virtual/"):
        return {
            "state": "conditional",
            "reason": "Virtual launcher target requires its declared runtime",
        }
    return {"state": "available", "reason": None}


def _qualification(scope: str) -> dict[str, Any]:
    return {
        "state": "unqualified",
        "scope": scope,
        "limitations": [
            "Catalog inclusion is not scientific validation or engineering approval."
        ],
    }


def _source_record(registry: str, path: str) -> dict[str, str]:
    return {"registry": registry, "path": path}


def _model_payload(model: LocalModelRecord) -> dict[str, Any]:
    launcher = model.launcher
    status = launcher.status if launcher is not None else "unclassified"
    return {
        "id": model.id,
        "name": model.name,
        "description": model.description,
        "category": launcher.category if launcher is not None else "external",
        "type": model.type,
        "entry_point": model.path,
        "engine_id": model.engine_type,
        "provider_id": (
            model.provider
            if model.provider not in (None, "", "local")
            else "upstreamdrift"
        ),
        "hidden": model.hidden,
        "legacy_statuses": [status],
        "source_records": [_source_record("model_registry", "src/config/models.yaml")],
    }


def _launcher_payload(tile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(tile["id"]),
        "name": str(tile["name"]),
        "description": str(tile["description"]),
        "category": str(tile["category"]),
        "type": str(tile["type"]),
        "entry_point": str(tile["path"]) if tile.get("path") else None,
        "engine_id": tile.get("engine_type"),
        "provider_id": (
            tile.get("provider")
            if tile.get("provider") not in (None, "", "local")
            else "upstreamdrift"
        ),
        "hidden": bool(tile.get("hidden", False)),
        "legacy_statuses": [str(tile.get("status", "unclassified"))],
        "source_records": [
            _source_record("launcher_manifest", "src/config/launcher_manifest.json")
        ],
    }


def _merge_programs(
    launcher_tiles: Sequence[Mapping[str, Any]],
    models: Sequence[LocalModelRecord],
    feature_ids_by_program: Mapping[str, list[str]],
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for raw in launcher_tiles:
        payload = _launcher_payload(raw)
        records[payload["id"]] = payload
    for model in models:
        payload = _model_payload(model)
        existing = records.get(model.id)
        if existing is None:
            records[model.id] = payload
            continue
        # models.yaml owns native launch semantics; the web manifest still
        # contributes provenance and legacy display status.
        existing.update(
            {
                key: payload[key]
                for key in (
                    "name",
                    "description",
                    "category",
                    "type",
                    "entry_point",
                    "engine_id",
                    "provider_id",
                    "hidden",
                )
            }
        )
        existing["legacy_statuses"] = sorted(
            set(existing["legacy_statuses"] + payload["legacy_statuses"])
        )
        existing["source_records"].extend(payload["source_records"])

    result: list[dict[str, Any]] = []
    for program_id in sorted(records):
        record = records[program_id]
        statuses = sorted(set(record.pop("legacy_statuses")))
        source_records = sorted(
            record.pop("source_records"),
            key=lambda item: (item["registry"], item["path"]),
        )
        engine_id = record["engine_id"]
        record.update(
            {
                "maturity": _maturity(statuses),
                "availability": _availability(
                    hidden=record["hidden"],
                    provider_id=record["provider_id"],
                    entry_point=record["entry_point"],
                ),
                "support_tier": _ENGINE_TIERS.get(engine_id, "not_applicable"),
                "scientific_qualification": _qualification(
                    "Program listing and launch metadata only"
                ),
                "feature_ids": sorted(feature_ids_by_program.get(program_id, [])),
                "legacy_statuses": statuses,
                "source_records": source_records,
            }
        )
        result.append(record)
    return result


def _features(
    raw_features: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]], int]:
    features: list[dict[str, Any]] = []
    feature_ids_by_program: dict[str, list[str]] = {}
    surface_count = 0
    for feature_id in sorted(raw_features):
        raw = raw_features[feature_id]
        surfaces = [
            {"surface": surface, "path": raw[surface]}
            for surface in ("pyqt", "api", "web")
            if isinstance(raw.get(surface), str)
        ]
        surface_count += len(surfaces)
        program_ids = sorted(set(raw.get("tiles", [])))
        for program_id in program_ids:
            feature_ids_by_program.setdefault(program_id, []).append(feature_id)
        features.append(
            {
                "id": feature_id,
                "title": str(raw["title"]),
                "parity": {
                    "state": str(raw["status"]),
                    "issue": raw.get("issue"),
                    "reason": raw.get("reason"),
                    "pending_decision": bool(raw.get("pending_decision", False)),
                },
                "program_ids": program_ids,
                "surfaces": surfaces,
                "notes": raw.get("notes"),
                "scientific_qualification": _qualification(
                    "Feature parity record only"
                ),
            }
        )
    return features, feature_ids_by_program, surface_count


def _source_provenance(
    repo_root: Path, payloads: Mapping[Path, bytes]
) -> dict[str, Any]:
    commit = _git_commit(repo_root)
    package = tomllib.loads(payloads[Path("pyproject.toml")].decode("utf-8"))["project"]
    return {
        "repository": "https://github.com/D-sorganization/UpstreamDrift",
        "commit": commit,
        "commit_timestamp": _run_git(repo_root, "show", "-s", "--format=%cI", commit),
        "package_version": str(package["version"]),
        "generator": {
            "path": "scripts/companion_catalog.py",
            "version": GENERATOR_VERSION,
        },
        "inputs": [
            {"path": relative.as_posix(), "sha256": _sha256(payloads[relative])}
            for relative in sorted(payloads, key=lambda path: path.as_posix())
        ],
    }


def _catalog_payload(
    *,
    launcher: Mapping[str, Any],
    parity: Mapping[str, Any],
    payloads: Mapping[Path, bytes],
    source: Mapping[str, Any],
    tools_commit: str,
    inventories: Mapping[str, Sequence[Mapping[str, Any]]],
    summary: Mapping[str, int],
) -> dict[str, Any]:
    """Assemble the strict schema shape from already validated facts."""
    project = tomllib.loads(payloads[Path("pyproject.toml")].decode("utf-8"))["project"]
    requires_python = str(project["requires-python"])
    providers = [
        {
            "id": "tools",
            "repository": "https://github.com/D-sorganization/Tools",
            "relationship": "vendored-provider",
            "pin_kind": "gitlink",
            "vendor_path": "vendor/ud-tools",
            "pinned_commit": tools_commit,
        },
        {
            "id": "upstreamdrift",
            "repository": source["repository"],
            "relationship": "authority",
            "pin_kind": "source-commit",
            "vendor_path": None,
            "pinned_commit": source["commit"],
        },
    ]
    registries = [
        {
            "id": "feature_parity",
            "path": "src/config/feature_parity.json",
            "version": str(parity["version"]),
            "discovery_mode": "local-only",
        },
        {
            "id": "launcher_manifest",
            "path": "src/config/launcher_manifest.json",
            "version": str(launcher["version"]),
            "discovery_mode": "local-only",
        },
        {
            "id": "model_registry",
            "path": "src/config/models.yaml",
            "version": None,
            "discovery_mode": "local-only",
        },
        {
            "id": "workflow_registry",
            "path": companion_workflows.REGISTRY_PATH.as_posix(),
            "version": companion_workflows.REGISTRY_VERSION,
            "discovery_mode": "local-only",
        },
    ]
    engines = [
        {
            "id": engine_id,
            "name": _ENGINE_DISPLAY_NAMES[engine_id],
            "support_tier": _ENGINE_TIERS[engine_id],
            "scientific_qualification": _qualification(
                "Installation/support tier only; no numerical capability claim"
            ),
        }
        for engine_id in sorted(_ENGINE_TIERS)
    ]
    return {
        "$schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "manifest_id": "upstreamdrift-companion",
        "publication": {
            "state": "draft",
            "blockers": [
                "Engine capability claims require qualification evidence.",
                "Documentation and screenshot inventories require their governed provider slices.",
            ],
        },
        "source": dict(source),
        "providers": providers,
        "registries": registries,
        "compatibility": {
            "requires_python": requires_python,
            "supported_python_minors": _supported_python_minors(requires_python),
            "verification_command": {
                "executable": "python",
                "arguments": ["scripts/ci/verify_installation.py"],
            },
        },
        "engines": engines,
        "programs": list(inventories["programs"]),
        "features": list(inventories["features"]),
        "documentation": [],
        "workflows": list(inventories["workflows"]),
        "screenshots": [],
        "summary": dict(summary),
    }


def _capability_ids_by_program(
    launcher_tiles: Sequence[Mapping[str, Any]],
    models: Sequence[LocalModelRecord],
) -> dict[str, list[str]]:
    """Union the launcher-manifest and models.yaml capability lists per program."""
    by_program: dict[str, set[str]] = {}
    for tile in launcher_tiles:
        by_program.setdefault(str(tile["id"]), set()).update(
            normalize_capability_ids(tile.get("capabilities"))
        )
    for model in models:
        by_program.setdefault(model.id, set()).update(model.capabilities)
    return {program_id: sorted(ids) for program_id, ids in sorted(by_program.items())}


def _capabilities_payload(
    catalog: Mapping[str, Any],
    capability_ids_by_program: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Derive the consumer capability catalog from an already built manifest.

    Pure: no I/O, no git. Every program id in the result exists in
    ``catalog["programs"]``; capability ids are sorted and unique.
    """
    program_ids = {program["id"] for program in catalog["programs"]}
    dangling = sorted(set(capability_ids_by_program) - program_ids)
    if dangling:
        raise CatalogAuthorityError(
            f"capability records reference unknown programs: {dangling}"
        )
    programs: list[dict[str, Any]] = []
    programs_by_capability: dict[str, list[str]] = {}
    capability_ids_by_engine: dict[str, set[str]] = {
        engine["id"]: set() for engine in catalog["engines"]
    }
    for program in catalog["programs"]:
        capability_ids = sorted(set(capability_ids_by_program.get(program["id"], ())))
        for capability_id in capability_ids:
            programs_by_capability.setdefault(capability_id, []).append(program["id"])
        engine_id = program["engine_id"]
        if engine_id in capability_ids_by_engine:
            capability_ids_by_engine[engine_id].update(capability_ids)
        programs.append(
            {
                "id": program["id"],
                "engine_id": engine_id,
                "support_tier": program["support_tier"],
                "maturity": program["maturity"],
                "availability_state": program["availability"]["state"],
                "capability_ids": capability_ids,
            }
        )
    return {
        "$schema": CAPABILITIES_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "manifest_id": CAPABILITIES_MANIFEST_ID,
        "source": {
            "repository": catalog["source"]["repository"],
            "commit": catalog["source"]["commit"],
        },
        "engines": [
            {
                "id": engine["id"],
                "name": engine["name"],
                "support_tier": engine["support_tier"],
                "capability_ids": sorted(capability_ids_by_engine[engine["id"]]),
            }
            for engine in catalog["engines"]
        ],
        "capabilities": [
            {"id": capability_id, "program_ids": sorted(program_ids)}
            for capability_id, program_ids in sorted(programs_by_capability.items())
        ],
        "programs": programs,
    }


def _screenshots_payload(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Emit metadata-only screenshot records; nothing is fabricated.

    Every non-hidden program gets one ``pending`` record with null asset
    fields and an explicit reason. Captured records require the governed
    capture workflow tracked by #9191, which does not exist at this commit.
    """
    records = [
        {
            "id": f"{program['id']}-primary",
            "program_id": program["id"],
            "status": "pending",
            "path": None,
            "sha256": None,
            "width": None,
            "height": None,
            "viewport": None,
            "theme": None,
            "capture_workflow_id": None,
            "capture_step_id": None,
            "alt_text": None,
            "caption": None,
            "visible_limitations": [],
            "artifact_class": "illustrative",
            "reason": SCREENSHOT_PENDING_REASON,
        }
        for program in catalog["programs"]
        if not program["hidden"]
    ]
    return {
        "$schema": SCREENSHOTS_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "manifest_id": SCREENSHOTS_MANIFEST_ID,
        "source": {
            "repository": catalog["source"]["repository"],
            "commit": catalog["source"]["commit"],
        },
        "records": records,
    }


@dataclass(frozen=True)
class CompanionPayloadSet:
    """The three consumer-facing JSON documents built from one input read."""

    catalog: dict[str, Any]
    capabilities: dict[str, Any]
    screenshots: dict[str, Any]


def build_payload_set(
    repo_root: Path, *, require_clean: bool = True
) -> CompanionPayloadSet:
    """Build the manifest plus its derived capability and screenshot payloads."""
    catalog, capability_ids_by_program = _build(repo_root, require_clean=require_clean)
    return CompanionPayloadSet(
        catalog=catalog,
        capabilities=_capabilities_payload(catalog, capability_ids_by_program),
        screenshots=_screenshots_payload(catalog),
    )


_REQUIRES_PYTHON_RE = re.compile(
    r">=(?P<floor_major>\d+)\.(?P<floor>\d+),<(?P<ceiling_major>\d+)\.(?P<ceiling>\d+)"
)


def _supported_python_minors(requires_python: str) -> list[str]:
    """Derive the advertised minor versions from ``requires-python``.

    The specifier in ``pyproject.toml`` is the single source of truth for the
    supported range; duplicating the minor list as a literal here (or as a
    schema ``const``/``enum``) is how the two drift apart.
    """
    match = _REQUIRES_PYTHON_RE.fullmatch(requires_python)
    if match is None:
        msg = (
            "requires-python must be a bounded range such as '>=3.11,<3.13', "
            f"got {requires_python!r}"
        )
        raise ValueError(msg)
    major = match.group("floor_major")
    if match.group("ceiling_major") != major:
        msg = f"requires-python must not span major versions, got {requires_python!r}"
        raise ValueError(msg)
    floor = int(match.group("floor"))
    ceiling = int(match.group("ceiling"))
    if ceiling <= floor:
        msg = (
            "requires-python upper bound must exceed the floor, got "
            f"{requires_python!r}"
        )
        raise ValueError(msg)
    return [f"{major}.{minor}" for minor in range(floor, ceiling)]


def build_catalog(repo_root: Path, *, require_clean: bool = True) -> dict[str, Any]:
    """Build a canonical catalog from committed, repository-local inputs.

    Preconditions:
        ``repo_root`` is a Git checkout at an exact commit; inputs are tracked,
        unchanged files; and authoritative exports use a clean tree.
    Postconditions:
        Discovery is explicitly local-only and all collections are ID-sorted.
    """
    return _build(repo_root, require_clean=require_clean)[0]


def _build(
    repo_root: Path, *, require_clean: bool
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    root = repo_root.resolve()
    _git_commit(root)
    if require_clean and (dirty := _dirty_paths(root)):
        raise CatalogAuthorityError(
            "authoritative export requires a clean tree; dirty paths: "
            + ", ".join(dirty)
        )
    payloads = {
        path: _read_input(root, path, require_committed=require_clean)
        for path in INPUT_PATHS
    }
    launcher = json.loads(
        payloads[Path("src/config/launcher_manifest.json")].decode("utf-8")
    )
    parity = json.loads(
        payloads[Path("src/config/feature_parity.json")].decode("utf-8")
    )
    models = read_local_models(payloads[Path("src/config/models.yaml")])
    features, feature_ids_by_program, surface_count = _features(parity["features"])
    programs = _merge_programs(launcher["tiles"], models, feature_ids_by_program)
    source_commit = _git_commit(root)
    workflows = companion_workflows.parse_registry(
        payloads[companion_workflows.REGISTRY_PATH],
        repo_root=root,
        source_commit=source_commit,
        program_ids={program["id"] for program in programs},
    )
    for fixture_path in companion_workflows.referenced_fixture_paths(workflows):
        payloads.setdefault(
            fixture_path,
            _read_input(root, fixture_path, require_committed=require_clean),
        )
    source = _source_provenance(root, payloads)
    tools_commit = _tools_gitlink(root)
    summary = {
        "raw_launcher_records": len(launcher["tiles"]),
        "local_model_records": len(models),
        "program_records": len(programs),
        "feature_records": len(features),
        "feature_surface_paths": surface_count,
        "workflow_records": len(workflows),
        "executable_workflow_records": sum(
            workflow["availability"]["state"] == "available" for workflow in workflows
        ),
    }
    catalog = _catalog_payload(
        launcher=launcher,
        parity=parity,
        payloads=payloads,
        source=source,
        tools_commit=tools_commit,
        inventories={
            "programs": programs,
            "features": features,
            "workflows": workflows,
        },
        summary=summary,
    )
    return catalog, _capability_ids_by_program(launcher["tiles"], models)


def render_catalog(catalog: Mapping[str, Any]) -> bytes:
    """Serialize the catalog to canonical, human-reviewable JSON bytes."""
    return (
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


# The derived payloads use the same canonical serialisation as the manifest.
render_json = render_catalog


def write_catalog(
    repo_root: Path,
    *,
    output: Path = DEFAULT_OUTPUT,
    require_clean: bool = True,
) -> Path:
    """Write a catalog and detached SHA-256 digest; return the digest path."""
    root = repo_root.resolve()
    destination = output if output.is_absolute() else root / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = render_catalog(build_catalog(root, require_clean=require_clean))
    destination.write_bytes(payload)
    digest_path = destination.with_suffix(destination.suffix + ".sha256")
    digest_path.write_text(
        f"{_sha256(payload)}  {destination.name}\n", encoding="ascii", newline="\n"
    )
    return digest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fail-closed authoritative export command."""
    args = _parser().parse_args(argv)
    try:
        digest_path = write_catalog(args.repo_root, output=args.output)
    except (CatalogAuthorityError, ValueError) as exc:
        sys.stderr.write(f"companion catalog export refused: {exc}\n")
        return 2
    sys.stdout.write(f"wrote {args.output} and {digest_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
