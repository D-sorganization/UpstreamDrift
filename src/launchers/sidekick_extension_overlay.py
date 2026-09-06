"""Expose exact, manifest-approved UpstreamDrift Sidekick extensions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_file_location
from pathlib import Path
import sys
from typing import Any

import yaml

_SUPPORTED_SCOPES = ("chat", "sidekick")
_UPSTREAM_OWNER = "UpstreamDrift"
_REQUIRED_PARENT_RUNTIME = (
    "chat/websocket_protocol.py",
    "sidekick/__main__.py",
    "sidekick/persistence/__init__.py",
    "sidekick/persistence/schema.py",
    "sidekick/persistence/state_profile.py",
    "sidekick/standalone/__init__.py",
    "sidekick/standalone/onboarding.py",
    "sidekick/standalone/preferences.py",
    "sidekick/standalone/runner.py",
    "sidekick/standalone/session_store.py",
    "sidekick/standalone/window.py",
)


class IncompleteParentSidekickRuntimeError(RuntimeError):
    """The selected Tools source does not provide the required runtime."""


@dataclass(frozen=True)
class _ExtensionSource:
    """An exact local module approved by the ownership manifest."""

    path: Path
    is_package: bool


class _ExtensionAliasLoader(Loader):
    """Coalesce direct and legacy spellings onto a canonical extension."""

    def __init__(self, canonical_name: str, aliases: tuple[str, ...]) -> None:
        self._canonical_name = canonical_name
        self._aliases = aliases

    def create_module(self, spec: ModuleSpec) -> Any:
        del spec
        return __import__(self._canonical_name, fromlist=["*"])

    def exec_module(self, module: Any) -> None:
        canonical = sys.modules[self._canonical_name]
        for alias in self._aliases:
            sys.modules[alias] = canonical


class ManifestGatedSidekickFinder(MetaPathFinder):
    """Resolve only exact approved modules without widening package paths."""

    def __init__(self, sources: Mapping[str, _ExtensionSource]) -> None:
        self._sources = dict(sources)
        self._aliases = {
            alias: canonical
            for canonical in sources
            for alias in self._module_aliases(canonical)
            if alias != canonical
        }

    @staticmethod
    def _module_aliases(canonical: str) -> tuple[str, ...]:
        direct = canonical.removeprefix("shared.python.")
        return canonical, direct, f"src.{canonical}"

    @property
    def approved_modules(self) -> frozenset[str]:
        """Return the immutable set of modules this finder may expose."""
        return frozenset(self._sources)

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> ModuleSpec | None:
        """Return a source spec only for an exact approved module."""
        del path, target
        source = self._sources.get(fullname)
        if source is None and fullname in self._aliases:
            canonical = self._aliases[fullname]
            canonical_source = self._sources[canonical]
            spec = ModuleSpec(
                fullname,
                _ExtensionAliasLoader(
                    canonical,
                    self._module_aliases(canonical),
                ),
                origin=str(canonical_source.path),
            )
            if canonical_source.is_package:
                spec.submodule_search_locations = []
            return spec
        if source is None:
            return None
        search_locations: list[str] | None = [] if source.is_package else None
        source_spec = spec_from_file_location(
            fullname,
            source.path,
            submodule_search_locations=search_locations,
        )
        if source_spec is not None and source.is_package:
            source_spec.submodule_search_locations = []
        return source_spec

    def uninstall(self) -> None:
        """Remove this finder and only the extension modules it loaded."""
        while self in sys.meta_path:
            sys.meta_path.remove(self)
        for canonical, source in self._sources.items():
            expected_path = source.path.resolve()
            for alias in self._module_aliases(canonical):
                module = sys.modules.get(alias)
                module_file = getattr(module, "__file__", None)
                if module_file is None:
                    continue
                try:
                    loaded_path = Path(module_file).resolve()
                except OSError:
                    continue
                if loaded_path == expected_path:
                    sys.modules.pop(alias, None)


def _load_manifest_owners(manifest_path: Path) -> dict[str, str]:
    """Return validated file-level ownership decisions."""
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError("Sidekick ownership manifest is unavailable") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise RuntimeError("Sidekick ownership manifest is malformed")
    entries = raw.get("paths")
    if entries is None:
        entries = {}
    if not isinstance(entries, dict):
        raise RuntimeError("Sidekick ownership manifest is malformed")

    owners: dict[str, str] = {}
    for relative, metadata in entries.items():
        if (
            not isinstance(relative, str)
            or not relative.endswith(".py")
            or not isinstance(metadata, dict)
            or not isinstance(metadata.get("owner"), str)
        ):
            raise RuntimeError("Sidekick ownership manifest is malformed")
        owners[relative] = metadata["owner"]
    unresolved = sorted(
        relative for relative, owner in owners.items() if owner != _UPSTREAM_OWNER
    )
    if unresolved:
        raise RuntimeError(
            "Sidekick ownership manifest must resolve every entry to "
            f"{_UPSTREAM_OWNER}: {', '.join(unresolved)}"
        )
    return owners


def _python_inventory(root: Path) -> set[str]:
    """Return Chat and Sidekick Python paths relative to ``root``."""
    inventory: set[str] = set()
    for scope in _SUPPORTED_SCOPES:
        scope_root = root / scope
        if not scope_root.is_dir():
            continue
        inventory.update(
            path.relative_to(root).as_posix()
            for path in scope_root.rglob("*.py")
            if path.is_file()
        )
    return inventory


def _module_name(relative: str) -> tuple[str, bool]:
    """Convert a validated relative Python path into its import name."""
    parts = Path(relative).with_suffix("").parts
    is_package = parts[-1] == "__init__"
    module_parts = parts[:-1] if is_package else parts
    if not module_parts or module_parts[0] != "sidekick":
        raise RuntimeError(
            "Manifest-gated runtime extensions must use the sidekick namespace: "
            f"{relative}"
        )
    return ".".join(module_parts), is_package


def _approved_sources(
    *,
    local_python_root: Path,
    parent_python_root: Path,
    manifest_path: Path,
) -> dict[str, _ExtensionSource]:
    """Validate ownership parity and construct the exact import allowlist."""
    local_root = local_python_root.resolve()
    parent_root = parent_python_root.resolve()
    if not local_root.is_dir() or not parent_root.is_dir():
        raise RuntimeError(
            "Manifest-gated Sidekick extensions require local and parent "
            "Python source roots"
        )

    owners = _load_manifest_owners(manifest_path.resolve())
    local_inventory = _python_inventory(local_root)
    parent_inventory = _python_inventory(parent_root)
    local_only = local_inventory - parent_inventory
    if set(owners) != local_only:
        missing = sorted(local_only - set(owners))
        stale = sorted(set(owners) - local_only)
        raise RuntimeError(
            "Sidekick ownership inventory mismatch; "
            f"unclassified={missing!r}, stale={stale!r}"
        )

    sources: dict[str, _ExtensionSource] = {}
    for relative in sorted(owners):
        direct_name, is_package = _module_name(relative)
        module_name = f"shared.python.{direct_name}"
        source_path = (local_root / relative).resolve()
        if not source_path.is_relative_to(local_root) or not source_path.is_file():
            raise RuntimeError(f"Approved Sidekick extension is missing: {relative}")
        if module_name in sources:
            raise RuntimeError(
                f"Duplicate Sidekick extension module mapping: {module_name}"
            )
        sources[module_name] = _ExtensionSource(source_path, is_package)
    return sources


def install_manifest_gated_sidekick_extensions(
    *,
    local_python_root: Path,
    parent_python_root: Path,
    manifest_path: Path,
) -> ManifestGatedSidekickFinder:
    """Install a fail-closed exact-module finder for source checkouts.

    Preconditions:
        Both roots exist and the manifest exactly classifies every local-only
        Chat/Sidekick Python file as owned by UpstreamDrift.

    Postconditions:
        Canonical parent packages keep their original ``__path__`` values.
        Only exact approved Sidekick module names can resolve from local source.
    """
    sources = _approved_sources(
        local_python_root=local_python_root,
        parent_python_root=parent_python_root,
        manifest_path=manifest_path,
    )
    aliases = {
        alias
        for canonical in sources
        for alias in ManifestGatedSidekickFinder._module_aliases(canonical)
    }
    preloaded = sorted(name for name in aliases if name in sys.modules)
    if preloaded:
        raise RuntimeError(
            "An UpstreamDrift Sidekick extension was already loaded before "
            f"ownership validation: {', '.join(preloaded)}"
        )

    finder = ManifestGatedSidekickFinder(sources)
    sys.meta_path.insert(0, finder)
    return finder


def validate_parent_sidekick_runtime(parent_python_root: Path) -> None:
    """Require every canonical module migrated out of UpstreamDrift."""
    parent_root = parent_python_root.resolve()
    missing = [
        relative
        for relative in _REQUIRED_PARENT_RUNTIME
        if not (parent_root / relative).is_file()
    ]
    if missing:
        raise IncompleteParentSidekickRuntimeError(
            "The canonical Sidekick runtime is incomplete in the selected "
            f"Tools source: {', '.join(missing)}"
        )


__all__ = [
    "IncompleteParentSidekickRuntimeError",
    "ManifestGatedSidekickFinder",
    "install_manifest_gated_sidekick_extensions",
    "validate_parent_sidekick_runtime",
]
