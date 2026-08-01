"""Shared import aliasing for legacy Tools package spellings.

The canonical import root for shared code is ``shared.python``. Older entry
points and downstream repos still import selected packages as top-level modules
(``sidekick``, ``theme``, ``ai``) or as ``src.shared.python``. This installer
keeps those spellings pointed at one canonical module object during the
deprecation window.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import warnings
from collections.abc import Mapping, Sequence
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, cast

__all__ = [
    "SharedImportAliasFinder",
    "install_aliases",
    "install_shared_import_aliases",
]

_SHARED_ROOTS = frozenset(
    {
        "ai",
        "calc_backend",
        "chat",
        "chat_contracts",
        "compatibility",
        "config",
        "contracts",
        "cors",
        "codemap",
        "deprecation",
        "humanoid_character_builder",
        "logging_pkg",
        "model_generation",
        "notes",
        "programmatic_pid",
        "rotation_transforms",
        "safe_eval",
        "sidekick",
        "signal_toolkit",
        "theme",
        "upstream_drift_tools",
    }
)
_DOWNSTREAM_SRC_ALIAS_ROOTS = frozenset({"chat", "sidekick", "upstream_drift_tools"})
_TOOLS_SRC_ROOT = Path(__file__).resolve().parents[2]


def alias_legacy_package(
    legacy_name: str,
    canonical_name: str,
    *,
    warning: str | None = None,
) -> types.ModuleType:
    """Bind a top-level compatibility package to its canonical package.

    Thin shim packages under ``src/`` use this during the deprecation window
    after ``src/shared/python`` stops being a package-discovery root. The shim
    preserves old import spellings without installing a second physical copy of
    the same shared package tree.
    """
    partial_shim = sys.modules.pop(legacy_name, None)
    if warning is not None:
        warnings.warn(warning, DeprecationWarning, stacklevel=2)
    try:
        install_shared_import_aliases()
        module = importlib.import_module(canonical_name)
        sys.modules[legacy_name] = module
    except Exception:
        if partial_shim is not None:
            sys.modules[legacy_name] = partial_shim
        raise
    return module


def _canonical_module(aliases: Sequence[str]) -> types.ModuleType | None:
    canonical_name = aliases[0]
    canonical = sys.modules.get(canonical_name)
    if canonical is not None:
        for alias in aliases:
            if alias in sys.modules:
                sys.modules[alias] = canonical
        return canonical

    for alias in aliases:
        module = sys.modules.get(alias)
        if module is not None:
            sys.modules[canonical_name] = module
            for name in aliases:
                if name in sys.modules:
                    sys.modules[name] = module
            return module
    return None


class _AliasLoader:
    def __init__(self, module: types.ModuleType) -> None:
        self.module = module

    def create_module(self, spec: ModuleSpec) -> types.ModuleType:
        return self.module

    def exec_module(self, module: types.ModuleType) -> None:
        return None


class _CanonicalAliasLoader:
    def __init__(self, canonical_spec: ModuleSpec, aliases: list[str]) -> None:
        self.canonical_spec = canonical_spec
        self.aliases = aliases
        self.canonical_name = aliases[0]

    def create_module(self, spec: ModuleSpec) -> types.ModuleType:
        return importlib.import_module(self.canonical_name)

    def get_code(self, fullname: str) -> types.CodeType | None:
        """Return canonical code when ``runpy`` executes an alias with ``-m``."""
        del fullname
        canonical_loader = self.canonical_spec.loader
        canonical_get_code = getattr(canonical_loader, "get_code", None)
        if canonical_get_code is None:
            return None
        return cast(types.CodeType | None, canonical_get_code(self.canonical_name))

    def exec_module(self, module: types.ModuleType) -> None:
        canonical = importlib.import_module(self.canonical_name)
        for alias in self.aliases:
            sys.modules[alias] = canonical


class SharedImportAliasFinder(MetaPathFinder):
    """Map deprecated shared import spellings to canonical module objects."""

    def _parse(self, fullname: str) -> tuple[str | None, str]:
        parts = fullname.split(".")
        if len(parts) >= 3 and parts[:2] == ["shared", "python"]:
            return (
                (parts[2], ".".join(parts[3:]))
                if parts[2] in _SHARED_ROOTS
                else (None, "")
            )
        if len(parts) >= 4 and parts[:3] == ["src", "shared", "python"]:
            allowed_roots = (
                _DOWNSTREAM_SRC_ALIAS_ROOTS
                if _external_src_package_is_available()
                else _SHARED_ROOTS
            )
            return (
                (parts[3], ".".join(parts[4:]))
                if parts[3] in allowed_roots
                else (None, "")
            )
        if parts and parts[0] in _SHARED_ROOTS:
            return parts[0], ".".join(parts[1:])
        return None, ""

    def _aliases(self, root: str, suffix: str) -> list[str]:
        suffix_part = f".{suffix}" if suffix else ""
        canonical_root = "sidekick" if root == "upstream_drift_tools" else root
        aliases = [
            f"shared.python.{canonical_root}{suffix_part}",
            f"src.shared.python.{canonical_root}{suffix_part}",
            f"{canonical_root}{suffix_part}",
        ]
        if root in {"sidekick", "upstream_drift_tools"}:
            aliases.extend(
                [
                    f"shared.python.upstream_drift_tools{suffix_part}",
                    f"src.shared.python.upstream_drift_tools{suffix_part}",
                    f"upstream_drift_tools{suffix_part}",
                ]
            )
        return list(dict.fromkeys(aliases))

    def _find_canonical_spec(self, aliases: list[str]) -> ModuleSpec | None:
        removed = False
        if self in sys.meta_path:
            sys.meta_path.remove(self)
            removed = True
        try:
            for alias in aliases:
                try:
                    spec = importlib.util.find_spec(alias)
                except Exception:
                    continue
                if spec is not None and spec.loader is not None:
                    return spec
            return None
        finally:
            if removed:
                sys.meta_path.insert(0, self)

    def _redirect_spec(
        self, fullname: str, loader: Any, canonical_spec: ModuleSpec
    ) -> ModuleSpec:
        spec = ModuleSpec(
            fullname,
            loader,
            origin=canonical_spec.origin,
            loader_state=canonical_spec.loader_state,
        )
        spec.submodule_search_locations = canonical_spec.submodule_search_locations
        spec.cached = canonical_spec.cached
        spec.has_location = canonical_spec.has_location
        return spec

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: types.ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname == "shared.python" or fullname.startswith("shared.python."):
            return None
        if ".tests." in fullname or fullname.endswith(".tests"):
            return None
        root, suffix = self._parse(fullname)
        if root is None:
            return None
        if root == "upstream_drift_tools" or fullname.startswith(
            "upstream_drift_tools."
        ):
            warnings.warn(
                "upstream_drift_tools is deprecated; import shared.python.sidekick.",
                DeprecationWarning,
                stacklevel=2,
            )
        aliases = self._aliases(root, suffix)
        module = _canonical_module(aliases)
        if module is not None:
            alias_spec = self._find_canonical_spec(aliases) or ModuleSpec(
                fullname, None
            )
            return self._redirect_spec(fullname, _AliasLoader(module), alias_spec)
        canonical_spec = self._find_canonical_spec(aliases)
        if canonical_spec is None:
            return None
        return self._redirect_spec(
            fullname, _CanonicalAliasLoader(canonical_spec, aliases), canonical_spec
        )


def _coalesce_loaded_aliases(finder: SharedImportAliasFinder) -> None:
    for fullname in list(sys.modules):
        root, suffix = finder._parse(fullname)
        if root is None:
            continue
        _canonical_module(finder._aliases(root, suffix))


def _src_search_locations() -> tuple[str, ...]:
    legacy_src = sys.modules.get("src")
    if legacy_src is not None:
        locations = getattr(legacy_src, "__path__", ())
        return tuple(str(location) for location in locations)
    try:
        spec = importlib.util.find_spec("src")
    except (ImportError, ValueError):
        return ()
    locations = None if spec is None else spec.submodule_search_locations
    return tuple(str(location) for location in locations or ())


def _external_src_package_is_available() -> bool:
    for location in _src_search_locations():
        try:
            if Path(location).resolve() != _TOOLS_SRC_ROOT:
                return True
        except OSError:
            return True
    return False


def _bind_legacy_src_namespaces() -> None:
    """Fill missing ``src.shared`` parents without replacing downstream packages."""
    shared = sys.modules.get("shared")
    shared_python = sys.modules.get("shared.python")
    if shared is None or shared_python is None or _external_src_package_is_available():
        return
    legacy_shared = sys.modules.setdefault("src.shared", shared)
    legacy_python = sys.modules.setdefault("src.shared.python", shared_python)
    legacy_src = sys.modules.get("src")
    if legacy_src is not None and not hasattr(legacy_src, "shared"):
        legacy_src.shared = legacy_shared  # type: ignore[attr-defined]
    if not hasattr(legacy_shared, "python"):
        legacy_shared.python = legacy_python  # type: ignore[attr-defined]


def install_shared_import_aliases() -> None:
    """Install the shared import alias finder once per interpreter."""
    _bind_legacy_src_namespaces()
    for finder in sys.meta_path:
        if isinstance(finder, SharedImportAliasFinder):
            _coalesce_loaded_aliases(finder)
            return
    finder = SharedImportAliasFinder()
    _coalesce_loaded_aliases(finder)
    sys.meta_path.insert(0, finder)
    _coalesce_loaded_aliases(finder)


def install_aliases(aliases: Mapping[str, str] | None = None) -> None:
    """Compatibility wrapper for shim packages that install shared aliases."""
    install_shared_import_aliases()
