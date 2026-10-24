"""UpstreamDrift policy on top of Tools ``SharedImportAliasFinder``.

``model_generation`` and ``humanoid_character_builder`` are ruled ud-canonical in
``docs/shared_tools/seam_rulings.v1.json``. Their implementations live under
``src/shared/python``; ``src.shared.python.<root>`` must resolve there instead
of being rewritten into the pinned Tools tree by the alias finder (issue #12177).

This module is UD-owned so we do not edit the Tools child copy
``import_aliases.py`` directly.
"""

from __future__ import annotations

import enum
import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from src.shared.python.import_aliases import (
    SharedImportAliasFinder,
    _AliasLoader,
    _bind_legacy_src_namespaces,
    _coalesce_loaded_aliases,
    _DOWNSTREAM_SRC_ALIAS_ROOTS,
)
from src.shared.python._contracts_level import (
    _contract_level_for_module,
    _contract_level_value,
)

__all__ = [
    "UD_CANONICAL_SRC_ROOTS",
    "UD_OWNED_SHARED_PYTHON_ROOTS",
    "UdSharedImportAliasFinder",
    "install_contract_level_alias_sync",
    "install_ud_canonical_shared_import_aliases",
]

_CONTRACT_MODULE_NAMES = frozenset(
    {
        "src.shared.python.contracts",
        "shared.python.contracts",
    }
)

UD_CANONICAL_SRC_ROOTS = frozenset({"humanoid_character_builder", "model_generation"})

# Packages UD owns that pinned Tools code nevertheless imports as
# ``shared.python.<root>`` (e.g. ``data_explorer._embed_adapter`` imports
# ``shared.python.launcher_embed``; UD #12895). Tools has no such module, so the
# spelling is served from UD's single copy instead of failing or loading a twin.
UD_OWNED_SHARED_PYTHON_ROOTS = frozenset({"launcher_embed"})

# Chat retains a UD-only _qt/runtime.py, so only this exact Tools-owned module
# is opted into canonical loading; the package itself must keep its local path.
_CHAT_UI_BUILDER_CANONICAL = "shared.python.chat._qt.ui_builder"
_CHAT_UI_BUILDER_LEGACY = "src.shared.python.chat._qt.ui_builder"
_CHAT_UI_BUILDER_RELATIVE = Path("chat/_qt/ui_builder.py")
_TOOLS_SHARED_PYTHON = (
    Path(__file__).resolve().parents[3]
    / "vendor"
    / "ud-tools"
    / "src"
    / "shared"
    / "python"
)


class UdSharedImportAliasFinder(SharedImportAliasFinder):
    """Apply UD ownership rules and exact pinned-provider aliases.

    Decline ``src.shared.python`` aliasing for ud-canonical package roots,
    serve UD-owned modules from UD's copy, and route the chat UI builder module
    to Tools without redirecting the rest of the locally extended chat package.
    """

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname == _CHAT_UI_BUILDER_CANONICAL:
            return self._pinned_chat_ui_builder_spec(fullname)
        if fullname == _CHAT_UI_BUILDER_LEGACY:
            canonical = importlib.import_module(_CHAT_UI_BUILDER_CANONICAL)
            return self._alias_spec(fullname, canonical)

        parts = fullname.split(".")
        if (
            len(parts) >= 3
            and parts[:2] == ["shared", "python"]
            and parts[2] in UD_OWNED_SHARED_PYTHON_ROOTS
        ):
            canonical = importlib.import_module(
                ".".join(["src", "shared", "python", *parts[2:]])
            )
            return self._alias_spec(fullname, canonical)
        return super().find_spec(fullname, path, target)

    @staticmethod
    def _alias_spec(fullname: str, canonical: ModuleType) -> Any:
        """Build an alias spec while preserving its canonical module origin."""
        spec = importlib.util.spec_from_loader(
            fullname,
            cast(Any, _AliasLoader(canonical)),
            origin=canonical.__file__,
        )
        if spec is not None:
            spec.submodule_search_locations = getattr(canonical, "__path__", None)
        return spec

    def _pinned_chat_ui_builder_spec(self, fullname: str) -> Any:
        """Load the exact Tools-owned builder from vendor or an install."""
        tools_root = _TOOLS_SHARED_PYTHON.resolve()
        vendored_provider = (tools_root / _CHAT_UI_BUILDER_RELATIVE).resolve()
        provider: Path | None = (
            vendored_provider
            if vendored_provider.is_relative_to(tools_root)
            and vendored_provider.is_file()
            else self._installed_chat_ui_builder_path()
        )
        if provider is None:
            raise ModuleNotFoundError(
                f"{fullname} requires the pinned Tools provider; no vendored or "
                "installed Tools chat UI builder was found",
                name=fullname,
            )
        spec = importlib.util.spec_from_file_location(fullname, provider)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load pinned Tools provider for {fullname}")
        return spec

    def _installed_chat_ui_builder_path(self) -> Path | None:
        """Find the exact canonical builder in installed Tools, never UD's copy."""
        finder_position = next(
            (index for index, finder in enumerate(sys.meta_path) if finder is self),
            None,
        )
        if finder_position is not None:
            sys.meta_path.pop(finder_position)
        try:
            spec = importlib.util.find_spec(_CHAT_UI_BUILDER_CANONICAL)
        except (ImportError, ValueError):
            return None
        finally:
            if finder_position is not None:
                sys.meta_path.insert(finder_position, self)

        if spec is None or spec.origin is None:
            return None
        provider = Path(spec.origin).resolve()
        ud_shared_python = Path(__file__).resolve().parent
        if (
            provider.is_relative_to(ud_shared_python)
            or provider.parts[-len(_CHAT_UI_BUILDER_RELATIVE.parts) :]
            != _CHAT_UI_BUILDER_RELATIVE.parts
            or not provider.is_file()
        ):
            return None
        return provider

    def _parse(self, fullname: str) -> tuple[str | None, str]:
        parts = fullname.split(".")
        if len(parts) >= 4 and parts[:3] == ["src", "shared", "python"]:
            root = parts[3]
            if root in UD_CANONICAL_SRC_ROOTS:
                return (None, "")
            allowed_roots = _DOWNSTREAM_SRC_ALIAS_ROOTS
            return (root, ".".join(parts[4:])) if root in allowed_roots else (None, "")
        return super()._parse(fullname)


def _sync_contract_level_aliases(
    level: Any,
    *,
    primary_state: Any,
) -> None:
    """Mirror DBC_LEVEL onto every loaded contracts module spelling."""
    updates: list[tuple[ModuleType, enum.Enum, bool, Any]] = []
    for mod_name in _CONTRACT_MODULE_NAMES:
        module = sys.modules.get(mod_name)
        if module is None:
            continue
        module_level, contracts_enabled, state = _contract_level_for_module(
            level, module
        )
        updates.append((module, module_level, contracts_enabled, state))
    for target_module, module_level, contracts_enabled, state in updates:
        target_module.DBC_LEVEL = module_level  # type: ignore[attr-defined]
        target_module.CONTRACTS_ENABLED = contracts_enabled  # type: ignore[attr-defined]
        if state is not None and state is not primary_state:
            state.level = module_level


def install_contract_level_alias_sync() -> None:
    """Keep ``set_contract_level`` consistent across Tools and UD spellings (#12177)."""
    contracts = importlib.import_module("src.shared.python.contracts")
    original: Callable[[Any], None] = contracts.set_contract_level
    if getattr(original, "_ud_contract_level_sync_installed", False):
        return
    primary_state = contracts._ContractState

    def set_contract_level_with_sync(level: Any) -> None:
        normalized = contracts.ContractLevel(_contract_level_value(level))
        original(normalized)
        _sync_contract_level_aliases(
            contracts.get_contract_level(),
            primary_state=primary_state,
        )

    set_contract_level_with_sync._ud_contract_level_sync_installed = True  # type: ignore[attr-defined]
    contracts.set_contract_level = set_contract_level_with_sync  # type: ignore[attr-defined]


def install_ud_canonical_shared_import_aliases() -> None:
    """Install the UD policy finder once per interpreter."""
    _bind_legacy_src_namespaces()
    for index, finder in enumerate(sys.meta_path):
        if isinstance(finder, SharedImportAliasFinder):
            if type(finder) is UdSharedImportAliasFinder:
                _coalesce_loaded_aliases(finder)
                install_contract_level_alias_sync()
                return
            policy_finder = UdSharedImportAliasFinder()
            sys.meta_path[index] = policy_finder
            _coalesce_loaded_aliases(policy_finder)
            install_contract_level_alias_sync()
            return
    finder = UdSharedImportAliasFinder()
    _coalesce_loaded_aliases(finder)
    sys.meta_path.insert(0, finder)
    _coalesce_loaded_aliases(finder)
    install_contract_level_alias_sync()
