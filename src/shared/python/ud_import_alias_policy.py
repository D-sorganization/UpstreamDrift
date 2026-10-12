"""UpstreamDrift policy on top of Tools ``SharedImportAliasFinder``.

``model_generation`` and ``humanoid_character_builder`` are ruled ud-canonical in
``docs/shared_tools/seam_rulings.v1.json``. Their implementations live under
``src/shared/python``; ``src.shared.python.<root>`` must resolve there instead
of being rewritten into the pinned Tools tree by the alias finder (issue #12177).

This module is UD-owned so we do not edit the Tools child copy
``import_aliases.py`` directly.
"""

from __future__ import annotations

import importlib
import enum
import sys
from collections.abc import Callable
from typing import Any

from src.shared.python.import_aliases import (
    SharedImportAliasFinder,
    _bind_legacy_src_namespaces,
    _coalesce_loaded_aliases,
    _DOWNSTREAM_SRC_ALIAS_ROOTS,
)
from src.shared.python._contracts_level import _contract_level_value

__all__ = [
    "UD_CANONICAL_SRC_ROOTS",
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


class UdSharedImportAliasFinder(SharedImportAliasFinder):
    """Decline ``src.shared.python`` aliasing for ud-canonical package roots."""

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
    """Mirror ``DBC_LEVEL`` onto every loaded contracts module spelling."""
    for mod_name in _CONTRACT_MODULE_NAMES:
        module = sys.modules.get(mod_name)
        if module is None:
            continue
        state = getattr(module, "_ContractState", None)
        target_enum = getattr(module, "ContractLevel", None)
        state_level = getattr(state, "level", None)
        if not (isinstance(target_enum, type) and issubclass(target_enum, enum.Enum)):
            target_enum = (
                type(state_level) if isinstance(state_level, enum.Enum) else None
            )
        module_level = (
            target_enum(_contract_level_value(level))
            if target_enum is not None
            else level
        )
        off_level = getattr(target_enum, "OFF", None)
        contracts_enabled = (
            module_level != off_level
            if off_level is not None
            else _contract_level_value(module_level) != "off"
        )
        module.DBC_LEVEL = module_level  # type: ignore[attr-defined]
        module.CONTRACTS_ENABLED = contracts_enabled  # type: ignore[attr-defined]
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
        original(level)
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
