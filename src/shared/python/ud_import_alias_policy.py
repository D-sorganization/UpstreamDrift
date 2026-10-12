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
import sys
from collections.abc import Callable
from typing import Any

from src.shared.python.import_aliases import (
    SharedImportAliasFinder,
    _bind_legacy_src_namespaces,
    _coalesce_loaded_aliases,
    _DOWNSTREAM_SRC_ALIAS_ROOTS,
)

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
    from src.shared.python._contracts_level import ContractLevel

    contracts_enabled = level != ContractLevel.OFF
    for mod_name in _CONTRACT_MODULE_NAMES:
        module = sys.modules.get(mod_name)
        if module is None:
            continue
        module.DBC_LEVEL = level  # type: ignore[attr-defined]
        module.CONTRACTS_ENABLED = contracts_enabled  # type: ignore[attr-defined]
        state = getattr(module, "_ContractState", None)
        if state is not None and state is not primary_state:
            state.level = level


def _unify_contracts_module_level_state() -> None:
    """Bind the Tools child-copy ``contracts`` exports to UD canonical level state."""
    from src.shared.python import _contracts_level as canonical

    contracts = importlib.import_module("src.shared.python.contracts")
    if getattr(contracts, "_ud_contract_level_unified", False):
        return

    contracts.ContractLevel = canonical.ContractLevel  # type: ignore[attr-defined]
    contracts._ContractState = canonical._ContractState  # type: ignore[attr-defined]
    contracts.DBC_LEVEL = canonical.DBC_LEVEL  # type: ignore[attr-defined]
    contracts.CONTRACTS_ENABLED = canonical.CONTRACTS_ENABLED  # type: ignore[attr-defined]
    contracts.get_contract_level = canonical.get_contract_level  # type: ignore[attr-defined]

    original_set = canonical.set_contract_level

    def set_contract_level_unified(level: Any) -> None:
        original_set(level)
        _sync_contract_level_aliases(level, primary_state=canonical._ContractState)

    set_contract_level_unified._ud_contract_level_sync_installed = True  # type: ignore[attr-defined]
    contracts.set_contract_level = set_contract_level_unified  # type: ignore[attr-defined]
    contracts._ud_contract_level_unified = True  # type: ignore[attr-defined]


def install_contract_level_alias_sync() -> None:
    """Keep ``set_contract_level`` consistent across Tools and UD spellings (#12177)."""
    _unify_contracts_module_level_state()


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
