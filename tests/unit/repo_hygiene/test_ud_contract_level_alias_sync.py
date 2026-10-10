"""UD-owned contract level sync when Tools and UD spellings diverge (#12177)."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_set_contract_level_syncs_pinned_tools_contracts_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``shared.python.contracts`` must track runtime level changes from the UD copy."""
    from src.shared.python.contracts import ContractLevel, set_contract_level
    from src.shared.python.ud_import_alias_policy import (
        install_contract_level_alias_sync,
    )

    src_mod = importlib.import_module("src.shared.python.contracts")
    tools_mod = types.ModuleType("shared.python.contracts")
    tools_mod.DBC_LEVEL = ContractLevel.ENFORCE
    tools_mod.CONTRACTS_ENABLED = True
    tools_mod._ContractState = types.SimpleNamespace(level=ContractLevel.ENFORCE)
    monkeypatch.setitem(sys.modules, "shared.python.contracts", tools_mod)

    install_contract_level_alias_sync()
    set_contract_level(ContractLevel.WARN)

    assert src_mod.DBC_LEVEL == ContractLevel.WARN
    assert tools_mod.DBC_LEVEL == ContractLevel.WARN
    assert tools_mod._ContractState.level == ContractLevel.WARN
