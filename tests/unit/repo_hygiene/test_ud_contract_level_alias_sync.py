"""UD-owned contract level sync when Tools and UD spellings diverge (#12177)."""

from __future__ import annotations

import importlib
import enum
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


def test_contract_level_sync_preserves_each_modules_enum_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A separately loaded provider keeps its own ContractLevel class."""
    from src.shared.python.contracts import (
        ContractLevel,
        get_contract_level,
        set_contract_level,
    )
    from src.shared.python.ud_import_alias_policy import (
        install_contract_level_alias_sync,
    )

    class ProviderContractLevel(enum.Enum):
        OFF = "off"
        WARN = "warn"
        ENFORCE = "enforce"

    provider = types.ModuleType("shared.python.contracts")
    provider.ContractLevel = ProviderContractLevel
    provider.DBC_LEVEL = ProviderContractLevel.ENFORCE
    provider.CONTRACTS_ENABLED = True
    provider._ContractState = types.SimpleNamespace(level=ProviderContractLevel.ENFORCE)
    monkeypatch.setitem(sys.modules, "shared.python.contracts", provider)

    original = get_contract_level()
    try:
        install_contract_level_alias_sync()
        set_contract_level(ContractLevel.WARN)
        assert provider.DBC_LEVEL is ProviderContractLevel.WARN
        assert provider._ContractState.level is ProviderContractLevel.WARN
        assert isinstance(provider._ContractState.level, provider.ContractLevel)
    finally:
        set_contract_level(original)


def test_contract_level_sync_normalizes_provider_input_for_local_state() -> None:
    """A provider enum passed through the legacy setter stays locally typed."""
    from src.shared.python.contracts import (
        ContractLevel,
        get_contract_level,
        set_contract_level,
    )
    from src.shared.python.ud_import_alias_policy import (
        install_contract_level_alias_sync,
    )

    class ProviderContractLevel(enum.Enum):
        OFF = "off"
        WARN = "warn"
        ENFORCE = "enforce"

    original = get_contract_level()
    try:
        install_contract_level_alias_sync()
        set_contract_level(ProviderContractLevel.WARN)
        assert get_contract_level() is ContractLevel.WARN
        assert isinstance(get_contract_level(), ContractLevel)
    finally:
        set_contract_level(original)
