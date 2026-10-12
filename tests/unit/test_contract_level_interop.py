"""Cross-module enum identity tests for contract-level setters."""

from __future__ import annotations

import enum
import logging
import sys
import types
from typing import Any, cast

import pytest

pytestmark = pytest.mark.unit


class UnsupportedLevel(enum.Enum):
    STRICT = "strict"


@pytest.mark.parametrize("level_name", ["ENFORCE", "WARN", "OFF"])
def test_public_setter_normalizes_core_enum(level_name: str) -> None:
    from src.shared.python import contracts as public_contracts
    from src.shared.python.core.contracts.level import ContractLevel as CoreLevel

    from src.shared.python.ud_import_alias_policy import (
        install_contract_level_alias_sync,
    )

    install_contract_level_alias_sync()
    original = public_contracts.get_contract_level()
    try:
        cast(Any, public_contracts.set_contract_level)(CoreLevel[level_name])
        level = public_contracts.get_contract_level()
        assert type(level) is public_contracts.ContractLevel
        assert level is public_contracts.ContractLevel[level_name]
        if level_name == "ENFORCE":
            with pytest.raises(public_contracts.PreconditionError):
                public_contracts.require(False, "interop guard")
        else:
            public_contracts.require(False, "non-enforcing interop level")
    finally:
        public_contracts.set_contract_level(original)


@pytest.mark.parametrize("level_name", ["ENFORCE", "WARN", "OFF"])
def test_core_setter_normalizes_public_enum(level_name: str) -> None:
    from src.shared.python.contracts import ContractLevel as PublicLevel
    from src.shared.python.core import contracts as core_contracts

    original = core_contracts.get_contract_level()
    try:
        core_contracts.set_contract_level(PublicLevel[level_name])
        level = core_contracts.get_contract_level()
        assert type(level) is core_contracts.ContractLevel
        assert level is core_contracts.ContractLevel[level_name]
        if level_name == "ENFORCE":
            with pytest.raises(core_contracts.PreconditionError):
                core_contracts.require(False, "interop guard")
        else:
            core_contracts.require(False, "non-enforcing interop level")
    finally:
        core_contracts.set_contract_level(original)


@pytest.mark.parametrize("level_name", ["ENFORCE", "WARN", "OFF"])
def test_shared_level_setter_normalizes_public_enum(level_name: str) -> None:
    from src.shared.python import _contracts_level as shared_level
    from src.shared.python.contracts import ContractLevel as PublicLevel

    original = shared_level.get_contract_level()
    try:
        shared_level.set_contract_level(PublicLevel[level_name])
        level = shared_level.get_contract_level()
        assert type(level) is shared_level.ContractLevel
        assert level is shared_level.ContractLevel[level_name]
    finally:
        shared_level.set_contract_level(original)


@pytest.mark.parametrize("level_name", ["ENFORCE", "WARN", "OFF"])
def test_alias_sync_preserves_alias_enum_type(
    monkeypatch: pytest.MonkeyPatch, level_name: str
) -> None:
    from src.shared.python import contracts as public_contracts
    from src.shared.python.ud_import_alias_policy import (
        install_contract_level_alias_sync,
    )

    class CompatibilityLevel(enum.Enum):
        OFF = "off"
        WARN = "warn"
        ENFORCE = "enforce"

    original = public_contracts.get_contract_level()
    alias: Any = types.ModuleType("shared.python.contracts")
    alias.ContractLevel = CompatibilityLevel
    alias.DBC_LEVEL = CompatibilityLevel.ENFORCE
    alias.CONTRACTS_ENABLED = True
    alias._ContractState = types.SimpleNamespace(level=CompatibilityLevel.ENFORCE)
    monkeypatch.setitem(sys.modules, alias.__name__, alias)

    try:
        install_contract_level_alias_sync()
        public_contracts.set_contract_level(public_contracts.ContractLevel[level_name])
        assert type(alias.DBC_LEVEL) is CompatibilityLevel
        assert type(alias._ContractState.level) is CompatibilityLevel
        assert alias.DBC_LEVEL is CompatibilityLevel[level_name]
        assert alias._ContractState.level is CompatibilityLevel[level_name]
        assert alias.CONTRACTS_ENABLED is (level_name != "OFF")
    finally:
        public_contracts.set_contract_level(original)


def test_public_warn_logs_and_off_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    from src.shared.python import contracts as public_contracts
    from src.shared.python.ud_import_alias_policy import (
        install_contract_level_alias_sync,
    )

    install_contract_level_alias_sync()
    original = public_contracts.get_contract_level()
    try:
        with caplog.at_level(logging.WARNING, logger="src.shared.python.contracts"):
            public_contracts.set_contract_level(public_contracts.ContractLevel.WARN)
            public_contracts.require(False, "warn interop")
            assert any("warn interop" in record.message for record in caplog.records)
            caplog.clear()
            public_contracts.set_contract_level(public_contracts.ContractLevel.OFF)
            public_contracts.require(False, "off interop")
            assert not caplog.records
    finally:
        public_contracts.set_contract_level(original)


@pytest.mark.parametrize("setter_name", ["public", "core", "shared"])
def test_setters_reject_unsupported_levels_without_mutation(setter_name: str) -> None:
    from src.shared.python import _contracts_level, contracts as public_contracts
    from src.shared.python.core import contracts as core_contracts

    module: Any = {
        "public": public_contracts,
        "core": core_contracts,
        "shared": _contracts_level,
    }[setter_name]
    if setter_name == "public":
        from src.shared.python.ud_import_alias_policy import (
            install_contract_level_alias_sync,
        )

        install_contract_level_alias_sync()
    original = module.get_contract_level()
    with pytest.raises(ValueError, match="contract level"):
        module.set_contract_level(UnsupportedLevel.STRICT)
    assert module.get_contract_level() is original
    with pytest.raises(TypeError, match="Enum"):
        module.set_contract_level(object())
    assert module.get_contract_level() is original
