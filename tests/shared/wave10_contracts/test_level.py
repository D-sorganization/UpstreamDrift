"""Tests for ContractLevel resolution, state singleton, and set/get helpers."""

from __future__ import annotations

import logging

import pytest

from src.shared.python import _contracts_level as level_mod
from src.shared.python._contracts_level import (
    CONTRACTS_ENABLED,
    ContractLevel,
    _ContractState,
    _resolve_contract_level,
    get_contract_level,
    set_contract_level,
)


class TestResolveLevel:
    @pytest.mark.parametrize(
        "env,expected",
        [
            ("off", ContractLevel.OFF),
            ("OFF", ContractLevel.OFF),
            ("warn", ContractLevel.WARN),
            (" Warn ", ContractLevel.WARN),
            ("enforce", ContractLevel.ENFORCE),
            ("ENFORCE", ContractLevel.ENFORCE),
        ],
    )
    def test_env_overrides(self, monkeypatch, env, expected):
        monkeypatch.setenv("DBC_LEVEL", env)
        assert _resolve_contract_level() == expected

    def test_missing_env_defaults_to_enforce_in_debug(self, monkeypatch):
        monkeypatch.delenv("DBC_LEVEL", raising=False)
        # __debug__ is True under standard pytest runs
        assert _resolve_contract_level() == ContractLevel.ENFORCE

    def test_unknown_env_falls_back_to_debug_default(self, monkeypatch):
        monkeypatch.setenv("DBC_LEVEL", "garbage")
        assert _resolve_contract_level() == ContractLevel.ENFORCE


class TestSetGet:
    def test_round_trip(self):
        for lvl in ContractLevel:
            set_contract_level(lvl)
            assert get_contract_level() == lvl
            assert _ContractState.level == lvl

    def test_set_logs_info(self, caplog):
        with caplog.at_level(logging.INFO, logger="src.shared.python._contracts_level"):
            set_contract_level(ContractLevel.WARN)
        assert any("WARN" in r.message.upper() for r in caplog.records)

    def test_set_propagates_to_aliases(self):
        # The contracts module is imported, aliases must reflect the new level
        import src.shared.python.contracts as contracts_mod

        set_contract_level(ContractLevel.OFF)
        assert contracts_mod.DBC_LEVEL == ContractLevel.OFF
        assert contracts_mod.CONTRACTS_ENABLED is False
        set_contract_level(ContractLevel.ENFORCE)
        assert contracts_mod.DBC_LEVEL == ContractLevel.ENFORCE
        assert contracts_mod.CONTRACTS_ENABLED is True


def test_contract_level_enum_values():
    assert ContractLevel.OFF.value == "off"
    assert ContractLevel.WARN.value == "warn"
    assert ContractLevel.ENFORCE.value == "enforce"


def test_module_constants_present():
    # Compile-time module-level constants
    assert isinstance(CONTRACTS_ENABLED, bool)
    assert hasattr(level_mod, "DBC_LEVEL")
