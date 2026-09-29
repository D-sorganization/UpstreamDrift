"""Tests for src.shared.python.core.contracts.level (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.core.contracts.level import (
    ContractLevel,
    contracts_enabled,
    disable_contracts,
    enable_contracts,
    get_contract_level,
    set_contract_level,
)


class TestContractLevelEnum:
    def test_off_value(self) -> None:
        assert ContractLevel.OFF.value == "off"

    def test_warn_value(self) -> None:
        assert ContractLevel.WARN.value == "warn"

    def test_enforce_value(self) -> None:
        assert ContractLevel.ENFORCE.value == "enforce"

    def test_three_members(self) -> None:
        assert len(ContractLevel) == 3

    def test_members_are_enum(self) -> None:
        for level in ContractLevel:
            assert isinstance(level, ContractLevel)


class TestSetGetContractLevel:
    def setup_method(self) -> None:
        # Save original level to restore after each test
        self._original = get_contract_level()

    def teardown_method(self) -> None:
        set_contract_level(self._original)

    def test_set_enforce(self) -> None:
        set_contract_level(ContractLevel.ENFORCE)
        assert get_contract_level() == ContractLevel.ENFORCE

    def test_set_warn(self) -> None:
        set_contract_level(ContractLevel.WARN)
        assert get_contract_level() == ContractLevel.WARN

    def test_set_off(self) -> None:
        set_contract_level(ContractLevel.OFF)
        assert get_contract_level() == ContractLevel.OFF

    def test_get_returns_contract_level_instance(self) -> None:
        result = get_contract_level()
        assert isinstance(result, ContractLevel)


class TestEnableDisableContracts:
    def setup_method(self) -> None:
        self._original = get_contract_level()

    def teardown_method(self) -> None:
        set_contract_level(self._original)

    def test_enable_contracts_sets_enforce(self) -> None:
        enable_contracts()
        assert get_contract_level() == ContractLevel.ENFORCE

    def test_disable_contracts_sets_off(self) -> None:
        disable_contracts()
        assert get_contract_level() == ContractLevel.OFF

    def test_contracts_enabled_true_when_enforce(self) -> None:
        enable_contracts()
        assert contracts_enabled() is True

    def test_contracts_enabled_false_when_off(self) -> None:
        disable_contracts()
        assert contracts_enabled() is False

    def test_contracts_enabled_true_when_warn(self) -> None:
        set_contract_level(ContractLevel.WARN)
        assert contracts_enabled() is True

    def test_toggle_enable_disable(self) -> None:
        enable_contracts()
        assert contracts_enabled() is True
        disable_contracts()
        assert contracts_enabled() is False
        enable_contracts()
        assert contracts_enabled() is True
