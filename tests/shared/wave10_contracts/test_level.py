"""Tests for ContractLevel resolution, state singleton, and set/get helpers."""

from __future__ import annotations

import logging
import os
import subprocess
import sys

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

pytestmark = pytest.mark.unit

OPTIMIZED_LEVEL_PROBE = """
import json
from src.shared.python import contracts as shared_contracts
from src.shared.python import _contracts_level as shared_level
from src.shared.python.core import contracts as core_contracts
from src.shared.python.core.contracts import level as core_level

def raises_precondition(require):
    try:
        require(False, "must fail")
    except Exception:
        return True
    return False

print(json.dumps({
    "debug": __debug__,
    "shared_level": shared_level.get_contract_level().value,
    "shared_enabled": shared_level.CONTRACTS_ENABLED,
    "shared_require_raises": raises_precondition(shared_contracts.require),
    "core_level": core_level.get_contract_level().value,
    "core_enabled": core_level.contracts_enabled(),
    "core_require_raises": raises_precondition(core_contracts.require),
}))
"""


def _run_optimized_level_probe(dbc_level: str | None) -> dict[str, object]:
    env = os.environ.copy()
    if dbc_level is None:
        env.pop("DBC_LEVEL", None)
    else:
        env["DBC_LEVEL"] = dbc_level
    result = subprocess.run(
        [sys.executable, "-O", "-c", OPTIMIZED_LEVEL_PROBE],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    import json

    return json.loads(result.stdout)


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

    def test_optimized_python_defaults_to_enforce_when_env_unset(self):
        state = _run_optimized_level_probe(dbc_level=None)

        assert state["debug"] is False
        assert state["shared_level"] == ContractLevel.ENFORCE.value
        assert state["shared_enabled"] is True
        assert state["shared_require_raises"] is True
        assert state["core_level"] == ContractLevel.ENFORCE.value
        assert state["core_enabled"] is True
        assert state["core_require_raises"] is True

    def test_optimized_python_respects_explicit_off(self):
        state = _run_optimized_level_probe(dbc_level="off")

        assert state["debug"] is False
        assert state["shared_level"] == ContractLevel.OFF.value
        assert state["shared_enabled"] is False
        assert state["shared_require_raises"] is False
        assert state["core_level"] == ContractLevel.OFF.value
        assert state["core_enabled"] is False
        assert state["core_require_raises"] is False


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
