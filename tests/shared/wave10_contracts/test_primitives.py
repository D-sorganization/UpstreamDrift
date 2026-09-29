"""Exhaustive tests for the DbC primitive helpers: require, ensure, invariant."""

from __future__ import annotations

import logging

import pytest

from src.shared.python._contracts_level import (
    ContractLevel,
    _ContractState,
    set_contract_level,
)
from src.shared.python._contracts_primitives import ensure, invariant, require
from src.shared.python.contracts import (
    InvariantError,
    PostconditionError,
    PreconditionError,
)


class TestRequire:
    def test_passes_when_condition_true(self):
        require(True, "always true")

    def test_raises_precondition_when_false(self):
        with pytest.raises(PreconditionError) as exc:
            require(False, "must be x", value=42)
        assert "must be x" in str(exc.value)
        assert "42" in str(exc.value)
        assert exc.value.condition_type == "pre-condition"
        assert exc.value.value == 42

    def test_raises_precondition_when_falsey_value(self):
        with pytest.raises(PreconditionError):
            require(0, "zero is falsey")  # type: ignore[arg-type]

    def test_off_level_skips_check(self):
        set_contract_level(ContractLevel.OFF)
        # Should not raise even with a false condition
        require(False, "skipped")

    def test_warn_level_logs(self, caplog):
        set_contract_level(ContractLevel.WARN)
        with caplog.at_level(logging.WARNING):
            require(False, "warn-only", value="v")
        # Find at least one warning containing the message
        assert any("warn-only" in r.message for r in caplog.records)

    def test_value_none_omitted_from_detail(self):
        with pytest.raises(PreconditionError) as exc:
            require(False, "no value here")
        assert "got:" not in str(exc.value)


class TestEnsure:
    def test_passes_when_true(self):
        ensure(True, "ok")

    def test_raises_postcondition_when_false(self):
        with pytest.raises(PostconditionError) as exc:
            ensure(False, "post fail", value=3.14)
        assert exc.value.condition_type == "post-condition"
        assert exc.value.value == 3.14

    def test_off_level_skips(self):
        set_contract_level(ContractLevel.OFF)
        ensure(False, "off")

    def test_warn_level_does_not_raise(self, caplog):
        set_contract_level(ContractLevel.WARN)
        with caplog.at_level(logging.WARNING):
            ensure(False, "warn-post")
        assert any("warn-post" in r.message for r in caplog.records)


class TestInvariantPrimitive:
    def test_passes_when_true(self):
        invariant(True, "ok")

    def test_raises_invariant_when_false(self):
        with pytest.raises(InvariantError) as exc:
            invariant(False, "broke", value=[1, 2])
        assert exc.value.condition_type == "invariant"
        assert exc.value.value == [1, 2]

    def test_off_level_skips(self):
        set_contract_level(ContractLevel.OFF)
        invariant(False, "off")

    def test_warn_level(self, caplog):
        set_contract_level(ContractLevel.WARN)
        with caplog.at_level(logging.WARNING):
            invariant(False, "warn-inv")
        assert any("warn-inv" in r.message for r in caplog.records)


def test_state_singleton_default_level():
    # ENFORCE under __debug__ is the documented default
    assert _ContractState.level in (ContractLevel.ENFORCE, ContractLevel.OFF)
