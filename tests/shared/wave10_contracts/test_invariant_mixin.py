"""Tests for ContractChecker mixin and invariant_checked decorator."""

from __future__ import annotations

import logging

import pytest

from src.shared.python._contracts_exceptions import InvariantError
from src.shared.python._contracts_invariant_mixin import (
    ContractChecker,
    invariant_checked,
)
from src.shared.python._contracts_level import ContractLevel, set_contract_level


class _Counter(ContractChecker):
    def __init__(self, count: int = 0):
        self.count = count

    def _get_invariants(self):
        return [
            (lambda: self.count >= 0, "count must be non-negative"),
            (lambda: isinstance(self.count, int), "count must be int"),
        ]

    @invariant_checked
    def decrement(self):
        self.count -= 1


class TestContractChecker:
    def test_default_invariants_empty(self):
        cc = ContractChecker()
        assert cc._get_invariants() == []
        assert cc.verify_invariants() is True

    def test_pass(self):
        c = _Counter(2)
        assert c.verify_invariants() is True

    def test_enforce_raises(self):
        c = _Counter(0)
        c.count = -1
        with pytest.raises(InvariantError) as exc:
            c.verify_invariants()
        assert "count must be non-negative" in str(exc.value)
        assert "_Counter" in str(exc.value)

    def test_warn_logs(self, caplog):
        set_contract_level(ContractLevel.WARN)
        c = _Counter(0)
        c.count = -1
        with caplog.at_level(logging.WARNING):
            c.verify_invariants()
        assert any("count must be non-negative" in r.message for r in caplog.records)

    def test_off_skipped(self):
        set_contract_level(ContractLevel.OFF)
        c = _Counter(0)
        c.count = -1
        # Should not raise
        assert c.verify_invariants() is True

    def test_condition_raises_typeerror_wraps(self):
        class Bad(ContractChecker):
            def _get_invariants(self):
                def boom():
                    raise TypeError("oops")

                return [(boom, "bad")]

        with pytest.raises(InvariantError) as exc:
            Bad().verify_invariants()
        assert "oops" in str(exc.value)

    def test_condition_raises_invariant_propagates(self):
        class Bad(ContractChecker):
            def _get_invariants(self):
                def boom():
                    raise InvariantError("explicit")

                return [(boom, "bad")]

        with pytest.raises(InvariantError) as exc:
            Bad().verify_invariants()
        assert "explicit" in str(exc.value)

    def test_condition_raises_in_warn_mode_logs(self, caplog):
        set_contract_level(ContractLevel.WARN)

        class Bad(ContractChecker):
            def _get_invariants(self):
                def boom():
                    raise RuntimeError("runtime")

                return [(boom, "bad")]

        # Should not raise; runtime errors in WARN mode are absorbed
        Bad().verify_invariants()


class TestInvariantCheckedDecorator:
    def test_passes(self):
        c = _Counter(2)
        c.decrement()
        assert c.count == 1

    def test_violation_after_method(self):
        c = _Counter(0)
        with pytest.raises(InvariantError):
            c.decrement()

    def test_off_returns_original(self):
        set_contract_level(ContractLevel.OFF)

        class C(ContractChecker):
            def __init__(self):
                self.n = 0

            def _get_invariants(self):
                return [(lambda: self.n >= 0, "n>=0")]

            @invariant_checked
            def bad(self):
                self.n = -1

        c = C()
        c.bad()  # would raise if wrapper active
        assert c.n == -1
