"""Tests for the DbC exception hierarchy and _handle_violation dispatch."""

from __future__ import annotations

import logging

import pytest

from src.shared.python._contracts_exceptions import (
    _VIOLATION_CLASSES,
    ContractEvaluationError,
    ContractViolationError,
    InvariantError,
    PostconditionError,
    PreconditionError,
    _handle_violation,
)
from src.shared.python._contracts_level import ContractLevel, set_contract_level


class TestExceptionConstruction:
    def test_base_records_fields(self):
        err = ContractViolationError("pre-condition", "msg", value=7)
        assert err.condition_type == "pre-condition"
        assert err.message == "msg"
        assert err.value == 7
        assert "pre-condition" in str(err)
        assert "got: 7" in str(err)

    def test_base_without_value_omits_got(self):
        err = ContractViolationError("invariant", "m")
        assert "got:" not in str(err)

    def test_subclass_precondition_sets_type(self):
        err = PreconditionError("p")
        assert err.condition_type == "pre-condition"
        assert isinstance(err, ContractViolationError)
        assert isinstance(err, AssertionError)
        assert isinstance(err, ValueError)

    def test_subclass_postcondition(self):
        err = PostconditionError("p", value=1)
        assert err.condition_type == "post-condition"
        assert err.value == 1

    def test_subclass_invariant(self):
        err = InvariantError("i")
        assert err.condition_type == "invariant"

    def test_subclass_evaluation_error(self):
        err = ContractEvaluationError("e")
        assert err.condition_type == "evaluation-error"
        assert isinstance(err, ContractViolationError)


class TestViolationDispatch:
    def test_enforce_raises_mapped_class(self):
        for ctype, cls in _VIOLATION_CLASSES.items():
            with pytest.raises(cls):
                _handle_violation(ctype, "x")

    def test_enforce_unknown_type_falls_back_to_base(self):
        with pytest.raises(ContractViolationError):
            _handle_violation("does-not-exist", "x")

    def test_warn_logs_only(self, caplog):
        set_contract_level(ContractLevel.WARN)
        with caplog.at_level(logging.WARNING):
            _handle_violation("pre-condition", "warn-msg", value=9)
        msgs = [r.message for r in caplog.records]
        assert any("warn-msg" in m and "got: 9" in m for m in msgs)

    def test_warn_without_value_no_got_segment(self, caplog):
        set_contract_level(ContractLevel.WARN)
        with caplog.at_level(logging.WARNING):
            _handle_violation("invariant", "novalue")
        assert any(
            "novalue" in r.message and "got:" not in r.message for r in caplog.records
        )

    def test_off_silent(self, caplog):
        set_contract_level(ContractLevel.OFF)
        with caplog.at_level(logging.WARNING):
            _handle_violation("pre-condition", "silent")
        assert not any("silent" in r.message for r in caplog.records)


def test_violation_classes_keys_are_canonical():
    assert set(_VIOLATION_CLASSES) == {
        "pre-condition",
        "post-condition",
        "invariant",
        "evaluation-error",
    }
