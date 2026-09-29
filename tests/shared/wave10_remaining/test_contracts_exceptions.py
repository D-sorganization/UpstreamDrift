"""Tests for src.shared.python._contracts_exceptions."""

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
from src.shared.python._contracts_level import (
    ContractLevel,
    _ContractState,
    set_contract_level,
)


@pytest.fixture
def restore_contract_level():
    original = _ContractState.level
    yield
    set_contract_level(original)


@pytest.mark.unit
def test_violation_classes_mapping_complete():
    assert _VIOLATION_CLASSES["pre-condition"] is PreconditionError
    assert _VIOLATION_CLASSES["post-condition"] is PostconditionError
    assert _VIOLATION_CLASSES["invariant"] is InvariantError
    assert _VIOLATION_CLASSES["evaluation-error"] is ContractEvaluationError


@pytest.mark.unit
def test_contract_violation_error_message_includes_value():
    err = ContractViolationError("pre-condition", "must be positive", value=-1)
    text = str(err)
    assert "pre-condition" in text
    assert "must be positive" in text
    assert "-1" in text


@pytest.mark.unit
def test_contract_violation_error_message_omits_none_value():
    err = ContractViolationError("invariant", "x must hold")
    assert "got:" not in str(err)


@pytest.mark.unit
def test_contract_violation_attributes():
    err = ContractViolationError("invariant", "msg", value=42)
    assert err.condition_type == "invariant"
    assert err.message == "msg"
    assert err.value == 42


@pytest.mark.unit
def test_contract_violation_is_assertion_and_value_error():
    # Subclasses should inherit from both AssertionError and ValueError
    err = ContractViolationError("pre-condition", "x")
    assert isinstance(err, AssertionError)
    assert isinstance(err, ValueError)


@pytest.mark.unit
def test_specific_subclasses_carry_correct_type():
    pre = PreconditionError("p")
    post = PostconditionError("po")
    inv = InvariantError("inv")
    ev = ContractEvaluationError("e")
    assert pre.condition_type == "pre-condition"
    assert post.condition_type == "post-condition"
    assert inv.condition_type == "invariant"
    assert ev.condition_type == "evaluation-error"


@pytest.mark.unit
def test_contract_violation_requires_condition_type():
    with pytest.raises(ValueError, match="condition_type must be provided"):
        ContractViolationError(None, "msg")  # type: ignore[arg-type]


@pytest.mark.unit
def test_precondition_error_requires_message():
    with pytest.raises(ValueError, match="message must be provided"):
        PreconditionError(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_postcondition_error_requires_message():
    with pytest.raises(ValueError, match="message must be provided"):
        PostconditionError(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_invariant_error_requires_message():
    with pytest.raises(ValueError, match="message must be provided"):
        InvariantError(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_evaluation_error_requires_message():
    with pytest.raises(ValueError, match="message must be provided"):
        ContractEvaluationError(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _handle_violation behaviour at the three enforcement levels
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_handle_violation_enforce_raises_specific_error(restore_contract_level):
    set_contract_level(ContractLevel.ENFORCE)
    with pytest.raises(PreconditionError):
        _handle_violation("pre-condition", "boom")
    with pytest.raises(PostconditionError):
        _handle_violation("post-condition", "boom")
    with pytest.raises(InvariantError):
        _handle_violation("invariant", "boom")
    with pytest.raises(ContractEvaluationError):
        _handle_violation("evaluation-error", "boom")


@pytest.mark.unit
def test_handle_violation_enforce_unknown_type_uses_base(restore_contract_level):
    set_contract_level(ContractLevel.ENFORCE)
    with pytest.raises(ContractViolationError):
        _handle_violation("mystery-type", "boom")


@pytest.mark.unit
def test_handle_violation_warn_logs_only(restore_contract_level, caplog):
    set_contract_level(ContractLevel.WARN)
    with caplog.at_level(logging.WARNING):
        # Must NOT raise.
        _handle_violation("pre-condition", "soft warn", value=7)
    joined = " ".join(rec.message for rec in caplog.records)
    assert "pre-condition" in joined
    assert "soft warn" in joined
    assert "7" in joined


@pytest.mark.unit
def test_handle_violation_off_does_nothing(restore_contract_level, caplog):
    set_contract_level(ContractLevel.OFF)
    with caplog.at_level(logging.WARNING):
        _handle_violation("pre-condition", "silent")
    # Should not raise, should not log a warning for the violation.
    assert all("silent" not in rec.message for rec in caplog.records)
