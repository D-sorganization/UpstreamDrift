import pytest
from src.shared.python.contracts import (
    ContractLevel,
    PreconditionError,
    _evaluate_precondition,
    precondition,
    set_contract_level,
)


def test_evaluate_precondition_fails_closed_on_broken_lambda():
    """DbC must raise when a precondition lambda cannot be evaluated."""
    set_contract_level(ContractLevel.ENFORCE)

    def example_func(x: int, y: int) -> int:
        return x + y

    # A broken lambda with wrong parameter names cannot be evaluated
    broken_condition = lambda wrong_name: wrong_name > 0  # noqa: E731

    with pytest.raises(PreconditionError, match="Precondition evaluation failed"):
        _evaluate_precondition(broken_condition, example_func, (1, 2), {})


def test_evaluate_precondition_fails_closed_on_type_error():
    """DbC must raise when a precondition raises TypeError."""
    set_contract_level(ContractLevel.ENFORCE)

    def example_func(x: int, y: int) -> int:
        return x + y

    # A lambda that will raise TypeError when called with wrong arity
    bad_condition = lambda a, b, c, d: a > 0  # noqa: E731

    with pytest.raises(PreconditionError, match="Precondition evaluation failed"):
        _evaluate_precondition(bad_condition, example_func, (1, 2), {})


def test_precondition_passes_with_matching_signature():
    """Valid preconditions with correct signatures should pass silently."""
    set_contract_level(ContractLevel.ENFORCE)

    @precondition(lambda x: x > 0, "x must be positive")
    def double(x: int) -> int:
        return x * 2

    assert double(5) == 10


def test_precondition_blocks_with_failing_condition():
    """Valid preconditions that evaluate to False must raise."""
    set_contract_level(ContractLevel.ENFORCE)

    @precondition(lambda x: x > 0, "x must be positive")
    def double(x: int) -> int:
        return x * 2

    with pytest.raises(PreconditionError, match="x must be positive"):
        double(-1)
