"""Tests for the higher-level numeric validators."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python._contracts_exceptions import (
    PostconditionError,
    PreconditionError,
)
from src.shared.python._contracts_level import ContractLevel, set_contract_level
from src.shared.python._contracts_validators import (
    check_non_negative,
    check_positive,
    check_pressure,
    check_range,
    check_temperature,
    ensure_valid_result,
    has_finite_elements,
    is_non_negative,
    is_positive,
    is_valid_result,
    require_finite,
    require_positive,
    require_unit_vector,
    set_contracts_enabled,
)


class TestCheckHelpers:
    def test_check_positive_pass(self):
        check_positive(1.0)

    def test_check_positive_fail(self):
        with pytest.raises(PreconditionError):
            check_positive(0)
        with pytest.raises(PreconditionError):
            check_positive(-1, name="thing")

    def test_check_non_negative(self):
        check_non_negative(0)
        check_non_negative(5)
        with pytest.raises(PreconditionError):
            check_non_negative(-0.01)

    def test_check_range_pass(self):
        check_range(5, 0, 10)
        check_range(0, 0, 10)
        check_range(10, 0, 10)

    def test_check_range_fail(self):
        with pytest.raises(PreconditionError):
            check_range(-1, 0, 10)
        with pytest.raises(PreconditionError):
            check_range(11, 0, 10, name="x")

    def test_check_temperature(self):
        check_temperature(300)
        with pytest.raises(PreconditionError):
            check_temperature(0)

    def test_check_pressure(self):
        check_pressure(101325)
        with pytest.raises(PreconditionError):
            check_pressure(-1)


class TestRequireHelpers:
    def test_require_positive_pass(self):
        require_positive(1)

    def test_require_positive_fail(self):
        with pytest.raises(PreconditionError) as exc:
            require_positive(-1, name="x")
        assert "x must be positive" in str(exc.value)

    def test_require_positive_off(self):
        set_contract_level(ContractLevel.OFF)
        require_positive(-5)  # no raise

    def test_require_finite_pass(self):
        require_finite(np.array([1.0, 2.0, 3.0]))

    def test_require_finite_nan(self):
        with pytest.raises(PreconditionError):
            require_finite(np.array([1.0, np.nan]))

    def test_require_finite_inf(self):
        with pytest.raises(PreconditionError):
            require_finite(np.array([1.0, np.inf]), name="vec")

    def test_require_finite_off(self):
        set_contract_level(ContractLevel.OFF)
        require_finite(np.array([np.nan]))

    def test_require_unit_vector_pass(self):
        require_unit_vector(np.array([1.0, 0.0, 0.0]))

    def test_require_unit_vector_fail(self):
        with pytest.raises(PreconditionError) as exc:
            require_unit_vector(np.array([2.0, 0.0, 0.0]))
        assert "unit vector" in str(exc.value)

    def test_require_unit_vector_tolerance(self):
        # Slightly off but within tol
        require_unit_vector(np.array([1.0 + 1e-9, 0.0, 0.0]), tol=1e-6)
        with pytest.raises(PreconditionError):
            require_unit_vector(np.array([1.0 + 1e-3, 0.0, 0.0]), tol=1e-6)

    def test_require_unit_vector_off(self):
        set_contract_level(ContractLevel.OFF)
        require_unit_vector(np.array([5.0, 0.0, 0.0]))


class _Result:
    def __init__(self, valid: bool, errors: list[str] | None = None):
        self.is_valid = valid
        self._errors = errors or []

    def get_error_messages(self):
        return self._errors


class TestEnsureValidResult:
    def test_valid_passes(self):
        ensure_valid_result(_Result(True))

    def test_invalid_raises(self):
        with pytest.raises(PostconditionError) as exc:
            ensure_valid_result(_Result(False, ["bad shape", "out of range"]))
        assert "bad shape" in str(exc.value)
        assert "out of range" in str(exc.value)

    def test_off_skips(self):
        set_contract_level(ContractLevel.OFF)
        ensure_valid_result(_Result(False, ["err"]))


class TestIsHelpers:
    def test_is_positive(self):
        assert is_positive(1) is True
        assert is_positive(0) is False
        assert is_positive(-1) is False

    def test_is_non_negative(self):
        assert is_non_negative(0) is True
        assert is_non_negative(1) is True
        assert is_non_negative(-1) is False

    def test_is_valid_result(self):
        assert is_valid_result(_Result(True)) is True
        assert is_valid_result(_Result(False)) is False

    def test_has_finite_elements(self):
        assert has_finite_elements(np.array([1.0, 2.0])) is True
        assert has_finite_elements(np.array([1.0, np.nan])) is False
        assert has_finite_elements(np.array([np.inf])) is False


class TestSetContractsEnabled:
    def test_enabled_true_maps_to_enforce(self):
        set_contracts_enabled(True)
        from src.shared.python._contracts_level import get_contract_level

        assert get_contract_level() == ContractLevel.ENFORCE

    def test_enabled_false_maps_to_off(self):
        set_contracts_enabled(False)
        from src.shared.python._contracts_level import get_contract_level

        assert get_contract_level() == ContractLevel.OFF
