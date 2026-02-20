"""Tests for tests.helpers.numerical — the shared numerical test helpers.

Follows TDD: every public function is exercised for both passing and
failing paths, plus edge cases and precondition violations.
"""

import pytest

from tests.helpers.numerical import (
    assert_close,
    assert_conserved,
    assert_jacobian_symmetry,
    assert_monotonic,
    assert_physics_state,
    is_finite,
)

# ── is_finite ────────────────────────────────────────────────────────────────


class TestIsFinite:
    """Tests for the is_finite helper."""

    def test_finite_int(self):
        assert is_finite(42) is True

    def test_finite_float(self):
        assert is_finite(3.14) is True

    def test_zero(self):
        assert is_finite(0) is True
        assert is_finite(0.0) is True

    def test_negative(self):
        assert is_finite(-1.5) is True

    def test_nan(self):
        assert is_finite(float("nan")) is False

    def test_positive_inf(self):
        assert is_finite(float("inf")) is False

    def test_negative_inf(self):
        assert is_finite(float("-inf")) is False

    def test_string_returns_false(self):
        assert is_finite("hello") is False

    def test_none_returns_false(self):
        assert is_finite(None) is False


# ── assert_close ─────────────────────────────────────────────────────────────


class TestAssertClose:
    """Tests for assert_close."""

    def test_exact_match(self):
        assert_close(1.0, 1.0)

    def test_within_default_rtol(self):
        assert_close(1.0, 1.0 + 1e-8)

    def test_outside_default_rtol_raises(self):
        with pytest.raises(AssertionError, match="Values not close"):
            assert_close(1.0, 1.1)

    def test_custom_atol(self):
        assert_close(1.0, 1.05, atol=0.1)

    def test_custom_rtol(self):
        assert_close(100.0, 101.0, rtol=0.02)

    def test_type_error_on_string(self):
        with pytest.raises(TypeError, match="actual must be a number"):
            assert_close("nope", 1.0)

    def test_negative_rtol_raises(self):
        with pytest.raises(ValueError, match="rtol must be non-negative"):
            assert_close(1.0, 1.0, rtol=-0.1)

    def test_negative_atol_raises(self):
        with pytest.raises(ValueError, match="atol must be non-negative"):
            assert_close(1.0, 1.0, atol=-0.1)

    def test_zero_expected(self):
        """When expected is 0, only atol matters."""
        assert_close(0.0, 0.0)
        with pytest.raises(AssertionError):
            assert_close(0.1, 0.0)  # default atol=0

    def test_integers(self):
        assert_close(3, 3)

    def test_diagnostic_message_contains_values(self):
        with pytest.raises(AssertionError, match=r"actual=2\.0.*expected=1\.0"):
            assert_close(2.0, 1.0)


# ── assert_conserved ─────────────────────────────────────────────────────────


class TestAssertConserved:
    """Tests for assert_conserved."""

    def test_identical_values(self):
        assert_conserved(100.0, 100.0, "energy")

    def test_within_tolerance(self):
        assert_conserved(1000.0, 1000.0005, "mass", rtol=1e-6)

    def test_violation_raises(self):
        with pytest.raises(AssertionError, match="energy not conserved"):
            assert_conserved(100.0, 110.0, "energy", rtol=1e-3)

    def test_both_zero(self):
        assert_conserved(0.0, 0.0, "nothing")

    def test_type_error(self):
        with pytest.raises(TypeError):
            assert_conserved("a", 1.0, "bad")

    def test_negative_rtol(self):
        with pytest.raises(ValueError, match="rtol must be non-negative"):
            assert_conserved(1.0, 1.0, "q", rtol=-1)


# ── assert_monotonic ─────────────────────────────────────────────────────────


class TestAssertMonotonic:
    """Tests for assert_monotonic."""

    def test_increasing(self):
        assert_monotonic([1, 2, 3, 4, 5])

    def test_decreasing(self):
        assert_monotonic([5, 4, 3, 2, 1], increasing=False)

    def test_non_strict_allows_equal(self):
        assert_monotonic([1, 2, 2, 3])

    def test_strict_rejects_equal(self):
        with pytest.raises(AssertionError, match="not strictly increasing"):
            assert_monotonic([1, 2, 2, 3], strict=True)

    def test_violation_reports_index(self):
        with pytest.raises(AssertionError, match=r"values\[2\]"):
            assert_monotonic([1, 2, 1, 4])

    def test_too_few_values(self):
        with pytest.raises(ValueError, match="at least 2"):
            assert_monotonic([1])

    def test_type_error(self):
        with pytest.raises(TypeError, match="values must be a sequence"):
            assert_monotonic(42)

    def test_non_number_element(self):
        with pytest.raises(TypeError, match="must be a number"):
            assert_monotonic([1, "two", 3])

    def test_decreasing_strict(self):
        assert_monotonic([5.0, 4.0, 3.0], increasing=False, strict=True)

    def test_label_in_message(self):
        with pytest.raises(AssertionError, match="temperature"):
            assert_monotonic([1, 0], label="temperature")


# ── assert_physics_state ─────────────────────────────────────────────────────


class TestAssertPhysicsState:
    """Tests for assert_physics_state."""

    def test_valid_3d_state(self):
        assert_physics_state([0, 1, 2], [3, 4, 5])

    def test_valid_with_acceleration(self):
        assert_physics_state([0, 0, 0], [1, 1, 1], [9.8, 0, 0])

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            assert_physics_state([1, 2, 3], [4, 5])

    def test_nan_in_position(self):
        with pytest.raises(ValueError, match="not finite"):
            assert_physics_state([float("nan"), 0, 0], [1, 2, 3])

    def test_inf_in_velocity(self):
        with pytest.raises(ValueError, match="not finite"):
            assert_physics_state([0, 0, 0], [float("inf"), 0, 0])

    def test_empty_position(self):
        with pytest.raises(ValueError, match="must not be empty"):
            assert_physics_state([], [1, 2, 3])

    def test_acceleration_length_mismatch(self):
        with pytest.raises(ValueError, match="same length as position"):
            assert_physics_state([1, 2], [3, 4], [5])

    def test_not_a_sequence(self):
        with pytest.raises(TypeError, match="must be a sequence"):
            assert_physics_state(42, [1, 2])


# ── assert_jacobian_symmetry ─────────────────────────────────────────────────


class TestAssertJacobianSymmetry:
    """Tests for assert_jacobian_symmetry."""

    def test_symmetric_matrix(self):
        J = [
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ]
        assert_jacobian_symmetry(J)

    def test_identity_matrix(self):
        J = [[1, 0], [0, 1]]
        assert_jacobian_symmetry(J)

    def test_1x1_matrix(self):
        assert_jacobian_symmetry([[42.0]])

    def test_asymmetric_raises(self):
        J = [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
        with pytest.raises(AssertionError, match="not symmetric"):
            assert_jacobian_symmetry(J)

    def test_non_square_raises(self):
        J = [
            [1, 2, 3],
            [4, 5],
        ]
        with pytest.raises(ValueError, match="must be square"):
            assert_jacobian_symmetry(J)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            assert_jacobian_symmetry([])

    def test_nearly_symmetric_within_tolerance(self):
        J = [
            [1.0, 2.0],
            [2.0 + 1e-8, 1.0],
        ]
        assert_jacobian_symmetry(J, rtol=1e-6)

    def test_not_a_sequence(self):
        with pytest.raises(TypeError):
            assert_jacobian_symmetry(42)
