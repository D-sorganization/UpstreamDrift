"""Coverage tests for ``motion_matching.validators``.

Pin every public-API branch in ``validators.py`` so refactors can't
silently relax the DbC contracts. Each test docstring identifies the
specific behaviour or error message it pins.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.motion_matching.validators import (
    REGULARIZER_KINDS,
    must_be_finite_vector,
    must_be_monotonic_time,
    must_be_regularizer_kind,
    must_be_unit_quaternion_rows,
    must_have_fields,
)


class TestMustHaveFields:
    """Pin: ``must_have_fields`` accepts dict, attribute object, dataclass."""

    def test_dict_with_all_fields_passes(self) -> None:
        """Pin: dict containing every requested key returns None silently."""
        must_have_fields({"a": 1, "b": 2}, ["a", "b"])

    def test_dict_missing_field_raises(self) -> None:
        """Pin: missing field message lists the missing names."""
        with pytest.raises(ValueError, match="missing required fields: c"):
            must_have_fields({"a": 1}, ["a", "c"])

    def test_object_with_attrs_passes(self) -> None:
        """Pin: arbitrary object with attributes is accepted."""

        class Bag:
            x = 1
            y = 2

        must_have_fields(Bag(), ["x", "y"])

    def test_object_dict_fallback(self) -> None:
        """Pin: empty hasattr probe falls back to ``__dict__`` keys."""

        class Bag:
            pass

        b = Bag()
        b.alpha = 1
        # Request a name we know is absent so ``present`` starts empty and
        # we exercise the ``__dict__`` fallback branch.
        with pytest.raises(ValueError, match="missing required fields"):
            must_have_fields(b, ["beta"])


class TestMustBeFiniteVector:
    """Pin: ``must_be_finite_vector`` enforces real-finite-1-D contract."""

    def test_returns_float64(self) -> None:
        """Pin: integer inputs are coerced to ``float64`` on success."""
        out = must_be_finite_vector([1, 2, 3])
        assert out.dtype == np.float64
        assert out.shape == (3,)

    def test_none_rejected(self) -> None:
        """Pin: ``None`` is rejected with a numeric-vector message."""
        with pytest.raises(ValueError, match="real numeric vector"):
            must_be_finite_vector(None)

    def test_complex_rejected(self) -> None:
        """Pin: complex dtype is rejected by the ``iscomplexobj`` branch."""
        with pytest.raises(ValueError, match="real numeric array"):
            must_be_finite_vector(np.array([1 + 0j, 2 + 0j]))

    def test_non_numeric_rejected(self) -> None:
        """Pin: object/string dtype is rejected."""
        with pytest.raises(ValueError, match="real numeric array"):
            must_be_finite_vector(np.array(["a", "b"]))

    def test_empty_rejected(self) -> None:
        """Pin: zero-length vectors are rejected (``size == 0``)."""
        with pytest.raises(ValueError, match="non-empty 1-D vector"):
            must_be_finite_vector(np.array([], dtype=float))

    def test_2d_rejected(self) -> None:
        """Pin: 2-D arrays fail the ``ndim != 1`` branch."""
        with pytest.raises(ValueError, match="non-empty 1-D vector"):
            must_be_finite_vector(np.zeros((2, 3)))

    def test_nan_rejected(self) -> None:
        """Pin: NaN entries are caught by the ``isfinite`` check."""
        with pytest.raises(ValueError, match="finite entries"):
            must_be_finite_vector(np.array([1.0, np.nan]))

    def test_inf_rejected(self) -> None:
        """Pin: ``+inf`` is rejected by the finiteness check."""
        with pytest.raises(ValueError, match="finite entries"):
            must_be_finite_vector(np.array([1.0, np.inf]))


class TestMustBeMonotonicTime:
    """Pin: ``must_be_monotonic_time`` enforces strict increase."""

    def test_strict_increase_passes(self) -> None:
        """Pin: strictly increasing input passes through unchanged."""
        out = must_be_monotonic_time([0.0, 0.1, 0.2])
        assert np.allclose(out, [0.0, 0.1, 0.2])

    def test_duplicate_rejected(self) -> None:
        """Pin: duplicate samples are rejected (``diff == 0``)."""
        with pytest.raises(ValueError, match="strictly increasing"):
            must_be_monotonic_time([0.0, 0.1, 0.1, 0.2])

    def test_decrease_rejected(self) -> None:
        """Pin: decreasing samples are rejected (``diff < 0``)."""
        with pytest.raises(ValueError, match="strictly increasing"):
            must_be_monotonic_time([0.0, 0.2, 0.1])


class TestMustBeUnitQuaternionRows:
    """Pin: ``must_be_unit_quaternion_rows`` shape + norm contract."""

    def test_unit_quat_passes(self) -> None:
        """Pin: ``[w,x,y,z]`` rows of unit norm pass."""
        q = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
        out = must_be_unit_quaternion_rows(q)
        assert out.dtype == np.float64
        assert out.shape == (3, 4)

    def test_non_positive_tol_rejected(self) -> None:
        """Pin: ``tol <= 0`` is rejected up-front."""
        with pytest.raises(ValueError, match="tol must be > 0"):
            must_be_unit_quaternion_rows(np.eye(4), tol=0.0)

    def test_wrong_shape_rejected(self) -> None:
        """Pin: shape ``(N, 4)`` is required."""
        with pytest.raises(ValueError, match=r"\(N, 4\) real matrix"):
            must_be_unit_quaternion_rows(np.zeros((3, 3)))

    def test_complex_rejected(self) -> None:
        """Pin: complex dtype quaternions are rejected."""
        q = np.tile([1.0 + 0j, 0, 0, 0], (2, 1))
        with pytest.raises(ValueError, match=r"\(N, 4\) real matrix"):
            must_be_unit_quaternion_rows(q)

    def test_nan_rejected(self) -> None:
        """Pin: NaN entries fail the finiteness check."""
        q = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
        q[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            must_be_unit_quaternion_rows(q)

    def test_non_unit_rejected(self) -> None:
        """Pin: non-unit-norm rows are rejected with worst-deviation msg."""
        q = np.tile([2.0, 0.0, 0.0, 0.0], (2, 1))
        with pytest.raises(ValueError, match="unit-norm"):
            must_be_unit_quaternion_rows(q)


class TestMustBeRegularizerKind:
    """Pin: ``must_be_regularizer_kind`` whitelist + lowercase contract."""

    def test_valid_lowercase(self) -> None:
        """Pin: every entry in ``REGULARIZER_KINDS`` round-trips."""
        for name in REGULARIZER_KINDS:
            assert must_be_regularizer_kind(name) == name

    def test_uppercase_normalised(self) -> None:
        """Pin: input is lower-cased before whitelist check."""
        assert must_be_regularizer_kind("TOTAL_WORK") == "total_work"

    def test_non_string_rejected(self) -> None:
        """Pin: non-string inputs raise a non-empty-string message."""
        with pytest.raises(ValueError, match="non-empty string"):
            must_be_regularizer_kind(42)

    def test_empty_string_rejected(self) -> None:
        """Pin: empty string is rejected."""
        with pytest.raises(ValueError, match="non-empty string"):
            must_be_regularizer_kind("")

    def test_unknown_kind_rejected(self) -> None:
        """Pin: unknown regularizer name is rejected with allowed list."""
        with pytest.raises(ValueError, match="not one of"):
            must_be_regularizer_kind("not_a_thing")
