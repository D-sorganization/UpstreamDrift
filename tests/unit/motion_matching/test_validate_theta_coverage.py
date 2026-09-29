"""Coverage tests for ``motion_matching.validate_theta``."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.motion_matching.validate_theta import (
    COEFFS_PER_JOINT,
    DEFAULT_THETA_BOUND_TABLE,
    validate_theta,
)


def _good(n_joints: int = 3) -> np.ndarray:
    """Return a within-bounds 1-D theta of the right length."""
    return np.zeros(n_joints * COEFFS_PER_JOINT, dtype=np.float64)


class TestValidateTheta:
    """Pin: every public branch in ``validate_theta``."""

    def test_returns_contiguous_float64(self) -> None:
        """Pin: success returns a flat float64 ndarray."""
        out = validate_theta(_good(2), n_joints=2)
        assert out.dtype == np.float64
        assert out.flags["C_CONTIGUOUS"]
        assert out.shape == (14,)

    def test_2d_njoints7_reshaped_flat(self) -> None:
        """Pin: a ``(n_joints, 7)`` matrix is auto-reshaped to flat."""
        m = np.zeros((4, 7))
        out = validate_theta(m, n_joints=4)
        assert out.shape == (28,)

    def test_n_joints_must_be_positive_int(self) -> None:
        """Pin: non-positive or non-int ``n_joints`` is rejected."""
        with pytest.raises(ValueError, match="positive int"):
            validate_theta(_good(1), n_joints=0)
        with pytest.raises(ValueError, match="positive int"):
            validate_theta(_good(1), n_joints="3")  # type: ignore[arg-type]

    def test_array_like_coercion_failure(self) -> None:
        """Pin: un-coercible inputs raise ``TypeError`` from coercion branch."""
        with pytest.raises(TypeError, match="array-like of floats"):
            validate_theta("hello", n_joints=1)

    def test_3d_rejected(self) -> None:
        """Pin: 3-D arrays fail the ``ndim != 1`` branch."""
        with pytest.raises(ValueError, match=r"1-D \(n_joints\*7,\)"):
            validate_theta(np.zeros((1, 1, 7)), n_joints=1)

    def test_wrong_length_rejected(self) -> None:
        """Pin: length mismatch error names both expected and observed."""
        with pytest.raises(ValueError, match="length 6 != n_joints"):
            validate_theta(np.zeros(6), n_joints=1)

    def test_nan_inf_count_in_message(self) -> None:
        """Pin: non-finite count is reported in error message."""
        bad = _good(1).copy()
        bad[0] = np.nan
        bad[1] = np.inf
        with pytest.raises(ValueError, match="NaN=1, Inf=1"):
            validate_theta(bad, n_joints=1)

    def test_bounds_pass(self) -> None:
        """Pin: a within-bounds theta passes optional bounds check."""
        validate_theta(_good(2), n_joints=2, bounds=DEFAULT_THETA_BOUND_TABLE)

    def test_bounds_violation_reports_letter(self) -> None:
        """Pin: bound violations cite the offending letter and column."""
        bad = _good(2).copy()
        # Letter A is column 0; default bound 1000.
        bad[0] = 5000.0
        with pytest.raises(ValueError, match="coefficient 'A' .column 0."):
            validate_theta(bad, n_joints=2, bounds=DEFAULT_THETA_BOUND_TABLE)

    def test_bounds_must_be_mapping(self) -> None:
        """Pin: a non-mapping ``bounds`` argument is rejected."""
        with pytest.raises(TypeError, match="must be a mapping"):
            validate_theta(_good(1), n_joints=1, bounds=[("A", -1.0, 1.0)])  # type: ignore[arg-type]

    def test_bounds_letter_must_be_single_char(self) -> None:
        """Pin: multi-char keys are rejected."""
        with pytest.raises(ValueError, match="single-character letters"):
            validate_theta(_good(1), n_joints=1, bounds={"AB": (-1.0, 1.0)})

    def test_bounds_pair_must_be_tuple_of_two_floats(self) -> None:
        """Pin: malformed pair raises TypeError."""
        with pytest.raises(TypeError, match=r"\(lo, hi\) tuple"):
            validate_theta(_good(1), n_joints=1, bounds={"A": (1.0,)})  # type: ignore[dict-item]

    def test_bounds_lo_gt_hi_rejected(self) -> None:
        """Pin: lo > hi is caught at coercion time."""
        with pytest.raises(ValueError, match="lo > hi"):
            validate_theta(_good(1), n_joints=1, bounds={"A": (1.0, -1.0)})

    def test_bounds_letter_outside_a_g(self) -> None:
        """Pin: a letter mapping to column >= 7 is rejected at enforce time."""
        # 'H' maps to column 7, which is out of range.
        with pytest.raises(ValueError, match="not in A..G"):
            validate_theta(_good(1), n_joints=1, bounds={"H": (-1.0, 1.0)})

    def test_custom_name_in_error(self) -> None:
        """Pin: ``name=`` argument appears in length-mismatch messages."""
        with pytest.raises(ValueError, match="theta_optimal length"):
            validate_theta(np.zeros(6), n_joints=1, name="theta_optimal")
