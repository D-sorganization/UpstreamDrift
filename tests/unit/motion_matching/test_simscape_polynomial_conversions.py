"""Unit tests for Simscape polynomial conversions (#9921 / #9964)."""

import numpy as np
import pytest

from src.shared.python.motion_matching.prefix_fit import (
    COEFFS_PER_JOINT,
    bernstein_to_simscape,
    reexpress_bernstein_basis,
    simscape_to_bernstein,
)

pytestmark = pytest.mark.unit


def test_simscape_to_bernstein_roundtrip() -> None:
    """Converting from Simscape descending to Bernstein and back must match to machine precision."""
    rng = np.random.default_rng(42)
    n_joints = 27
    coeffs = rng.uniform(-100.0, 100.0, size=(n_joints, COEFFS_PER_JOINT))

    duration_s = 1.813889
    bernstein = simscape_to_bernstein(coeffs, duration_s=duration_s)
    coeffs_back = bernstein_to_simscape(bernstein, duration_s=duration_s)

    np.testing.assert_allclose(coeffs_back, coeffs, atol=1e-11, rtol=1e-11)

    # Test numerical evaluation at test points t in [0, duration_s]
    t_test = np.linspace(0.0, duration_s, 20)
    for t in t_test:
        val_orig = np.polyval(coeffs.T, t)
        val_back = np.polyval(coeffs_back.T, t)
        np.testing.assert_allclose(val_back, val_orig, atol=1e-11, rtol=1e-11)


def test_simscape_to_bernstein_reexpress_basis() -> None:
    """Re-expressing across different durations must preserve physical torque exactly."""
    rng = np.random.default_rng(123)
    coeffs_080 = rng.uniform(-50.0, 50.0, size=(5, COEFFS_PER_JOINT))

    b_080 = simscape_to_bernstein(coeffs_080, duration_s=0.80)
    b_basis = reexpress_bernstein_basis(
        b_080, source_duration_s=0.80, target_duration_s=1.813889
    )
    coeffs_basis = bernstein_to_simscape(b_basis, duration_s=1.813889)

    t_eval = np.linspace(0.0, 0.80, 25)
    for t in t_eval:
        val_orig = np.polyval(coeffs_080.T, t)
        val_re = np.polyval(coeffs_basis.T, t)
        np.testing.assert_allclose(val_re, val_orig, atol=1e-10, rtol=1e-10)


def test_simscape_to_bernstein_validation() -> None:
    """Test defensive checks on inputs."""
    with pytest.raises(ValueError, match="duration_s must be strictly positive"):
        simscape_to_bernstein(np.ones((2, 7)), duration_s=-0.5)

    with pytest.raises(ValueError, match="shape"):
        simscape_to_bernstein(np.ones((2, 5)), duration_s=1.0)
