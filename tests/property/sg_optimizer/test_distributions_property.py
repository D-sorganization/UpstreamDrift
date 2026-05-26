"""Hypothesis property tests for TiltedBivariateGaussian."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.shared.python.sg_optimizer.shot_model.distributions import (
    TiltedBivariateGaussian,
)


sigma = st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False)
rho = st.floats(min_value=-0.95, max_value=0.95, allow_nan=False, allow_infinity=False)
bias = st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False)


@given(sigma_long=sigma, sigma_lat=sigma, rho_=rho, bl=bias, bp=bias)
@settings(max_examples=40, deadline=None)
def test_covariance_is_positive_semidefinite(sigma_long, sigma_lat, rho_, bl, bp):
    d = TiltedBivariateGaussian(sigma_long, sigma_lat, rho_, bl, bp)
    eigvals = np.linalg.eigvalsh(d.covariance_matrix())
    assert eigvals.min() > -1e-9


@given(sigma_long=sigma, sigma_lat=sigma, rho_=rho)
@settings(max_examples=20, deadline=None)
def test_scaled_preserves_correlation(sigma_long, sigma_lat, rho_):
    d = TiltedBivariateGaussian(sigma_long, sigma_lat, rho_)
    s = d.scaled(2.5, 0.5)
    assert math.isclose(s.rho, d.rho)
    assert math.isclose(s.sigma_long, d.sigma_long * 2.5)
    assert math.isclose(s.sigma_lat, d.sigma_lat * 0.5)
