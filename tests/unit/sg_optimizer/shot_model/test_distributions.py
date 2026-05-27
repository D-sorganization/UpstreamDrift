"""Unit tests for TiltedBivariateGaussian."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.shot_model.distributions import (
    TiltedBivariateGaussian,
)


class TestInvariants:
    def test_rejects_zero_sigma_long(self):
        with pytest.raises(ContractViolationError):
            TiltedBivariateGaussian(sigma_long=0.0, sigma_lat=1.0)

    def test_rejects_negative_sigma_lat(self):
        with pytest.raises(ContractViolationError):
            TiltedBivariateGaussian(sigma_long=1.0, sigma_lat=-1.0)

    def test_rejects_unbounded_rho(self):
        with pytest.raises(ContractViolationError):
            TiltedBivariateGaussian(sigma_long=1.0, sigma_lat=1.0, rho=0.999)


class TestSampling:
    def test_sample_shape(self):
        d = TiltedBivariateGaussian(sigma_long=2.0, sigma_lat=1.0, rho=0.2)
        rng = np.random.default_rng(0)
        samples = d.sample(1000, rng)
        assert samples.shape == (1000, 2)

    def test_empirical_mean_close_to_bias(self):
        d = TiltedBivariateGaussian(
            sigma_long=2.0,
            sigma_lat=3.0,
            rho=0.3,
            bias_long=5.0,
            bias_lat=-2.0,
        )
        rng = np.random.default_rng(1)
        samples = d.sample(20_000, rng)
        assert np.allclose(samples.mean(axis=0), [5.0, -2.0], atol=0.1)

    def test_empirical_covariance_close_to_analytic(self):
        d = TiltedBivariateGaussian(sigma_long=4.0, sigma_lat=2.0, rho=0.4)
        rng = np.random.default_rng(2)
        samples = d.sample(50_000, rng)
        emp_cov = np.cov(samples.T)
        analytic = d.covariance_matrix()
        np.testing.assert_allclose(emp_cov, analytic, rtol=0.05, atol=0.1)


class TestGeometry:
    def test_tilt_angle_matches_eigendecomposition(self):
        d = TiltedBivariateGaussian(sigma_long=5.0, sigma_lat=2.0, rho=0.4)
        # Eigendecomposition principal axis angle.
        eigvals, eigvecs = np.linalg.eigh(d.covariance_matrix())
        idx = int(np.argmax(eigvals))
        principal = eigvecs[:, idx]
        analytic_angle = math.degrees(math.atan2(principal[1], principal[0]))
        assert math.isclose(
            d.tilt_angle_degrees() % 180,
            analytic_angle % 180,
            abs_tol=1e-6,
        )

    def test_confidence_ellipse_contains_expected_fraction(self):
        d = TiltedBivariateGaussian(sigma_long=2.0, sigma_lat=1.5, rho=0.3)
        rng = np.random.default_rng(3)
        samples = d.sample(40_000, rng)
        ell = d.confidence_ellipse(0.95)
        # Rotate samples into ellipse frame.
        ang = math.radians(ell["angle_deg"])
        c, s = math.cos(-ang), math.sin(-ang)
        x = samples[:, 0] - ell["cx"]
        y = samples[:, 1] - ell["cy"]
        xr = c * x - s * y
        yr = s * x + c * y
        inside = (xr / ell["a"]) ** 2 + (yr / ell["b"]) ** 2 <= 1.0
        frac = inside.mean()
        assert 0.93 < frac < 0.97


class TestTransformations:
    def test_scaled_multiplies_sigmas(self):
        d = TiltedBivariateGaussian(sigma_long=2.0, sigma_lat=3.0, rho=0.25)
        scaled = d.scaled(1.5, 2.0)
        assert math.isclose(scaled.sigma_long, 3.0)
        assert math.isclose(scaled.sigma_lat, 6.0)
        assert math.isclose(scaled.rho, 0.25)

    def test_shifted_adds_to_bias(self):
        d = TiltedBivariateGaussian(sigma_long=1.0, sigma_lat=1.0, bias_long=1.0)
        s = d.shifted(2.0, -3.0)
        assert math.isclose(s.bias_long, 3.0)
        assert math.isclose(s.bias_lat, -3.0)
