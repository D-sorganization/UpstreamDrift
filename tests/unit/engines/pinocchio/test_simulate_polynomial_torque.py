"""Unit tests for the polynomial-torque controller (issue #4118).

These tests exercise the canonical polynomial-torque math without
importing ``pinocchio``. They run on every Python install and serve as
the executable spec for the
``tau_j(t) = sum_{k=0}^{6} a_{j,k} * t^k`` contract.

The forward-simulation tests that depend on the ABA + URDF live in
``tests/heavy_integration/test_pinocchio_simulate.py`` and are gated on
``pytest.mark.requires_pinocchio``.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

simulate_mod = importlib.import_module(
    "src.engines.physics_engines.pinocchio.python.motion_matching.simulate"
)
evaluate_polynomial_torque = simulate_mod.evaluate_polynomial_torque
COEFFS_PER_JOINT = simulate_mod.COEFFS_PER_JOINT
POLY_DEGREE = simulate_mod.POLY_DEGREE


pytestmark = pytest.mark.unit


class TestPolynomialTorqueContract:
    """tau_j(t) = sum_{k=0}^{6} a_{j,k} * t^k."""

    def test_canonical_constants_are_seven_and_six(self) -> None:
        assert COEFFS_PER_JOINT == 7
        assert POLY_DEGREE == 6

    def test_zero_coefficients_yield_zero_torque(self) -> None:
        coeffs = np.zeros((10, COEFFS_PER_JOINT))
        out = evaluate_polynomial_torque(coeffs, 0.5)
        assert out.shape == (10,)
        np.testing.assert_array_equal(out, 0.0)

    def test_constant_term_only(self) -> None:
        # tau_j(t) = a_{j,0}; evaluation is independent of t.
        coeffs = np.zeros((4, COEFFS_PER_JOINT))
        coeffs[:, 0] = np.array([1.0, -2.0, 3.5, 0.0])
        for t in (0.0, 0.1, 1.7):
            out = evaluate_polynomial_torque(coeffs, t)
            np.testing.assert_allclose(out, [1.0, -2.0, 3.5, 0.0])

    def test_linear_term_only(self) -> None:
        coeffs = np.zeros((3, COEFFS_PER_JOINT))
        coeffs[:, 1] = np.array([2.0, 0.0, -1.0])
        out = evaluate_polynomial_torque(coeffs, 0.5)
        np.testing.assert_allclose(out, [1.0, 0.0, -0.5])

    def test_high_degree_term_only(self) -> None:
        coeffs = np.zeros((2, COEFFS_PER_JOINT))
        coeffs[:, POLY_DEGREE] = np.array([1.0, -1.0])
        # tau_j(t) = a_{j,6} * t^6; at t=2 -> 64*a_{j,6}
        out = evaluate_polynomial_torque(coeffs, 2.0)
        np.testing.assert_allclose(out, [64.0, -64.0])

    def test_matches_manual_horner(self) -> None:
        rng = np.random.default_rng(seed=12345)
        coeffs = rng.standard_normal((5, COEFFS_PER_JOINT))
        for t in (-0.3, 0.0, 0.05, 0.5, 1.0, 2.7):
            expected = np.zeros(5)
            for k in range(COEFFS_PER_JOINT):
                expected += coeffs[:, k] * (t**k)
            actual = evaluate_polynomial_torque(coeffs, t)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_matches_numpy_polyval_per_joint(self) -> None:
        # numpy.polyval expects highest-degree-first; our coeffs[:, k] is
        # the coefficient on t^k (lowest-first). Confirm via per-joint
        # reverse + polyval.
        rng = np.random.default_rng(seed=7)
        coeffs = rng.standard_normal((3, COEFFS_PER_JOINT))
        t = 0.42
        actual = evaluate_polynomial_torque(coeffs, t)
        for j in range(3):
            expected_j = np.polyval(coeffs[j, ::-1], t)
            assert actual[j] == pytest.approx(expected_j, rel=1e-12)


class TestPolynomialTorquePreconditions:
    """DbC: bad inputs raise ValueError with informative messages."""

    def test_rejects_1d_coeffs(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            evaluate_polynomial_torque(np.zeros(7), 0.5)

    def test_rejects_wrong_column_count(self) -> None:
        with pytest.raises(ValueError, match="columns"):
            evaluate_polynomial_torque(np.zeros((3, 5)), 0.5)

    def test_rejects_nonfinite_t(self) -> None:
        coeffs = np.zeros((1, COEFFS_PER_JOINT))
        with pytest.raises(ValueError, match="finite"):
            evaluate_polynomial_torque(coeffs, float("nan"))
        with pytest.raises(ValueError, match="finite"):
            evaluate_polynomial_torque(coeffs, float("inf"))


class TestSimOptionsContract:
    """SimOptions enforces the parity-spec invariants."""

    def test_defaults_are_valid(self) -> None:
        opts = simulate_mod.SimOptions()
        assert opts.t_final == pytest.approx(1.0)
        assert opts.dt == pytest.approx(1e-3)
        assert opts.integrator == "rk4"
        assert opts.compute_energy is True

    def test_rejects_nonpositive_t_final(self) -> None:
        with pytest.raises(ValueError, match="t_final"):
            simulate_mod.SimOptions(t_final=0.0)
        with pytest.raises(ValueError, match="t_final"):
            simulate_mod.SimOptions(t_final=-0.5)

    def test_rejects_dt_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            simulate_mod.SimOptions(t_final=1.0, dt=0.0)
        with pytest.raises(ValueError, match="dt"):
            simulate_mod.SimOptions(t_final=1.0, dt=2.0)

    def test_rejects_unknown_integrator(self) -> None:
        with pytest.raises(ValueError, match="integrator"):
            simulate_mod.SimOptions(integrator="euler")  # type: ignore[arg-type]

    def test_rejects_bad_gravity_shape(self) -> None:
        with pytest.raises(ValueError, match="gravity"):
            simulate_mod.SimOptions(gravity=np.zeros(2))
