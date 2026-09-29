"""Unit tests for the pure-numpy gradient math in fit_swing_pinocchio.

These tests exercise the polynomial-torque chain rule and the rotation
matrix -> quaternion helper without importing pinocchio. They run on
every CI lane and serve as the executable spec for the analytical
Jacobian's polynomial layer.

The full LM driver is exercised in
``tests/heavy_integration/test_pinocchio_fit_swing.py``.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# Lazy import so unit lane does not pull pinocchio (we only touch the
# pure-numpy helpers).
fit_swing_mod = importlib.import_module(
    "src.engines.physics_engines.pinocchio.python.motion_matching.fit_swing"
)
polynomial_basis = fit_swing_mod.polynomial_basis
polynomial_torque_chain_rule = fit_swing_mod.polynomial_torque_chain_rule
rotmat_to_quat_wxyz = fit_swing_mod.rotmat_to_quat_wxyz

simulate_mod = importlib.import_module(
    "src.engines.physics_engines.pinocchio.python.motion_matching.simulate"
)
COEFFS_PER_JOINT = simulate_mod.COEFFS_PER_JOINT
evaluate_polynomial_torque = simulate_mod.evaluate_polynomial_torque


pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# polynomial_basis
# --------------------------------------------------------------------------- #


class TestPolynomialBasis:
    """B[i, k] = t[i]**k contract."""

    def test_shape_and_values_at_t_one(self) -> None:
        B = polynomial_basis(np.array([1.0]))
        assert B.shape == (1, COEFFS_PER_JOINT)
        # 1**k == 1 for all k.
        np.testing.assert_array_equal(B[0], 1.0)

    def test_t_zero_only_constant(self) -> None:
        B = polynomial_basis(np.array([0.0]))
        # 0**0 conventionally 1; t**k for k>=1 is zero.
        np.testing.assert_array_equal(B[0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_powers_are_correct(self) -> None:
        ts = np.array([0.0, 0.5, 1.0, 2.0])
        B = polynomial_basis(ts)
        for i, t in enumerate(ts):
            for k in range(COEFFS_PER_JOINT):
                assert B[i, k] == pytest.approx(t**k)

    def test_rejects_nonfinite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            polynomial_basis(np.array([np.nan]))


# --------------------------------------------------------------------------- #
# polynomial_torque_chain_rule
# --------------------------------------------------------------------------- #


class TestPolynomialTorqueChainRule:
    """``∂tau / ∂theta`` is block-diagonal with [1, t, t^2, ..., t^6] blocks."""

    @pytest.mark.parametrize("n_joints", [1, 3, 7, 23])
    def test_shape(self, n_joints: int) -> None:
        J = polynomial_torque_chain_rule(0.5, n_joints)
        assert J.shape == (n_joints, n_joints * COEFFS_PER_JOINT)

    def test_block_diagonal_structure(self) -> None:
        n_joints = 4
        t = 0.7
        J = polynomial_torque_chain_rule(t, n_joints)
        for j in range(n_joints):
            for jp in range(n_joints):
                block = J[j, jp * COEFFS_PER_JOINT : (jp + 1) * COEFFS_PER_JOINT]
                if j == jp:
                    expected = np.array([t**k for k in range(COEFFS_PER_JOINT)])
                    np.testing.assert_allclose(block, expected)
                else:
                    np.testing.assert_array_equal(block, 0.0)

    def test_chain_rule_matches_finite_difference(self) -> None:
        """Numerical sanity: J @ delta_theta ≈ tau(theta+delta) - tau(theta)."""
        rng = np.random.default_rng(0)
        n_joints = 5
        t = 0.4
        nx = n_joints * COEFFS_PER_JOINT
        theta = rng.standard_normal(nx)
        delta = 1e-6 * rng.standard_normal(nx)

        J = polynomial_torque_chain_rule(t, n_joints)

        coeffs = theta.reshape(n_joints, COEFFS_PER_JOINT)
        coeffs_p = (theta + delta).reshape(n_joints, COEFFS_PER_JOINT)
        tau = evaluate_polynomial_torque(coeffs, t)
        tau_p = evaluate_polynomial_torque(coeffs_p, t)

        analytic = J @ delta
        numeric = tau_p - tau
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-9)

    def test_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            polynomial_torque_chain_rule(np.nan, 3)
        with pytest.raises(ValueError, match="positive int"):
            polynomial_torque_chain_rule(0.5, 0)
        with pytest.raises(ValueError, match="positive int"):
            polynomial_torque_chain_rule(0.5, -1)


# --------------------------------------------------------------------------- #
# rotmat_to_quat_wxyz
# --------------------------------------------------------------------------- #


class TestRotmatToQuatWxyz:
    """Round-trip: known rotations -> known quaternions, scalar-positive."""

    def test_identity_is_unit_w(self) -> None:
        R = np.eye(3)
        q = rotmat_to_quat_wxyz(R)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_rotation_about_x_pi_over_2(self) -> None:
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        R = np.array(
            [
                [1, 0, 0],
                [0, np.cos(np.pi / 2), -np.sin(np.pi / 2)],
                [0, np.sin(np.pi / 2), np.cos(np.pi / 2)],
            ]
        )
        q = rotmat_to_quat_wxyz(R)
        np.testing.assert_allclose(q, [c, s, 0.0, 0.0], atol=1e-9)

    def test_unit_norm_invariant(self) -> None:
        rng = np.random.default_rng(7)
        # Build a stack of random rotations from random axis-angles.
        n = 16
        R_stack = np.empty((n, 3, 3))
        for i in range(n):
            axis = rng.standard_normal(3)
            axis = axis / np.linalg.norm(axis)
            angle = rng.uniform(-np.pi, np.pi)
            K = np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
            )
            R_stack[i] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        Q = rotmat_to_quat_wxyz(R_stack)
        norms = np.linalg.norm(Q, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-9)
        # Scalar-positive convention:
        assert np.all(Q[:, 0] >= 0.0)

    def test_single_matrix_returns_1d(self) -> None:
        q = rotmat_to_quat_wxyz(np.eye(3))
        assert q.shape == (4,)

    def test_stacked_matrices_returns_2d(self) -> None:
        R = np.tile(np.eye(3), (5, 1, 1))
        Q = rotmat_to_quat_wxyz(R)
        assert Q.shape == (5, 4)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match=r"\(\.\.\., 3, 3\)"):
            rotmat_to_quat_wxyz(np.zeros((3, 3, 3, 3)))


# --------------------------------------------------------------------------- #
# FitOptions / FitResult invariants
# --------------------------------------------------------------------------- #


class TestFitOptionsInvariants:
    """Frozen dataclass + sane defaults."""

    def test_default_jac_mode_is_analytical(self) -> None:
        FitOptions = fit_swing_mod.FitOptions
        opts = FitOptions()
        assert opts.jac_mode == "analytical"

    def test_default_max_iter_under_spec_target(self) -> None:
        # Spec target: < 50 LM outer iterations on the recovery problem.
        FitOptions = fit_swing_mod.FitOptions
        opts = FitOptions()
        assert opts.max_iter <= 50

    def test_options_are_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        FitOptions = fit_swing_mod.FitOptions
        opts = FitOptions()
        with pytest.raises(FrozenInstanceError):
            opts.max_iter = 100  # type: ignore[misc]
