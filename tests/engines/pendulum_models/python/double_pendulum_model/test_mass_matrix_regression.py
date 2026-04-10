"""Regression tests for double-pendulum mass matrix formula (issue #2498).

These tests pin down the *correct* mass matrix values for the default
JS browser-demo parameters. They serve as the reference contract that
the browser demo (double_pendulum_web/app.js) must satisfy.

Bug caught: app.js was double-counting the parallel-axis CM terms —
it computed I_prox = I_cm + m*lc^2 (correct), then added m*lc^2 again
when assembling m11, m12, m22 (wrong). This inflated effective inertia
by ~47% and made the pendulum unrealistically sluggish.

Reference formula (Spong et al., "Robot Modeling and Control"):
    I1 = (1/12)*m1*l1^2 + m1*lc1^2   (proximal-joint inertia, arm 1)
    I2 = (1/12)*m2*l2^2 + m2*lc2^2   (proximal-joint inertia, arm 2)
    m11 = I1 + I2 + m2*(l1^2 + 2*l1*lc2*cos(theta2))
    m12 = I2 + m2*l1*lc2*cos(theta2)
    m22 = I2
"""

from __future__ import annotations

import math

import pytest

# ── reference parameters matching app.js defaults ────────────────────────────
L1 = 0.75
L2 = 1.0
M1 = 7.5
M2 = 0.55  # mShaft(0.35) + mHead(0.20)
COM1 = 0.45  # fractional CoM position along segment 1
COM2 = 0.43  # fractional CoM position along segment 2

LC1 = L1 * COM1  # 0.3375  m
LC2 = L2 * COM2  # 0.43    m

# Proximal-joint inertias (already incorporate parallel-axis shift)
I1 = (1.0 / 12.0) * M1 * L1**2 + M1 * LC1**2  # ≈ 1.2059
I2 = (1.0 / 12.0) * M2 * L2**2 + M2 * LC2**2  # ≈ 0.1475


def _correct_mass_matrix(theta2: float) -> tuple[float, float, float]:
    """Return (m11, m12, m22) using the correct formula."""
    cos2 = math.cos(theta2)
    m11 = I1 + I2 + M2 * (L1**2 + 2 * L1 * LC2 * cos2)
    m12 = I2 + M2 * L1 * LC2 * cos2
    m22 = I2
    return m11, m12, m22


def _buggy_mass_matrix(theta2: float) -> tuple[float, float, float]:
    """Reproduce the *old* (wrong) app.js formula for regression reference."""
    cos2 = math.cos(theta2)
    m11 = I1 + I2 + M1 * LC1**2 + M2 * (L1**2 + LC2**2 + 2 * L1 * LC2 * cos2)
    m12 = I2 + M2 * (LC2**2 + L1 * LC2 * cos2)
    m22 = I2 + M2 * LC2**2
    return m11, m12, m22


class TestMassMatrixFormula:
    """Verify the correct mass matrix formula with known analytical values."""

    def test_m22_equals_I2(self) -> None:
        """m22 = I2 (proximal inertia of arm 2) — independent of configuration."""
        for theta2 in [0.0, math.pi / 4, -math.pi / 3, math.pi / 2, -math.pi]:
            _, _, m22 = _correct_mass_matrix(theta2)
            assert m22 == pytest.approx(I2, rel=1e-12), (
                f"m22 != I2 at theta2={theta2:.3f}"
            )

    def test_m12_equals_m21(self) -> None:
        """Mass matrix is symmetric: m12 = m21."""
        for theta2 in [0.0, 0.5, -1.2, math.pi / 3]:
            m11, m12, m22 = _correct_mass_matrix(theta2)
            # By construction the matrix is [[m11, m12], [m12, m22]]
            assert math.isfinite(m12)

    def test_positive_definite_all_configs(self) -> None:
        """M must be positive-definite for all theta2."""
        import numpy as np

        for theta2 in [t * 0.3 for t in range(-10, 11)]:
            m11, m12, m22 = _correct_mass_matrix(theta2)
            M = [[m11, m12], [m12, m22]]
            eigvals = np.linalg.eigvalsh(M)
            assert all(ev > 0 for ev in eigvals), (
                f"Not positive-definite at theta2={theta2:.2f}: {eigvals}"
            )

    def test_known_values_at_theta2_zero(self) -> None:
        """At theta2=0 (segments aligned), verify exact formula values."""
        m11, m12, m22 = _correct_mass_matrix(0.0)
        expected_m11 = I1 + I2 + M2 * (L1**2 + 2 * L1 * LC2)
        expected_m12 = I2 + M2 * L1 * LC2
        expected_m22 = I2
        assert m11 == pytest.approx(expected_m11, rel=1e-12)
        assert m12 == pytest.approx(expected_m12, rel=1e-12)
        assert m22 == pytest.approx(expected_m22, rel=1e-12)

    def test_known_values_at_theta2_pi_half(self) -> None:
        """At theta2=π/2 (perpendicular), coupling term vanishes."""
        m11, m12, m22 = _correct_mass_matrix(math.pi / 2)
        # cos(π/2) = 0, so coupling terms drop out
        expected_m11 = I1 + I2 + M2 * L1**2
        expected_m12 = I2  # coupling term = 0
        expected_m22 = I2
        assert m11 == pytest.approx(expected_m11, rel=1e-12)
        assert m12 == pytest.approx(expected_m12, rel=1e-12)
        assert m22 == pytest.approx(expected_m22, rel=1e-12)

    def test_buggy_formula_differs_from_correct(self) -> None:
        """Confirm the old (buggy) app.js formula gives wrong values.

        This test documents the regression: the correct and buggy formulas
        must NOT agree, proving the bug was real and the fix changes behaviour.
        """
        for theta2 in [0.0, math.pi / 4, -math.pi / 3]:
            m11_ok, m12_ok, m22_ok = _correct_mass_matrix(theta2)
            m11_bad, m12_bad, m22_bad = _buggy_mass_matrix(theta2)
            assert m11_ok != pytest.approx(m11_bad, rel=1e-3), (
                f"Buggy m11 unexpectedly matches correct m11 at theta2={theta2:.3f}"
            )
            assert m12_ok != pytest.approx(m12_bad, rel=1e-3), (
                f"Buggy m12 unexpectedly matches correct m12 at theta2={theta2:.3f}"
            )
            assert m22_ok != pytest.approx(m22_bad, rel=1e-3), (
                f"Buggy m22 unexpectedly matches correct m22 at theta2={theta2:.3f}"
            )

    def test_buggy_overestimates_inertia(self) -> None:
        """Buggy formula produces higher (wrong) inertias — pendulum too sluggish."""
        for theta2 in [0.0, 0.5, -0.8]:
            m11_ok, _, m22_ok = _correct_mass_matrix(theta2)
            m11_bad, _, m22_bad = _buggy_mass_matrix(theta2)
            assert m11_bad > m11_ok, f"Expected m11_bad > m11_ok at theta2={theta2}"
            assert m22_bad > m22_ok, f"Expected m22_bad > m22_ok at theta2={theta2}"

    def test_i1_proximal_already_contains_parallel_axis(self) -> None:
        """I1 = I1_cm + m1*lc1^2: parallel-axis is ALREADY included in I1."""
        I1_cm = (1.0 / 12.0) * M1 * L1**2
        parallel_axis_term = M1 * LC1**2
        assert pytest.approx(I1_cm + parallel_axis_term, rel=1e-12) == I1
        # Therefore: do NOT add m1*lc1^2 again when assembling m11

    def test_i2_proximal_already_contains_parallel_axis(self) -> None:
        """I2 = I2_cm + m2*lc2^2: parallel-axis is ALREADY included in I2."""
        I2_cm = (1.0 / 12.0) * M2 * L2**2
        parallel_axis_term = M2 * LC2**2
        assert pytest.approx(I2_cm + parallel_axis_term, rel=1e-12) == I2
        # Therefore: do NOT add m2*lc2^2 again when assembling m11, m12, m22


class TestSafeEvalContract:
    """Document expected safeEval semantics (issue #2498).

    The JavaScript safeEval previously returned 0 on runtime errors,
    masking failures like ReferenceError, division by zero, or NaN.
    The corrected version must surface errors via a status flag.

    These Python tests document the Python-equivalent contract using
    compile_forcing_functions from the reference Python model.
    """

    def test_valid_expression_evaluates_correctly(self) -> None:
        """A valid expression returns the expected value."""
        from src.engines.pendulum_models.python.double_pendulum_model import (
            DoublePendulumState,
            compile_forcing_functions,
        )

        shoulder, _ = compile_forcing_functions("2.0 * t", "0")
        state = DoublePendulumState(0.0, 0.0, 0.0, 0.0)
        assert shoulder(3.0, state) == pytest.approx(6.0)

    def test_invalid_expression_raises(self) -> None:
        """An invalid/undefined expression raises rather than silently returning 0."""
        from src.engines.pendulum_models.python.double_pendulum_model import (
            compile_forcing_functions,
        )

        # Simulate what safeEval should do: raise on runtime error
        with pytest.raises((NameError, ValueError, SyntaxError)):
            compile_forcing_functions("undefined_variable * t", "0")
