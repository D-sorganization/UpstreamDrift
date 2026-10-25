"""Tests for Benettin largest Lyapunov exponent check on canonical 3DOF models.

Acceptance criteria:
1. Harmonic oscillator asserts LLE ~ 0 (|lambda| < 0.05 by 30s).
2. Lorenz-63 asserts lambda ~ 0.9 +- 0.1.
3. Triple-pendulum check reports LLE vs horizon curve to a receipt.
   High-energy case is positive; low-energy case is decreasing.
4. DbC validation for invalid horizons, step sizes, and renormalization intervals.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pendulum_simulator.benettin_lyapunov import (
    BenettinReceipt,
    compute_benettin_lle,
    compute_canonical_triple_pendulum_lle,
)

pytestmark = pytest.mark.unit


def test_benettin_dbc_validates_inputs() -> None:
    """DbC checks fail closed on invalid parameters."""
    with pytest.raises(ValueError, match="horizon_s"):
        compute_benettin_lle(
            initial_state=np.array([1.0, 0.0]),
            horizon_s=-1.0,
            deriv_fn=lambda t, y: y,
        )

    with pytest.raises(ValueError, match="renormalization_interval_s"):
        compute_benettin_lle(
            initial_state=np.array([1.0, 0.0]),
            horizon_s=10.0,
            renormalization_interval_s=0.0,
            deriv_fn=lambda t, y: y,
        )

    with pytest.raises(ValueError, match="dt"):
        compute_benettin_lle(
            initial_state=np.array([1.0, 0.0]),
            horizon_s=10.0,
            dt=0.0,
            deriv_fn=lambda t, y: y,
        )

    with pytest.raises(ValueError, match="d0"):
        compute_benettin_lle(
            initial_state=np.array([1.0, 0.0]),
            horizon_s=10.0,
            d0=-1e-5,
            deriv_fn=lambda t, y: y,
        )

    with pytest.raises(ValueError, match="step_fn or deriv_fn"):
        compute_benettin_lle(
            initial_state=np.array([1.0, 0.0]),
            horizon_s=10.0,
        )


def test_harmonic_oscillator_lle_vanishes() -> None:
    """Harmonic oscillator is regular: LLE should be near zero (|lambda| < 0.05)."""

    def _harmonic_deriv(_t: float, y: np.ndarray) -> np.ndarray:
        q, p = y
        return np.array([p, -q], dtype=float)

    receipt = compute_benettin_lle(
        initial_state=np.array([1.0, 0.0]),
        horizon_s=30.0,
        renormalization_interval_s=0.1,
        dt=0.01,
        d0=1e-8,
        deriv_fn=_harmonic_deriv,
        system_name="harmonic_oscillator",
    )

    assert isinstance(receipt, BenettinReceipt)
    assert abs(receipt.largest_lyapunov_exponent) < 0.05
    assert len(receipt.convergence_history) > 0


def test_lorenz_chaotic_reference_lle() -> None:
    """Lorenz-63 chaotic benchmark (sigma=10, rho=28, beta=8/3) has lambda ~ 0.9 +- 0.1."""
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    def _lorenz_deriv(_t: float, y: np.ndarray) -> np.ndarray:
        x, y_coord, z = y
        return np.array(
            [
                sigma * (y_coord - x),
                x * (rho - z) - y_coord,
                x * y_coord - beta * z,
            ],
            dtype=float,
        )

    receipt = compute_benettin_lle(
        initial_state=np.array([1.0, 1.0, 1.0]),
        horizon_s=40.0,
        transient_horizon_s=10.0,
        renormalization_interval_s=0.05,
        dt=0.005,
        d0=1e-8,
        deriv_fn=_lorenz_deriv,
        system_name="lorenz_1963",
    )

    assert 0.8 <= receipt.largest_lyapunov_exponent <= 1.0
    assert receipt.largest_lyapunov_exponent == pytest.approx(0.9, abs=0.1)


def test_canonical_triple_pendulum_lle_fast_qualitative() -> None:
    """Fast check on canonical triple pendulum: high-energy > low-energy exponent."""
    high_receipt = compute_canonical_triple_pendulum_lle(
        initial_angles=(2.0, 2.0, 2.0),
        horizon_s=4.0,
        renormalization_interval_s=0.1,
        dt=0.002,
    )
    low_receipt = compute_canonical_triple_pendulum_lle(
        initial_angles=(0.2, 0.2, 0.2),
        horizon_s=4.0,
        renormalization_interval_s=0.1,
        dt=0.002,
    )

    assert (
        high_receipt.largest_lyapunov_exponent > low_receipt.largest_lyapunov_exponent
    )
    assert len(high_receipt.convergence_history) == 40
    assert len(low_receipt.convergence_history) == 40
    # Values are finite and qualitative numbers are captured
    assert np.isfinite(high_receipt.largest_lyapunov_exponent)
    assert np.isfinite(low_receipt.largest_lyapunov_exponent)


@pytest.mark.slow
def test_canonical_triple_pendulum_lle_long_horizon() -> None:
    """Long-horizon 30s check for SAP-7: high energy stays positive; low energy decreases."""
    high_receipt = compute_canonical_triple_pendulum_lle(
        initial_angles=(2.0, 2.0, 2.0),
        horizon_s=30.0,
        renormalization_interval_s=0.1,
        dt=0.002,
    )
    low_receipt = compute_canonical_triple_pendulum_lle(
        initial_angles=(0.2, 0.2, 0.2),
        horizon_s=30.0,
        renormalization_interval_s=0.1,
        dt=0.002,
    )

    # High energy settles positive (> 1.0 /s, typically ~2.6 /s)
    assert high_receipt.largest_lyapunov_exponent > 1.0
    # Low energy decreases to near-zero (< 0.5 /s, typically ~0.15 /s at 30s)
    assert low_receipt.largest_lyapunov_exponent < 0.5
    assert (
        high_receipt.largest_lyapunov_exponent
        > 3.0 * low_receipt.largest_lyapunov_exponent
    )
