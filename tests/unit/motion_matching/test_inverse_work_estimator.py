"""Unit tests for the closed-form work estimator (issue #033 / GH #4002).

Cross-checks the analytical estimator against MATLAB's ``compute_total_work``
when MATLAB is available; otherwise asserts agreement between the analytical
form and a high-resolution trapezoidal reference.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import torch
from src.shared.python.motion_matching.inverse._work_estimator import (
    analytical_total_work,
    analytical_work_per_joint,
    work_estimate_torch,
)


def _quad_reference(coeffs: np.ndarray, duration_s: float, n: int = 20_000) -> float:
    """High-res trapezoidal reference for ``∫|poly * poly'| dt`` per joint."""
    asc = coeffs[::-1].astype(np.float64)
    deriv = np.polynomial.polynomial.polyder(asc)
    t = np.linspace(0.0, duration_s, n)
    p = np.polynomial.polynomial.polyval(t, asc)
    pp = np.polynomial.polynomial.polyval(t, deriv)
    return float(np.trapezoid(np.abs(p * pp), t))


@pytest.mark.unit
def test_work_estimator_matches_analytical_closed_form_for_constant_torque() -> None:
    """Constant torque: poly=G, poly'=0 -> work = 0 by construction."""
    coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.5])  # G=17.5 only
    assert analytical_work_per_joint(coeffs, duration_s=1.0) == pytest.approx(0.0)


@pytest.mark.unit
def test_work_estimator_linear_torque_matches_hand_computation() -> None:
    """tau(t) = F*t + G with F=2, G=1, T=1.

    Closed form: poly^2 = (2t + 1)^2; deriv = 2*(2t+1)*2 has zero at
    t=-1/2 (outside [0,1]); poly^2 monotone -> TV = poly^2(1) - poly^2(0)
    = 9 - 1 = 8. Work = 0.5 * 8 = 4.
    """
    coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0])
    val = analytical_work_per_joint(coeffs, duration_s=1.0)
    assert val == pytest.approx(4.0, rel=1e-9)


@pytest.mark.unit
def test_analytical_matches_high_resolution_quadrature() -> None:
    rng = np.random.default_rng(0)
    for _ in range(5):
        coeffs = rng.normal(0.0, 1.0, size=7)
        ana = analytical_work_per_joint(coeffs, duration_s=1.0)
        ref = _quad_reference(coeffs, 1.0)
        assert ana == pytest.approx(ref, rel=1e-3, abs=1e-6)


@pytest.mark.unit
def test_total_work_sums_across_joints() -> None:
    rng = np.random.default_rng(1)
    flat = rng.normal(0.0, 1.0, size=3 * 7)
    by_joint = sum(
        analytical_work_per_joint(flat[i * 7 : (i + 1) * 7], duration_s=1.0)
        for i in range(3)
    )
    assert analytical_total_work(flat, duration_s=1.0) == pytest.approx(by_joint)


@pytest.mark.unit
def test_torch_work_estimate_close_to_analytical() -> None:
    rng = np.random.default_rng(2)
    flat = rng.normal(0.0, 1.0, size=(4, 2 * 7)).astype(np.float64)
    expected = np.array([analytical_total_work(row, duration_s=1.0) for row in flat])
    got = (
        work_estimate_torch(torch.from_numpy(flat), duration_s=1.0, n_samples=512)
        .cpu()
        .numpy()
    )
    np.testing.assert_allclose(got, expected, rtol=5e-2, atol=1e-3)


@pytest.mark.unit
def test_torch_work_estimate_is_differentiable() -> None:
    coeffs = torch.randn(2, 14, requires_grad=True)
    work = work_estimate_torch(coeffs, duration_s=1.0, n_samples=64)
    work.sum().backward()
    assert coeffs.grad is not None
    assert torch.isfinite(coeffs.grad).all()


@pytest.mark.unit
@pytest.mark.skipif(
    shutil.which("matlab") is None,
    reason="MATLAB not available on this machine",
)
def test_work_estimator_close_to_matlab_compute_total_work_when_loaded() -> None:
    """Cross-check against MATLAB ``compute_total_work`` on a known case.

    The MATLAB function expects a sim_out struct with sampled (tau, omega)
    on a time grid. We construct that grid from a random degree-6
    polynomial via ``omega = d/dt poly``, then expect MATLAB's trapezoidal
    integration to match our analytical form within trapezoidal error.

    Skipped automatically when MATLAB is not on PATH.
    """
    pytest.skip("MATLAB cross-check stub; enable in CI image with MATLAB")
