"""Cross-language parity test for ``upstream_motion_matching.finite_diff_q_to_qdot_qddot``.

Verifies the Rust port (issue #5218 first slice) reproduces the Python
``PinocchioInverseDynMatchingSolver._finite_difference`` outputs to
within bit-level rounding noise on a deterministic sine-wave trajectory.

Skipped when the Rust extension is not installed — local builds need
``maturin develop -m rust_core/upstream-motion-matching/Cargo.toml``;
CI builds the wheel via the rust-quality-gate workflow.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _python_finite_difference(
    q: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reference Python finite-difference, mirroring the
    ``PinocchioInverseDynMatchingSolver._finite_difference`` numerics
    when the input timestamps are uniform with spacing ``dt``.

    Lifted into the test module so we don't have to instantiate a
    ``JointTrajectory`` (which would pull in the whole contracts +
    pinocchio import surface) for what is purely a numerical comparison.
    """
    n = q.shape[0]
    qdot = np.zeros_like(q)
    qddot = np.zeros_like(q)
    if n < 2:
        return qdot, qddot

    # Interior central differences.
    for i in range(1, n - 1):
        # Matches the non-uniform formula at uniform dt:
        # qdot[i] = (q[i+1] - q[i-1]) / (2*dt)
        # qddot[i] = (q[i+1] - 2*q[i] + q[i-1]) / dt^2
        qdot[i] = (q[i + 1] - q[i - 1]) / (2.0 * dt)
        qddot[i] = (q[i + 1] - 2.0 * q[i] + q[i - 1]) / (dt * dt)

    # Boundary qdot: one-sided.
    qdot[0] = (q[1] - q[0]) / dt
    qdot[-1] = (q[-1] - q[-2]) / dt

    # Boundary qddot: copy interior. Matches the Python
    # `if len(times) >= 3` branch exactly.
    if n >= 3:
        qddot[0] = qddot[1]
        qddot[-1] = qddot[-2]

    return qdot, qddot


def _make_sine_trajectory(n: int, d: int, dt: float) -> np.ndarray:
    """Deterministic 100-frame fallback fixture: each DOF picks up a
    distinct frequency / phase so the finite-difference output is
    non-trivial."""
    times = np.arange(n) * dt
    out = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        f = 1.0 + 0.5 * j
        out[:, j] = np.sin(2.0 * math.pi * f * times + 0.1 * j)
    return out


def test_rust_finite_diff_matches_python_within_1e_minus_10() -> None:
    """Rust output must match Python to 1e-10 absolute on every entry."""
    pytest.importorskip(
        "upstream_motion_matching",
        reason="upstream_motion_matching wheel not installed; run "
        "`maturin develop -m rust_core/upstream-motion-matching/Cargo.toml`",
    )
    import upstream_motion_matching as umm

    dt = 1.0 / 240.0
    q = _make_sine_trajectory(n=100, d=15, dt=dt)

    py_qdot, py_qddot = _python_finite_difference(q, dt)
    rs_qdot, rs_qddot = umm.finite_diff_q_to_qdot_qddot(q.tolist(), dt)
    rs_qdot_np = np.asarray(rs_qdot, dtype=np.float64)
    rs_qddot_np = np.asarray(rs_qddot, dtype=np.float64)

    # Both implementations evaluate the same arithmetic in the same
    # order at uniform dt — we expect bit-level parity, but allow 1e-10
    # absolute as a guard against future refactors that introduce a
    # different (still mathematically equivalent) factoring.
    assert rs_qdot_np.shape == py_qdot.shape
    assert rs_qddot_np.shape == py_qddot.shape
    np.testing.assert_allclose(rs_qdot_np, py_qdot, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(rs_qddot_np, py_qddot, atol=1e-10, rtol=0.0)


def test_rust_finite_diff_rejects_invalid_dt() -> None:
    """Rust binding surfaces dt validation as ``ValueError``."""
    pytest.importorskip("upstream_motion_matching")
    import upstream_motion_matching as umm

    q = _make_sine_trajectory(n=10, d=3, dt=0.01).tolist()
    with pytest.raises(ValueError):
        umm.finite_diff_q_to_qdot_qddot(q, 0.0)
    with pytest.raises(ValueError):
        umm.finite_diff_q_to_qdot_qddot(q, -0.01)
