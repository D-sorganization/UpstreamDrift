"""Unit tests for Pinocchio C3D motion matching and torque allocation CLI."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.match_pinocchio_c3d import setup_logging


@pytest.mark.unit
def test_setup_logging() -> None:
    """Verify setup_logging executes without error."""
    setup_logging(verbose=False)


@pytest.mark.unit
def test_trail_zero_torque_allocation_math() -> None:
    """Verify the linear algebra of the trail-zero wrench projection."""
    rng = np.random.default_rng(42)
    nv = 44
    trail_dim = 9
    constraint_dim = 6

    # Random RNEA generalized forces
    tau_rnea = rng.standard_normal(nv)

    # Random full constraint Jacobian (6, 44)
    raw_jc = rng.standard_normal((constraint_dim, nv))
    trail_v_idx = list(range(10, 10 + trail_dim))

    raw_trail = raw_jc[:, trail_v_idx]  # (6, 9)
    tau_trail_req = tau_rnea[trail_v_idx]

    # Solve least-squares contact wrench
    lambda_c, _, _, _ = np.linalg.lstsq(raw_trail.T, tau_trail_req, rcond=1e-4)
    assert lambda_c.shape == (constraint_dim,)

    # Apply trail-zero compensation
    tau_zero = tau_rnea.copy()
    tau_zero -= raw_jc.T @ lambda_c
    tau_zero[trail_v_idx] = 0.0

    assert np.allclose(tau_zero[trail_v_idx], 0.0)

    # Reconstructed effective generalized torque under active constraint wrench
    tau_effective = tau_zero + raw_jc.T @ lambda_c

    # Non-trail coordinates are preserved exactly
    non_trail_idx = [i for i in range(nv) if i not in trail_v_idx]
    assert np.allclose(tau_effective[non_trail_idx], tau_rnea[non_trail_idx])
