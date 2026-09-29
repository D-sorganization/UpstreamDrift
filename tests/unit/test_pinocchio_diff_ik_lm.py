"""Unit tests for the pure-pinocchio differential IK fallback.

These tests exercise the Levenberg-Marquardt iteration math purely in
numpy and do **not** require ``pinocchio`` to be installed. The full
end-to-end IK convergence tests live in
``tests/heavy_integration/test_pinocchio_diff_ik.py``.

Closes issue #4138.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.engines.physics_engines.pinocchio.python.pinocchio_golf.diff_ik import (
    lm_step,
)


@pytest.mark.unit
class TestLMStep:
    """Numpy-only LM iteration math."""

    def test_zero_error_returns_zero_step(self) -> None:
        """Zero task error should produce a zero joint step."""
        rng = np.random.default_rng(0)
        jac = rng.standard_normal((6, 7))
        err = np.zeros(6)
        dq = lm_step(jac, err, damping=1e-3)
        assert np.allclose(dq, 0.0)

    def test_full_rank_square_matches_undamped_inverse(self) -> None:
        """With negligible damping and a full-rank square Jacobian,
        the LM step matches the exact inverse."""
        rng = np.random.default_rng(1)
        jac = rng.standard_normal((4, 4))
        err = rng.standard_normal(4)
        dq = lm_step(jac, err, damping=1e-12)
        expected = np.linalg.solve(jac, err)
        assert np.allclose(dq, expected, atol=1e-6)

    def test_decreases_linearised_residual(self) -> None:
        """A single LM step must decrease the linearised residual norm."""
        rng = np.random.default_rng(2)
        jac = rng.standard_normal((6, 12))
        err = rng.standard_normal(6)
        dq = lm_step(jac, err, damping=1e-4)
        residual_before = float(np.linalg.norm(err))
        residual_after = float(np.linalg.norm(err - jac @ dq))
        assert residual_after < residual_before

    def test_singular_jacobian_does_not_blow_up(self) -> None:
        """A rank-deficient Jacobian with damping returns a finite step."""
        # Two identical rows → rank deficient.
        jac = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )
        err = np.array([1.0, 1.0, 0.5])
        dq = lm_step(jac, err, damping=1e-3)
        assert np.all(np.isfinite(dq))
        # Damping caps the step magnitude — sanity bound.
        assert float(np.linalg.norm(dq)) < 1e6

    def test_determinism(self) -> None:
        """Identical inputs produce identical outputs (no RNG inside)."""
        rng = np.random.default_rng(3)
        jac = rng.standard_normal((6, 9))
        err = rng.standard_normal(6)
        dq1 = lm_step(jac, err, damping=1e-3)
        dq2 = lm_step(jac, err, damping=1e-3)
        assert np.array_equal(dq1, dq2)

    def test_negative_damping_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            lm_step(np.eye(3), np.zeros(3), damping=-1.0)

    def test_pinocchio_diff_ik_lm_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="incompatible"):
            lm_step(np.eye(3), np.zeros(4), damping=1e-3)

    def test_non_array_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            lm_step([[1.0, 0.0], [0.0, 1.0]], np.zeros(2), damping=1e-3)  # type: ignore[arg-type]

    def test_increasing_damping_shrinks_step(self) -> None:
        """Larger damping should produce a smaller-magnitude step."""
        rng = np.random.default_rng(4)
        jac = rng.standard_normal((4, 6))
        err = rng.standard_normal(4)
        dq_small = lm_step(jac, err, damping=1e-4)
        dq_large = lm_step(jac, err, damping=1.0)
        assert float(np.linalg.norm(dq_large)) < float(np.linalg.norm(dq_small))
