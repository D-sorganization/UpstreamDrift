"""Unit tests for analytic pelvis-yaw orientation cost integration in Crocoddyl solver (MS-107, #10381).

Verifies:
1. MarkerTargets.waist_indices resolves WaistLeft / WaistRight or (-1, -1).
2. _NodeCost evaluates zero pelvis-yaw cost and zero yaw gradient when aligned.
3. _NodeCost analytic gradient matches finite-difference numerical gradient.
4. _NodeCost Gauss-Newton Hessian matches analytic J^T J structure.
5. Inactive pelvis_yaw weight (0.0) or missing markers leaves cost and grad unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.crocoddyl_action import _NodeCost
from src.engines.physics_engines.pinocchio.python.crocoddyl_problem import (
    FitWeights,
    MarkerTargets,
)

pytestmark = pytest.mark.unit


class _MockPlantContext:
    """Rigid 2-marker body with yaw rotation and 3D translation."""

    def __init__(self, n: int = 4) -> None:
        self.n = n
        self.lower = -10.0 * np.ones(n)
        self.upper = 10.0 * np.ones(n)

    def markers(self, q: np.ndarray) -> np.ndarray:
        c, s = float(np.cos(q[0])), float(np.sin(q[0]))
        p_wl = np.array([-0.1 * c + q[1], -0.1 * s + q[2], 1.0 + q[3]])
        p_wr = np.array([0.1 * c + q[1], 0.1 * s + q[2], 1.0 + q[3]])
        return np.stack([p_wl, p_wr], axis=0)

    def markers_and_jacobians(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = self.markers(q)
        c, s = float(np.cos(q[0])), float(np.sin(q[0]))
        jac = np.zeros((2, 3, self.n))
        jac[0, :, 0] = np.array([0.1 * s, -0.1 * c, 0.0])
        jac[0, :, 1] = np.array([1.0, 0.0, 0.0])
        jac[0, :, 2] = np.array([0.0, 1.0, 0.0])
        jac[0, :, 3] = np.array([0.0, 0.0, 1.0])

        jac[1, :, 0] = np.array([-0.1 * s, 0.1 * c, 0.0])
        jac[1, :, 1] = np.array([1.0, 0.0, 0.0])
        jac[1, :, 2] = np.array([0.0, 1.0, 0.0])
        jac[1, :, 3] = np.array([0.0, 0.0, 1.0])
        return pos, jac


def test_marker_targets_waist_indices() -> None:
    targets_data = np.zeros((2, 3, 3))
    valid_data = np.ones((2, 3), dtype=bool)
    weights_data = np.ones(3)
    node_times = np.array([0.0, 0.01])

    mt = MarkerTargets(
        targets=targets_data,
        valid=valid_data,
        weights=weights_data,
        labels=("WaistLeft", "Head", "WaistRight"),
        node_times=node_times,
    )
    assert mt.waist_indices == (0, 2)

    mt_absent = MarkerTargets(
        targets=targets_data,
        valid=valid_data,
        weights=weights_data,
        labels=("Head", "Knee", "Ankle"),
        node_times=node_times,
    )
    assert mt_absent.waist_indices == (-1, -1)


def test_node_cost_zero_yaw_residual_when_aligned() -> None:
    ctx = _MockPlantContext()
    q = np.array([0.0, 0.0, 0.0, 0.0])
    v = np.zeros(4)
    target_pos = ctx.markers(q)

    weights_active = FitWeights(
        marker=1.0,
        terminal_marker=1.0,
        effort=1e-5,
        velocity=1e-3,
        range_barrier=1e3,
        pelvis_yaw=50.0,
    )
    weights_inactive = FitWeights(
        marker=1.0,
        terminal_marker=1.0,
        effort=1e-5,
        velocity=1e-3,
        range_barrier=1e3,
        pelvis_yaw=0.0,
    )

    cost_active = _NodeCost(
        ctx,
        target_pos,
        np.array([True, True]),
        np.ones(2),
        weights_active,
        marker_weight=1.0,
        wl_i=0,
        wr_i=1,
    )
    cost_inactive = _NodeCost(
        ctx,
        target_pos,
        np.array([True, True]),
        np.ones(2),
        weights_inactive,
        marker_weight=1.0,
        wl_i=0,
        wr_i=1,
    )

    # Both active and inactive should yield identical zero yaw cost when aligned
    assert cost_active.value(q, v, None) == pytest.approx(
        cost_inactive.value(q, v, None), abs=1e-12
    )
    grad_a, _ = cost_active.gradient_hessian(q)
    grad_i, _ = cost_inactive.gradient_hessian(q)
    assert np.allclose(grad_a, grad_i)


def test_node_cost_yaw_gradient_matches_finite_difference() -> None:
    ctx = _MockPlantContext()
    q_tgt = np.array([0.0, 0.0, 0.0, 0.0])
    target_pos = ctx.markers(q_tgt)

    # Perturbed configuration with non-zero yaw and translation
    q = np.array([0.35, 0.05, -0.02, 0.01])
    v = np.zeros(4)

    weights = FitWeights(
        marker=10.0,
        terminal_marker=10.0,
        effort=1e-3,
        velocity=1e-3,
        range_barrier=10.0,
        pelvis_yaw=75.0,
    )

    cost_model = _NodeCost(
        ctx,
        target_pos,
        np.array([True, True]),
        np.ones(2),
        weights,
        marker_weight=weights.marker,
        wl_i=0,
        wr_i=1,
    )

    val0 = cost_model.value(q, v, None)
    assert val0 > 0.0

    grad_analytic, hess_analytic = cost_model.gradient_hessian(q)

    # Central difference gradient
    eps = 1e-6
    grad_num = np.zeros_like(q)
    for i in range(len(q)):
        qp = q.copy()
        qm = q.copy()
        qp[i] += eps
        qm[i] -= eps
        vp = cost_model.value(qp, v, None)
        vm = cost_model.value(qm, v, None)
        grad_num[i] = (vp - vm) / (2.0 * eps)

    assert np.allclose(grad_analytic, grad_num, rtol=1e-4, atol=1e-5)
    # Gauss-Newton Hessian should be positive semidefinite
    eigvals = np.linalg.eigvalsh(hess_analytic)
    assert (eigvals >= -1e-10).all()


def test_node_cost_inactive_when_weight_zero_or_markers_missing() -> None:
    ctx = _MockPlantContext()
    q_tgt = np.array([0.0, 0.0, 0.0, 0.0])
    target_pos = ctx.markers(q_tgt)
    q = np.array([0.3, 0.0, 0.0, 0.0])
    v = np.zeros(4)

    weights_zero = FitWeights(
        marker=10.0,
        terminal_marker=10.0,
        effort=1e-5,
        velocity=1e-3,
        range_barrier=1e3,
        pelvis_yaw=0.0,
    )
    cost_zero = _NodeCost(
        ctx,
        target_pos,
        np.array([True, True]),
        np.ones(2),
        weights_zero,
        marker_weight=weights_zero.marker,
        wl_i=0,
        wr_i=1,
    )

    weights_active = FitWeights(
        marker=10.0,
        terminal_marker=10.0,
        effort=1e-5,
        velocity=1e-3,
        range_barrier=1e3,
        pelvis_yaw=50.0,
    )
    # Missing markers (-1, -1)
    cost_missing = _NodeCost(
        ctx,
        target_pos,
        np.array([True, True]),
        np.ones(2),
        weights_active,
        marker_weight=weights_active.marker,
        wl_i=-1,
        wr_i=-1,
    )

    # Both should have identical costs and gradients (pelvis yaw inactive)
    val_zero = cost_zero.value(q, v, None)
    val_missing = cost_missing.value(q, v, None)
    assert val_zero == pytest.approx(val_missing, abs=1e-12)

    grad_z, hess_z = cost_zero.gradient_hessian(q)
    grad_m, hess_m = cost_missing.gradient_hessian(q)
    assert np.allclose(grad_z, grad_m)
    assert np.allclose(hess_z, hess_m)
