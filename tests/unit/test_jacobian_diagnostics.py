"""Tests for cross-engine Jacobian diagnostics (Issue #760).

Validates JacobianDiagnostics, ConstraintDiagnostics, nullspace
projection, cross-engine validation, and task-point diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.common.jacobian_diagnostics import (
    GOLF_TASK_POINTS,
    compute_constraint_diagnostics,
    compute_jacobian_diagnostics,
    compute_nullspace_projection,
    diagnose_task_points,
    validate_jacobians_cross_engine,
)


class TestJacobianDiagnostics:
    """Tests for single-Jacobian diagnostics."""

    def test_full_rank_jacobian(self) -> None:
        """Full-rank Jacobian should have zero nullspace dimension."""
        J = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        diag = compute_jacobian_diagnostics(J, body_name="test")

        assert diag.shape == (3, 2)
        assert diag.rank == 2
        assert diag.nullspace_dim == 0
        assert diag.condition_number < 1e6
        assert not diag.is_near_singular
        assert diag.manipulability > 0

    def test_rank_deficient_jacobian(self) -> None:
        """Rank-deficient Jacobian should report correct nullspace."""
        # Rank-1 matrix (second column is 2x first)
        J = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
        diag = compute_jacobian_diagnostics(J, body_name="singular")

        assert diag.rank == 1
        assert diag.nullspace_dim == 1
        assert diag.is_near_singular

    def test_identity_jacobian(self) -> None:
        """Identity Jacobian should have perfect conditioning."""
        J = np.eye(6)
        diag = compute_jacobian_diagnostics(J, body_name="identity")

        assert diag.rank == 6
        assert diag.nullspace_dim == 0
        assert diag.condition_number == pytest.approx(1.0, abs=1e-10)
        assert diag.manipulability == pytest.approx(1.0, abs=1e-10)

    def test_empty_jacobian(self) -> None:
        """Empty Jacobian should return degenerate diagnostics."""
        J = np.array([]).reshape(0, 0)
        diag = compute_jacobian_diagnostics(J, body_name="empty")

        assert diag.shape == (0, 0)
        assert diag.rank == 0
        assert diag.condition_number == float("inf")
        assert diag.manipulability == 0.0
        assert diag.is_near_singular

    def test_rectangular_tall_jacobian(self) -> None:
        """Tall (overdetermined) Jacobian should work correctly."""
        # 6x3 Jacobian (more rows than columns)
        rng = np.random.default_rng(42)
        J = rng.standard_normal((6, 3))
        diag = compute_jacobian_diagnostics(J, body_name="tall")

        assert diag.shape == (6, 3)
        assert diag.rank <= 3
        assert len(diag.singular_values) == 3

    def test_rectangular_wide_jacobian(self) -> None:
        """Wide (underdetermined) Jacobian should report nullspace."""
        # 3x6 Jacobian (redundant robot)
        rng = np.random.default_rng(42)
        J = rng.standard_normal((3, 6))
        diag = compute_jacobian_diagnostics(J, body_name="wide")

        assert diag.shape == (3, 6)
        assert diag.rank <= 3
        assert diag.nullspace_dim >= 3  # At least 3-dimensional nullspace

    def test_near_singular_detection(self) -> None:
        """Near-singular Jacobian should be flagged."""
        J = np.array([[1.0, 0.0], [0.0, 1e-8]])
        diag = compute_jacobian_diagnostics(J, body_name="near_singular")

        assert diag.is_near_singular
        assert diag.condition_number > 1e6

    def test_singular_values_descending(self) -> None:
        """Singular values should be in descending order."""
        rng = np.random.default_rng(42)
        J = rng.standard_normal((6, 4))
        diag = compute_jacobian_diagnostics(J)

        for i in range(len(diag.singular_values) - 1):
            assert diag.singular_values[i] >= diag.singular_values[i + 1]


class TestConstraintDiagnostics:
    """Tests for constraint Jacobian analysis."""

    def test_full_rank_constraints(self) -> None:
        """Full-rank constraint Jacobian should have minimal nullspace."""
        # 3 constraints on 6 DOF → 3-dim nullspace
        rng = np.random.default_rng(42)
        J_c = rng.standard_normal((3, 6))
        diag = compute_constraint_diagnostics(J_c)

        assert diag.constraint_rank == 3
        assert diag.nullspace_dim == 3
        assert diag.nullspace_basis.shape == (6, 3)

    def test_nullspace_basis_orthonormal(self) -> None:
        """Nullspace basis vectors should be orthonormal."""
        rng = np.random.default_rng(42)
        J_c = rng.standard_normal((2, 5))
        diag = compute_constraint_diagnostics(J_c)

        if diag.nullspace_dim > 0:
            # Check orthonormality: V^T V = I
            VtV = diag.nullspace_basis.T @ diag.nullspace_basis
            np.testing.assert_allclose(VtV, np.eye(diag.nullspace_dim), atol=1e-10)

    def test_nullspace_satisfies_constraint(self) -> None:
        """Nullspace vectors should satisfy J_c @ v = 0."""
        rng = np.random.default_rng(42)
        J_c = rng.standard_normal((3, 6))
        diag = compute_constraint_diagnostics(J_c)

        if diag.nullspace_dim > 0:
            # J_c @ nullspace_basis should be near zero
            result = J_c @ diag.nullspace_basis
            np.testing.assert_allclose(result, 0.0, atol=1e-8)

    def test_rank_plus_nullspace_equals_n(self) -> None:
        """Rank + nullspace dimension should equal number of DOF."""
        rng = np.random.default_rng(42)
        J_c = rng.standard_normal((4, 7))
        diag = compute_constraint_diagnostics(J_c)

        assert diag.constraint_rank + diag.nullspace_dim == 7

    def test_overconstrained_detection(self) -> None:
        """Overconstrained system should be flagged."""
        # 4 constraints on 4 DOF, expect 2 unconstrained DOF
        rng = np.random.default_rng(42)
        J_c = rng.standard_normal((4, 4))
        diag = compute_constraint_diagnostics(J_c, expected_dof=2)

        # rank=4 > n-expected=2, so overconstrained
        assert diag.is_overconstrained

    def test_empty_constraint_jacobian(self) -> None:
        """Empty constraint Jacobian should return zero diagnostics."""
        J_c = np.array([]).reshape(0, 0)
        diag = compute_constraint_diagnostics(J_c)

        assert diag.constraint_rank == 0
        assert diag.nullspace_dim == 0


class TestNullspaceProjection:
    """Tests for nullspace projection matrix."""

    def test_projection_idempotent(self) -> None:
        """Projection matrix should be idempotent: P^2 = P."""
        rng = np.random.default_rng(42)
        J = rng.standard_normal((3, 6))
        P = compute_nullspace_projection(J)

        np.testing.assert_allclose(P @ P, P, atol=1e-10)

    def test_projection_annihilates_task(self) -> None:
        """J @ P should be zero (projection is in nullspace of J)."""
        rng = np.random.default_rng(42)
        J = rng.standard_normal((3, 6))
        P = compute_nullspace_projection(J)

        np.testing.assert_allclose(J @ P, 0.0, atol=1e-10)

    def test_projection_preserves_nullspace_vectors(self) -> None:
        """Nullspace vectors should be preserved by projection."""
        # Create a Jacobian with known nullspace
        J = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        P = compute_nullspace_projection(J)

        # [0, 0, 1] is in the nullspace
        v_null = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(P @ v_null, v_null, atol=1e-10)

    def test_square_full_rank_projection_is_zero(self) -> None:
        """Full-rank square Jacobian should have zero nullspace projection."""
        J = np.eye(3)
        P = compute_nullspace_projection(J)

        np.testing.assert_allclose(P, np.zeros((3, 3)), atol=1e-10)


class TestCrossEngineValidation:
    """Tests for cross-engine Jacobian comparison."""

    def test_identical_jacobians_pass(self) -> None:
        """Identical Jacobians should pass validation."""
        J = np.array([[1.0, 0.5], [0.0, 1.0], [0.3, 0.7]])
        report = validate_jacobians_cross_engine(
            {"engine_a": J.copy(), "engine_b": J.copy()},
            body_name="test",
        )

        assert report.passed
        assert report.shape_match
        assert report.rank_match
        assert report.max_element_diff == 0.0

    def test_different_shapes_fail(self) -> None:
        """Different Jacobian shapes should fail validation."""
        J_a = np.zeros((3, 4))
        J_b = np.zeros((3, 5))
        report = validate_jacobians_cross_engine(
            {"engine_a": J_a, "engine_b": J_b},
            body_name="test",
        )

        assert not report.shape_match
        assert not report.passed

    def test_different_ranks_fail(self) -> None:
        """Different ranks should fail validation."""
        J_a = np.array([[1.0, 0.0], [0.0, 1.0]])
        J_b = np.array([[1.0, 2.0], [2.0, 4.0]])  # Rank 1
        report = validate_jacobians_cross_engine(
            {"engine_a": J_a, "engine_b": J_b},
            body_name="test",
        )

        assert not report.rank_match
        assert not report.passed

    def test_small_differences_pass(self) -> None:
        """Small numerical differences should pass validation."""
        rng = np.random.default_rng(42)
        J_base = rng.standard_normal((3, 4))
        J_noisy = J_base + 1e-5 * rng.standard_normal((3, 4))

        report = validate_jacobians_cross_engine(
            {"engine_a": J_base, "engine_b": J_noisy},
            body_name="test",
            atol=1e-3,
        )

        assert report.passed
        assert report.max_element_diff < 1e-3

    def test_single_engine_passes(self) -> None:
        """Single engine should trivially pass."""
        J = np.eye(3)
        report = validate_jacobians_cross_engine({"only_engine": J}, body_name="test")

        assert report.passed

    def test_three_engine_comparison(self) -> None:
        """Three-way comparison should work."""
        J = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        report = validate_jacobians_cross_engine(
            {"a": J.copy(), "b": J.copy(), "c": J.copy()},
            body_name="test",
        )

        assert report.passed
        assert len(report.engines) == 3
        assert len(report.condition_numbers) == 3


class TestTaskPointDiagnostics:
    """Tests for task-point diagnostic sweep."""

    def test_golf_task_points_defined(self) -> None:
        """Golf task points should include clubhead and grip."""
        assert "clubhead" in GOLF_TASK_POINTS
        assert "grip" in GOLF_TASK_POINTS
        assert "left_hand" in GOLF_TASK_POINTS
        assert "right_hand" in GOLF_TASK_POINTS

    def test_diagnose_with_mock_engine(self) -> None:
        """Diagnose task points using a mock compute_jacobian."""

        def mock_compute_jacobian(body_name: str) -> dict[str, np.ndarray] | None:
            if body_name == "unknown_body":
                return None
            rng = np.random.default_rng(hash(body_name) % 2**32)
            J = rng.standard_normal((6, 4))
            return {"spatial": J, "linear": J[:3], "angular": J[3:]}

        results = diagnose_task_points(
            mock_compute_jacobian,
            task_points=["clubhead", "grip", "unknown_body"],
        )

        assert "clubhead" in results
        assert "grip" in results
        assert "unknown_body" not in results

        for name, diag in results.items():
            assert diag.body_name == name
            assert diag.shape == (6, 4)
            assert diag.rank <= 4

    def test_diagnose_prefers_spatial(self) -> None:
        """Diagnostics should prefer spatial Jacobian over linear."""

        def mock_compute_jacobian(body_name: str) -> dict[str, np.ndarray]:
            return {
                "spatial": np.eye(6, 4),
                "linear": np.eye(3, 4),
            }

        results = diagnose_task_points(mock_compute_jacobian, task_points=["clubhead"])

        assert results["clubhead"].shape == (6, 4)

    def test_diagnose_falls_back_to_linear(self) -> None:
        """Diagnostics should fall back to linear if no spatial."""

        def mock_compute_jacobian(body_name: str) -> dict[str, np.ndarray]:
            return {"linear": np.eye(3, 4)}

        results = diagnose_task_points(mock_compute_jacobian, task_points=["grip"])

        assert results["grip"].shape == (3, 4)


class TestPendulumJacobianDiagnostics:
    """Tests using the real PendulumPhysicsEngine (no external deps)."""

    def test_pendulum_jacobian_diagnostics(self) -> None:
        """PendulumPhysicsEngine Jacobian should have correct diagnostics."""
        from src.engines.physics_engines.pendulum.python.pendulum_physics_engine import (
            PendulumPhysicsEngine,
        )

        engine = PendulumPhysicsEngine()
        engine.set_state(np.array([0.5, -0.3]), np.array([0.0, 0.0]))
        engine.forward()

        jac = engine.compute_jacobian("link2")
        if jac is None:
            pytest.skip("PendulumPhysicsEngine doesn't support body 'link2'")

        J = jac.get("spatial", jac.get("linear"))
        if J is None:
            pytest.skip("No Jacobian returned")

        diag = compute_jacobian_diagnostics(J, body_name="link2")

        assert diag.rank > 0
        assert diag.rank <= min(J.shape)
        assert diag.rank + diag.nullspace_dim == J.shape[1]
        assert diag.manipulability >= 0

    def test_pendulum_nullspace_projection(self) -> None:
        """Nullspace projection for pendulum should satisfy J*P=0."""
        from src.engines.physics_engines.pendulum.python.pendulum_physics_engine import (
            PendulumPhysicsEngine,
        )

        engine = PendulumPhysicsEngine()
        engine.set_state(np.array([0.5, -0.3]), np.array([0.0, 0.0]))
        engine.forward()

        jac = engine.compute_jacobian("link2")
        if jac is None:
            pytest.skip("PendulumPhysicsEngine doesn't support body 'link2'")

        J = jac.get("spatial", jac.get("linear"))
        if J is None:
            pytest.skip("No Jacobian returned")

        P = compute_nullspace_projection(J)

        # J @ P should be zero
        np.testing.assert_allclose(J @ P, 0.0, atol=1e-8)
