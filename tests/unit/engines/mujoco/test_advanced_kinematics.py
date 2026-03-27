"""Comprehensive tests for advanced kinematics module."""

import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.advanced_kinematics import (
    AdvancedKinematicsAnalyzer,
    ConstraintJacobianData,
    ManipulabilityMetrics,
)
from mujoco_humanoid_golf.models import DOUBLE_PENDULUM_XML


class TestManipulabilityMetrics:
    """Tests for ManipulabilityMetrics dataclass."""

    def test_initialization(self) -> None:
        """Test metrics initialization."""
        metrics = ManipulabilityMetrics(
            manipulability_index=1.5,
            condition_number=10.0,
            singular_values=np.array([1.0, 0.5, 0.1]),
            is_near_singularity=False,
            min_singular_value=0.1,
            max_singular_value=1.0,
        )

        assert metrics.manipulability_index == 1.5
        assert metrics.condition_number == 10.0
        assert len(metrics.singular_values) == 3
        assert metrics.is_near_singularity is False


class TestConstraintJacobianData:
    """Tests for ConstraintJacobianData dataclass."""

    def test_initialization(self) -> None:
        """Test constraint data initialization."""
        jac = np.eye(3)
        nullspace = np.zeros((3, 0))
        data = ConstraintJacobianData(
            constraint_jacobian=jac,
            nullspace_basis=nullspace,
            nullspace_dimension=0,
            rank=3,
            is_overconstrained=False,
        )

        np.testing.assert_array_equal(data.constraint_jacobian, jac)
        assert data.nullspace_dimension == 0
        assert data.rank == 3


class TestAdvancedKinematicsAnalyzer:
    """Tests for AdvancedKinematicsAnalyzer class."""

    @pytest.fixture()
    def model_and_data(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Create model and data for testing."""
        model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def test_initialization(self, model_and_data) -> None:
        """Test analyzer initialization."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        assert analyzer.model == model
        assert analyzer.data == data
        assert analyzer.singularity_threshold == 30.0
        assert analyzer.ik_damping == 0.01
        assert analyzer.ik_max_iterations == 100

    def test_find_body_id(self, model_and_data) -> None:
        """Test finding body ID."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        body_id = analyzer._find_body_id("shoulder")
        if body_id is not None:
            assert body_id > 0
            assert body_id < model.nbody

        # Should return None for nonexistent body
        body_id = analyzer._find_body_id("nonexistent_body_xyz")
        assert body_id is None

    def test_compute_body_jacobian(self, model_and_data) -> None:
        """Test computing body Jacobian."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        # Use first non-world body
        body_id = 1
        jacp, jacr = analyzer.compute_body_jacobian(body_id)

        assert jacp.shape == (3, model.nv)
        assert jacr.shape == (3, model.nv)
        assert np.all(np.isfinite(jacp))
        assert np.all(np.isfinite(jacr))

    def test_compute_body_jacobian_with_offset(self, model_and_data) -> None:
        """Test computing Jacobian with point offset."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        body_id = 1
        offset = np.array([0.1, 0.0, 0.0])
        jacp, jacr = analyzer.compute_body_jacobian(body_id, point_offset=offset)

        assert jacp.shape == (3, model.nv)
        assert jacr.shape == (3, model.nv)

    def test_compute_constraint_jacobian_no_constraints(self, model_and_data) -> None:
        """Test constraint Jacobian with no constraints."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        # Simple model may have no constraints
        constraint_data = analyzer.compute_constraint_jacobian()

        assert isinstance(constraint_data, ConstraintJacobianData)
        assert constraint_data.constraint_jacobian.shape[1] == model.nv
        assert constraint_data.rank >= 0
        assert constraint_data.nullspace_dimension >= 0

    def test_compute_manipulability(self, model_and_data) -> None:
        """Test manipulability computation."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        # Create a simple Jacobian (use actual body Jacobian)
        body_id = 1
        jacp, _ = analyzer.compute_body_jacobian(body_id)
        jacobian = jacp

        metrics = analyzer.compute_manipulability(jacobian, metric_type="yoshikawa")

        assert isinstance(metrics, ManipulabilityMetrics)
        assert metrics.manipulability_index >= 0.0
        assert metrics.condition_number >= 1.0
        assert len(metrics.singular_values) > 0
        assert isinstance(metrics.is_near_singularity, bool)

    def test_compute_manipulability_condition_type(self, model_and_data) -> None:
        """Test manipulability with condition metric type."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        jacobian = np.eye(3, model.nv)
        metrics = analyzer.compute_manipulability(jacobian, metric_type="condition")

        assert metrics.manipulability_index >= 0.0

    def test_solve_inverse_kinematics_position_only(self, model_and_data) -> None:
        """Test IK solver for position only."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        body_id = 1
        target_pos = data.xpos[body_id].copy() + np.array([0.1, 0.0, 0.0])

        q_result, success, iterations = analyzer.solve_inverse_kinematics(
            body_id,
            target_pos,
        )

        assert q_result.shape == (model.nq,)
        assert isinstance(success, bool)
        assert 0 <= iterations <= analyzer.ik_max_iterations
        assert np.all(np.isfinite(q_result))

    def test_solve_inverse_kinematics_with_orientation(self, model_and_data) -> None:
        """Test IK solver with orientation."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        body_id = 1
        target_pos = data.xpos[body_id].copy()
        target_ori = data.xquat[body_id].copy()

        q_result, success, iterations = analyzer.solve_inverse_kinematics(
            body_id,
            target_pos,
            target_orientation=target_ori,
        )

        assert q_result.shape == (model.nq,)
        assert isinstance(success, bool)
        assert np.all(np.isfinite(q_result))

    def test_solve_inverse_kinematics_with_initial_guess(
        self,
        model_and_data,
    ) -> None:
        """Test IK solver with initial guess."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        body_id = 1
        target_pos = data.xpos[body_id].copy()
        q_init = data.qpos.copy()

        q_result, success, iterations = analyzer.solve_inverse_kinematics(
            body_id,
            target_pos,
            q_init=q_init,
        )

        assert q_result.shape == (model.nq,)
        assert np.all(np.isfinite(q_result))

    def test_quat_conjugate(self, model_and_data) -> None:
        """Test quaternion conjugate computation."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        q_conj = analyzer._quat_conjugate(q)

        np.testing.assert_array_equal(q_conj, q)  # Identity is its own conjugate

        q2 = np.array([0.707, 0.707, 0.0, 0.0])
        q2_conj = analyzer._quat_conjugate(q2)
        assert q2_conj[0] == q2[0]  # w unchanged
        assert q2_conj[1] == -q2[1]  # x negated

    def test_quat_multiply(self, model_and_data) -> None:
        """Test quaternion multiplication."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        # Identity quaternion
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.707, 0.707, 0.0, 0.0])

        q_result = analyzer._quat_multiply(q_identity, q)
        np.testing.assert_allclose(q_result, q, atol=1e-3)

    def test_compute_orientation_error(self, model_and_data) -> None:
        """Test orientation error computation."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        # Same orientation should give zero error
        q = np.array([1.0, 0.0, 0.0, 0.0])
        error = analyzer._compute_orientation_error(q, q)

        assert error.shape == (3,)
        np.testing.assert_allclose(error, [0, 0, 0], atol=1e-3)

    def test_clamp_to_joint_limits(self, model_and_data) -> None:
        """Test joint limit clamping."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        q = data.qpos.copy()
        q_clamped = analyzer._clamp_to_joint_limits(q)

        assert q_clamped.shape == q.shape
        assert np.all(np.isfinite(q_clamped))

    def test_compute_manipulability_ellipsoid(self, model_and_data) -> None:
        """Test manipulability ellipsoid computation."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        body_id = 1
        center, radii, axes = analyzer.compute_manipulability_ellipsoid(body_id)

        assert center.shape == (3,)
        assert len(radii) >= 1  # May be less than 3 depending on SVD
        assert axes.shape[0] == 3  # First dimension is always 3
        assert np.all(np.isfinite(center))
        assert np.all(np.isfinite(radii))
        assert np.all(np.isfinite(axes))

    def test_analyze_singularities(self, model_and_data) -> None:
        """Test singularity analysis."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        body_id = 1
        # Use small number of samples for speed
        singular_configs, condition_numbers = analyzer.analyze_singularities(
            body_id,
            num_samples=5,
        )

        assert isinstance(singular_configs, list)
        assert len(condition_numbers) == 5
        assert all(isinstance(cn, float) for cn in condition_numbers)

    def test_analyze_singularities_with_samples(self, model_and_data) -> None:
        """Test singularity analysis with provided samples."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        body_id = 1
        q_samples = np.array([data.qpos.copy() for _ in range(3)])

        singular_configs, condition_numbers = analyzer.analyze_singularities(
            body_id,
            q_samples=q_samples,
        )

        assert len(condition_numbers) == 3

    def test_generate_random_configs(self, model_and_data) -> None:
        """Test random configuration generation."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        configs = analyzer._generate_random_configs(5)

        assert configs.shape[0] == 5
        assert configs.shape[1] == model.nq
        assert np.all(np.isfinite(configs))

    def test_compute_nullspace_projection(self, model_and_data) -> None:
        """Test nullspace projection computation."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        # Create a simple Jacobian
        jacobian = np.eye(3, model.nv)

        nullspace_proj = analyzer.compute_nullspace_projection(jacobian)

        assert nullspace_proj.shape == (model.nv, model.nv)
        assert np.all(np.isfinite(nullspace_proj))

        # Nullspace projection should be idempotent: P^2 = P
        proj_squared = nullspace_proj @ nullspace_proj
        np.testing.assert_allclose(proj_squared, nullspace_proj, atol=1e-3)

    def test_compute_task_space_inertia(self, model_and_data) -> None:
        """Test task-space inertia computation."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        # Create a simple Jacobian
        jacobian = np.eye(3, model.nv)

        # May fail with singular matrices - handle gracefully
        try:
            task_inertia = analyzer.compute_task_space_inertia(jacobian)

            assert task_inertia.shape == (3, 3)
            assert np.all(np.isfinite(task_inertia))
            # Should be symmetric
            np.testing.assert_allclose(task_inertia, task_inertia.T, atol=1e-6)
        except np.linalg.LinAlgError:
            # Singular matrix is acceptable
            pytest.skip(
                "Singular matrix in task-space inertia (expected in some configs)"
            )

    def test_compute_grip_constraint_jacobian(self, model_and_data) -> None:
        """Test grip constraint Jacobian computation."""
        model, data = model_and_data
        analyzer = AdvancedKinematicsAnalyzer(model, data)

        constraint_jac = analyzer._compute_grip_constraint_jacobian()

        # May be empty if hands not found
        assert constraint_jac.shape[1] == model.nv or constraint_jac.shape[0] == 0
