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
