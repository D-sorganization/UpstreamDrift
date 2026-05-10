"""Tests for DrakeMotionOptimizer."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.motion_optimization import (
    DrakeMotionOptimizer,
    OptimizationConstraint,
    OptimizationObjective,
    OptimizationResult,
)


@pytest.fixture
def optimizer() -> DrakeMotionOptimizer:
    """Fixture providing a fresh DrakeMotionOptimizer instance."""
    return DrakeMotionOptimizer()


@pytest.fixture
def initial_trajectory() -> np.ndarray:
    """Fixture providing a dummy initial trajectory."""
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


class TestDrakeMotionOptimizerSetup:
    """Tests for setup methods of DrakeMotionOptimizer."""

    def test_init(self, optimizer: DrakeMotionOptimizer) -> None:
        """Test initialization."""
        assert len(optimizer.objectives) == 0
        assert len(optimizer.constraints) == 0

    def test_add_objective(self, optimizer: DrakeMotionOptimizer) -> None:
        """Test adding an objective."""
        dummy_cost = lambda x: 0.0
        optimizer.add_objective("test_obj", 1.5, dummy_cost, target_value=10.0)
        
        assert len(optimizer.objectives) == 1
        obj = optimizer.objectives[0]
        assert obj.name == "test_obj"
        assert obj.weight == 1.5
        assert obj.target_value == 10.0
        assert obj.cost_function is dummy_cost

    def test_add_objective_none_name(self, optimizer: DrakeMotionOptimizer) -> None:
        """Test adding an objective with None name raises ValueError."""
        with pytest.raises(ValueError, match="name must be provided"):
            optimizer.add_objective(None, 1.0, lambda x: 0.0)  # type: ignore

    def test_add_constraint(self, optimizer: DrakeMotionOptimizer) -> None:
        """Test adding a constraint."""
        dummy_constraint = lambda x: 0.0
        optimizer.add_constraint(
            "test_con", "inequality", dummy_constraint, lower_bound=-1.0, upper_bound=1.0
        )
        
        assert len(optimizer.constraints) == 1
        con = optimizer.constraints[0]
        assert con.name == "test_con"
        assert con.constraint_type == "inequality"
        assert con.lower_bound == -1.0
        assert con.upper_bound == 1.0
        assert con.constraint_function is dummy_constraint

    def test_add_constraint_none_name(self, optimizer: DrakeMotionOptimizer) -> None:
        """Test adding a constraint with None name raises ValueError."""
        with pytest.raises(ValueError, match="name must be provided"):
            optimizer.add_constraint(None, "equality", lambda x: 0.0)  # type: ignore

    def test_setup_standard_golf_objectives(self, optimizer: DrakeMotionOptimizer) -> None:
        """Test setting up standard golf objectives."""
        optimizer.setup_standard_golf_objectives()
        assert len(optimizer.objectives) == 3
        names = {obj.name for obj in optimizer.objectives}
        assert names == {"ball_speed", "accuracy", "smoothness"}
        
        # Test objective functions directly to hit coverage
        traj = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
        
        speed_obj = next(o for o in optimizer.objectives if o.name == "ball_speed")
        assert speed_obj.cost_function is not None
        assert speed_obj.cost_function(traj) < 0  # Negative peak speed
        
        acc_obj = next(o for o in optimizer.objectives if o.name == "accuracy")
        assert acc_obj.cost_function is not None
        assert acc_obj.cost_function(traj) == 1.0  # y-deviation
        
        smooth_obj = next(o for o in optimizer.objectives if o.name == "smoothness")
        assert smooth_obj.cost_function is not None
        assert smooth_obj.cost_function(traj) > 0

    def test_setup_standard_golf_constraints(self, optimizer: DrakeMotionOptimizer) -> None:
        """Test setting up standard golf constraints."""
        optimizer.setup_standard_golf_constraints()
        assert len(optimizer.constraints) == 2
        names = {con.name for con in optimizer.constraints}
        assert names == {"joint_limits", "impact_timing"}
        
        traj = np.array([[0.0, 0.0, 0.0]])
        for con in optimizer.constraints:
            assert con.constraint_function is not None
            assert con.constraint_function(traj) == 0.0


class TestDrakeMotionOptimizerOptimization:
    """Tests for optimization methods."""

    def test_build_total_cost_function(self, optimizer: DrakeMotionOptimizer) -> None:
        """Test building the total cost function."""
        optimizer.add_objective("obj1", 2.0, lambda x: np.sum(x))
        optimizer.add_objective("obj2", 1.0, lambda x: np.sum(x**2))
        
        shape = (2, 2)
        total_cost_fn = optimizer._build_total_cost_function(shape)
        
        flat_x = np.array([1.0, 2.0, 3.0, 4.0])
        cost = total_cost_fn(flat_x)
        
        # obj1 = 2.0 * (1+2+3+4) = 20.0
        # obj2 = 1.0 * (1+4+9+16) = 30.0
        # total = 50.0
        assert cost == 50.0

    def test_build_scipy_constraints(self, optimizer: DrakeMotionOptimizer) -> None:
        """Test building scipy constraints dictionaries."""
        optimizer.add_constraint("eq_con", "equality", lambda x: np.sum(x))
        optimizer.add_constraint(
            "ineq_upper", "inequality", lambda x: x[0, 0], upper_bound=5.0
        )
        optimizer.add_constraint(
            "ineq_lower", "inequality", lambda x: x[0, 0], lower_bound=1.0
        )
        optimizer.add_constraint(
            "ineq_both", "inequality", lambda x: x[0, 0], lower_bound=0.0, upper_bound=10.0
        )
        # Constraint with no function should be skipped
        optimizer.constraints.append(
            OptimizationConstraint(name="empty", constraint_type="equality")
        )
        
        shape = (1, 1)
        scipy_cons = optimizer._build_scipy_constraints(shape)
        
        assert len(scipy_cons) == 5  # 1 eq + 1 ineq upper + 1 ineq lower + 2 for both
        
        flat_x = np.array([2.0])
        
        # Test eq
        assert scipy_cons[0]["type"] == "eq"
        assert scipy_cons[0]["fun"](flat_x) == 2.0
        
        # Test ineq upper (5.0 - x)
        assert scipy_cons[1]["type"] == "ineq"
        assert scipy_cons[1]["fun"](flat_x) == 3.0
        
        # Test ineq lower (x - 1.0)
        assert scipy_cons[2]["type"] == "ineq"
        assert scipy_cons[2]["fun"](flat_x) == 1.0

    @patch("scipy.optimize.minimize")
    def test_optimize_trajectory(
        self,
        mock_minimize: MagicMock,
        optimizer: DrakeMotionOptimizer,
        initial_trajectory: np.ndarray,
    ) -> None:
        """Test optimize_trajectory calls scipy minimize correctly."""
        # Setup mock minimize result
        mock_result = MagicMock()
        mock_result.x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        mock_result.success = True
        mock_result.fun = 10.5
        mock_result.nit = 5
        mock_result.message = "Optimization terminated successfully."
        mock_minimize.return_value = mock_result
        
        optimizer.add_objective("obj1", 1.0, lambda x: 0.0)
        optimizer.add_constraint("con1", "equality", lambda x: 0.0)
        
        result = optimizer.optimize_trajectory(initial_trajectory)
        
        assert result.success is True
        assert result.optimal_cost == 10.5
        assert result.iterations == 5
        assert result.convergence_message == "Optimization terminated successfully."
        assert result.optimal_trajectory.shape == initial_trajectory.shape
        np.testing.assert_array_equal(
            result.optimal_trajectory,
            np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        )
        
        mock_minimize.assert_called_once()

    def test_optimize_trajectory_none_initial(self, optimizer: DrakeMotionOptimizer) -> None:
        """Test optimize_trajectory with None initial trajectory raises ValueError."""
        with pytest.raises(ValueError, match="initial_trajectory must be provided"):
            optimizer.optimize_trajectory(None)  # type: ignore

    @patch.object(DrakeMotionOptimizer, "optimize_trajectory")
    def test_optimize_for_distance(
        self,
        mock_optimize: MagicMock,
        optimizer: DrakeMotionOptimizer,
        initial_trajectory: np.ndarray,
    ) -> None:
        """Test optimize_for_distance sets up objectives correctly."""
        mock_result = MagicMock(spec=OptimizationResult)
        mock_optimize.return_value = mock_result
        
        result = optimizer.optimize_for_distance(initial_trajectory, target_distance=300.0)
        
        assert result is mock_result
        assert len(optimizer.objectives) == 1
        assert optimizer.objectives[0].name == "carry_distance"
        assert optimizer.objectives[0].target_value == 300.0
        assert optimizer.objectives[0].cost_function is not None
        assert optimizer.objectives[0].cost_function(initial_trajectory) == -300.0

    @patch.object(DrakeMotionOptimizer, "optimize_trajectory")
    def test_optimize_for_accuracy(
        self,
        mock_optimize: MagicMock,
        optimizer: DrakeMotionOptimizer,
        initial_trajectory: np.ndarray,
    ) -> None:
        """Test optimize_for_accuracy sets up objectives correctly."""
        mock_result = MagicMock(spec=OptimizationResult)
        mock_optimize.return_value = mock_result
        
        target = np.array([5.0, 0.0, 0.0])
        result = optimizer.optimize_for_accuracy(initial_trajectory, target_point=target)
        
        assert result is mock_result
        assert len(optimizer.objectives) == 1
        assert optimizer.objectives[0].name == "target_accuracy"
        assert optimizer.objectives[0].cost_function is not None
        
        # Test the cost function (distance from final position to target)
        # initial_trajectory final pos is [2.0, 0.0, 0.0]
        # target is [5.0, 0.0, 0.0]
        # distance = 3.0
        assert optimizer.objectives[0].cost_function(initial_trajectory) == 3.0

    def test_export_optimization_results(
        self, optimizer: DrakeMotionOptimizer, tmp_path: Path
    ) -> None:
        """Test exporting optimization results to JSON."""
        result = OptimizationResult(
            success=True,
            optimal_trajectory=np.array([[1.0, 2.0], [3.0, 4.0]]),
            optimal_cost=42.0,
            iterations=10,
            convergence_message="Success",
            objective_values={"obj1": 42.0},
            constraint_violations={"con1": 0.0},
        )
        
        optimizer.add_objective("obj1", 1.0, lambda x: 0.0)
        
        output_path = tmp_path / "results.json"
        optimizer.export_optimization_results(result, str(output_path))
        
        assert output_path.exists()
        
        with open(output_path) as f:
            data = json.load(f)
            
        assert data["optimization_result"]["success"] is True
        assert data["optimization_result"]["optimal_cost"] == 42.0
        assert data["trajectory"]["num_points"] == 2
        assert data["engine"] == "drake"
        assert data["optimization_setup"]["objective_names"] == ["obj1"]
