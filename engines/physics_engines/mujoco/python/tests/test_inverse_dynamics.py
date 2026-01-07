"""Comprehensive tests for inverse dynamics module."""

import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.inverse_dynamics import (
    ForceDecomposition,
    InverseDynamicsAnalyzer,
    InverseDynamicsResult,
    InverseDynamicsSolver,
    RecursiveNewtonEuler,
    export_inverse_dynamics_to_csv,
)
from mujoco_humanoid_golf.models import DOUBLE_PENDULUM_XML


class TestInverseDynamicsResult:
    """Tests for InverseDynamicsResult dataclass."""

    def test_initialization(self) -> None:
        """Test result initialization."""
        torques = np.array([1.0, -0.5])
        result = InverseDynamicsResult(joint_torques=torques)

        np.testing.assert_array_equal(result.joint_torques, torques)
        assert result.constraint_forces is None
        assert result.is_feasible is True
        assert result.residual_norm == 0.0

    def test_initialization_with_decomposition(self) -> None:
        """Test initialization with force decomposition."""
        torques = np.array([1.0, -0.5])
        inertial = np.array([0.8, -0.3])
        coriolis = np.array([0.1, -0.1])
        gravity = np.array([0.1, -0.1])

        result = InverseDynamicsResult(
            joint_torques=torques,
            inertial_torques=inertial,
            coriolis_torques=coriolis,
            gravity_torques=gravity,
        )

        np.testing.assert_array_equal(result.inertial_torques, inertial)
        np.testing.assert_array_equal(result.coriolis_torques, coriolis)
        np.testing.assert_array_equal(result.gravity_torques, gravity)


class TestForceDecomposition:
    """Tests for ForceDecomposition dataclass."""

    def test_initialization(self) -> None:
        """Test force decomposition initialization."""
        total = np.array([1.0, -0.5])
        inertial = np.array([0.8, -0.3])
        coriolis = np.array([0.1, -0.1])
        centrifugal = np.array([0.05, -0.05])
        gravity = np.array([0.05, -0.05])

        decomp = ForceDecomposition(
            total=total,
            inertial=inertial,
            coriolis=coriolis,
            centrifugal=centrifugal,
            gravity=gravity,
        )

        np.testing.assert_array_equal(decomp.total, total)
        np.testing.assert_array_equal(decomp.inertial, inertial)


class TestInverseDynamicsSolver:
    """Tests for InverseDynamicsSolver class."""

    @pytest.fixture()
    def model_and_data(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Create model and data for testing."""
        model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def test_initialization(self, model_and_data) -> None:
        """Test solver initialization."""
        model, data = model_and_data
        solver = InverseDynamicsSolver(model, data)

        assert solver.model == model
        assert solver.data == data
        assert isinstance(solver.has_constraints, bool)

    def test_detect_closed_chains(self, model_and_data) -> None:
        """Test closed chain detection."""
        model, data = model_and_data
        solver = InverseDynamicsSolver(model, data)

        has_chains = solver._detect_closed_chains()
        assert isinstance(has_chains, bool)

    def test_compute_required_torques(self, model_and_data) -> None:
        """Test computing required torques."""
        model, data = model_and_data
        solver = InverseDynamicsSolver(model, data)

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        qacc = np.zeros(model.nv)

        result = solver.compute_required_torques(qpos, qvel, qacc)

        assert isinstance(result, InverseDynamicsResult)
        assert result.joint_torques.shape == (model.nv,)
        assert np.all(np.isfinite(result.joint_torques))

    def test_compute_required_torques_with_external_forces(
        self,
        model_and_data,
    ) -> None:
        """Test computing torques with external forces."""
        model, data = model_and_data
        solver = InverseDynamicsSolver(model, data)

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        qacc = np.zeros(model.nv)
        external = np.array([0.5, -0.3])

        result = solver.compute_required_torques(
            qpos, qvel, qacc, external_forces=external
        )

        assert result.joint_torques.shape == (model.nv,)
        assert np.all(np.isfinite(result.joint_torques))

    def test_compute_required_torques_force_decomposition(
        self,
        model_and_data,
    ) -> None:
        """Test that force decomposition is computed."""
        model, data = model_and_data
        solver = InverseDynamicsSolver(model, data)

        qpos = data.qpos.copy()
        qvel = np.array([0.1, -0.05])
        qacc = np.array([0.01, -0.005])

        result = solver.compute_required_torques(qpos, qvel, qacc)

        # Check that decomposition components are present
        assert result.inertial_torques is not None
        assert result.coriolis_torques is not None
        assert result.gravity_torques is not None

        assert result.inertial_torques.shape == (model.nv,)
        assert result.coriolis_torques.shape == (model.nv,)
        assert result.gravity_torques.shape == (model.nv,)

    def test_solve_inverse_dynamics_trajectory(self, model_and_data) -> None:
        """Test solving inverse dynamics for trajectory."""
        model, data = model_and_data
        solver = InverseDynamicsSolver(model, data)

        times = np.array([0.0, 0.01, 0.02])
        positions = np.array([data.qpos.copy() for _ in range(3)])
        velocities = np.array([data.qvel.copy() for _ in range(3)])
        accelerations = np.zeros((3, model.nv))

        results = solver.solve_inverse_dynamics_trajectory(
            times,
            positions,
            velocities,
            accelerations,
        )

        assert len(results) == 3
        assert all(isinstance(r, InverseDynamicsResult) for r in results)

    def test_compute_partial_inverse_dynamics(self, model_and_data) -> None:
        """Test partial inverse dynamics for parallel mechanisms."""
        model, data = model_and_data
        solver = InverseDynamicsSolver(model, data)

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        qacc = np.zeros(model.nv)
        constrained_joints = [0]  # Constrain first joint

        result = solver.compute_partial_inverse_dynamics(
            qpos,
            qvel,
            qacc,
            constrained_joints,
        )

        assert isinstance(result, InverseDynamicsResult)
        assert result.joint_torques.shape == (model.nv,)

    def test_decompose_forces(self, model_and_data) -> None:
        """Test force decomposition."""
        model, data = model_and_data
        solver = InverseDynamicsSolver(model, data)

        qpos = data.qpos.copy()
        qvel = np.array([0.1, -0.05])
        qacc = np.array([0.01, -0.005])

        decomp = solver.decompose_forces(qpos, qvel, qacc)

        assert isinstance(decomp, ForceDecomposition)
        assert decomp.total.shape == (model.nv,)
        assert decomp.inertial.shape == (model.nv,)
        assert decomp.coriolis.shape == (model.nv,)
        assert decomp.centrifugal.shape == (model.nv,)
        assert decomp.gravity.shape == (model.nv,)

    def test_compute_end_effector_forces(self, model_and_data) -> None:
        """Test computing end-effector forces."""
        model, data = model_and_data
        solver = InverseDynamicsSolver(model, data)

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        qacc = np.zeros(model.nv)
        body_id = 1

        ee_force = solver.compute_end_effector_forces(qpos, qvel, qacc, body_id)

        assert ee_force.shape == (3,)
        assert np.all(np.isfinite(ee_force))

    def test_validate_solution(self, model_and_data) -> None:
        """Test solution validation."""
        model, data = model_and_data
        solver = InverseDynamicsSolver(model, data)

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        qacc = np.zeros(model.nv)

        result = solver.compute_required_torques(qpos, qvel, qacc)
        metrics = solver.validate_solution(qpos, qvel, qacc, result.joint_torques)

        assert "acceleration_error" in metrics
        assert "relative_error" in metrics
        assert "max_torque" in metrics
        assert "mean_torque" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_compute_actuator_efficiency(self, model_and_data) -> None:
        """Test actuator efficiency computation."""
        model, data = model_and_data
        solver = InverseDynamicsSolver(model, data)

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        qacc = np.zeros(model.nv)

        result = solver.compute_required_torques(qpos, qvel, qacc)
        efficiency = solver.compute_actuator_efficiency(result)

        assert "inertial_ratio" in efficiency
        assert "gravity_ratio" in efficiency
        assert "coriolis_ratio" in efficiency
        assert "efficiency_index" in efficiency
        assert all(isinstance(v, float) for v in efficiency.values())


class TestRecursiveNewtonEuler:
    """Tests for RecursiveNewtonEuler class."""

    @pytest.fixture()
    def model_and_data(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Create model and data for testing."""
        model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def test_initialization(self, model_and_data) -> None:
        """Test RNE initialization."""
        model, data = model_and_data
        rne = RecursiveNewtonEuler(model, data)

        assert rne.model == model
        assert rne.data == data

    def test_compute(self, model_and_data) -> None:
        """Test RNE computation."""
        model, data = model_and_data
        rne = RecursiveNewtonEuler(model, data)

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        qacc = np.zeros(model.nv)

        torques = rne.compute(qpos, qvel, qacc)

        assert torques.shape == (model.nv,)
        assert np.all(np.isfinite(torques))


class TestInverseDynamicsAnalyzer:
    """Tests for InverseDynamicsAnalyzer class."""

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
        analyzer = InverseDynamicsAnalyzer(model, data)

        assert analyzer.id_solver is not None
        assert analyzer.kin_analyzer is not None

    def test_analyze_captured_motion(self, model_and_data) -> None:
        """Test analyzing captured motion."""
        model, data = model_and_data
        analyzer = InverseDynamicsAnalyzer(model, data)

        times = np.array([0.0, 0.01, 0.02])
        positions = np.array([data.qpos.copy() for _ in range(3)])
        velocities = np.array([data.qvel.copy() for _ in range(3)])
        accelerations = np.zeros((3, model.nv))

        analysis = analyzer.analyze_captured_motion(
            times,
            positions,
            velocities,
            accelerations,
        )

        assert "kinematic_forces" in analysis
        assert "inverse_dynamics" in analysis
        assert "statistics" in analysis
        assert len(analysis["inverse_dynamics"]) == 3

    def test_compare_swings(self, model_and_data) -> None:
        """Test comparing two swings."""
        model, data = model_and_data
        analyzer = InverseDynamicsAnalyzer(model, data)

        times = np.array([0.0, 0.01])
        positions = np.array([data.qpos.copy() for _ in range(2)])
        velocities = np.array([data.qvel.copy() for _ in range(2)])
        accelerations = np.zeros((2, model.nv))

        swing1 = analyzer.analyze_captured_motion(
            times, positions, velocities, accelerations
        )
        swing2 = analyzer.analyze_captured_motion(
            times, positions, velocities, accelerations
        )

        comparison = analyzer.compare_swings(swing1, swing2)

        assert "coriolis_power_diff" in comparison
        assert "torque_diff" in comparison
        assert "duration_diff" in comparison


class TestExportFunctions:
    """Tests for export functions."""

    def test_export_inverse_dynamics_to_csv(self) -> None:
        """Test exporting inverse dynamics to CSV."""
        times = np.array([0.0, 0.01, 0.02])
        results = [
            InverseDynamicsResult(
                joint_torques=np.array([1.0, -0.5]),
                inertial_torques=np.array([0.8, -0.3]),
                coriolis_torques=np.array([0.1, -0.1]),
                gravity_torques=np.array([0.1, -0.1]),
            )
            for _ in range(3)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

        try:
            export_inverse_dynamics_to_csv(times, results, str(csv_path))

            assert csv_path.exists()
            content = csv_path.read_text()
            assert "time" in content
            assert "torque_0" in content

        finally:
            if csv_path.exists():
                csv_path.unlink()
