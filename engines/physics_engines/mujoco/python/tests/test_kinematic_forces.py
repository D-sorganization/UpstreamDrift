"""Comprehensive tests for kinematic forces module."""

import warnings
import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.kinematic_forces import (
    KinematicForceAnalyzer,
    KinematicForceData,
)
from mujoco_humanoid_golf.models import DOUBLE_PENDULUM_XML


class TestKinematicForceData:
    """Tests for KinematicForceData dataclass."""

    def test_initialization(self) -> None:
        """Test force data initialization."""
        coriolis = np.array([1.0, -0.5])
        gravity = np.array([0.5, -0.3])

        data = KinematicForceData(
            time=1.0,
            coriolis_forces=coriolis,
            gravity_forces=gravity,
        )

        assert data.time == 1.0
        np.testing.assert_array_equal(data.coriolis_forces, coriolis)
        np.testing.assert_array_equal(data.gravity_forces, gravity)
        assert data.coriolis_power == 0.0


class TestKinematicForceAnalyzer:
    """Tests for KinematicForceAnalyzer class."""

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
        analyzer = KinematicForceAnalyzer(model, data)

        assert analyzer.model == model
        assert analyzer.data == data

    def test_find_body_id(self, model_and_data) -> None:
        """Test finding body ID."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        body_id = analyzer._find_body_id("shoulder")
        if body_id is not None:
            assert body_id > 0
            assert body_id < model.nbody

        # Should return None for nonexistent body
        body_id = analyzer._find_body_id("nonexistent_body_xyz")
        assert body_id is None

    def test_compute_coriolis_forces(self, model_and_data) -> None:
        """Test computing Coriolis forces."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()
        qvel = np.array([0.1, -0.05])

        coriolis = analyzer.compute_coriolis_forces(qpos, qvel)

        assert coriolis.shape == (model.nv,)
        assert np.all(np.isfinite(coriolis))

    def test_compute_coriolis_forces_zero_velocity(self, model_and_data) -> None:
        """Test Coriolis forces with zero velocity."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()
        qvel = np.zeros(model.nv)

        coriolis = analyzer.compute_coriolis_forces(qpos, qvel)

        # With zero velocity, Coriolis should be approximately zero
        assert coriolis.shape == (model.nv,)
        # May have small numerical errors
        assert np.all(np.abs(coriolis) < 1e-3)

    def test_compute_gravity_forces(self, model_and_data) -> None:
        """Test computing gravity forces."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()

        gravity = analyzer.compute_gravity_forces(qpos)

        assert gravity.shape == (model.nv,)
        assert np.all(np.isfinite(gravity))

    def test_decompose_coriolis_forces(self, model_and_data) -> None:
        """Test decomposing Coriolis forces."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()
        qvel = np.array([0.1, -0.05])

        centrifugal, coupling = analyzer.decompose_coriolis_forces(qpos, qvel)

        assert centrifugal.shape == (model.nv,)
        assert coupling.shape == (model.nv,)
        assert np.all(np.isfinite(centrifugal))
        assert np.all(np.isfinite(coupling))

    def test_compute_mass_matrix(self, model_and_data) -> None:
        """Test computing mass matrix."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()

        M = analyzer.compute_mass_matrix(qpos)

        assert M.shape == (model.nv, model.nv)
        assert np.allclose(M, M.T)  # Symmetric
        assert np.all(np.linalg.eigvals(M) > 0)  # Positive definite
        assert np.all(np.isfinite(M))

    def test_compute_club_head_apparent_forces(self, model_and_data) -> None:
        """Test computing club head apparent forces."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()
        qvel = np.array([0.1, -0.05])
        qacc = np.zeros(model.nv)  # qacc needed for apparent force

        # May not have club head in simple model
        if analyzer.club_head_id is not None:
            coriolis, centrifugal, apparent = (
                analyzer.compute_club_head_apparent_forces(qpos, qvel, qacc)
            )

            assert coriolis.shape == (3,)
            assert centrifugal.shape == (3,)
            assert apparent.shape == (3,)

            # Check finiteness
            assert np.all(np.isfinite(coriolis))
            assert np.all(np.isfinite(centrifugal))
            assert np.all(np.isfinite(apparent))

    def test_analyze_trajectory(self, model_and_data) -> None:
        """Test analyzing trajectory."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        times = np.array([0.0, 0.01, 0.02])
        positions = np.array([data.qpos.copy() for _ in range(3)])
        velocities = np.array([data.qvel.copy() for _ in range(3)])
        accelerations = np.zeros((3, model.nv))

        results = analyzer.analyze_trajectory(
            times, positions, velocities, accelerations
        )

        assert len(results) == 3
        assert all(isinstance(r, KinematicForceData) for r in results)
        assert all(r.time == t for r, t in zip(results, times, strict=False))

    def test_compute_effective_mass(self, model_and_data) -> None:
        """Test computing effective mass."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        # Set a non-zero configuration to avoid singularities
        qpos = np.array([0.5, 0.5])  # Arbitrary non-zero angles
        direction = np.array([0.0, 1.0, 0.0])  # Y direction (tangential movement)

        body_id = analyzer._find_body_id("club_body")
        assert body_id is not None, "club_body not found in model"

        # Filter runtime warning for rank deficient Jacobian (planar robot in 3D world)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="Jacobian is rank deficient"
            )
            m_eff = analyzer.compute_effective_mass(qpos, direction, body_id)

        assert isinstance(m_eff, float)
        assert m_eff >= 0.0
        assert np.isfinite(m_eff)

    def test_compute_centripetal_acceleration(self, model_and_data) -> None:
        """Test computing centripetal acceleration.

        This function is deprecated and should raise NotImplementedError.
        """
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()
        qvel = np.array([0.1, -0.05])
        body_id = 1

        with pytest.raises(NotImplementedError, match="fundamental physics error"):
            analyzer.compute_centripetal_acceleration(qpos, qvel, body_id)

    def test_compute_kinematic_power(self, model_and_data) -> None:
        """Test computing kinematic power."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()
        qvel = np.array([0.1, -0.05])

        power_data = analyzer.compute_kinematic_power(qpos, qvel)

        assert "coriolis_power" in power_data
        assert "centrifugal_power" in power_data
        assert all(isinstance(v, float) for v in power_data.values())
