"""Tests for energy conservation monitoring.

Tests the energy drift detection system that implements Guideline O3
for ensuring physical validity of conservative system integrations.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.energy_monitor import (
    ENERGY_DRIFT_CRITICAL_PCT,
    ENERGY_DRIFT_TOLERANCE_PCT,
    ConservationMonitor,
    EnergySnapshot,
    IntegrationFailureError,
)
from shared.python.tests.mock_physics_engine import (
    MockPhysicsEngine,
    as_physics_engine,
)


class TestEnergySnapshot:
    """Test EnergySnapshot dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        snapshot = EnergySnapshot(time=1.0, kinetic=5.0, potential=10.0)
        assert snapshot.time == 1.0
        assert snapshot.kinetic == 5.0
        assert snapshot.potential == 10.0

    def test_total_energy_property(self):
        """Test that total property returns KE + PE."""
        snapshot = EnergySnapshot(time=0.0, kinetic=3.0, potential=7.0)
        assert snapshot.total == 10.0

    def test_total_energy_with_negative_potential(self):
        """Test total energy with negative potential energy."""
        snapshot = EnergySnapshot(time=0.0, kinetic=15.0, potential=-5.0)
        assert snapshot.total == 10.0

    def test_zero_energy(self):
        """Test with zero energy components."""
        snapshot = EnergySnapshot(time=0.0, kinetic=0.0, potential=0.0)
        assert snapshot.total == 0.0

    def test_total_is_computed_property(self):
        """Test that total is a computed property, not stored."""
        snapshot = EnergySnapshot(time=0.0, kinetic=5.0, potential=5.0)
        assert snapshot.total == 10.0

        # Modify kinetic energy
        snapshot.kinetic = 8.0
        # Total should update automatically
        assert snapshot.total == 13.0


class TestConservationMonitorInitialization:
    """Test ConservationMonitor initialization."""

    def test_basic_initialization(self):
        """Test basic monitor initialization."""
        engine = MockPhysicsEngine()
        monitor = ConservationMonitor(as_physics_engine(engine))

        assert monitor.engine is engine
        assert monitor.E_initial is None
        assert len(monitor.drift_history) == 0
        assert monitor.max_drift_pct == ENERGY_DRIFT_TOLERANCE_PCT
        assert monitor.critical_drift_pct == ENERGY_DRIFT_CRITICAL_PCT

    def test_custom_drift_thresholds(self):
        """Test initialization with custom drift thresholds."""
        engine = MockPhysicsEngine()
        monitor = ConservationMonitor(
            as_physics_engine(engine),
            max_drift_pct=0.5,
            critical_drift_pct=2.0,
        )

        assert monitor.max_drift_pct == 0.5
        assert monitor.critical_drift_pct == 2.0

    def test_default_tolerance_values(self):
        """Test that default tolerance values match Guideline O3."""
        engine = MockPhysicsEngine()
        monitor = ConservationMonitor(as_physics_engine(engine))

        # Per Guideline O3
        assert monitor.max_drift_pct == 1.0  # 1% warning threshold
        assert monitor.critical_drift_pct == 5.0  # 5% critical threshold


class TestMonitorInitialize:
    """Test ConservationMonitor.initialize() method."""

    def test_initialize_sets_initial_energy(self):
        """Test that initialize() sets E_initial."""
        from shared.python.constants import GRAVITY_M_S2

        engine = MockPhysicsEngine()
        engine.set_state(q=np.array([1.0, 2.0]), v=np.array([0.5, 0.5]))
        engine.set_mass_matrix(np.eye(2))
        engine.set_gravity_forces(np.array([0.0, -GRAVITY_M_S2]))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        assert monitor.E_initial is not None
        assert isinstance(monitor.E_initial, float)

    def test_initialize_clears_drift_history(self):
        """Test that initialize() clears drift history."""
        engine = MockPhysicsEngine()
        engine.set_state(np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        monitor = ConservationMonitor(as_physics_engine(engine))

        # Add some fake history
        monitor.drift_history.append((0.0, 0.5))
        monitor.drift_history.append((1.0, 1.0))

        # Initialize should clear it
        monitor.initialize()
        assert len(monitor.drift_history) == 0

    def test_initialize_computes_correct_energy(self):
        """Test that initialize() computes energy correctly."""
        engine = MockPhysicsEngine()

        # Set up simple state: KE = 0.5 * m * v^2
        m = 2.0
        v_val = 3.0
        engine.set_state(q=np.array([0.0]), v=np.array([v_val]))
        engine.set_mass_matrix(np.array([[m]]))
        engine.set_gravity_forces(np.array([0.0]))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # KE = 0.5 * m * v^2 = 0.5 * 2.0 * 3.0^2 = 9.0 J
        expected_KE = 0.5 * m * v_val**2
        assert monitor.E_initial is not None
        np.testing.assert_allclose(monitor.E_initial, expected_KE, rtol=1e-10)


class TestGetEnergySnapshot:
    """Test ConservationMonitor.get_energy_snapshot() method."""

    def test_snapshot_captures_current_time(self):
        """Test that snapshot captures current simulation time."""
        engine = MockPhysicsEngine()
        engine.time = 5.5
        engine.set_state(np.array([0.0, 0.0]), np.array([0.0, 0.0]))

        monitor = ConservationMonitor(as_physics_engine(engine))
        snapshot = monitor.get_energy_snapshot()

        assert snapshot.time == 5.5

    def test_snapshot_computes_kinetic_energy(self):
        """Test kinetic energy computation: KE = 0.5 * v^T * M * v."""
        engine = MockPhysicsEngine()

        # Simple case: 1D, m=2.0, v=3.0 -> KE = 0.5 * 2.0 * 3.0^2 = 9.0
        engine.set_state(q=np.array([0.0]), v=np.array([3.0]))
        engine.set_mass_matrix(np.array([[2.0]]))
        engine.set_gravity_forces(np.array([0.0]))

        monitor = ConservationMonitor(as_physics_engine(engine))
        snapshot = monitor.get_energy_snapshot()

        expected_KE = 0.5 * 2.0 * 3.0**2
        np.testing.assert_allclose(snapshot.kinetic, expected_KE, rtol=1e-10)

    def test_snapshot_computes_potential_energy(self):
        """Test potential energy computation: PE = -q^T * g."""
        from shared.python.constants import GRAVITY_M_S2

        engine = MockPhysicsEngine()

        # q = [1.0], g = [-GRAVITY_M_S2] -> PE = -1.0 * (-GRAVITY_M_S2) = GRAVITY_M_S2
        engine.set_state(q=np.array([1.0]), v=np.array([0.0]))
        engine.set_mass_matrix(np.array([[1.0]]))
        engine.set_gravity_forces(np.array([-GRAVITY_M_S2]))

        monitor = ConservationMonitor(as_physics_engine(engine))
        snapshot = monitor.get_energy_snapshot()

        expected_PE = -1.0 * (-GRAVITY_M_S2)
        np.testing.assert_allclose(snapshot.potential, expected_PE, rtol=1e-10)

    def test_snapshot_with_multidof_system(self):
        """Test energy computation for multi-DOF system."""
        from shared.python.constants import GRAVITY_M_S2

        engine = MockPhysicsEngine(n_dof=3)

        q = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, 1.0, 1.5])
        M = np.diag([1.0, 2.0, 3.0])
        g = np.array([-GRAVITY_M_S2, -GRAVITY_M_S2, -GRAVITY_M_S2])

        engine.set_state(q, v)
        engine.set_mass_matrix(M)
        engine.set_gravity_forces(g)

        monitor = ConservationMonitor(as_physics_engine(engine))
        snapshot = monitor.get_energy_snapshot()

        # KE = 0.5 * v^T * M * v
        # = 0.5 * (0.5^2 * 1.0 + 1.0^2 * 2.0 + 1.5^2 * 3.0)
        # = 0.5 * (0.25 + 2.0 + 6.75) = 0.5 * 9.0 = 4.5
        expected_KE = 0.5 * (v * M.diagonal() * v).sum()

        # PE = -q^T * g = -(1.0 * -GRAVITY_M_S2 + 2.0 * -GRAVITY_M_S2 + 3.0 * -GRAVITY_M_S2)
        # = -(-GRAVITY_M_S2 - 2*GRAVITY_M_S2 - 3*GRAVITY_M_S2) = 6*GRAVITY_M_S2
        expected_PE = -np.dot(q, g)

        np.testing.assert_allclose(snapshot.kinetic, expected_KE, rtol=1e-10)
        np.testing.assert_allclose(snapshot.potential, expected_PE, rtol=1e-10)


class TestCheckAndWarn:
    """Test ConservationMonitor.check_and_warn() method."""

    def test_requires_initialization(self):
        """Test that check_and_warn() requires initialization first."""
        engine = MockPhysicsEngine()
        monitor = ConservationMonitor(as_physics_engine(engine))

        with pytest.raises(RuntimeError, match="not initialized"):
            monitor.check_and_warn()

    def test_zero_drift_returns_zero(self):
        """Test that zero drift returns 0.0%."""
        engine = MockPhysicsEngine()
        engine.set_state(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        engine.set_mass_matrix(np.eye(2))
        engine.set_gravity_forces(np.zeros(2))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # No change in state
        drift_pct = monitor.check_and_warn()

        np.testing.assert_allclose(drift_pct, 0.0, atol=1e-10)

    def test_drift_calculation_positive(self):
        """Test drift calculation when energy increases."""
        engine = MockPhysicsEngine()

        # Initial: v = 1.0, KE = 0.5 * 1.0^2 = 0.5 J
        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()  # E_initial = 0.5 J

        # Change: v = 1.02, KE = 0.5 * 1.02^2 = 0.5202 J (small increase, < 5%)
        v_new = 1.02
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        drift_pct = monitor.check_and_warn()

        # Drift = (0.5202 - 0.5) / 0.5 * 100 = 4.04%
        E_new = 0.5 * v_new**2
        expected_drift = (E_new - 0.5) / 0.5 * 100
        np.testing.assert_allclose(drift_pct, expected_drift, rtol=1e-6)

    def test_drift_calculation_negative(self):
        """Test drift calculation when energy decreases."""
        engine = MockPhysicsEngine()

        # Initial: v = 2.0, KE = 0.5 * 2.0^2 = 2.0 J
        engine.set_state(q=np.array([0.0]), v=np.array([2.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Change: v = 1.96, KE = 0.5 * 1.96^2 = 1.9208 J (small decrease, < 5%)
        v_new = 1.96
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        drift_pct = monitor.check_and_warn()

        # Drift = (1.9208 - 2.0) / 2.0 * 100 = -3.96%
        E_new = 0.5 * v_new**2
        expected_drift = (E_new - 2.0) / 2.0 * 100
        np.testing.assert_allclose(drift_pct, expected_drift, rtol=1e-6)

    def test_drift_history_accumulation(self):
        """Test that drift history is accumulated."""
        engine = MockPhysicsEngine()
        engine.set_state(np.array([0.0]), np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Multiple checks
        engine.advance_time(1.0)
        monitor.check_and_warn()

        engine.advance_time(1.0)
        monitor.check_and_warn()

        engine.advance_time(1.0)
        monitor.check_and_warn()

        # Should have 3 entries
        assert len(monitor.drift_history) == 3
        assert monitor.drift_history[0][0] == 1.0  # First time
        assert monitor.drift_history[1][0] == 2.0  # Second time
        assert monitor.drift_history[2][0] == 3.0  # Third time

    def test_warning_at_tolerance_threshold(self, caplog):
        """Test that warning is logged at 1% drift threshold."""
        engine = MockPhysicsEngine()

        # Initial: KE = 0.5 J
        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Create exactly 1.1% drift (just above threshold)
        # E_new = E_initial * 1.011 = 0.5 * 1.011 = 0.5055 J
        # v_new^2 = 2 * E_new = 1.011 -> v_new = sqrt(1.011) ≈ 1.00548
        v_new = np.sqrt(1.011)
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        with caplog.at_level("WARNING"):
            drift_pct = monitor.check_and_warn()

        # Should warn because drift > 1%
        assert drift_pct > 1.0
        assert (
            "Energy conservation violated" in caplog.text
            or "conservation" in caplog.text.lower()
        )

    def test_critical_error_at_5_percent_drift(self):
        """Test that IntegrationFailureError is raised at 5% drift."""
        engine = MockPhysicsEngine()

        # Initial: KE = 0.5 J
        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Create 6% drift (above critical threshold)
        # E_new = E_initial * 1.06 = 0.5 * 1.06 = 0.53 J
        # v_new = sqrt(2 * 0.53) = sqrt(1.06)
        v_new = np.sqrt(1.06)
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        with pytest.raises(IntegrationFailureError, match="INTEGRATION FAILURE"):
            monitor.check_and_warn()

    def test_no_warning_below_tolerance(self, caplog):
        """Test that no warning is logged below 1% drift."""
        engine = MockPhysicsEngine()

        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Create 0.5% drift (below threshold)
        v_new = np.sqrt(1.005)
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        with caplog.at_level("WARNING"):
            drift_pct = monitor.check_and_warn()

        # Should not warn
        assert drift_pct < 1.0
        # Check no warning in logs (might have other logs, so check specifically)
        energy_warnings = [
            record
            for record in caplog.records
            if "energy conservation violated" in record.message.lower()
        ]
        assert len(energy_warnings) == 0

    def test_negative_drift_triggers_warning(self, caplog):
        """Test that negative drift (energy loss) also triggers warning."""
        engine = MockPhysicsEngine()

        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Create -1.5% drift
        v_new = np.sqrt(0.985)  # E = 0.985 * E_initial
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        with caplog.at_level("WARNING"):
            drift_pct = monitor.check_and_warn()

        assert drift_pct < -1.0
        assert (
            "Energy conservation violated" in caplog.text
            or "conservation" in caplog.text.lower()
        )


class TestEstimateMaxStableTimestep:
    """Test estimate_max_stable_timestep() method."""

    def test_slow_motion_recommendation(self):
        """Test timestep recommendation for slow motion."""
        engine = MockPhysicsEngine()
        engine.set_state(q=np.array([0.0, 0.0]), v=np.array([0.1, 0.2]))  # ||v|| < 1.0

        monitor = ConservationMonitor(as_physics_engine(engine))
        dt_max = monitor.estimate_max_stable_timestep()

        # For slow motion, should recommend dt = 0.01s
        assert dt_max == 0.01

    def test_normal_motion_recommendation(self):
        """Test timestep recommendation for normal motion."""
        engine = MockPhysicsEngine()
        engine.set_state(
            q=np.array([0.0, 0.0]), v=np.array([3.0, 4.0])  # ||v|| = 5.0, in [1, 10)
        )

        monitor = ConservationMonitor(as_physics_engine(engine))
        dt_max = monitor.estimate_max_stable_timestep()

        # For normal motion, should recommend dt = 0.001s
        assert dt_max == 0.001

    def test_high_speed_motion_recommendation(self):
        """Test timestep recommendation for high-speed motion."""
        engine = MockPhysicsEngine()
        engine.set_state(
            q=np.array([0.0, 0.0, 0.0]),
            v=np.array([50.0, 50.0, 50.0]),  # ||v|| ~ 86.6, >> 10
        )

        monitor = ConservationMonitor(as_physics_engine(engine))
        dt_max = monitor.estimate_max_stable_timestep()

        # For high-speed motion, should recommend dt = 0.0001s
        assert dt_max == 0.0001

    def test_zero_velocity(self):
        """Test timestep recommendation with zero velocity."""
        engine = MockPhysicsEngine()
        engine.set_state(q=np.array([0.0, 0.0]), v=np.array([0.0, 0.0]))

        monitor = ConservationMonitor(as_physics_engine(engine))
        dt_max = monitor.estimate_max_stable_timestep()

        # Zero velocity -> slow motion regime
        assert dt_max == 0.01


class TestProjectToEnergyManifold:
    """Test project_to_energy_manifold() method."""

    def test_requires_initialization(self):
        """Test that projection requires initialization first."""
        engine = MockPhysicsEngine()
        monitor = ConservationMonitor(as_physics_engine(engine))

        with pytest.raises(RuntimeError, match="not initialized"):
            monitor.project_to_energy_manifold()

    def test_projection_scales_velocity(self):
        """Test that projection scales velocity to restore energy."""
        engine = MockPhysicsEngine()

        # Initial: v = 1.0, E = 0.5
        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()  # E_initial = 0.5

        # Perturb: v = 1.5, E = 1.125 (2.25x increase)
        engine.set_state(q=np.array([0.0]), v=np.array([1.5]))

        # Project back
        monitor.project_to_energy_manifold()

        # Check that energy is restored
        q, v = engine.get_state()
        E_restored = 0.5 * v[0] ** 2

        np.testing.assert_allclose(E_restored, 0.5, rtol=1e-6)

    def test_projection_does_not_change_position(self):
        """Test that projection only changes velocity, not position."""
        engine = MockPhysicsEngine()

        q_initial = np.array([1.5, 2.5])
        v_initial = np.array([1.0, 1.0])

        engine.set_state(q_initial, v_initial)
        engine.set_mass_matrix(np.eye(2))
        engine.set_gravity_forces(np.zeros(2))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Perturb velocity
        engine.set_state(q_initial, v_initial * 1.2)

        # Project
        monitor.project_to_energy_manifold()

        # Check position unchanged
        q, v = engine.get_state()
        np.testing.assert_allclose(q, q_initial, rtol=1e-10)

    def test_projection_with_near_zero_energy(self, caplog):
        """Test projection behavior when current energy is near zero."""
        engine = MockPhysicsEngine()

        # Initial energy
        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Set energy to near zero
        engine.set_state(q=np.array([0.0]), v=np.array([1e-10]))

        with caplog.at_level("WARNING"):
            monitor.project_to_energy_manifold()

        # Should warn about inability to project
        assert "Cannot project to energy manifold" in caplog.text

    def test_projection_preserves_direction(self):
        """Test that projection preserves velocity direction (only scales magnitude)."""
        engine = MockPhysicsEngine(n_dof=2)

        v_initial = np.array([3.0, 4.0])  # ||v|| = 5.0
        engine.set_state(q=np.zeros(2), v=v_initial)
        engine.set_mass_matrix(np.eye(2))
        engine.set_gravity_forces(np.zeros(2))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Perturb magnitude
        engine.set_state(q=np.zeros(2), v=v_initial * 1.3)

        # Project
        monitor.project_to_energy_manifold()

        # Check direction preserved
        q, v_projected = engine.get_state()
        v_initial_normalized = v_initial / np.linalg.norm(v_initial)
        v_projected_normalized = v_projected / np.linalg.norm(v_projected)

        np.testing.assert_allclose(
            v_projected_normalized,
            v_initial_normalized,
            rtol=1e-6,
            err_msg="Projection should preserve velocity direction",
        )


class TestIntegrationFailureError:
    """Test IntegrationFailureError exception."""

    def test_exception_is_raised(self):
        """Test that IntegrationFailureError can be raised."""
        with pytest.raises(IntegrationFailureError):
            raise IntegrationFailureError("Test error")

    def test_exception_inherits_from_exception(self):
        """Test that IntegrationFailureError is an Exception."""
        assert issubclass(IntegrationFailureError, Exception)

    def test_exception_message(self):
        """Test that exception message is preserved."""
        msg = "Critical energy drift detected"
        try:
            raise IntegrationFailureError(msg)
        except IntegrationFailureError as e:
            assert str(e) == msg


class TestPhysicalRealism:
    """Test physical realism of energy monitoring."""

    def test_conservative_system_maintains_energy(self):
        """Test that a true conservative system shows zero drift."""
        engine = MockPhysicsEngine()

        # Set up conservative system (no external forces, no damping)
        engine.set_state(q=np.array([1.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Simulate conservation (no actual dynamics, just checking)
        # In real simulation, energy would be conserved
        drift_pct = monitor.check_and_warn()

        # Should have zero drift
        np.testing.assert_allclose(drift_pct, 0.0, atol=1e-10)

    def test_drift_detection_sensitivity(self):
        """Test that monitor detects small energy changes."""
        engine = MockPhysicsEngine()

        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()  # E = 0.5 J

        # Create tiny 0.1% drift
        v_new = np.sqrt(1.001)
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        drift_pct = monitor.check_and_warn()

        # Should detect this small drift
        assert 0.09 < drift_pct < 0.11  # ~0.1%


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_initial_energy(self):
        """Test behavior with zero initial energy.

        Note: This is a pathological case. Division by zero occurs when
        calculating drift percentage. This test documents the current behavior.
        """
        engine = MockPhysicsEngine()

        # Zero energy state
        engine.set_state(q=np.array([0.0]), v=np.array([0.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Drift calculation with zero denominator causes division by zero
        # Perturb slightly
        engine.set_state(q=np.array([0.0]), v=np.array([0.01]))

        # Current implementation raises ZeroDivisionError for zero initial energy
        # This is acceptable as it's a pathological case (no energy to conserve)
        with pytest.raises(ZeroDivisionError):
            monitor.check_and_warn()

    def test_very_large_energy_drift(self):
        """Test behavior with extremely large drift."""
        engine = MockPhysicsEngine()

        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Create 1000% drift
        v_new = np.sqrt(10.0)  # 10x energy
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        # Should raise critical error
        with pytest.raises(IntegrationFailureError):
            monitor.check_and_warn()

    def test_negative_energy_total(self):
        """Test with negative total energy (PE dominates)."""
        engine = MockPhysicsEngine()

        # Large negative potential, small kinetic
        # KE = 0.5, PE = -10.0, Total = -9.5
        engine.set_state(q=np.array([10.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.array([-1.0]))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Drift calculation should work with negative energy
        drift_pct = monitor.check_and_warn()

        assert np.isfinite(drift_pct)

    def test_multiple_initializations(self):
        """Test that re-initialization resets the monitor."""
        engine = MockPhysicsEngine()

        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()
        E_first = monitor.E_initial

        # Add some drift history
        monitor.check_and_warn()
        assert len(monitor.drift_history) == 1

        # Change state and re-initialize
        engine.set_state(q=np.array([0.0]), v=np.array([2.0]))
        monitor.initialize()
        E_second = monitor.E_initial

        # Energy should be different
        assert E_second != E_first
        # History should be cleared
        assert len(monitor.drift_history) == 0
