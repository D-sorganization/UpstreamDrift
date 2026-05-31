"""Tests for Ground Reaction Force Analysis.

Guideline E5 implementation tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from src.shared.python.physics.ground_reaction_forces import (
    FootSide,
    GRFAnalyzer,
    GRFTimeSeries,
    compute_angular_impulse,
    compute_cop_from_grf,
    compute_cop_trajectory_length,
    compute_linear_impulse,
    extract_grf_from_contacts,
    validate_grf_cross_engine,
)


class TestLinearImpulse:
    """Tests for linear impulse computation."""

    def test_constant_force_impulse(self) -> None:
        """Constant force over time should give F*t impulse."""
        force_magnitude = 100.0  # [N]
        duration = 0.5  # [s]

        timestamps = np.linspace(0, duration, 100)
        forces = np.zeros((100, 3))
        forces[:, 2] = force_magnitude  # Vertical force

        impulse = compute_linear_impulse(forces, timestamps)

        expected_impulse = force_magnitude * duration
        np.testing.assert_allclose(impulse[2], expected_impulse, rtol=0.01)
        np.testing.assert_allclose(impulse[:2], 0.0, atol=1e-10)

    def test_zero_force_gives_zero_impulse(self) -> None:
        """Zero force should give zero impulse."""
        timestamps = np.linspace(0, 1, 100)
        forces = np.zeros((100, 3))

        impulse = compute_linear_impulse(forces, timestamps)

        np.testing.assert_allclose(impulse, 0.0, atol=1e-10)

    def test_trapezoidal_integration_accuracy(self) -> None:
        """Trapezoidal integration should be accurate for linear ramp."""
        timestamps = np.linspace(0, 1, 100)
        forces = np.zeros((100, 3))
        forces[:, 2] = timestamps * 1000  # Linear ramp to 1000 N

        impulse = compute_linear_impulse(forces, timestamps)

        # Integral of linear ramp from 0 to 1: ∫ 1000*t dt = 500
        np.testing.assert_allclose(impulse[2], 500.0, rtol=0.01)


class TestAngularImpulse:
    """Tests for angular impulse computation."""

    def test_force_at_arm_produces_angular_impulse(self) -> None:
        """Force applied at distance from reference should produce torque."""
        timestamps = np.linspace(0, 1, 100)
        forces = np.zeros((100, 3))
        forces[:, 2] = 100.0  # Vertical force

        # COP at 1m in X direction
        cops = np.zeros((100, 3))
        cops[:, 0] = 1.0

        ref_point = np.zeros(3)  # Origin

        angular_impulse = compute_angular_impulse(forces, cops, timestamps, ref_point)

        # Torque = r × F = [1, 0, 0] × [0, 0, 100] = [0, -100, 0]
        # Angular impulse = torque * time = [0, -100, 0] * 1 = [0, -100, 0]
        np.testing.assert_allclose(angular_impulse[1], -100.0, rtol=0.01)

    def test_force_through_reference_gives_zero_angular_impulse(self) -> None:
        """Force acting through reference point should give zero torque."""
        timestamps = np.linspace(0, 1, 100)
        forces = np.zeros((100, 3))
        forces[:, 2] = 100.0

        # COP at reference point
        cops = np.zeros((100, 3))
        ref_point = np.zeros(3)

        angular_impulse = compute_angular_impulse(forces, cops, timestamps, ref_point)

        np.testing.assert_allclose(angular_impulse, 0.0, atol=1e-10)

    def test_dynamic_reference_point_produces_angular_impulse(self) -> None:
        """Dynamic reference trajectory should produce correct angular impulse."""
        n = 100
        timestamps = np.linspace(0, 1, n)
        forces = np.zeros((n, 3))
        forces[:, 2] = 100.0  # Vertical force

        # COP at origin
        cops = np.zeros((n, 3))

        # Reference point moving in X direction at 1 m/s (r_x = -t)
        ref_trajectory = np.zeros((n, 3))
        ref_trajectory[:, 0] = timestamps

        angular_impulse = compute_angular_impulse(
            forces, cops, timestamps, ref_trajectory
        )

        # Torque = r × F = [-t, 0, 0] × [0, 0, 100] = [0, 100*t, 0]
        # Angular impulse = int(100*t dt) from 0 to 1 = 100 * (1^2 / 2) = 50.0
        np.testing.assert_allclose(angular_impulse[1], 50.0, rtol=0.01)


class TestCOPComputation:
    """Tests for center of pressure computation."""

    def test_cop_from_pure_vertical_force(self) -> None:
        """Pure vertical force with moment should give correct COP."""
        force = np.array([0.0, 0.0, 1000.0])  # 1000 N vertical
        moment = np.array([100.0, -200.0, 0.0])  # M_x = 100, M_y = -200

        cop = compute_cop_from_grf(force, moment)

        # COP_x = -M_y / F_z = 200/1000 = 0.2
        # COP_y = M_x / F_z = 100/1000 = 0.1
        np.testing.assert_allclose(cop[0], 0.2, atol=1e-10)
        np.testing.assert_allclose(cop[1], 0.1, atol=1e-10)

    def test_low_vertical_force_gives_zero_cop(self) -> None:
        """Very small vertical force should return origin COP."""
        force = np.array([0.0, 0.0, 5.0])  # Below threshold
        moment = np.array([100.0, 100.0, 0.0])

        cop = compute_cop_from_grf(force, moment)

        np.testing.assert_allclose(cop, np.array([0.0, 0.0, 0.0]), atol=1e-10)


class TestCOPTrajectoryLength:
    """Tests for COP trajectory length computation."""

    def test_stationary_cop_has_zero_length(self) -> None:
        """Stationary COP should have zero path length."""
        cops = np.zeros((100, 3))

        length = compute_cop_trajectory_length(cops)

        assert length == 0.0

    def test_linear_motion_cop(self) -> None:
        """Linear COP motion should give straight-line distance."""
        cops = np.column_stack(
            [
                np.linspace(0, 1, 100),
                np.zeros(100),
                np.zeros(100),
            ]
        )

        length = compute_cop_trajectory_length(cops)

        np.testing.assert_allclose(length, 1.0, rtol=0.01)

    def test_circular_motion_cop(self) -> None:
        """Circular COP motion should give circumference."""
        radius = 0.1  # [m]
        theta = np.linspace(0, 2 * np.pi, 100)
        cops = np.column_stack(
            [
                radius * np.cos(theta),
                radius * np.sin(theta),
                np.zeros(100),
            ]
        )

        length = compute_cop_trajectory_length(cops)

        expected_circumference = 2 * np.pi * radius
        np.testing.assert_allclose(length, expected_circumference, rtol=0.05)


class TestGRFAnalyzer:
    """Tests for GRF analyzer class."""

    @pytest.fixture
    def sample_grf_data(self) -> GRFTimeSeries:
        """Create sample GRF time series data."""
        n = 100
        timestamps = np.linspace(0, 1, n)
        forces = np.zeros((n, 3))
        forces[:, 2] = 800.0 + 200 * np.sin(np.pi * timestamps)  # Varying vertical

        moments = np.zeros((n, 3))
        cops = np.zeros((n, 3))
        cops[:, 0] = 0.1 * np.sin(2 * np.pi * timestamps)  # Oscillating X

        return GRFTimeSeries(
            timestamps=timestamps,
            forces=forces,
            moments=moments,
            cops=cops,
            foot_side=FootSide.COMBINED,
        )

    def test_analyzer_computes_impulse(self, sample_grf_data: GRFTimeSeries) -> None:
        """Analyzer should compute impulse metrics."""
        analyzer = GRFAnalyzer()
        analyzer.add_grf_data(sample_grf_data)

        metrics = analyzer.compute_impulse_metrics(FootSide.COMBINED)

        assert metrics.linear_impulse_magnitude > 0
        assert metrics.duration > 0

    def test_analyzer_full_analysis(self, sample_grf_data: GRFTimeSeries) -> None:
        """Analyzer should produce full summary."""
        analyzer = GRFAnalyzer()
        analyzer.add_grf_data(sample_grf_data)

        summary = analyzer.analyze(FootSide.COMBINED)

        assert summary.peak_vertical_force > 0
        assert summary.cop_trajectory_length > 0
        assert summary.linear_impulse is not None

    def test_analyzer_uses_dynamic_com(self, sample_grf_data: GRFTimeSeries) -> None:
        """Analyzer should compute moments using dynamic COM if lengths match."""
        analyzer = GRFAnalyzer()
        analyzer.add_grf_data(sample_grf_data)

        # Dynamic COM trajectory exactly matching timestamps
        n = len(sample_grf_data.timestamps)
        com_traj = np.zeros((n, 3))
        com_traj[:, 0] = 1.0  # 1m offset in X

        analyzer.set_com_trajectories(com_traj)
        summary = analyzer.analyze(FootSide.COMBINED)

        # The angular impulse should not be zero
        assert np.any(summary.angular_impulse_about_golfer_com)

        # Test fallback to static
        short_com_traj = np.zeros((1, 3))
        short_com_traj[0, 0] = 1.0
        analyzer.set_com_trajectories(short_com_traj)
        summary_fallback = analyzer.analyze(FootSide.COMBINED)

        # The outputs should be identical since the dynamic and fallback offset are the same 1.0 in X
        np.testing.assert_allclose(
            summary.angular_impulse_about_golfer_com,
            summary_fallback.angular_impulse_about_golfer_com,
        )


class TestExtractGRFFromContacts:
    """Tests for extract_grf_from_contacts with engine contact solver."""

    def _make_engine(
        self,
        contact_force: np.ndarray | None = None,
        gravity: np.ndarray | None = None,
        time: float = 0.0,
    ) -> MagicMock:
        """Create a mock engine with configurable contact forces."""
        engine = MagicMock()
        engine.get_time.return_value = time

        if contact_force is not None:
            engine.compute_contact_forces.return_value = contact_force
        else:
            engine.compute_contact_forces.return_value = np.zeros(3)

        if gravity is not None:
            engine.compute_gravity_forces.return_value = gravity
        else:
            engine.compute_gravity_forces.return_value = np.array([-9.81])

        # Jacobian returns a dict with a linear key
        jac = {"linear": np.array([[0.0, 0.1], [0.0, 0.0], [0.0, 0.0]])}
        engine.compute_jacobian.return_value = jac
        return engine

    def test_uses_contact_solver_when_available(self) -> None:
        """When engine returns non-zero contact forces, use them."""
        contact = np.array([10.0, 5.0, 800.0])
        engine = self._make_engine(contact_force=contact)

        grf = extract_grf_from_contacts(engine, ["left_foot", "right_foot"])

        np.testing.assert_allclose(grf.force[:3], contact, atol=1e-10)
        # Should NOT call compute_gravity_forces (primary path used)
        engine.compute_gravity_forces.assert_not_called()

    def test_falls_back_to_gravity_when_no_contacts(self) -> None:
        """When engine returns zero contact forces, fall back to gravity."""
        engine = self._make_engine(contact_force=np.zeros(3))

        grf = extract_grf_from_contacts(engine, ["left_foot"])

        # Should call gravity fallback
        engine.compute_gravity_forces.assert_called()
        # Force should be non-zero (gravity-based estimate)
        assert grf.force[2] > 0

    def test_timestamp_from_engine(self) -> None:
        """GRF timestamp should come from engine time."""
        engine = self._make_engine(time=1.5)

        grf = extract_grf_from_contacts(engine, ["foot"])

        assert grf.timestamp == 1.5

    def test_cop_at_ground_height(self) -> None:
        """COP z-coordinate should equal the ground_height argument."""
        engine = self._make_engine(contact_force=np.array([0.0, 0.0, 500.0]))

        grf = extract_grf_from_contacts(engine, ["foot"], ground_height=0.05)

        assert grf.cop[2] == 0.05

    def test_empty_contact_bodies_returns_zero(self) -> None:
        """No contact bodies should give zero force."""
        engine = self._make_engine(contact_force=np.zeros(3))

        grf = extract_grf_from_contacts(engine, [])

        np.testing.assert_allclose(grf.force, 0.0, atol=1e-10)

    def test_gravity_fallback_not_doubled_for_two_bodies(self) -> None:
        """Gravity fallback must report total weight once, not once per contact
        body (issue #6894). Two feet must not double the vertical force."""
        gravity = np.array([0.0, 0.0, -800.0])  # total weight magnitude 800 N

        engine_one = self._make_engine(contact_force=np.zeros(3), gravity=gravity)
        engine_two = self._make_engine(contact_force=np.zeros(3), gravity=gravity)

        grf_one = extract_grf_from_contacts(engine_one, ["left_foot"])
        grf_two = extract_grf_from_contacts(engine_two, ["left_foot", "right_foot"])

        np.testing.assert_allclose(grf_one.force[2], 800.0, rtol=1e-9)
        np.testing.assert_allclose(grf_two.force[2], 800.0, rtol=1e-9)

    def test_moment_computed_from_contact_data(self) -> None:
        """When contact data is available, moment should be computed from COP x force."""
        contact = np.array([0.0, 0.0, 1000.0])
        engine = self._make_engine(contact_force=contact)

        grf = extract_grf_from_contacts(engine, ["foot"])

        # Moment is cross(cop - ground_origin, force)
        # With default ground_height=0.0, moment depends on COP position
        assert grf.moment is not None
        assert grf.moment.shape == (3,)


class TestCrossEngineValidation:
    """Tests for cross-engine GRF validation."""

    def test_identical_data_passes_validation(self) -> None:
        """Identical GRF data should pass all validations."""
        n = 100
        timestamps = np.linspace(0, 1, n)
        forces = np.zeros((n, 3))
        forces[:, 2] = 800.0
        cops = np.zeros((n, 3))

        data_a = GRFTimeSeries(
            timestamps=timestamps,
            forces=forces.copy(),
            moments=np.zeros((n, 3)),
            cops=cops.copy(),
        )
        data_b = GRFTimeSeries(
            timestamps=timestamps,
            forces=forces.copy(),
            moments=np.zeros((n, 3)),
            cops=cops.copy(),
        )

        results = validate_grf_cross_engine(data_a, data_b)

        assert results["force_magnitude"] is True
        assert results["cop_position"] is True
        assert results["angular_impulse"] is True

    def test_different_forces_fails_validation(self) -> None:
        """Significantly different forces should fail validation."""
        n = 100
        timestamps = np.linspace(0, 1, n)
        forces_a = np.zeros((n, 3))
        forces_a[:, 2] = 800.0
        forces_b = np.zeros((n, 3))
        forces_b[:, 2] = 1000.0  # 25% different - exceeds 5% tolerance

        data_a = GRFTimeSeries(
            timestamps=timestamps,
            forces=forces_a,
            moments=np.zeros((n, 3)),
            cops=np.zeros((n, 3)),
        )
        data_b = GRFTimeSeries(
            timestamps=timestamps,
            forces=forces_b,
            moments=np.zeros((n, 3)),
            cops=np.zeros((n, 3)),
        )

        results = validate_grf_cross_engine(data_a, data_b)

        assert results["force_magnitude"] is False

    def test_different_cop_fails_validation(self) -> None:
        """COP difference > 10mm should fail validation."""
        n = 100
        timestamps = np.linspace(0, 1, n)
        forces = np.zeros((n, 3))
        forces[:, 2] = 800.0

        cops_a = np.zeros((n, 3))
        cops_b = np.zeros((n, 3))
        cops_b[:, 0] = 0.02  # 20mm difference - exceeds 10mm tolerance

        data_a = GRFTimeSeries(
            timestamps=timestamps,
            forces=forces.copy(),
            moments=np.zeros((n, 3)),
            cops=cops_a,
        )
        data_b = GRFTimeSeries(
            timestamps=timestamps,
            forces=forces.copy(),
            moments=np.zeros((n, 3)),
            cops=cops_b,
        )

        results = validate_grf_cross_engine(data_a, data_b)

        assert results["cop_position"] is False

    def test_unsupported_engine_zero_force_fallback(self) -> None:
        """When engine returns zero contact forces, should fall back to gravity."""
        # Create an engine that returns zero (unsupported contact queries like Pinocchio)
        engine = MagicMock()
        engine.get_time.return_value = 0.5
        engine.compute_contact_forces.return_value = np.zeros(3)
        engine.compute_gravity_forces.return_value = np.array([-9.81])
        jac = {"linear": np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])}
        engine.compute_jacobian.return_value = jac

        # extract_grf_from_contacts should handle zero forces gracefully
        grf = extract_grf_from_contacts(engine, ["left_foot"])

        # Should have fallen back to gravity-based estimate (z > 0)
        engine.compute_gravity_forces.assert_called()
        assert grf.force[2] > 0  # Vertical force from gravity fallback
        assert grf.timestamp == 0.5
