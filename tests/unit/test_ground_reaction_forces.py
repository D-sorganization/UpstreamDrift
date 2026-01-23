"""Tests for Ground Reaction Force Analysis.

Guideline E5 implementation tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.ground_reaction_forces import (
    FootSide,
    GRFAnalyzer,
    GRFTimeSeries,
    compute_angular_impulse,
    compute_cop_from_grf,
    compute_cop_trajectory_length,
    compute_linear_impulse,
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
