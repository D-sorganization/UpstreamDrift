"""Comprehensive tests for src.shared.python.biomechanics package.

Covers hill_muscle (MuscleParameters, MuscleState, HillMuscleModel),
activation_dynamics (ActivationDynamics), and swing_plane_analysis
(SwingPlaneAnalyzer, SwingPlaneMetrics).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics
from src.shared.python.biomechanics.hill_muscle import (
    HillMuscleModel,
    MuscleParameters,
    MuscleState,
)
from src.shared.python.biomechanics.swing_plane_analysis import (
    SwingPlaneAnalyzer,
    SwingPlaneMetrics,
)
from src.shared.python.core.contracts import PreconditionError

# ============================================================================
# Tests for MuscleParameters
# ============================================================================


class TestMuscleParameters:
    """Tests for MuscleParameters dataclass validation."""

    def test_valid_parameters(self) -> None:
        params = MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=0.20)
        assert params.F_max == 1000.0
        assert params.l_opt == 0.15
        assert params.l_slack == 0.20
        assert params.v_max == 10.0  # Default
        assert params.pennation_angle == 0.0
        assert params.damping == 0.05

    def test_custom_optional_params(self) -> None:
        params = MuscleParameters(
            F_max=500.0,
            l_opt=0.10,
            l_slack=0.15,
            v_max=8.0,
            pennation_angle=0.1,
            damping=0.02,
        )
        assert params.v_max == 8.0
        assert params.pennation_angle == 0.1
        assert params.damping == 0.02

    def test_negative_fmax_raises(self) -> None:
        with pytest.raises(ValueError, match="F_max"):
            MuscleParameters(F_max=-100.0, l_opt=0.15, l_slack=0.20)

    def test_zero_fmax_raises(self) -> None:
        with pytest.raises(ValueError, match="F_max"):
            MuscleParameters(F_max=0.0, l_opt=0.15, l_slack=0.20)

    def test_negative_lopt_raises(self) -> None:
        with pytest.raises(ValueError, match="l_opt"):
            MuscleParameters(F_max=1000.0, l_opt=-0.1, l_slack=0.20)

    def test_negative_lslack_raises(self) -> None:
        with pytest.raises(ValueError, match="l_slack"):
            MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=-0.1)


# ============================================================================
# Tests for MuscleState
# ============================================================================


class TestMuscleState:
    """Tests for MuscleState dataclass."""

    def test_defaults(self) -> None:
        state = MuscleState()
        assert state.activation == 0.0
        assert state.l_CE == 0.0
        assert state.v_CE == 0.0
        assert state.l_MT == 0.0

    def test_custom_values(self) -> None:
        state = MuscleState(activation=0.8, l_CE=0.15, v_CE=-0.5, l_MT=0.35)
        assert state.activation == 0.8
        assert state.v_CE == -0.5


# ============================================================================
# Tests for HillMuscleModel
# ============================================================================


class TestHillMuscleModel:
    """Tests for Hill-type muscle model."""

    @pytest.fixture()
    def params(self) -> MuscleParameters:
        return MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=0.20)

    @pytest.fixture()
    def muscle(self, params: MuscleParameters) -> HillMuscleModel:
        return HillMuscleModel(params)

    # -- Force-Length Active --

    def test_force_length_active_at_optimal(self, muscle: HillMuscleModel) -> None:
        """f_l(1.0) should be 1.0 (max force at optimal length)."""
        assert muscle.force_length_active(1.0) == pytest.approx(1.0)

    def test_force_length_active_falls_away(self, muscle: HillMuscleModel) -> None:
        """Force should decrease when l_norm departs from 1.0."""
        f_at_opt = muscle.force_length_active(1.0)
        f_short = muscle.force_length_active(0.5)
        f_long = muscle.force_length_active(1.5)
        assert f_short < f_at_opt
        assert f_long < f_at_opt

    def test_force_length_active_symmetric(self, muscle: HillMuscleModel) -> None:
        """Gaussian curve should be symmetric around 1.0."""
        delta = 0.2
        f_below = muscle.force_length_active(1.0 - delta)
        f_above = muscle.force_length_active(1.0 + delta)
        assert f_below == pytest.approx(f_above, abs=1e-10)

    def test_force_length_active_nonnegative(self, muscle: HillMuscleModel) -> None:
        for l_norm in [0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
            assert muscle.force_length_active(l_norm) >= 0.0

    # -- Force-Length Passive --

    def test_force_length_passive_zero_below_optimal(
        self, muscle: HillMuscleModel
    ) -> None:
        """Passive force is zero when l_norm <= 1.0."""
        assert muscle.force_length_passive(0.5) == 0.0
        assert muscle.force_length_passive(1.0) == 0.0

    def test_force_length_passive_increases_above_optimal(
        self, muscle: HillMuscleModel
    ) -> None:
        """Passive force should increase when stretched beyond optimal."""
        f_1_1 = muscle.force_length_passive(1.1)
        f_1_5 = muscle.force_length_passive(1.5)
        assert f_1_1 > 0
        assert f_1_5 > f_1_1

    # -- Force-Velocity --

    def test_force_velocity_isometric(self, muscle: HillMuscleModel) -> None:
        """At zero velocity (isometric), f_v should be ~1.0."""
        assert muscle.force_velocity(0.0) == pytest.approx(1.0)

    def test_force_velocity_concentric_decreases(self, muscle: HillMuscleModel) -> None:
        """Shortening (negative v) should reduce force."""
        f_v = muscle.force_velocity(-0.5)
        assert f_v < 1.0
        assert f_v > 0.0

    def test_force_velocity_eccentric_increases(self, muscle: HillMuscleModel) -> None:
        """Lengthening (positive v) should increase force above 1.0."""
        f_v = muscle.force_velocity(0.5)
        assert f_v > 1.0

    # -- Tendon Force --

    def test_tendon_force_slack(self, muscle: HillMuscleModel) -> None:
        """Tendon force is zero when at or below slack length."""
        assert muscle.tendon_force(0.9) == 0.0
        assert muscle.tendon_force(1.0) == 0.0

    def test_tendon_force_stretched(self, muscle: HillMuscleModel) -> None:
        """Tendon force increases with strain (above slack length)."""
        f_small = muscle.tendon_force(1.01)
        f_large = muscle.tendon_force(1.05)
        assert f_small > 0
        assert f_large > f_small

    # -- compute_force --

    def test_compute_force_isometric_full_activation(
        self,
        muscle: HillMuscleModel,
        params: MuscleParameters,
    ) -> None:
        """Full activation at optimal length, zero velocity → force ≈ F_max."""
        state = MuscleState(activation=1.0, l_CE=params.l_opt, v_CE=0.0, l_MT=0.35)
        force = muscle.compute_force(state)
        # At l_norm=1.0, f_l=1.0, f_v=1.0, a=1.0 → F = F_max
        assert force == pytest.approx(params.F_max, rel=0.01)

    def test_compute_force_zero_activation(
        self,
        muscle: HillMuscleModel,
        params: MuscleParameters,
    ) -> None:
        """Zero activation at optimal length → only passive + damping."""
        state = MuscleState(activation=0.0, l_CE=params.l_opt, v_CE=0.0, l_MT=0.35)
        force = muscle.compute_force(state)
        # No active force, passive is 0 at l_norm=1.0, damping zero → 0
        assert force == pytest.approx(0.0)

    def test_compute_force_nonnegative(
        self,
        muscle: HillMuscleModel,
        params: MuscleParameters,
    ) -> None:
        """Force should always be non-negative (postcondition)."""
        for act in [0.0, 0.3, 0.5, 0.8, 1.0]:
            for l_factor in [0.8, 1.0, 1.2]:
                state = MuscleState(
                    activation=act,
                    l_CE=params.l_opt * l_factor,
                    v_CE=0.0,
                    l_MT=0.35,
                )
                assert muscle.compute_force(state) >= 0.0

    def test_compute_force_invalid_activation_raises(
        self, muscle: HillMuscleModel
    ) -> None:
        """Activation outside [0, 1] should raise error."""
        state = MuscleState(activation=1.5, l_CE=0.15, v_CE=0.0, l_MT=0.35)
        with pytest.raises(Exception, match="activation"):
            muscle.compute_force(state)

    def test_compute_force_negative_activation_raises(
        self, muscle: HillMuscleModel
    ) -> None:
        state = MuscleState(activation=-0.1, l_CE=0.15, v_CE=0.0, l_MT=0.35)
        with pytest.raises(Exception, match="activation"):
            muscle.compute_force(state)

    def test_pennation_angle_reduces_force(self) -> None:
        """Non-zero pennation angle should reduce tendon force."""
        params_no_pen = MuscleParameters(
            F_max=1000.0, l_opt=0.15, l_slack=0.20, pennation_angle=0.0
        )
        params_with_pen = MuscleParameters(
            F_max=1000.0, l_opt=0.15, l_slack=0.20, pennation_angle=0.3
        )
        m1 = HillMuscleModel(params_no_pen)
        m2 = HillMuscleModel(params_with_pen)

        state = MuscleState(activation=1.0, l_CE=0.15, v_CE=0.0, l_MT=0.35)
        f1 = m1.compute_force(state)
        f2 = m2.compute_force(state)
        assert f2 < f1  # cos(0.3) < 1.0 reduces force


# ============================================================================
# Tests for ActivationDynamics
# ============================================================================


class TestActivationDynamics:
    """Tests for neural excitation to muscle activation dynamics."""

    @pytest.fixture()
    def dynamics(self) -> ActivationDynamics:
        return ActivationDynamics(tau_act=0.010, tau_deact=0.040)

    def test_default_params(self) -> None:
        d = ActivationDynamics()
        assert d.tau_act == 0.010
        assert d.tau_deact == 0.040
        assert d.min_activation == 0.001

    def test_invalid_tau_act_raises(self) -> None:
        with pytest.raises(Exception, match="tau_act"):
            ActivationDynamics(tau_act=-0.01)

    def test_invalid_tau_deact_raises(self) -> None:
        with pytest.raises(Exception, match="tau_deact"):
            ActivationDynamics(tau_deact=0.0)

    def test_invalid_min_activation_raises(self) -> None:
        with pytest.raises(Exception, match="min_activation"):
            ActivationDynamics(min_activation=0.0)
        with pytest.raises(Exception, match="min_activation"):
            ActivationDynamics(min_activation=1.0)

    # -- compute_derivative --

    def test_derivative_positive_when_u_gt_a(
        self, dynamics: ActivationDynamics
    ) -> None:
        """When excitation > activation, derivative should be positive."""
        dadt = dynamics.compute_derivative(u=1.0, a=0.1)
        assert dadt > 0

    def test_derivative_negative_when_u_lt_a(
        self, dynamics: ActivationDynamics
    ) -> None:
        """When excitation < activation, derivative should be negative."""
        dadt = dynamics.compute_derivative(u=0.0, a=0.5)
        assert dadt < 0

    def test_derivative_zero_at_equilibrium(self, dynamics: ActivationDynamics) -> None:
        """When u == a, derivative should be ~0."""
        dadt = dynamics.compute_derivative(u=0.5, a=0.5)
        assert abs(dadt) < 1e-6

    def test_derivative_finite(self, dynamics: ActivationDynamics) -> None:
        """Derivative should always be finite."""
        for u in [0.0, 0.5, 1.0]:
            for a in [0.0, 0.5, 1.0]:
                dadt = dynamics.compute_derivative(u, a)
                assert np.isfinite(dadt)

    # -- update --

    def test_update_negative_dt_raises(self, dynamics: ActivationDynamics) -> None:
        with pytest.raises(Exception, match="dt"):
            dynamics.update(u=1.0, a=0.0, dt=-0.001)

    def test_update_zero_dt_raises(self, dynamics: ActivationDynamics) -> None:
        with pytest.raises(Exception, match="dt"):
            dynamics.update(u=1.0, a=0.0, dt=0.0)

    def test_update_bounded(self, dynamics: ActivationDynamics) -> None:
        """Result must be in [min_activation, 1.0]."""
        a = 0.5
        for u in [0.0, 0.5, 1.0]:
            a_new = dynamics.update(u, a, dt=0.001)
            assert dynamics.min_activation <= a_new <= 1.0

    def test_step_response_rises(self, dynamics: ActivationDynamics) -> None:
        """Applying u=1 should drive activation toward 1."""
        a = 0.0
        for _ in range(200):
            a = dynamics.update(u=1.0, a=a, dt=0.001)
        assert a > 0.95  # Should be near 1.0 after 200ms

    def test_step_response_falls(self, dynamics: ActivationDynamics) -> None:
        """Applying u=0 should drive activation toward min_activation."""
        a = 1.0
        for _ in range(500):
            a = dynamics.update(u=0.0, a=a, dt=0.001)
        assert a < 0.05  # Should be near 0 after 500ms

    def test_activation_faster_than_deactivation(
        self, dynamics: ActivationDynamics
    ) -> None:
        """Activation (rise) should be faster than deactivation (fall)."""
        # Rise: 0 → 0.5
        a_rise = 0.0
        steps_rise = 0
        while a_rise < 0.5:
            a_rise = dynamics.update(u=1.0, a=a_rise, dt=0.001)
            steps_rise += 1

        # Fall: 1.0 → 0.5
        a_fall = 1.0
        steps_fall = 0
        while a_fall > 0.5:
            a_fall = dynamics.update(u=0.0, a=a_fall, dt=0.001)
            steps_fall += 1

        assert steps_rise < steps_fall  # Rise is faster


# ============================================================================
# Tests for SwingPlaneAnalyzer
# ============================================================================


class TestSwingPlaneAnalyzer:
    """Tests for swing plane analysis from 3D trajectory."""

    @pytest.fixture()
    def analyzer(self) -> SwingPlaneAnalyzer:
        return SwingPlaneAnalyzer()

    def _make_planar_arc(
        self,
        n: int = 100,
        normal: np.ndarray | None = None,
    ) -> np.ndarray:
        """Create an arc of points lying in a plane with given normal."""
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0])  # XY plane
        t = np.linspace(0, np.pi, n)
        # Arc in XY: (cos(t), sin(t), 0)
        points = np.column_stack([np.cos(t), np.sin(t), np.zeros(n)])
        return points

    def test_fit_plane_xy(self, analyzer: SwingPlaneAnalyzer) -> None:
        """Points in XY plane should have normal ≈ (0, 0, 1)."""
        points = self._make_planar_arc()
        centroid, normal = analyzer.fit_plane(points)

        assert centroid.shape == (3,)
        assert normal.shape == (3,)
        # Normal should be close to ±(0, 0, 1)
        assert abs(abs(normal[2]) - 1.0) < 0.01

    def test_fit_plane_unit_normal(self, analyzer: SwingPlaneAnalyzer) -> None:
        """Normal vector must be unit length."""
        points = self._make_planar_arc()
        _, normal = analyzer.fit_plane(points)
        assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-6)

    def test_fit_plane_too_few_points(self, analyzer: SwingPlaneAnalyzer) -> None:
        """Fewer than 3 points should raise an error."""
        with pytest.raises(PreconditionError, match="at least 3"):
            analyzer.fit_plane(np.array([[0, 0, 0], [1, 0, 0]]))

    def test_deviation_on_plane_is_zero(self, analyzer: SwingPlaneAnalyzer) -> None:
        """Points on the fitted plane should have ~0 deviation."""
        points = self._make_planar_arc()
        centroid, normal = analyzer.fit_plane(points)
        devs = analyzer.calculate_deviation(points, centroid, normal)
        np.testing.assert_allclose(devs, 0, atol=1e-10)

    def test_deviation_off_plane(self, analyzer: SwingPlaneAnalyzer) -> None:
        """Adding offset in normal direction should produce non-zero deviation."""
        points = self._make_planar_arc()
        centroid, normal = analyzer.fit_plane(points)

        offset_points = points + 0.1 * normal  # Shift 0.1 in normal dir
        devs = analyzer.calculate_deviation(offset_points, centroid, normal)
        np.testing.assert_allclose(devs, 0.1, atol=1e-10)

    def test_analyze_planar_data(self, analyzer: SwingPlaneAnalyzer) -> None:
        """Full analysis on perfect planar data."""
        points = self._make_planar_arc()
        metrics = analyzer.analyze(points)

        assert isinstance(metrics, SwingPlaneMetrics)
        assert metrics.rmse == pytest.approx(0.0, abs=1e-8)
        assert metrics.max_deviation == pytest.approx(0.0, abs=1e-8)
        assert 0.0 <= metrics.steepness_deg <= 180.0

    def test_analyze_noisy_data(self, analyzer: SwingPlaneAnalyzer) -> None:
        """With noise, RMSE should be small but nonzero."""
        rng = np.random.default_rng(42)
        points = self._make_planar_arc(n=200)
        noise = rng.normal(scale=0.01, size=points.shape)
        noisy_points = points + noise

        metrics = analyzer.analyze(noisy_points)
        assert metrics.rmse > 0
        assert metrics.rmse < 0.05  # Small noise → small RMSE
        assert metrics.max_deviation >= metrics.rmse

    def test_analyze_tilted_plane(self, analyzer: SwingPlaneAnalyzer) -> None:
        """Tilted plane should produce non-zero steepness."""
        t = np.linspace(0, np.pi, 100)
        # Arc tilted 45° from horizontal: XZ plane
        points = np.column_stack([np.cos(t), np.zeros(100), np.sin(t)])
        metrics = analyzer.analyze(points)

        # Normal should be along Y → steepness ≈ 90°
        assert metrics.steepness_deg == pytest.approx(90.0, abs=5.0)

    def test_analyze_postconditions(self, analyzer: SwingPlaneAnalyzer) -> None:
        """Verify postconditions: rmse >= 0, max_dev >= 0, steepness in [0, 180]."""
        rng = np.random.default_rng(123)
        points = rng.normal(size=(50, 3))
        metrics = analyzer.analyze(points)
        assert metrics.rmse >= 0
        assert metrics.max_deviation >= 0
        assert 0.0 <= metrics.steepness_deg <= 180.0


# ============================================================================
# Tests for SwingPlaneMetrics dataclass
# ============================================================================


class TestSwingPlaneMetrics:
    """Tests for SwingPlaneMetrics dataclass fields."""

    def test_instantiation(self) -> None:
        m = SwingPlaneMetrics(
            normal_vector=np.array([0, 0, 1.0]),
            point_on_plane=np.zeros(3),
            steepness_deg=45.0,
            direction_deg=90.0,
            rmse=0.01,
            max_deviation=0.03,
        )
        assert m.steepness_deg == 45.0
        assert m.direction_deg == 90.0
        assert m.rmse == 0.01
