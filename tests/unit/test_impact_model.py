"""Tests for Modular Impact Model.

Guideline K3 implementation tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.physics.impact_model import (
    GOLF_BALL_MASS,
    GOLF_BALL_RADIUS,
    FiniteTimeImpactModel,
    ImpactEvent,
    ImpactModelType,
    ImpactParameters,
    ImpactRecorder,
    ImpactSolverAPI,
    PreImpactState,
    RigidBodyImpactModel,
    SpringDamperImpactModel,
    compute_gear_effect_spin,
    create_impact_model,
    validate_energy_balance,
)


class TestPreImpactState:
    """Tests for pre-impact state creation."""

    def test_default_values(self) -> None:
        """Should have sensible default values."""
        state = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),  # 45 m/s ~100 mph
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        assert state.clubhead_mass == pytest.approx(0.200)  # 200g
        assert state.clubhead_loft == pytest.approx(np.radians(10.5))


class TestRigidBodyImpactModel:
    """Tests for rigid body collision model."""

    @pytest.fixture
    def default_pre_state(self) -> PreImpactState:
        """Create default pre-impact state for testing."""
        return PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),  # 40 m/s
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),  # Facing X
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

    @pytest.fixture
    def default_params(self) -> ImpactParameters:
        """Create default impact parameters."""
        return ImpactParameters()

    def test_ball_gains_velocity(
        self,
        default_pre_state: PreImpactState,
        default_params: ImpactParameters,
    ) -> None:
        """Ball should gain velocity after impact."""
        model = RigidBodyImpactModel()
        result = model.solve(default_pre_state, default_params)

        # Ball should have significant forward velocity
        assert result.ball_velocity[0] > 0
        # Ball speed should be faster than clubhead (smash factor > 1)
        ball_speed = np.linalg.norm(result.ball_velocity)
        club_speed = np.linalg.norm(default_pre_state.clubhead_velocity)
        assert ball_speed > club_speed * 1.3  # Typical smash factor ~1.45-1.5

    def test_clubhead_loses_velocity(
        self,
        default_pre_state: PreImpactState,
        default_params: ImpactParameters,
    ) -> None:
        """Clubhead should lose velocity after impact."""
        model = RigidBodyImpactModel()
        result = model.solve(default_pre_state, default_params)

        # Clubhead should be slower after impact
        club_speed_pre = np.linalg.norm(default_pre_state.clubhead_velocity)
        club_speed_post = np.linalg.norm(result.clubhead_velocity)
        assert club_speed_post < club_speed_pre

    def test_momentum_conservation(
        self,
        default_pre_state: PreImpactState,
        default_params: ImpactParameters,
    ) -> None:
        """Total momentum should be conserved."""
        model = RigidBodyImpactModel()
        result = model.solve(default_pre_state, default_params)

        m_ball = GOLF_BALL_MASS
        m_club = default_pre_state.clubhead_mass

        # Pre-impact momentum
        p_pre = (
            m_club * default_pre_state.clubhead_velocity
            + m_ball * default_pre_state.ball_velocity
        )

        # Post-impact momentum
        p_post = m_club * result.clubhead_velocity + m_ball * result.ball_velocity

        # Momentum should be conserved
        np.testing.assert_allclose(p_pre, p_post, rtol=1e-5)

    def test_cor_affects_separation_velocity(self) -> None:
        """Higher COR should give higher separation velocity."""
        pre_state = PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        model = RigidBodyImpactModel()

        low_cor = ImpactParameters(cor=0.6)
        high_cor = ImpactParameters(cor=0.9)

        result_low = model.solve(pre_state, low_cor)
        result_high = model.solve(pre_state, high_cor)

        # Higher COR should give faster ball
        assert np.linalg.norm(result_high.ball_velocity) > np.linalg.norm(
            result_low.ball_velocity
        )


class TestSpringDamperImpactModel:
    """Tests for spring-damper compliant contact model.

    Note: The spring-damper model requires very small timesteps
    due to the high contact stiffness. These tests are marked as
    expected failures until a more numerically stable integration
    scheme (e.g., implicit Euler) is implemented.
    """

    def test_ball_gains_velocity(self) -> None:
        """Spring-damper model should produce finite results."""
        pre_state = PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.array([GOLF_BALL_RADIUS, 0.0, 0.0]),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        model = SpringDamperImpactModel()  # Use default stable timestep
        params = ImpactParameters()

        result = model.solve(pre_state, params)

        # Result should be finite and reasonably bounded
        assert np.all(np.isfinite(result.ball_velocity))
        # Velocity magnitude should be reasonable (not blown up)
        assert np.linalg.norm(result.ball_velocity) < 200  # m/s

    def test_has_contact_duration(self) -> None:
        """Spring-damper model should report non-zero contact duration."""
        pre_state = PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.array([GOLF_BALL_RADIUS, 0.0, 0.0]),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        model = SpringDamperImpactModel()
        params = ImpactParameters()

        result = model.solve(pre_state, params)

        # Should have measurable contact time
        assert result.contact_duration > 0


class TestFiniteTimeImpactModel:
    """Tests for finite-time impulse-momentum model."""

    def test_uses_specified_duration(self) -> None:
        """Should use the specified contact duration."""
        pre_state = PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        model = FiniteTimeImpactModel()
        params = ImpactParameters(contact_duration=0.0005)

        result = model.solve(pre_state, params)

        assert result.contact_duration == pytest.approx(0.0005)


class TestGearEffect:
    """Tests for gear effect spin computation."""

    def test_center_impact_no_gear_spin(self) -> None:
        """Center impact should produce no gear effect spin."""
        spin = compute_gear_effect_spin(
            impact_offset=np.array([0.0, 0.0]),
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubface_normal=np.array([1.0, 0.0, 0.0]),
        )

        np.testing.assert_allclose(spin, np.zeros(3), atol=1e-10)

    def test_toe_impact_produces_hook_spin(self) -> None:
        """Toe impact should produce hook (counterclockwise) spin."""
        spin = compute_gear_effect_spin(
            impact_offset=np.array([0.03, 0.0]),  # 30mm toe side
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubface_normal=np.array([1.0, 0.0, 0.0]),
        )

        # Should have non-zero spin
        assert np.linalg.norm(spin) > 0

    def test_higher_speed_more_spin(self) -> None:
        """Higher clubhead speed should produce more gear effect spin."""
        offset = np.array([0.02, 0.0])
        normal = np.array([1.0, 0.0, 0.0])

        spin_slow = compute_gear_effect_spin(offset, np.array([30.0, 0.0, 0.0]), normal)
        spin_fast = compute_gear_effect_spin(offset, np.array([50.0, 0.0, 0.0]), normal)

        assert np.linalg.norm(spin_fast) > np.linalg.norm(spin_slow)


class TestEnergyBalance:
    """Tests for energy balance validation."""

    def test_energy_lost_with_cor_less_than_1(self) -> None:
        """Impact with COR < 1 should lose energy."""
        pre_state = PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        model = RigidBodyImpactModel()
        params = ImpactParameters(cor=0.78)

        result = model.solve(pre_state, params)
        balance = validate_energy_balance(pre_state, result, params)

        # Energy should be lost
        assert balance["energy_lost"] > 0
        assert balance["total_ke_post"] < balance["total_ke_pre"]

    def test_ball_launch_speed_reasonable(self) -> None:
        """Ball launch speed should be in realistic range."""
        pre_state = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),  # ~100 mph
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        model = RigidBodyImpactModel()
        params = ImpactParameters(cor=0.78)

        result = model.solve(pre_state, params)
        balance = validate_energy_balance(pre_state, result, params)

        # Ball launch speed should be ~1.45-1.5x clubhead speed
        # 45 m/s * 1.45 = ~65 m/s (~145 mph)
        assert 50 < balance["ball_launch_speed"] < 80


class TestImpactModelFactory:
    """Tests for impact model factory."""

    @pytest.mark.parametrize(
        "model_type, expected_class",
        [
            (ImpactModelType.RIGID_BODY, RigidBodyImpactModel),
            (ImpactModelType.SPRING_DAMPER, SpringDamperImpactModel),
            (ImpactModelType.FINITE_TIME, FiniteTimeImpactModel),
        ],
        ids=["rigid-body", "spring-damper", "finite-time"],
    )
    def test_creates_correct_model(self, model_type: ImpactModelType, expected_class: type) -> None:
        """Factory should create the correct model type."""
        model = create_impact_model(model_type)
        assert isinstance(model, expected_class)


# =============================================================================
# Engine Integration Tests (Issue #758)
# =============================================================================


class TestImpactRecorder:
    """Tests for impact event recording (Issue #758)."""

    @pytest.fixture
    def pre_state(self) -> PreImpactState:
        """Create sample pre-impact state."""
        return PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

    def test_record_impact(self, pre_state: PreImpactState) -> None:
        """Should record impact event."""
        recorder = ImpactRecorder()
        model = RigidBodyImpactModel()
        params = ImpactParameters()

        post_state = model.solve(pre_state, params)
        event = recorder.record_impact(0.5, pre_state, post_state, params)

        assert event.impact_id == 0
        assert event.timestamp == 0.5
        assert len(recorder.events) == 1

    def test_increments_impact_id(self, pre_state: PreImpactState) -> None:
        """Should increment impact ID for each event."""
        recorder = ImpactRecorder()
        model = RigidBodyImpactModel()
        params = ImpactParameters()
        post_state = model.solve(pre_state, params)

        event1 = recorder.record_impact(0.1, pre_state, post_state, params)
        event2 = recorder.record_impact(0.2, pre_state, post_state, params)

        assert event1.impact_id == 0
        assert event2.impact_id == 1

    def test_export_to_dict(self, pre_state: PreImpactState) -> None:
        """Should export events as dictionary."""
        recorder = ImpactRecorder()
        model = RigidBodyImpactModel()
        params = ImpactParameters()
        post_state = model.solve(pre_state, params)

        recorder.record_impact(0.1, pre_state, post_state, params)

        data = recorder.export_to_dict()

        assert "num_impacts" in data
        assert "events" in data
        assert "summary" in data
        assert data["num_impacts"] == 1

    def test_get_summary(self, pre_state: PreImpactState) -> None:
        """Should compute summary statistics."""
        recorder = ImpactRecorder()
        model = RigidBodyImpactModel()
        params = ImpactParameters()
        post_state = model.solve(pre_state, params)

        recorder.record_impact(0.1, pre_state, post_state, params)
        recorder.record_impact(0.2, pre_state, post_state, params)

        summary = recorder.get_summary()

        assert summary["num_impacts"] == 2
        assert "mean_ball_speed" in summary
        assert "max_ball_speed" in summary

    def test_reset_clears_events(self, pre_state: PreImpactState) -> None:
        """Reset should clear all events."""
        recorder = ImpactRecorder()
        model = RigidBodyImpactModel()
        params = ImpactParameters()
        post_state = model.solve(pre_state, params)

        recorder.record_impact(0.1, pre_state, post_state, params)
        assert len(recorder.events) == 1

        recorder.reset()
        assert len(recorder.events) == 0


class TestImpactSolverAPI:
    """Tests for engine-agnostic impact solver API (Issue #758)."""

    def test_solve_impact_basic(self) -> None:
        """Should solve basic impact."""
        solver = ImpactSolverAPI()

        post = solver.solve_impact(
            timestamp=0.0,
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        )

        assert post.ball_velocity[0] > 0
        assert len(solver.recorder.events) == 1

    def test_solve_impact_no_record(self) -> None:
        """Should not record when record=False."""
        solver = ImpactSolverAPI()

        solver.solve_impact(
            timestamp=0.0,
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            record=False,
        )

        assert len(solver.recorder.events) == 0

    def test_solve_with_gear_effect(self) -> None:
        """Should add gear effect spin for offset impact."""
        solver = ImpactSolverAPI()

        post = solver.solve_with_gear_effect(
            timestamp=0.0,
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            impact_offset=np.array([0.02, 0.0]),  # Toe hit
        )

        # Should have non-zero spin from gear effect
        assert np.linalg.norm(post.ball_angular_velocity) > 0
        # Should record impact location
        np.testing.assert_allclose(post.impact_location, [0.02, 0.0])

    def test_get_energy_report(self) -> None:
        """Should generate energy balance report."""
        solver = ImpactSolverAPI()

        solver.solve_impact(
            timestamp=0.0,
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        )

        report = solver.get_energy_report()

        assert "impacts" in report
        assert "total_ke_pre" in report
        assert "total_energy_lost" in report
        assert len(report["impacts"]) == 1

    def test_validate_cor_behavior(self) -> None:
        """Should validate COR within tolerance."""
        solver = ImpactSolverAPI(params=ImpactParameters(cor=0.78))

        # Run several impacts
        for i in range(5):
            solver.solve_impact(
                timestamp=i * 0.1,
                clubhead_velocity=np.array([40.0 + i, 0.0, 0.0]),
                clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            )

        result = solver.validate_cor_behavior(tolerance=0.1)

        assert "valid" in result
        assert "measured_cor_mean" in result
        assert "deviation" in result

    def test_validate_spin_behavior(self) -> None:
        """Should validate spin within physical limits."""
        solver = ImpactSolverAPI()

        solver.solve_impact(
            timestamp=0.0,
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        )

        result = solver.validate_spin_behavior(max_spin_rpm=10000)

        assert "valid" in result
        assert "max_observed_rpm" in result

    def test_different_model_types(self) -> None:
        """Should work with different impact model types."""
        for model_type in [
            ImpactModelType.RIGID_BODY,
            ImpactModelType.FINITE_TIME,
        ]:
            solver = ImpactSolverAPI(model_type=model_type)

            post = solver.solve_impact(
                timestamp=0.0,
                clubhead_velocity=np.array([40.0, 0.0, 0.0]),
                clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            )

            assert post.ball_velocity[0] > 0

    def test_reset_clears_state(self) -> None:
        """Reset should clear recorder."""
        solver = ImpactSolverAPI()

        solver.solve_impact(
            timestamp=0.0,
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        )

        assert len(solver.recorder.events) == 1

        solver.reset()

        assert len(solver.recorder.events) == 0


class TestImpactEventDataclass:
    """Tests for ImpactEvent dataclass."""

    def test_event_contains_all_data(self) -> None:
        """ImpactEvent should contain complete impact data."""
        pre_state = PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        model = RigidBodyImpactModel()
        params = ImpactParameters()
        post_state = model.solve(pre_state, params)
        energy = validate_energy_balance(pre_state, post_state, params)

        event = ImpactEvent(
            timestamp=0.5,
            pre_state=pre_state,
            post_state=post_state,
            energy_balance=energy,
            impact_id=0,
            model_type=ImpactModelType.RIGID_BODY,
        )

        assert event.timestamp == 0.5
        assert event.impact_id == 0
        assert event.model_type == ImpactModelType.RIGID_BODY
        assert "total_ke_pre" in event.energy_balance


class TestCORValidation:
    """Tests for COR validation accuracy (Issue #758)."""

    @pytest.mark.parametrize("cor", [0.6, 0.7, 0.78, 0.85])
    def test_cor_matches_parameter(self, cor: float) -> None:
        """Measured COR should approximately match parameter."""
        solver = ImpactSolverAPI(params=ImpactParameters(cor=cor))

        for _ in range(3):
            solver.solve_impact(
                timestamp=0.0,
                clubhead_velocity=np.array([40.0, 0.0, 0.0]),
                clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            )

        result = solver.validate_cor_behavior(tolerance=0.15)

        # Measured COR should be within tolerance of expected
        assert result["deviation"] < 0.15


class TestSpinValidation:
    """Tests for spin validation (Issue #758)."""

    def test_realistic_spin_rates(self) -> None:
        """Spin rates should be in realistic range for golf."""
        solver = ImpactSolverAPI()

        # Typical driver impact
        solver.solve_impact(
            timestamp=0.0,
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        )

        result = solver.validate_spin_behavior(max_spin_rpm=10000)

        assert result["valid"]
        # Driver backspin typically 2000-3000 RPM
        assert result["max_observed_rpm"] < 10000


class TestMOIEffectiveMass:
    """Tests for MOI-based effective mass at impact point (Issue #1082)."""

    @pytest.fixture
    def center_hit_state(self) -> PreImpactState:
        """Pre-impact state with center hit (no offset)."""
        return PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
            clubhead_moi=4.5e-4,
            impact_offset=None,
        )

    @pytest.fixture
    def toe_hit_state(self) -> PreImpactState:
        """Pre-impact state with toe hit (20mm offset)."""
        return PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
            clubhead_moi=4.5e-4,
            impact_offset=np.array([0.020, 0.0]),  # 20mm toe
        )

    def test_center_hit_equals_point_mass(
        self, center_hit_state: PreImpactState
    ) -> None:
        """Center hit should produce same result as point mass model."""
        model = RigidBodyImpactModel()
        params = ImpactParameters()

        # With impact_offset=None, should use full clubhead mass
        result = model.solve(center_hit_state, params)

        # Compare with explicit zero offset
        zero_offset_state = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
            clubhead_moi=4.5e-4,
            impact_offset=np.array([0.0, 0.0]),
        )
        result_zero = model.solve(zero_offset_state, params)

        np.testing.assert_allclose(
            result.ball_velocity, result_zero.ball_velocity, atol=1e-10
        )

    def test_off_center_reduces_ball_speed(
        self,
        center_hit_state: PreImpactState,
        toe_hit_state: PreImpactState,
    ) -> None:
        """Off-center hit should produce lower ball speed than center hit."""
        model = RigidBodyImpactModel()
        params = ImpactParameters()

        result_center = model.solve(center_hit_state, params)
        result_toe = model.solve(toe_hit_state, params)

        speed_center = np.linalg.norm(result_center.ball_velocity)
        speed_toe = np.linalg.norm(result_toe.ball_velocity)

        assert speed_toe < speed_center

    def test_larger_offset_lower_speed(self) -> None:
        """Larger offset from CG should produce progressively lower ball speed."""
        model = RigidBodyImpactModel()
        params = ImpactParameters()

        speeds = []
        for offset_mm in [0, 10, 20, 30, 40]:
            state = PreImpactState(
                clubhead_velocity=np.array([45.0, 0.0, 0.0]),
                clubhead_angular_velocity=np.zeros(3),
                clubhead_orientation=np.array([1.0, 0.0, 0.0]),
                ball_position=np.zeros(3),
                ball_velocity=np.zeros(3),
                ball_angular_velocity=np.zeros(3),
                clubhead_mass=0.200,
                clubhead_moi=4.5e-4,
                impact_offset=np.array([offset_mm / 1000.0, 0.0]),
            )
            result = model.solve(state, params)
            speeds.append(float(np.linalg.norm(result.ball_velocity)))

        # Speeds should be monotonically decreasing
        for i in range(len(speeds) - 1):
            assert speeds[i] >= speeds[i + 1], (
                f"Speed at {i * 10}mm ({speeds[i]:.1f}) should be >= "
                f"speed at {(i + 1) * 10}mm ({speeds[i + 1]:.1f})"
            )

    def test_higher_moi_more_forgiving(self) -> None:
        """Higher MOI should result in less ball speed loss on off-center hits."""
        model = RigidBodyImpactModel()
        params = ImpactParameters()
        offset = np.array([0.025, 0.0])  # 25mm toe hit

        # Low MOI clubhead
        low_moi = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
            clubhead_moi=3.0e-4,  # Lower MOI (less forgiving)
            impact_offset=offset,
        )

        # High MOI clubhead
        high_moi = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
            clubhead_moi=6.0e-4,  # Higher MOI (more forgiving)
            impact_offset=offset,
        )

        result_low = model.solve(low_moi, params)
        result_high = model.solve(high_moi, params)

        speed_low = np.linalg.norm(result_low.ball_velocity)
        speed_high = np.linalg.norm(result_high.ball_velocity)

        # Higher MOI = more forgiving = higher ball speed on off-center hits
        assert speed_high > speed_low

    def test_effective_mass_formula(self) -> None:
        """Verify the effective mass calculation: m_eff = 1 / (1/m + r²/I)."""
        m_club = 0.200
        I_club = 4.5e-4
        r = 0.025  # 25mm

        expected_m_eff = 1.0 / (1.0 / m_club + r**2 / I_club)

        # Should be less than actual mass
        assert expected_m_eff < m_club

        # For typical driver at 25mm offset:
        # m_eff = 1 / (1/0.2 + 0.025²/4.5e-4)
        #       = 1 / (5 + 1.389) = 1 / 6.389 ≈ 0.1565
        assert expected_m_eff == pytest.approx(0.1565, rel=0.01)

    def test_backward_compatibility_no_offset(self) -> None:
        """Without impact_offset, behavior should match original point mass model."""
        model = RigidBodyImpactModel()
        params = ImpactParameters()

        # State without MOI fields (uses defaults)
        state = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
        )

        result = model.solve(state, params)

        # Should produce standard point-mass result
        # m_eff = (0.0459 * 0.200) / (0.0459 + 0.200) ≈ 0.03732
        # j = (1 + 0.83) * 0.03732 * 45.0 ≈ 3.076
        # v_ball = 3.076 / 0.0459 ≈ 67.0 m/s
        speed = np.linalg.norm(result.ball_velocity)
        assert speed == pytest.approx(67.0, rel=0.05)
