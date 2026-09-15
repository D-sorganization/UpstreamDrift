"""Regression tests for GS-04 (Issue #10193): Impact State Preservation and Contact Qualification.

Demonstrates and verifies:
1. Missing field regression: clubhead_angular_velocity, clubhead_loft, clubhead_lie, clubhead_moi
   must be preserved through the public solver API.
2. Direct PreImpactState solving via ImpactSolverAPI.solve_pre_impact_state without discarding fields.
3. SwingBallFlightPipeline preserves full clubhead angular velocity and parameters.
4. Contact qualification logic: distinguishing MODEL_CONTACT vs DEMO_PEAK_SPEED vs MANUAL,
   single contact deduplication, rejection of no-contact or non-physical approach.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.golf_simulator.contracts import (
    AimContext,
    ContactStatus,
    NumericalStatus,
    ScientificStatus,
    ShotMetadata,
    ShotQualification,
    SourceKind,
)
from src.shared.python.golf_simulator.launch_bridge import (
    qualify_impact_contact,
    pipeline_result_to_shot_envelope,
    post_impact_state_to_shot_envelope,
)
from src.shared.python.physics.impact_model import (
    ImpactParameters,
    ImpactSolverAPI,
    PostImpactState,
    PreImpactState,
)
from src.shared.python.physics.swing_ball_flight_pipeline import (
    PipelineResult,
    SwingBallFlightPipeline,
    SwingState,
)

pytestmark = pytest.mark.unit


def _identity_aim() -> AimContext:
    return AimContext(
        source_to_target_rotation=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        revision=1,
        provenance="identity",
    )


class TestImpactStatePreservationRegression:
    """Regression tests verifying that solver APIs preserve full pre-impact state."""

    def test_solve_impact_preserves_angular_velocity_and_club_parameters(self) -> None:
        """ImpactSolverAPI.solve_impact must preserve non-zero angular velocity and parameters."""
        solver = ImpactSolverAPI()
        omega_club = np.array([10.0, 20.0, 30.0])
        loft = np.radians(12.5)
        lie = np.radians(58.0)
        moi = 5.2e-4

        post = solver.solve_impact(
            timestamp=0.15,
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            clubhead_angular_velocity=omega_club,
            clubhead_loft=loft,
            clubhead_lie=lie,
            clubhead_moi=moi,
            record=True,
        )

        # In RigidBodyImpactModel, post clubhead angular velocity retains pre-state angular velocity
        np.testing.assert_allclose(post.clubhead_angular_velocity, omega_club)

        # Recorded event must also preserve all fields in pre_state
        assert len(solver.recorder.events) == 1
        recorded_pre = solver.recorder.events[0].pre_state
        np.testing.assert_allclose(recorded_pre.clubhead_angular_velocity, omega_club)
        assert recorded_pre.clubhead_loft == pytest.approx(loft)
        assert recorded_pre.clubhead_lie == pytest.approx(lie)
        assert recorded_pre.clubhead_moi == pytest.approx(moi)

    def test_solve_pre_impact_state_public_contract(self) -> None:
        """ImpactSolverAPI.solve_pre_impact_state solves directly without slicing/dropping."""
        solver = ImpactSolverAPI()
        omega_club = np.array([5.0, 15.0, 25.0])
        loft = np.radians(9.0)
        lie = np.radians(59.0)
        moi = 4.8e-4
        offset = np.array([0.015, -0.005])

        pre = PreImpactState(
            clubhead_velocity=np.array([48.0, -1.0, 2.0]),
            clubhead_angular_velocity=omega_club,
            clubhead_orientation=np.array([0.98, 0.0, 0.198]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.205,
            clubhead_loft=loft,
            clubhead_lie=lie,
            clubhead_moi=moi,
            impact_offset=offset,
        )

        post = solver.solve_pre_impact_state(timestamp=0.22, pre_state=pre, record=True)

        np.testing.assert_allclose(post.clubhead_angular_velocity, omega_club)
        np.testing.assert_allclose(post.impact_location, offset)

        assert len(solver.recorder.events) == 1
        recorded_event = solver.recorder.events[0]
        assert recorded_event.timestamp == pytest.approx(0.22)
        np.testing.assert_allclose(
            recorded_event.pre_state.clubhead_angular_velocity, omega_club
        )
        assert recorded_event.pre_state.clubhead_loft == pytest.approx(loft)
        assert recorded_event.pre_state.clubhead_lie == pytest.approx(lie)
        assert recorded_event.pre_state.impact_offset is not None
        np.testing.assert_allclose(recorded_event.pre_state.impact_offset, offset)

    def test_solve_with_gear_effect_preserves_angular_velocity_and_parameters(
        self,
    ) -> None:
        """solve_with_gear_effect preserves angular velocity and passes parameters through."""
        solver = ImpactSolverAPI()
        omega_club = np.array([8.0, 12.0, 16.0])
        loft = np.radians(10.5)
        lie = np.radians(60.0)
        moi = 5.0e-4
        offset = np.array([0.01, 0.0])

        post = solver.solve_with_gear_effect(
            timestamp=0.1,
            clubhead_velocity=np.array([44.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            impact_offset=offset,
            clubhead_angular_velocity=omega_club,
            clubhead_loft=loft,
            clubhead_lie=lie,
            clubhead_moi=moi,
            record=True,
        )

        np.testing.assert_allclose(post.clubhead_angular_velocity, omega_club)
        assert len(solver.recorder.events) == 1
        rec_pre = solver.recorder.events[0].pre_state
        np.testing.assert_allclose(rec_pre.clubhead_angular_velocity, omega_club)
        assert rec_pre.clubhead_loft == pytest.approx(loft)
        assert rec_pre.clubhead_lie == pytest.approx(lie)
        assert rec_pre.clubhead_moi == pytest.approx(moi)

    def test_swing_ball_flight_pipeline_forwards_full_pre_state(self) -> None:
        """SwingBallFlightPipeline forwards pre-impact angular velocity and MOI to solver."""
        from src.shared.python.physics.ball_launch_conditions import (
            LaunchConditions,
            TrajectoryPoint,
        )

        class _MockSim:
            def simulate_trajectory(
                self,
                launch: LaunchConditions,
                max_time: float = 15.0,
                dt: float = 0.01,
            ) -> list[TrajectoryPoint]:
                return [
                    TrajectoryPoint(
                        time=0.0,
                        position=np.zeros(3),
                        velocity=np.array([60.0, 0.0, 10.0]),
                        acceleration=np.zeros(3),
                        forces={},
                    ),
                    TrajectoryPoint(
                        time=1.0,
                        position=np.array([60.0, 0.0, 5.0]),
                        velocity=np.array([55.0, 0.0, -2.0]),
                        acceleration=np.zeros(3),
                        forces={},
                    ),
                ]

        pipeline = SwingBallFlightPipeline(flight_simulator=_MockSim())
        omega_club = np.array([12.0, 24.0, 36.0])
        swing = SwingState(
            clubhead_velocity=np.array([46.0, 0.0, 0.0]),
            clubhead_angular_velocity=omega_club,
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            clubhead_mass=0.200,
            clubhead_loft_deg=10.5,
            clubhead_moi=5.1e-4,
            impact_offset=np.array([0.01, 0.0]),
            engine_name="mujoco",
        )

        result = pipeline.run(swing)
        # Verify that post-impact state retained clubhead angular velocity
        np.testing.assert_allclose(
            result.impact_state.clubhead_angular_velocity, omega_club
        )


class TestContactQualificationAndExtraction:
    """Tests for qualifying contact and provenance linking."""

    def test_qualify_impact_contact_model_contact(self) -> None:
        """Legitimate model contact produces qualified status with physical smash factor."""
        pre = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
            clubhead_loft=np.radians(10.5),
            clubhead_moi=5.0e-4,
        )
        post = PostImpactState(
            ball_velocity=np.array([65.0, 0.0, 10.0]),
            ball_angular_velocity=np.array([0.0, -300.0, 0.0]),
            clubhead_velocity=np.array([32.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            contact_duration=0.00045,
            energy_transfer=140.0,
            impact_location=np.zeros(2),
        )

        qual = qualify_impact_contact(
            pre_state=pre,
            post_state=post,
            source_kind=SourceKind.MODEL_CONTACT,
            engine_name="mujoco",
        )

        assert qual.contact == ContactStatus.QUALIFIED
        assert qual.numerical == NumericalStatus.CONVERGED
        assert qual.scientific == ScientificStatus.BENCHMARKED
        assert "engine:mujoco" in qual.evidence_refs
        assert any(ref.startswith("smash_factor:") for ref in qual.evidence_refs)

    def test_qualify_impact_contact_demo_peak_speed(self) -> None:
        """Demo peak speed heuristic is explicitly flagged with DEMO_ONLY and PEAK_SPEED_HEURISTIC."""
        pre = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )
        post = PostImpactState(
            ball_velocity=np.array([65.0, 0.0, 10.0]),
            ball_angular_velocity=np.array([0.0, -300.0, 0.0]),
            clubhead_velocity=np.array([32.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            contact_duration=0.00045,
            energy_transfer=140.0,
            impact_location=np.zeros(2),
        )

        qual = qualify_impact_contact(
            pre_state=pre,
            post_state=post,
            source_kind=SourceKind.DEMO_PEAK_SPEED,
            engine_name="mujoco",
        )

        assert qual.contact == ContactStatus.DEMO_ONLY
        assert qual.scientific == ScientificStatus.PEAK_SPEED_HEURISTIC

    def test_qualify_impact_contact_rejects_negative_approach_velocity(self) -> None:
        """Negative approach velocity (club moving away from ball) is not a contact."""
        pre = PreImpactState(
            clubhead_velocity=np.array([-10.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )
        post = PostImpactState(
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_velocity=np.array([-10.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            contact_duration=0.0,
            energy_transfer=0.0,
            impact_location=np.zeros(2),
        )

        with pytest.raises(ValueError, match="negative or zero approach velocity"):
            qualify_impact_contact(
                pre_state=pre,
                post_state=post,
                source_kind=SourceKind.MODEL_CONTACT,
                engine_name="mujoco",
            )

    def test_qualify_impact_contact_rejects_unphysical_smash_factor(self) -> None:
        """Smash factor beyond theoretical conservation (> 1.60 for golf ball) raises ValueError."""
        pre = PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )
        post = PostImpactState(
            ball_velocity=np.array([100.0, 0.0, 0.0]),  # 100 / 40 = 2.5 > 1.6
            ball_angular_velocity=np.zeros(3),
            clubhead_velocity=np.array([30.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            contact_duration=0.00045,
            energy_transfer=500.0,
            impact_location=np.zeros(2),
        )

        with pytest.raises(ValueError, match="Smash factor"):
            qualify_impact_contact(
                pre_state=pre,
                post_state=post,
                source_kind=SourceKind.MODEL_CONTACT,
                engine_name="mujoco",
            )

    def test_single_contact_event_deduplication(self) -> None:
        """Multiple contact samples within an impact window produce only one qualified shot event."""
        from src.shared.python.golf_simulator.launch_bridge import (
            extract_single_contact_event,
        )

        # Simulate a contact time series with multiple consecutive sample points in contact
        samples = [
            {"time_s": 0.200, "in_contact": False, "normal_force": 0.0},
            {"time_s": 0.201, "in_contact": True, "normal_force": 1200.0},
            {"time_s": 0.202, "in_contact": True, "normal_force": 2500.0},  # Peak
            {"time_s": 0.203, "in_contact": True, "normal_force": 800.0},
            {"time_s": 0.204, "in_contact": False, "normal_force": 0.0},
            {"time_s": 0.205, "in_contact": False, "normal_force": 0.0},
        ]

        events = extract_single_contact_event(samples)
        assert len(events) == 1
        assert events[0]["peak_time_s"] == pytest.approx(0.202)
        assert events[0]["max_normal_force"] == pytest.approx(2500.0)

    def test_unsupported_engine_providers_remain_unavailable(self) -> None:
        """Drake and Pinocchio providers remain explicitly unavailable."""
        from src.shared.python.physics.swing_state_providers import (
            UnimplementedEngineProvider,
            available_swing_state_providers,
        )

        providers = {p.provider_id: p for p in available_swing_state_providers()}
        assert "drake" in providers
        assert not providers["drake"].is_available()
        assert "pinocchio" in providers
        assert not providers["pinocchio"].is_available()

    def test_simscape_acceptance_explicitly_requires_matlab_r2025b(self) -> None:
        """Simscape qualification policy requires MATLAB R2025b explicitly."""
        from src.shared.python.golf_simulator.launch_bridge import (
            check_simscape_eligibility,
        )

        assert not check_simscape_eligibility(matlab_release="R2024b")
        assert not check_simscape_eligibility(matlab_release=None)
        assert check_simscape_eligibility(matlab_release="R2025b")
