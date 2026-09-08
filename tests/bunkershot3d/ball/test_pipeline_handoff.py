"""Bunker shot to flight-pipeline handoff (issues #8613, #8657).

The handoff carries the ball's launch into ``SwingBallFlightPipeline``. Since
#8657 it also carries the *head's* exit: the clubhead state after the sand is
the solver's measured exit speed and engagement time, not the flat 35 % energy
retention and 5 ms the previous version assumed.

What a :class:`BunkerShotState` may no longer be built from is as much the
point as what it produces. ``entry_depth_m``, ``sole_width_m`` and
``sole_length_m`` fed the deleted box-volume estimate of displaced sand; they
are gone, and a state without a measured
:class:`~bunkershot3d.ball.splash.SandDelivery` cannot be constructed at all.
"""

import json
import math

import numpy as np
import pytest

from bunkershot3d.ball.lie import BallLie, BallProperties
from bunkershot3d.ball.pipeline import (
    HEAD_FRAME_TO_FLIGHT_TRANSFORM,
    BunkerShotState,
    ExitVectorProvenance,
    PostImpactEnvelope,
    compute_bunker_launch,
    post_impact_envelope,
    to_post_impact_state,
)
from bunkershot3d.solvers import EnvelopeStatus, FidelityTier, evaluate_envelope
from src.shared.python.physics.impact_model import PostImpactState

from .test_splash_transfer import delivery

pytestmark = pytest.mark.unit

#: The three feature scales the F0 tier judges every shot at, as in
#: ``test_splash_transfer``.
_FEATURE_LENGTHS_M = {"clubhead": 0.1, "sole width": 0.03, "leading edge": 0.005}


def nominal_state(**overrides: object) -> BunkerShotState:
    """Return a greenside bunker shot with a measured strike behind it.

    Args:
        **overrides: Fields to replace.

    Returns:
        The state.
    """
    fields: dict[str, object] = {
        "club_loft_deg": 56.0,
        "ball_lie": BallLie(depth_m=0.005),
        "delivery": delivery(),
        "club_mass_kg": 0.30,
    }
    fields.update(overrides)
    return BunkerShotState(**fields)  # type: ignore[arg-type]


class TestBunkerShotState:
    def test_a_state_carries_the_measured_strike(self) -> None:
        state = nominal_state()
        assert state.club_loft_deg == 56.0
        assert state.delivery.impulse_n_s > 0.0
        assert state.delivery.displaced_mass_kg > 0.0

    def test_the_deleted_box_volume_inputs_are_gone(self) -> None:
        for gone in ("entry_depth_m", "sole_width_m", "sole_length_m"):
            assert not hasattr(nominal_state(), gone)

    def test_a_state_without_a_measured_strike_is_refused(self) -> None:
        with pytest.raises(ValueError, match="measured"):
            BunkerShotState(
                club_loft_deg=56.0,
                ball_lie=BallLie(depth_m=0.005),
                delivery=None,  # type: ignore[arg-type]
            )


class TestComputeBunkerLaunch:
    def test_produces_valid_launch(self) -> None:
        result = compute_bunker_launch(nominal_state())
        assert result.ball_speed_m_s > 0
        assert 0 < result.launch_angle_rad < math.pi / 2

    def test_produces_energy_accounting(self) -> None:
        result = compute_bunker_launch(nominal_state())
        assert 0 < result.energy_transfer_fraction < 1

    def test_the_launch_carries_its_verdict_and_provenance(self) -> None:
        result = compute_bunker_launch(nominal_state())
        assert result.verdict.groups
        assert result.measured_constants() == ()

    def test_a_harder_strike_on_the_same_divot_launches_the_ball_faster(self) -> None:
        """The #8657 defect, checked through the pipeline entry point."""
        soft = compute_bunker_launch(
            nominal_state(delivery=delivery(impulse_n_s=3.0, displaced_mass_kg=0.25))
        )
        hard = compute_bunker_launch(
            nominal_state(delivery=delivery(impulse_n_s=6.0, displaced_mass_kg=0.25))
        )
        assert hard.ball_speed_m_s > soft.ball_speed_m_s


class TestToPostImpactState:
    def test_converts_to_post_impact_state(self) -> None:
        state = nominal_state()
        post = to_post_impact_state(compute_bunker_launch(state), state)
        assert isinstance(post, PostImpactState)
        assert len(post.ball_velocity) == 3
        assert len(post.ball_angular_velocity) == 3

    def test_ball_velocity_matches_launch(self) -> None:
        state = nominal_state()
        result = compute_bunker_launch(state)
        post = to_post_impact_state(result, state)
        speed = np.linalg.norm(post.ball_velocity)
        assert speed == pytest.approx(result.ball_speed_m_s, rel=1e-6)

    def test_clubhead_velocity_is_the_solver_s_exit_speed(self) -> None:
        state = nominal_state()
        post = to_post_impact_state(compute_bunker_launch(state), state)
        clubhead_speed = float(np.linalg.norm(post.clubhead_velocity))
        assert clubhead_speed == pytest.approx(state.delivery.exit_speed_m_s)
        assert clubhead_speed < state.delivery.entry_speed_m_s

    def test_preserves_actual_exit_velocity_and_angular_velocity(self) -> None:
        """Issue #9542: Non-zero lateral velocity and spin must not be flattened or dropped."""
        custom_exit_vel = (12.0, 3.5, -4.2)
        custom_exit_spin = (0.0, 45.0, -10.0)
        state = nominal_state(
            delivery=delivery(
                speed_m_s=25.0,
                exit_speed_m_s=math.hypot(12.0, 3.5, 4.2),
                exit_velocity_m_s=custom_exit_vel,
                exit_angular_velocity_rad_s=custom_exit_spin,
            )
        )
        post = to_post_impact_state(compute_bunker_launch(state), state)
        np.testing.assert_allclose(
            post.clubhead_velocity, custom_exit_vel, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            post.clubhead_angular_velocity, custom_exit_spin, rtol=0, atol=1e-12
        )


class TestEnergyAccounting:
    """Head kinetic energy in = sand dissipation + ball energy + head out."""

    def test_energy_accounting_closes(self) -> None:
        state = nominal_state()
        ball = BallProperties()
        result = compute_bunker_launch(state)
        post = to_post_impact_state(result, state)

        head_ke_in = 0.5 * state.club_mass_kg * state.delivery.entry_speed_m_s**2
        ball_ke = 0.5 * ball.mass_kg * result.ball_speed_m_s**2
        ball_rot_ke = (
            0.5 * ball.moi_kg_m2 * (result.spin_rate_rpm * 2 * math.pi / 60) ** 2
        )
        head_ke_out = (
            0.5
            * state.club_mass_kg
            * float(np.linalg.norm(post.clubhead_velocity)) ** 2
        )
        sand_dissipation = head_ke_in - (ball_ke + ball_rot_ke + head_ke_out)

        assert sand_dissipation > 0, "sand must dissipate energy"
        assert sand_dissipation < head_ke_in, "cannot dissipate more than went in"
        assert sand_dissipation > 0.5 * head_ke_in, "a splash shot is mostly sand"


class TestLaunchConditionsSanity:
    """Order-of-magnitude checks. Not validation: no measurement exists."""

    def test_a_greenside_shot_is_physically_sane(self) -> None:
        result = compute_bunker_launch(nominal_state())
        assert 0.5 < result.ball_speed_m_s < 25
        assert 25 < math.degrees(result.launch_angle_rad) < 70
        assert 200 < result.spin_rate_rpm < 12000


class TestExitProvenance:
    """Issue #9542 (reopen), defect 2: actual vs modelled, typed, not implied."""

    def test_a_synthesised_direction_is_labelled_modeled_convention(self) -> None:
        state = nominal_state()
        envelope = post_impact_envelope(compute_bunker_launch(state), state)
        assert (
            envelope.clubhead_velocity_provenance
            is ExitVectorProvenance.MODELED_CONVENTION
        )
        assert (
            envelope.clubhead_angular_velocity_provenance
            is ExitVectorProvenance.MODELED_CONVENTION
        )

    def test_supplied_exit_vectors_are_labelled_actual(self) -> None:
        state = nominal_state(
            delivery=delivery(
                speed_m_s=25.0,
                exit_speed_m_s=math.hypot(12.0, 3.5, 4.2),
                exit_velocity_m_s=(-12.0, 3.5, -4.2),
                exit_angular_velocity_rad_s=(0.0, 45.0, -10.0),
            )
        )
        envelope = post_impact_envelope(compute_bunker_launch(state), state)
        assert (
            envelope.clubhead_velocity_provenance
            is ExitVectorProvenance.ACTUAL_EXIT_STATE
        )
        assert (
            envelope.clubhead_angular_velocity_provenance
            is ExitVectorProvenance.ACTUAL_EXIT_STATE
        )

    def test_a_legitimately_zero_twist_is_actual_not_modeled(self) -> None:
        state = nominal_state(
            delivery=delivery(exit_angular_velocity_rad_s=(0.0, 0.0, 0.0))
        )
        envelope = post_impact_envelope(compute_bunker_launch(state), state)
        assert (
            envelope.clubhead_angular_velocity_provenance
            is ExitVectorProvenance.ACTUAL_EXIT_STATE
        )

    def test_the_ball_launch_is_never_advertised_as_measured(self) -> None:
        state = nominal_state()
        envelope = post_impact_envelope(compute_bunker_launch(state), state)
        assert (
            envelope.ball_launch_provenance is ExitVectorProvenance.MODELED_CONVENTION
        )


class TestEnvelopeVerdictAndFidelity:
    """Issue #9542 (reopen), defect 1: the verdict survives the boundary."""

    def test_the_envelope_carries_the_launch_verdict(self) -> None:
        state = nominal_state()
        result = compute_bunker_launch(state)
        envelope = post_impact_envelope(result, state)
        assert envelope.verdict.status is result.verdict.status

    def test_the_nominal_shot_is_beyond_validation_and_f0(self) -> None:
        envelope = post_impact_envelope(
            compute_bunker_launch(nominal_state()), nominal_state()
        )
        assert envelope.verdict.status is EnvelopeStatus.BEYOND_VALIDATION
        assert envelope.fidelity_tier is FidelityTier.F0


class TestFrameTransform:
    """Issue #9542 (reopen), defect 5: the frames are declared, not assumed."""

    def test_the_declared_transform_is_a_proper_nontrivial_rotation(self) -> None:
        matrix = np.array(HEAD_FRAME_TO_FLIGHT_TRANSFORM, dtype=float)
        assert matrix.shape == (3, 3)
        assert np.all(np.isfinite(matrix))
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-12)
        assert not np.allclose(matrix, np.eye(3))

    def test_head_frame_travel_maps_to_forward(self) -> None:
        matrix = np.array(HEAD_FRAME_TO_FLIGHT_TRANSFORM, dtype=float)
        np.testing.assert_allclose(
            matrix @ (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), atol=1e-12
        )

    def test_the_transform_applies_to_linear_and_angular_vectors(self) -> None:
        linear = (-12.0, 3.5, -4.2)
        angular = (1.0, -2.0, 3.0)
        matrix = np.array(HEAD_FRAME_TO_FLIGHT_TRANSFORM, dtype=float)
        np.testing.assert_allclose(matrix @ linear, (12.0, -3.5, -4.2), atol=1e-12)
        np.testing.assert_allclose(matrix @ angular, (-1.0, 2.0, 3.0), atol=1e-12)


class TestEnvelopeHandoff:
    """The envelope around the Tools-owned state: frames, digests, round-trip."""

    def test_actual_exit_vectors_can_be_expressed_in_the_flight_frame(self) -> None:
        state = nominal_state(
            delivery=delivery(
                speed_m_s=25.0,
                exit_speed_m_s=math.hypot(12.0, 3.5, 4.2),
                exit_velocity_m_s=(-12.0, 3.5, -4.2),
                exit_angular_velocity_rad_s=(1.0, -2.0, 3.0),
            )
        )
        envelope = post_impact_envelope(compute_bunker_launch(state), state)
        np.testing.assert_allclose(
            envelope.clubhead_velocity_in_flight_frame(), (12.0, -3.5, -4.2), atol=1e-12
        )
        np.testing.assert_allclose(
            envelope.clubhead_angular_velocity_in_flight_frame(),
            (-1.0, 2.0, 3.0),
            atol=1e-12,
        )

    def test_modeled_convention_vectors_are_already_flight_aligned(self) -> None:
        state = nominal_state()
        post = to_post_impact_state(compute_bunker_launch(state), state)
        envelope = post_impact_envelope(compute_bunker_launch(state), state)
        np.testing.assert_allclose(
            envelope.clubhead_velocity_in_flight_frame(),
            post.clubhead_velocity,
            atol=1e-12,
        )

    def test_envelope_state_vectors_are_owned(self) -> None:
        envelope = post_impact_envelope(
            compute_bunker_launch(nominal_state()), nominal_state()
        )
        with pytest.raises(ValueError):
            envelope.post_impact_state.ball_velocity[0] = float("nan")

    def test_serialization_round_trip_retains_verdict_and_provenance(self) -> None:
        state = nominal_state(
            delivery=delivery(
                speed_m_s=25.0,
                exit_speed_m_s=math.hypot(12.0, 3.5, 4.2),
                exit_velocity_m_s=(-12.0, 3.5, -4.2),
                exit_angular_velocity_rad_s=(0.0, 45.0, -10.0),
            )
        )
        envelope = post_impact_envelope(compute_bunker_launch(state), state)
        reloaded = PostImpactEnvelope.from_dict(
            json.loads(json.dumps(envelope.to_dict()))
        )
        assert reloaded.verdict.status is envelope.verdict.status
        assert reloaded.fidelity_tier is envelope.fidelity_tier
        assert (
            reloaded.clubhead_velocity_provenance
            is envelope.clubhead_velocity_provenance
        )
        assert reloaded.source_digest == envelope.source_digest
        np.testing.assert_allclose(
            reloaded.post_impact_state.ball_velocity,
            envelope.post_impact_state.ball_velocity,
            atol=1e-12,
        )

    def test_a_tampered_serialized_envelope_is_refused(self) -> None:
        envelope = post_impact_envelope(
            compute_bunker_launch(nominal_state()), nominal_state()
        )
        payload = envelope.to_dict()
        payload["post_impact_state"]["ball_velocity"][0] = 123.0
        with pytest.raises(ValueError, match="digest"):
            PostImpactEnvelope.from_dict(payload)

    def test_a_refused_verdict_survives_serialization(self) -> None:
        refused = evaluate_envelope(
            speed_m_s=25.0,
            feature_lengths_m=_FEATURE_LENGTHS_M,
            grain_diameter_m=0.0005,
            element_size_m=0.002,
            dynamic_terms_active=False,
        )
        assert refused.status is EnvelopeStatus.REFUSED
        state = nominal_state(delivery=delivery(verdict=refused))
        envelope = post_impact_envelope(compute_bunker_launch(state), state)
        reloaded = PostImpactEnvelope.from_dict(
            json.loads(json.dumps(envelope.to_dict()))
        )
        assert reloaded.verdict.status is EnvelopeStatus.REFUSED

    def test_an_unknown_schema_version_is_refused(self) -> None:
        envelope = post_impact_envelope(
            compute_bunker_launch(nominal_state()), nominal_state()
        )
        payload = envelope.to_dict()
        payload["schema_version"] = 999
        payload["source_digest"] = envelope.source_digest
        with pytest.raises(ValueError, match="schema_version"):
            PostImpactEnvelope.from_dict(payload)
