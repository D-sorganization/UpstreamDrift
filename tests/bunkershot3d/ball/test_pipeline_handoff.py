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

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.ball.lie import BallLie, BallProperties
from bunkershot3d.ball.pipeline import (
    BunkerShotState,
    compute_bunker_launch,
    to_post_impact_state,
)
from src.shared.python.physics.impact_model import PostImpactState

from .test_splash_transfer import delivery

pytestmark = pytest.mark.unit


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

    def test_the_contact_duration_is_the_measured_one(self) -> None:
        state = nominal_state(delivery=delivery(contact_duration_s=0.0072))
        post = to_post_impact_state(compute_bunker_launch(state), state)
        assert post.contact_duration == pytest.approx(0.0072)

    def test_energy_transfer_reported(self) -> None:
        state = nominal_state()
        post = to_post_impact_state(compute_bunker_launch(state), state)
        assert post.energy_transfer > 0


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
