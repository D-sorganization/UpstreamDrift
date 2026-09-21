"""Contact regimes: no hit, direct strike, splash, buried no-release (#9544).

The splash partition is the only launch model this package has, and it
is only a model of a *splash*. A leading edge that reaches the ball
before it reaches the sand, a head that never comes out of the bed, and
a head that never met the bed at all are three different outcomes with
no launch in common, and none of them may be routed through the splash
partition to manufacture a carry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.ball import (
    BallLie,
    BallProperties,
    BunkerShotState,
    ContactType,
    RegimeClassification,
    StrikeOutcome,
    UnsupportedContactRegimeError,
    classify_contact_regime,
    compute_bunker_launch,
)
from bunkershot3d.geometry import build_wedge_mesh, get_preset, preset_names
from bunkershot3d.sand import PlayingCondition, playing_condition
from bunkershot3d.solvers import (
    DRFTSolver,
    HeadKinematics,
    MaterialResponse,
    RefusalPolicy,
    ShotSettings,
    SurfaceElements,
    simulate_shot,
)
from bunkershot3d.solvers.exceptions import OutOfEnvelopeError
from bunkershot3d.solvers.mpm.ball import BallSection
from bunkershot3d.solvers.mpm.envelope import RefusedQuantity

from .test_pipeline_handoff import nominal_state

pytestmark = pytest.mark.unit

_BALL = BallProperties()
_RADIUS = _BALL.radius_m


def _outcome(**overrides: object) -> StrikeOutcome:
    """A nominal greenside splash: in 40 mm behind, out 60 mm past."""
    fields: dict[str, object] = {
        "engaged": True,
        "exited": True,
        "entry_distance_behind_ball_m": 0.040,
        "exit_distance_past_ball_m": 0.060,
    }
    fields.update(overrides)
    return StrikeOutcome(**fields)  # type: ignore[arg-type]


def _classify(outcome: StrikeOutcome, depth_m: float = 0.005) -> RegimeClassification:
    return classify_contact_regime(outcome, lie=BallLie(depth_m=depth_m), ball=_BALL)


class TestTheFourOutcomesAreDistinguished:
    def test_a_head_that_never_met_the_bed_is_no_hit(self) -> None:
        verdict = _classify(_outcome(engaged=False, exited=True))
        assert verdict.regime is ContactType.NO_HIT
        assert not verdict.launch_derivable

    def test_a_head_that_never_came_out_is_buried_no_release(self) -> None:
        verdict = _classify(
            _outcome(
                exited=False,
                entry_distance_behind_ball_m=None,
                exit_distance_past_ball_m=None,
            )
        )
        assert verdict.regime is ContactType.BURIED_NO_RELEASE
        assert not verdict.launch_derivable

    def test_an_entry_inside_the_ball_footprint_is_a_direct_strike(self) -> None:
        verdict = _classify(_outcome(entry_distance_behind_ball_m=_RADIUS * 0.5))
        assert verdict.regime is ContactType.THIN
        assert not verdict.launch_derivable
        assert any("leading edge" in reason for reason in verdict.reasons)

    def test_an_entry_exactly_at_the_footprint_edge_is_still_direct(self) -> None:
        assert _classify(_outcome(entry_distance_behind_ball_m=_RADIUS)).regime is (
            ContactType.THIN
        )

    def test_a_sole_that_leaves_the_sand_short_of_a_proud_ball_is_direct(
        self,
    ) -> None:
        verdict = _classify(_outcome(exit_distance_past_ball_m=-2.0 * _RADIUS))
        assert verdict.regime is ContactType.THIN

    def test_a_sole_that_leaves_short_of_a_fully_plugged_ball_hits_nothing(
        self,
    ) -> None:
        # Sunk a full diameter: the top is level with the surface, so a head
        # that clears the sand behind it passes over the ball.
        verdict = _classify(
            _outcome(exit_distance_past_ball_m=-2.0 * _RADIUS),
            depth_m=_BALL.diameter_m,
        )
        assert verdict.regime is ContactType.NO_HIT

    def test_the_nominal_greenside_strike_is_a_splash(self) -> None:
        verdict = _classify(_outcome())
        assert verdict.regime is ContactType.SPLASH
        assert verdict.launch_derivable

    def test_an_exited_strike_without_a_divot_cannot_be_classified(self) -> None:
        with pytest.raises(ValueError, match="divot"):
            _classify(
                _outcome(
                    entry_distance_behind_ball_m=None, exit_distance_past_ball_m=None
                )
            )

    def test_every_verdict_names_its_own_limitation(self) -> None:
        for outcome in (
            _outcome(),
            _outcome(engaged=False),
            _outcome(entry_distance_behind_ball_m=0.0),
        ):
            assert _classify(outcome).reasons


class TestTheOutcomeIsReadOffTheShot:
    def test_a_windowed_march_that_stays_buried_reads_as_no_release(self) -> None:
        # A wedge at delivery speed brings its sole back out at 12-18 ms
        # (ShotSettings.max_time_s); a 5 ms window ends with it still down.
        preset = get_preset(preset_names()[0])
        wedge = SurfaceElements.from_mesh(
            build_wedge_mesh(preset.geometry, n_profile_points=24, n_stations=11)
        )
        solver = DRFTSolver(
            material=MaterialResponse.from_sand_state(
                playing_condition(PlayingCondition.FIRM)
            ),
            refusal_policy=RefusalPolicy.REPORT,
        )
        angle = math.radians(6.0)
        shot = simulate_shot(
            solver,
            wedge,
            head_mass_kg=0.30,
            kinematics=HeadKinematics(
                velocity_m_s=25.0 * np.array([-math.cos(angle), 0.0, -math.sin(angle)])
            ),
            settings=ShotSettings(max_time_s=0.005, require_exit=False),
        )
        outcome = StrikeOutcome.from_shot(shot, divot=None)
        assert outcome.engaged and not outcome.exited
        assert _classify(outcome).regime is ContactType.BURIED_NO_RELEASE


class TestNoForcedCarry:
    def test_the_launch_refuses_every_regime_but_splash(self) -> None:
        for regime in (
            ContactType.NO_HIT,
            ContactType.THIN,
            ContactType.BURIED_NO_RELEASE,
        ):
            with pytest.raises(UnsupportedContactRegimeError, match=regime.value):
                compute_bunker_launch(nominal_state(contact_regime=regime))

    def test_the_refusal_is_a_value_error_the_workbench_can_report(self) -> None:
        assert issubclass(UnsupportedContactRegimeError, ValueError)

    def test_a_declared_splash_still_launches(self) -> None:
        result = compute_bunker_launch(nominal_state(contact_regime=ContactType.SPLASH))
        assert result.contact_type is ContactType.SPLASH
        assert result.ball_speed_m_s > 0.0

    def test_the_state_records_the_regime_it_was_declared_under(self) -> None:
        assert nominal_state().contact_regime is ContactType.SPLASH
        assert isinstance(nominal_state(), BunkerShotState)


class TestF1StillCannotQuoteALaunch:
    """The plane-strain ball may draw sand arriving; it may not launch."""

    def test_launch_velocity_and_heel_toe_split_stay_refused(self) -> None:
        section = BallSection.at((0.0, 0.0), n_facets=16)
        with pytest.raises(OutOfEnvelopeError) as launch:
            section.launch_velocity_m_s()
        with pytest.raises(OutOfEnvelopeError) as lateral:
            section.heel_toe_split()
        assert RefusedQuantity.BALL_LAUNCH.value in str(launch.value)
        assert RefusedQuantity.OUT_OF_PLANE.value in str(lateral.value)
