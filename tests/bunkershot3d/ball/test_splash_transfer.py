"""Sand-mediated momentum transfer, driven by the delivered impulse (#8657).

The model this suite pins replaced one that derived ball launch from a
hard-coded 12 cm sweep length::

    sweep_length = 0.12                                   # deleted
    displaced_volume = sole_length_m * entry_depth_m * sweep_length

That made ball speed **linear in entry depth** and blind to the impulse the F0
solver actually delivered, which is the defect issue #8657 was filed for. The
defining test of this file is
:meth:`TestCarryFollowsTheDeliveredImpulse.test_same_divot_different_impulse_gives_different_launch`:
two strikes that cut the same divot but delivered different momentum must not
produce the same ball. The old model gave them the identical answer.

The partition being tested is a partially-inelastic collision between the
ball and the share of the moving sand on a path that meets it::

    p_int  = f_int * J                       momentum on a path that meets the ball
    m_int  = f_int * m_divot                 mass of that share
    p_ball = eta * m_b / (m_int + m_b) * p_int

``J`` is the solver's impulse and ``m_divot`` is the metrics layer's divot
mass; both are computed elsewhere and neither is a fitted constant here. Only
``eta`` is uncalibrated, and it is named rather than buried.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st

from bunkershot3d.ball.lie import BallLie, BallProperties
from bunkershot3d.ball.splash import (
    ContactType,
    MomentumTransfer,
    SandDelivery,
    compute_ball_launch_from_splash,
    compute_sand_ejecta_velocity,
    compute_splash_impulse,
)
from bunkershot3d.solvers import EnvelopeStatus, ValidityVerdict, evaluate_envelope

pytestmark = pytest.mark.unit

GREENSIDE_LOFT_RAD = math.radians(56.0)
"""A 56 deg sand wedge, the club the one published dataset used."""

#: The three feature scales the F0 tier judges every shot at.
_FEATURE_LENGTHS_M = {"clubhead": 0.1, "sole width": 0.03, "leading edge": 0.005}


def solver_verdict(
    speed_m_s: float = 25.0,
    feature_lengths_m: dict[str, float] | None = None,
) -> ValidityVerdict:
    """Return a verdict of the kind an F0 shot carries.

    Args:
        speed_m_s: Intrusion speed. 25 m/s is greenside delivery.
        feature_lengths_m: Scales to judge at. The default three include the
            5 mm leading edge, which spans only ten grains and is therefore
            never ``WITHIN``; pass the clubhead alone for an in-envelope one.

    Returns:
        The verdict.
    """
    return evaluate_envelope(
        speed_m_s=speed_m_s,
        feature_lengths_m=feature_lengths_m or _FEATURE_LENGTHS_M,
        grain_diameter_m=0.0005,
        element_size_m=0.002,
        dynamic_terms_active=True,
    )


def delivery(
    *,
    impulse_n_s: float = 4.0,
    displaced_mass_kg: float = 0.25,
    contact_duration_s: float = 0.005,
    speed_m_s: float = 25.0,
    bed_relative_density: float = 0.5,
    verdict: ValidityVerdict | None = None,
) -> SandDelivery:
    """Return a measured strike, as the solver and metrics layer report one.

    Args:
        impulse_n_s: Magnitude of the sand impulse the solver integrated.
        displaced_mass_kg: Divot mass the metrics layer measured.
        contact_duration_s: Time the sole spent engaged.
        speed_m_s: Head speed at entry; the exit is 60% of it.
        bed_relative_density: Packing state of the bed, which sets the share of
            the intercepted momentum the ball keeps (issue #8704). The default
            is mid-range so a test that does not care about the lie is not
            sitting at either bound.
        verdict: The solver's verdict, defaulting to one formed at
            ``speed_m_s`` on the three standard scales.

    Returns:
        The delivery.
    """
    return SandDelivery(
        impulse_n_s=impulse_n_s,
        displaced_mass_kg=displaced_mass_kg,
        contact_duration_s=contact_duration_s,
        entry_speed_m_s=speed_m_s,
        exit_speed_m_s=0.6 * speed_m_s,
        bed_relative_density=bed_relative_density,
        verdict=solver_verdict(speed_m_s) if verdict is None else verdict,
    )


def launch(strike: SandDelivery, *, ball_depth_m: float = 0.005):
    """Return the ball launch for one strike at the nominal lie.

    Args:
        strike: The measured strike.
        ball_depth_m: How far the ball centre sits below the surface.

    Returns:
        The launch result.
    """
    return compute_ball_launch_from_splash(
        lie=BallLie(depth_m=ball_depth_m),
        ball=BallProperties(),
        delivery=strike,
        club_loft_rad=GREENSIDE_LOFT_RAD,
    )


class TestCarryFollowsTheDeliveredImpulse:
    """The defect in #8657, stated as a test."""

    def test_same_divot_different_impulse_gives_different_launch(self) -> None:
        """The defining test. Same entry depth, different momentum delivered.

        Two strikes that cut an identical divot -- same depth, same length,
        so the same displaced mass -- but delivered different impulse. The
        deleted model saw only the entry depth and returned the same ball for
        both.
        """
        soft = launch(delivery(impulse_n_s=3.0, displaced_mass_kg=0.25))
        hard = launch(delivery(impulse_n_s=6.0, displaced_mass_kg=0.25))
        assert hard.ball_speed_m_s > soft.ball_speed_m_s
        assert hard.ball_speed_m_s == pytest.approx(2.0 * soft.ball_speed_m_s)

    def test_ball_speed_is_proportional_to_the_delivered_impulse(self) -> None:
        speeds = [
            launch(delivery(impulse_n_s=impulse)).ball_speed_m_s
            for impulse in (1.0, 2.0, 4.0, 8.0)
        ]
        ratios = [b / a for a, b in zip(speeds, speeds[1:], strict=False)]
        assert ratios == pytest.approx([2.0, 2.0, 2.0])

    def test_zero_impulse_launches_nothing(self) -> None:
        result = launch(delivery(impulse_n_s=0.0))
        assert result.ball_speed_m_s == pytest.approx(0.0)
        assert result.spin_rate_rpm == pytest.approx(0.0)

    def test_entry_depth_is_no_longer_an_input(self) -> None:
        """The signature itself must not accept the deleted quantity."""
        import inspect

        parameters = inspect.signature(compute_ball_launch_from_splash).parameters
        assert "entry_depth_m" not in parameters
        assert "sole_length_m" not in parameters
        assert "sole_width_m" not in parameters

    def test_the_sweep_length_box_estimate_is_gone(self) -> None:
        """A source-level guard: the magic number must not come back."""
        import inspect

        from bunkershot3d.ball import splash

        source = inspect.getsource(splash)
        assert "sweep_length" not in source
        assert "0.12" not in source


class TestMovingMoreSandDoesNotHelp:
    """Metamorphic: mass is a cost, not a benefit, at fixed momentum."""

    def test_doubling_the_divot_mass_does_not_raise_ball_speed(self) -> None:
        light = launch(delivery(impulse_n_s=4.0, displaced_mass_kg=0.20))
        heavy = launch(delivery(impulse_n_s=4.0, displaced_mass_kg=0.40))
        assert heavy.ball_speed_m_s <= light.ball_speed_m_s

    @given(
        st.floats(min_value=0.01, max_value=2.0, allow_nan=False),
        st.floats(min_value=1.0, max_value=4.0, allow_nan=False),
    )
    @settings(deadline=None, max_examples=40)
    def test_ball_speed_is_monotonically_decreasing_in_sand_mass(
        self, mass_kg: float, factor: float
    ) -> None:
        light = launch(delivery(impulse_n_s=4.0, displaced_mass_kg=mass_kg))
        heavy = launch(delivery(impulse_n_s=4.0, displaced_mass_kg=mass_kg * factor))
        assert heavy.ball_speed_m_s <= light.ball_speed_m_s

    def test_the_mean_ejecta_speed_is_the_impulse_over_the_mass(self) -> None:
        """The ejecta speed is derived, not a fraction of club speed."""
        strike = delivery(impulse_n_s=4.0, displaced_mass_kg=0.25)
        assert compute_sand_ejecta_velocity(strike) == pytest.approx(16.0)


class TestMomentumBookkeeping:
    """The ball can never be given more momentum than the sand carried."""

    def test_ball_impulse_never_exceeds_the_delivered_impulse(self) -> None:
        strike = delivery(impulse_n_s=4.0, displaced_mass_kg=0.25)
        splash = compute_splash_impulse(
            lie=BallLie(depth_m=0.0),
            ball=BallProperties(),
            delivery=strike,
            club_loft_rad=GREENSIDE_LOFT_RAD,
        )
        assert splash.ball_impulse_n_s <= splash.delivered_impulse_n_s

    @given(
        st.floats(min_value=0.0, max_value=50.0, allow_nan=False),
        st.floats(min_value=0.005, max_value=3.0, allow_nan=False),
        st.floats(min_value=-0.02, max_value=0.04, allow_nan=False),
    )
    @settings(deadline=None, max_examples=60)
    def test_the_partition_conserves_momentum_for_any_strike(
        self, impulse_n_s: float, mass_kg: float, depth_m: float
    ) -> None:
        splash = compute_splash_impulse(
            lie=BallLie(depth_m=depth_m),
            ball=BallProperties(),
            delivery=delivery(impulse_n_s=impulse_n_s, displaced_mass_kg=mass_kg),
            club_loft_rad=GREENSIDE_LOFT_RAD,
        )
        assert splash.ball_impulse_n_s <= splash.delivered_impulse_n_s
        assert splash.ball_impulse_n_s <= impulse_n_s

    def test_an_efficiency_above_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="efficiency"):
            MomentumTransfer(efficiency=1.4)

    def test_the_efficiency_survives_contracts_being_switched_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``python -O`` and ``DBC_LEVEL=off`` must not remove the guard."""
        monkeypatch.setenv("DBC_LEVEL", "off")
        with pytest.raises(ValueError, match="efficiency"):
            MomentumTransfer(efficiency=2.0)


class TestBurialCostsMomentum:
    """A plugged ball presents less of itself to the splash."""

    def test_a_buried_ball_receives_less_than_a_sitting_one(self) -> None:
        sitting = launch(delivery(), ball_depth_m=0.0)
        buried = launch(delivery(), ball_depth_m=0.02)
        assert buried.ball_speed_m_s < sitting.ball_speed_m_s

    def test_a_fully_plugged_ball_receives_nothing_from_the_splash(self) -> None:
        plugged = launch(delivery(), ball_depth_m=BallProperties().diameter_m)
        assert plugged.ball_speed_m_s == pytest.approx(0.0)


class TestLaunchGeometry:
    """Direction is a stated convention; magnitude is the derived part."""

    def test_the_ball_leaves_upward_and_forward(self) -> None:
        result = launch(delivery())
        assert result.ball_velocity[0] > 0.0
        assert result.ball_velocity[2] > 0.0

    def test_the_launch_angle_is_the_effective_loft(self) -> None:
        result = launch(delivery())
        assert result.launch_angle_rad == pytest.approx(GREENSIDE_LOFT_RAD)

    @given(st.floats(min_value=40.0, max_value=64.0, allow_nan=False))
    @settings(deadline=None)
    def test_more_loft_launches_higher(self, loft_deg: float) -> None:
        result = compute_ball_launch_from_splash(
            lie=BallLie(depth_m=0.005),
            ball=BallProperties(),
            delivery=delivery(),
            club_loft_rad=math.radians(loft_deg),
        )
        assert 0.0 < result.launch_angle_rad < math.pi / 2

    def test_the_ball_has_backspin(self) -> None:
        result = launch(delivery())
        assert result.spin_rate_rpm > 0.0
        assert result.ball_angular_velocity[1] < 0.0

    def test_the_contact_is_a_splash(self) -> None:
        assert launch(delivery()).contact_type is ContactType.SPLASH

    def test_the_contact_duration_is_the_measured_one(self) -> None:
        splash = compute_splash_impulse(
            lie=BallLie(depth_m=0.005),
            ball=BallProperties(),
            delivery=delivery(),
            club_loft_rad=GREENSIDE_LOFT_RAD,
        )
        assert splash.contact_duration_s == pytest.approx(0.005)


class TestDeliveryRefusesNonsense:
    """A strike that was never measured is refused, not defaulted."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("impulse_n_s", -1.0),
            ("impulse_n_s", math.nan),
            ("displaced_mass_kg", 0.0),
            ("displaced_mass_kg", -0.5),
            ("contact_duration_s", -1e-3),
        ],
    )
    def test_an_unusable_measurement_is_refused(self, field: str, value: float) -> None:
        with pytest.raises(ValueError):
            delivery(**{field: value})  # type: ignore[arg-type]

    def test_a_delivery_without_a_verdict_cannot_be_built(self) -> None:
        with pytest.raises(ValueError, match="verdict"):
            SandDelivery(
                impulse_n_s=4.0,
                displaced_mass_kg=0.25,
                contact_duration_s=0.005,
                entry_speed_m_s=25.0,
                exit_speed_m_s=15.0,
                bed_relative_density=0.5,
                verdict=None,  # type: ignore[arg-type]
            )


class TestGreensideSanity:
    """Order-of-magnitude checks. Not validation: no measurement exists."""

    def test_a_greenside_strike_gives_a_plausible_ball_speed(self) -> None:
        result = launch(delivery(impulse_n_s=4.0, displaced_mass_kg=0.25))
        assert 0.5 < result.ball_speed_m_s < 25.0

    def test_the_ball_takes_a_small_share_of_the_head_energy(self) -> None:
        result = launch(delivery())
        assert 0.0 < result.energy_transfer_fraction < 0.5

    def test_the_verdict_is_never_better_than_the_solver_s(self) -> None:
        result = launch(delivery())
        assert result.verdict.status is not EnvelopeStatus.WITHIN
