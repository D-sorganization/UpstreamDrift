"""Ball speed must fall as the lie softens (issue #8704).

The demo's sand-condition sweep held geometry and delivery fixed and swept the
bed, and got the ordering **backwards**: firm 12.13 m/s, fluffy 12.55 m/s,
plugged 12.60 m/s. A plugged lie came out fastest.

The sand model was not at fault. The delivered impulse is nearly invariant
across the four conditions (2.848-2.855 N.s, 0.2 %) because the head gives up
almost the same momentum whatever it is decelerating against -- head speed loss
moved 1.1 % while peak sand force moved 22 %. What did change was the divot
**mass**, which scales with bulk density: 65.4 g in firm sand against 61.2 g in
plugged. The partition's added-mass term ``m_b / (m_int + m_b)`` therefore
*penalised* the firm bed, and with a lie-independent efficiency there was
nothing to pay it back.

A transfer model whose efficiency does not depend on the lie cannot represent a
lie-dependent splash. So ``eta`` is now a function of the bed's packing state::

    eta(D_r) = efficiency * (1 - packing_sensitivity * (1 - D_r))

increasing in relative density, equal to the stated dense-bed value at
``D_r = 1``, and never above it -- which keeps the momentum-budget proof of
issue #8657 intact. The mechanism assumed is that a loose bed spends a larger
share of the delivered momentum rearranging grains, so less of it leaves as a
directed stream. **That mechanism is assumed, not measured**: per issue #8616
no published measurement of ball speed out of sand exists, so the sensitivity
is a stated placeholder and the provenance record says so.

The defining test here is
:meth:`TestTheOrderingIsPhysicallyRight.test_firm_sand_launches_the_ball_faster_than_fluffy_or_plugged`,
which runs the real F0 solver at one geometry and one delivery across three
beds. That is the exact comparison the demo got backwards.
"""

from __future__ import annotations

import dataclasses as dc
import math

import numpy as np
import pytest

from bunkershot3d.ball.lie import BallLie, BallProperties
from bunkershot3d.ball.splash import (
    BED_PACKING_DEPENDENCE_REASON,
    BED_PACKING_TRANSFER_SENSITIVITY,
    MomentumTransfer,
    SandDelivery,
    compute_ball_launch_from_splash,
    compute_splash_impulse,
)
from bunkershot3d.geometry import build_wedge_mesh, get_preset
from bunkershot3d.metrics import HeadModel, StrikeScene, StrikeTrace, divot_metrics
from bunkershot3d.sand import PlayingCondition, ProvenanceBasis, playing_condition
from bunkershot3d.solvers import (
    DRFTSolver,
    HeadKinematics,
    MaterialResponse,
    RefusalPolicy,
    ShotSettings,
    SurfaceElements,
    simulate_shot,
)

from .test_splash_transfer import GREENSIDE_LOFT_RAD, delivery, solver_verdict

pytestmark = pytest.mark.unit

BALL = BallProperties()
NOMINAL_LIE = BallLie(depth_m=0.005)


def ball_impulse_at(bed_relative_density: float) -> float:
    """Return the ball impulse for one bed, everything else held fixed.

    Args:
        bed_relative_density: Packing state of the bed the strike was made in.

    Returns:
        The ball impulse [N.s].
    """
    return compute_splash_impulse(
        lie=NOMINAL_LIE,
        ball=BALL,
        delivery=delivery(bed_relative_density=bed_relative_density),
        club_loft_rad=GREENSIDE_LOFT_RAD,
    ).ball_impulse_n_s


class TestTheEfficiencyDependsOnTheBed:
    """``eta`` is a function of the packing state, not a constant."""

    def test_a_denser_bed_transfers_a_larger_share(self) -> None:
        transfer = MomentumTransfer()

        loose = transfer.efficiency_for(0.0)
        mid = transfer.efficiency_for(0.5)
        dense = transfer.efficiency_for(1.0)

        assert loose < mid < dense

    def test_a_fully_dense_bed_recovers_the_stated_efficiency(self) -> None:
        """The shipped constant keeps a statable meaning: the dense-bed value."""
        transfer = MomentumTransfer()

        assert transfer.efficiency_for(1.0) == pytest.approx(transfer.efficiency)

    def test_the_effective_efficiency_never_exceeds_the_stated_one(self) -> None:
        """What keeps the #8657 momentum budget provable."""
        transfer = MomentumTransfer()

        for tenth in range(11):
            assert transfer.efficiency_for(tenth / 10.0) <= transfer.efficiency

    def test_a_fully_loose_bed_costs_the_stated_share(self) -> None:
        transfer = MomentumTransfer()

        assert transfer.efficiency_for(0.0) == pytest.approx(
            transfer.efficiency * (1.0 - BED_PACKING_TRANSFER_SENSITIVITY)
        )

    @pytest.mark.parametrize("relative_density", [-0.01, 1.01, float("nan")])
    def test_a_bed_outside_the_relative_density_range_is_refused(
        self, relative_density: float
    ) -> None:
        """A plain raise: the budget must survive ``python -O``."""
        with pytest.raises(ValueError, match="relative density"):
            MomentumTransfer().efficiency_for(relative_density)

    def test_the_sensitivity_cannot_be_set_beyond_a_total_loss(self) -> None:
        with pytest.raises(ValueError, match="packing_sensitivity"):
            MomentumTransfer(packing_sensitivity=1.5)


class TestTheDeliveryCarriesTheBedItWasStruckIn:
    """The default path cannot be the lie-independent one."""

    def test_a_delivery_must_declare_the_bed(self) -> None:
        with pytest.raises(TypeError):
            SandDelivery(  # type: ignore[call-arg]
                impulse_n_s=4.0,
                displaced_mass_kg=0.25,
                contact_duration_s=0.005,
                entry_speed_m_s=25.0,
                exit_speed_m_s=15.0,
                verdict=solver_verdict(),
            )

    @pytest.mark.parametrize("relative_density", [-0.01, 1.01])
    def test_a_bed_outside_the_range_is_refused(self, relative_density: float) -> None:
        with pytest.raises(ValueError, match="bed_relative_density"):
            delivery(bed_relative_density=relative_density)

    def test_the_ball_impulse_rises_with_the_bed_density(self) -> None:
        """At a fixed delivered impulse and divot mass, only the bed changed."""
        assert ball_impulse_at(0.1) < ball_impulse_at(0.5) < ball_impulse_at(0.9)

    def test_the_effective_efficiency_is_reported(self) -> None:
        """A caller can see the number the partition actually used."""
        splash = compute_splash_impulse(
            lie=NOMINAL_LIE,
            ball=BALL,
            delivery=delivery(bed_relative_density=0.25),
            club_loft_rad=GREENSIDE_LOFT_RAD,
        )

        assert splash.transfer_efficiency == pytest.approx(
            MomentumTransfer().efficiency_for(0.25)
        )


class TestTheDependenceIsDeclaredAssumed:
    """Assumed-not-measured, in the shapes the package already uses."""

    def test_the_launch_reports_the_efficiency_it_used(self) -> None:
        launch = compute_ball_launch_from_splash(
            lie=NOMINAL_LIE,
            ball=BALL,
            delivery=delivery(bed_relative_density=0.25),
            club_loft_rad=GREENSIDE_LOFT_RAD,
        )

        assert launch.transfer_efficiency == pytest.approx(
            MomentumTransfer().efficiency_for(0.25)
        )

    def test_the_bed_dependence_carries_a_provenance_entry(self) -> None:
        launch = compute_ball_launch_from_splash(
            lie=NOMINAL_LIE,
            ball=BALL,
            delivery=delivery(bed_relative_density=0.25),
            club_loft_rad=GREENSIDE_LOFT_RAD,
        )
        entry = launch.provenance.entry("bed_packing_dependence")

        assert entry.basis is not ProvenanceBasis.MEASURED
        assert "assumed" in entry.note.lower()

    def test_the_verdict_says_the_dependence_is_assumed(self) -> None:
        launch = compute_ball_launch_from_splash(
            lie=NOMINAL_LIE,
            ball=BALL,
            delivery=delivery(bed_relative_density=0.25),
            club_loft_rad=GREENSIDE_LOFT_RAD,
        )

        assert BED_PACKING_DEPENDENCE_REASON in launch.verdict.reasons
        assert "8704" in BED_PACKING_DEPENDENCE_REASON


# --------------------------------------------------------------------------
# The end-to-end ordering, through the real F0 solver.
# --------------------------------------------------------------------------

_SHOT_SETTINGS = ShotSettings(max_time_s=0.020)
_ENTRY_SPEED_M_S = 25.0
_ATTACK_ANGLE_DEG = -6.0
_MAX_COAST_STEPS = 400


def _coasted_positions(
    positions_m: np.ndarray,
    velocities_m_s: np.ndarray,
    times_s: np.ndarray,
    sole_offset_z_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Continue the path ballistically until the sole clears the surface.

    ``simulate_shot`` stops when no element is still engaged, which happens
    while the sole is geometrically inside the divot, and ``divot_metrics``
    needs the exit crossing. The solver reports exactly zero wrench there, so
    straight-line motion is its own force model rather than a new assumption.

    Args:
        positions_m: ``(n, 3)`` recorded body-origin positions.
        velocities_m_s: ``(n, 3)`` recorded velocities.
        times_s: ``(n,)`` recorded sample times.
        sole_offset_z_m: Sole reference offset below the body origin.

    Returns:
        ``(times_s, positions_m)`` including the continuation.
    """
    step_s = float(np.diff(times_s).mean())
    velocity = velocities_m_s[-1]
    extra_t: list[float] = []
    extra_p: list[np.ndarray] = []
    position = positions_m[-1].copy()
    time_s = float(times_s[-1])
    if position[2] + sole_offset_z_m < 0.0 and velocity[2] > 0.0:
        for _ in range(_MAX_COAST_STEPS):
            time_s += step_s
            position = position + step_s * velocity
            extra_t.append(time_s)
            extra_p.append(position.copy())
            if position[2] + sole_offset_z_m >= 0.0:
                break
    return (
        np.concatenate([times_s, np.asarray(extra_t)]),
        np.vstack([positions_m, np.asarray(extra_p).reshape(-1, 3)]),
    )


def _ball_speed_in(condition: PlayingCondition, elements: SurfaceElements) -> float:
    """Run one F0 shot in one bed and return the launched ball speed.

    Args:
        condition: The playing condition, which fixes the bed.
        elements: The surface discretisation of the head, shared across beds so
            geometry is provably identical.

    Returns:
        Ball speed [m/s].
    """
    geometry = get_preset("sm9_54_f").geometry
    sand = playing_condition(condition)
    solver = DRFTSolver(
        material=MaterialResponse.from_sand_state(sand),
        refusal_policy=RefusalPolicy.REPORT,
    )
    attack_rad = math.radians(_ATTACK_ANGLE_DEG)
    shot = simulate_shot(
        solver,
        elements,
        head_mass_kg=geometry.head_mass_kg,
        kinematics=HeadKinematics(
            velocity_m_s=_ENTRY_SPEED_M_S
            * np.array([math.cos(attack_rad), 0.0, math.sin(attack_rad)])
        ),
        settings=_SHOT_SETTINGS,
    )
    sole_offset_z = float(elements.centroids_m[:, 2].min())
    head = HeadModel(
        mass_kg=geometry.head_mass_kg,
        centre_of_mass_body_m=np.zeros(3),
        sole_reference_body_m=np.array([0.0, 0.0, sole_offset_z]),
        shaft_axis_body=np.array([0.0, 0.0, 1.0]),
    )
    times, positions = _coasted_positions(
        shot.positions_m, shot.velocities_m_s, shot.times_s, sole_offset_z
    )
    pad = ((0, times.size - shot.n_steps), (0, 0))
    trace = StrikeTrace(
        time_s=times,
        head_position_m=positions,
        head_orientation_quat=np.tile([1.0, 0.0, 0.0, 0.0], (times.size, 1)),
        sand_force_N=np.pad(shot.forces_n, pad),
        sand_moment_Nm=np.pad(shot.torques_n_m, pad),
    )
    scene = StrikeScene(
        sand_surface_height_m=0.0,
        ball_position_m=np.array([float(positions[0, 0]) + 0.10, 0.0, 0.0]),
        travel_axis=np.array([1.0, 0.0, 0.0]),
    )
    divot = divot_metrics(
        trace,
        head,
        scene,
        width_m=geometry.sole_width_m,
        bulk_density_kg_m3=sand.bulk_density_kg_m3,
    )
    launch = compute_ball_launch_from_splash(
        lie=NOMINAL_LIE,
        ball=BALL,
        delivery=SandDelivery(
            impulse_n_s=float(np.linalg.norm(shot.impulse_n_s)),
            displaced_mass_kg=divot.mass_kg,
            contact_duration_s=shot.contact_duration_s,
            entry_speed_m_s=shot.entry_speed_m_s,
            exit_speed_m_s=shot.exit_speed_m_s,
            bed_relative_density=sand.relative_density,
            verdict=shot.verdict,
        ),
        club_loft_rad=GREENSIDE_LOFT_RAD,
        club_mass_kg=geometry.head_mass_kg,
    )
    return launch.ball_speed_m_s


class TestTheOrderingIsPhysicallyRight:
    """The comparison the demo got backwards, run through the real solver."""

    @pytest.fixture(scope="class")
    def ball_speeds(self) -> dict[PlayingCondition, float]:
        """Ball speed per bed at one geometry and one delivery.

        The head is meshed once and the discretisation is shared, so the only
        thing that differs between the three shots is the sand.
        """
        elements = SurfaceElements.from_mesh(
            build_wedge_mesh(dc.replace(get_preset("sm9_54_f").geometry))
        )
        return {
            condition: _ball_speed_in(condition, elements)
            for condition in (
                PlayingCondition.FIRM,
                PlayingCondition.FLUFFY,
                PlayingCondition.PLUGGED,
            )
        }

    def test_firm_sand_launches_the_ball_faster_than_fluffy_or_plugged(
        self, ball_speeds
    ) -> None:
        """Firm > fluffy > plugged, at fixed geometry and delivery."""
        assert (
            ball_speeds[PlayingCondition.FIRM]
            > ball_speeds[PlayingCondition.FLUFFY]
            > ball_speeds[PlayingCondition.PLUGGED]
        )

    def test_a_plugged_lie_is_the_slowest_by_a_visible_margin(
        self, ball_speeds
    ) -> None:
        """Not a rounding-width difference: the ordering has to be readable."""
        assert (
            ball_speeds[PlayingCondition.PLUGGED]
            < 0.9 * (ball_speeds[PlayingCondition.FIRM])
        )
