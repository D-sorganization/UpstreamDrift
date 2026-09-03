"""The mass a strike accelerated must be able to carry the impulse (#8659).

This is the acceptance test for the contradiction issue #8659 filed. Issue
#8657 made ball launch divide the solver's delivered impulse by the metrics
layer's divot mass, and at the workbench's nominal greenside shot that
arithmetic came out at 45.8 m/s of ejecta from a head arriving at 25.0 m/s.

The relation being pinned here is not a convention and not a threshold. Sand
is set moving by the head; momentum is carried by mass moving; so the mean
speed of the sand that shared the delivered momentum cannot exceed the speed
of the thing that delivered it::

    J / m_accelerated <= v_entry

Three things are asserted, and the sweep is what makes the first two mean
something:

* **the relation holds over real shots**, run through ``simulate_shot`` at
  default settings across delivery speeds, attack angles and sand conditions
  -- so a future change to the solver, the divot model or the entrainment
  factors cannot quietly reintroduce the contradiction;
* **it holds because the mass is right, not because a speed was capped** --
  ball speed still responds linearly to the delivered impulse, which is the
  property #8657 exists to protect and the reason a clamp was rejected;
* **the prism on its own does not satisfy it**, so the test would have failed
  before the fix rather than passing vacuously.

Nothing here is a validation. The entrainment factors were read off the F1
MPM tier, which is ``BEYOND_VALIDATION`` with a 1.44 m/s published-speed
ceiling and 0 of 4 on NASA-STD-7009B, so this is a consistency check between
two uncalibrated models; see
:data:`~bunkershot3d.metrics.accelerated_mass.ACCELERATED_MASS_CONSISTENCY_REASON`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.ball.splash import SandDelivery
from bunkershot3d.geometry import (
    build_wedge_mesh,
    compute_mass_properties,
    get_preset,
    preset_names,
    shaft_axis,
)
from bunkershot3d.metrics import (
    DivotMetrics,
    HeadModel,
    StrikeScene,
    StrikeTrace,
    divot_metrics,
)
from bunkershot3d.sand import PlayingCondition, SandState, playing_condition
from bunkershot3d.solvers import (
    DRFTSolver,
    HeadKinematics,
    MaterialResponse,
    RefusalPolicy,
    ShotResult,
    SurfaceElements,
    simulate_shot,
)

pytestmark = pytest.mark.integration

_HEAD_MASS_KG = 0.30
_SOLE_WIDTH_M = 0.020
_BALL_AHEAD_M = 0.030

#: Delivery speeds spanning the greenside band and well below it. The relation
#: is speed-dependent on both sides -- the impulse rises with speed and so does
#: the ceiling it is judged against -- so a single speed would prove nothing.
_SPEEDS_M_S = (12.0, 18.0, 25.0, 30.0)

#: Attack angles from a shallow sweep to the steepest blow whose divot is
#: measurable at all. This is the axis that moves the divot's *depth*, the
#: other half of the sweep the definition of done asks for: over the four
#: below the sole cuts 5.5 mm to 12.7 mm at 25 m/s.
#:
#: The ceiling is not this file's to choose. Past about 10 degrees the F0
#: head is turned back far enough that the trace reverses, and
#: ``divot_metrics`` refuses a reversing trace outright because its section
#: integral would double back -- a pre-existing refusal that predates issue
#: #8659 and has nothing to do with the mass. A test that swallowed it would
#: be reporting a skip as a pass, so the sweep stops where the measurement
#: does.
_ATTACK_ANGLES_DEG = (3.0, 5.0, 6.0, 8.0)

#: The three bunker presets that build without an explicit dilation suction.
#: ``WET`` is saturated and needs one, and inventing a value here to widen a
#: sweep would be the opposite of what this file is for.
_CONDITIONS = (
    PlayingCondition.FIRM,
    PlayingCondition.FLUFFY,
    PlayingCondition.PLUGGED,
)


@pytest.fixture(scope="module")
def wedge() -> SurfaceElements:
    """A real lofted wedge head, discretised. Module-scoped: lofting is slow."""
    preset = get_preset(preset_names()[0])
    mesh = build_wedge_mesh(preset.geometry, n_profile_points=24, n_stations=11)
    return SurfaceElements.from_mesh(mesh)


@pytest.fixture(scope="module")
def head(wedge: SurfaceElements) -> HeadModel:
    """The designer's description of the same head, for the metrics layer."""
    preset = get_preset(preset_names()[0])
    mesh = build_wedge_mesh(preset.geometry, n_profile_points=24, n_stations=11)
    mass = compute_mass_properties(mesh, mass_kg=preset.geometry.head_mass_kg)
    _, axis = shaft_axis(preset.geometry)
    lowest = int(np.argmin(wedge.centroids_m[:, 2]))
    return HeadModel(
        mass_kg=preset.geometry.head_mass_kg,
        centre_of_mass_body_m=mass.centroid_m,
        sole_reference_body_m=wedge.centroids_m[lowest],
        shaft_axis_body=np.asarray(axis, dtype=np.float64),
        inertia_body_kg_m2=mass.inertia_kg_m2,
    )


@pytest.fixture(scope="module")
def scene() -> StrikeScene:
    """A flat lie with the ball 30 mm down the travel axis."""
    return StrikeScene(
        sand_surface_height_m=0.0,
        ball_position_m=(-_BALL_AHEAD_M, 0.0, 0.0),
        travel_axis=(-1.0, 0.0, 0.0),
    )


def _shot(
    wedge: SurfaceElements,
    sand: SandState,
    *,
    speed_m_s: float,
    attack_angle_deg: float,
) -> ShotResult:
    """Run one shot at default settings, refusing nothing.

    Args:
        wedge: The discretised head.
        sand: The bed.
        speed_m_s: Delivery speed.
        attack_angle_deg: Descending-blow angle, positive downward here
            because the travel axis is ``-x``.

    Returns:
        The record.
    """
    angle = math.radians(attack_angle_deg)
    return simulate_shot(
        DRFTSolver(
            material=MaterialResponse.from_sand_state(sand),
            refusal_policy=RefusalPolicy.REPORT,
        ),
        wedge,
        head_mass_kg=_HEAD_MASS_KG,
        kinematics=HeadKinematics(
            velocity_m_s=(
                -speed_m_s * math.cos(angle),
                0.0,
                -speed_m_s * math.sin(angle),
            )
        ),
    )


def _divot(
    result: ShotResult, head: HeadModel, scene: StrikeScene, sand: SandState
) -> DivotMetrics:
    """Reduce one record to its divot, at the bed's own constants."""
    return divot_metrics(
        StrikeTrace.from_shot(result),
        head,
        scene,
        width_m=_SOLE_WIDTH_M,
        bulk_density_kg_m3=sand.bulk_density_kg_m3,
        friction_angle_deg=sand.friction_angle_deg,
    )


class TestTheImpliedEjectaSpeedIsAdmissible:
    """``J / m <= v_entry``, over real shots rather than one nominal case."""

    @pytest.mark.parametrize("speed_m_s", _SPEEDS_M_S)
    @pytest.mark.parametrize("attack_angle_deg", _ATTACK_ANGLES_DEG)
    def test_over_speeds_and_depths(
        self,
        wedge: SurfaceElements,
        head: HeadModel,
        scene: StrikeScene,
        speed_m_s: float,
        attack_angle_deg: float,
    ) -> None:
        sand = playing_condition(PlayingCondition.FIRM)
        result = _shot(
            wedge, sand, speed_m_s=speed_m_s, attack_angle_deg=attack_angle_deg
        )
        divot = _divot(result, head, scene, sand)
        impulse = float(np.linalg.norm(result.impulse_n_s))

        implied = impulse / divot.accelerated_mass.central_kg

        assert divot.max_depth_m > 0.0
        assert implied <= result.entry_speed_m_s, (
            f"{speed_m_s} m/s at {attack_angle_deg} deg cut a "
            f"{divot.max_depth_m * 1e3:.3g} mm divot and implies {implied:.4g} m/s "
            f"of ejecta from a {result.entry_speed_m_s:.4g} m/s head"
        )

    @pytest.mark.parametrize("condition", _CONDITIONS)
    def test_over_sand_conditions(
        self,
        wedge: SurfaceElements,
        head: HeadModel,
        scene: StrikeScene,
        condition: PlayingCondition,
    ) -> None:
        sand = playing_condition(condition)
        # 6 degrees rather than the 8 the firm-bed cases use: a looser bed
        # turns the head back sooner, and 8 degrees reverses the trace in
        # fluffy and plugged sand.
        result = _shot(wedge, sand, speed_m_s=25.0, attack_angle_deg=6.0)
        divot = _divot(result, head, scene, sand)
        impulse = float(np.linalg.norm(result.impulse_n_s))

        assert impulse / divot.accelerated_mass.central_kg <= result.entry_speed_m_s

    def test_the_whole_interval_is_not_required_to_be_admissible(
        self, wedge: SurfaceElements, head: HeadModel, scene: StrikeScene
    ) -> None:
        """Only the mass actually used has to be admissible.

        Truncating the interval at ``J / v_entry`` would be the clamp #8657
        rejected, wearing a different name. The lower edge is allowed to sit
        below the momentum floor; when it does, the launch says so.
        """
        sand = playing_condition(PlayingCondition.FIRM)
        result = _shot(wedge, sand, speed_m_s=25.0, attack_angle_deg=8.0)
        divot = _divot(result, head, scene, sand)
        accelerated = divot.accelerated_mass

        assert accelerated.lower_kg < accelerated.central_kg < accelerated.upper_kg
        assert accelerated.lower_kg > accelerated.prismatic_kg


class TestTheDefectItselfIsStillDetectable:
    """The test would have failed before the fix, rather than passing empty."""

    def test_the_swept_prism_alone_is_inadmissible_at_the_nominal_shot(
        self, wedge: SurfaceElements, head: HeadModel, scene: StrikeScene
    ) -> None:
        """The number in issue #8659's table, reproduced.

        If a future change ever makes this pass, the prism has stopped being
        an under-count and the correction should be revisited rather than
        left in place.
        """
        sand = playing_condition(PlayingCondition.FIRM)
        result = _shot(wedge, sand, speed_m_s=25.0, attack_angle_deg=8.0)
        divot = _divot(result, head, scene, sand)
        impulse = float(np.linalg.norm(result.impulse_n_s))

        assert impulse / divot.mass_kg > result.entry_speed_m_s

    def test_building_a_delivery_on_the_prism_is_refused(
        self, wedge: SurfaceElements, head: HeadModel, scene: StrikeScene
    ) -> None:
        """The refusal is what stops the contradiction shipping again."""
        sand = playing_condition(PlayingCondition.FIRM)
        result = _shot(wedge, sand, speed_m_s=25.0, attack_angle_deg=8.0)
        divot = _divot(result, head, scene, sand)

        with pytest.raises(ValueError, match="cannot leave faster"):
            SandDelivery(
                impulse_n_s=float(np.linalg.norm(result.impulse_n_s)),
                displaced_mass_kg=divot.mass_kg,
                contact_duration_s=result.contact_duration_s,
                entry_speed_m_s=result.entry_speed_m_s,
                exit_speed_m_s=result.exit_speed_m_s,
                bed_relative_density=sand.relative_density,
                verdict=result.verdict,
            )

    def test_the_accelerated_mass_still_lets_launch_track_the_impulse(
        self, wedge: SurfaceElements, head: HeadModel, scene: StrikeScene
    ) -> None:
        """Not a clamp: doubling the impulse still doubles the ejecta speed.

        The quantity checked is the one a clamp would have flattened. Both
        deliveries below are built on the same accelerated mass, so ``J / m``
        is exactly proportional to ``J`` -- which is precisely what capping the
        ejecta speed at the head speed would have destroyed.
        """
        sand = playing_condition(PlayingCondition.FIRM)
        result = _shot(wedge, sand, speed_m_s=25.0, attack_angle_deg=8.0)
        divot = _divot(result, head, scene, sand)
        mass = divot.accelerated_mass.central_kg

        def _built(impulse_n_s: float) -> SandDelivery:
            return SandDelivery(
                impulse_n_s=impulse_n_s,
                displaced_mass_kg=mass,
                contact_duration_s=result.contact_duration_s,
                entry_speed_m_s=result.entry_speed_m_s,
                exit_speed_m_s=result.exit_speed_m_s,
                bed_relative_density=sand.relative_density,
                verdict=result.verdict,
            )

        soft = _built(1.0)
        hard = _built(2.0)
        assert hard.mean_ejecta_speed_m_s == pytest.approx(
            2.0 * soft.mean_ejecta_speed_m_s
        )
