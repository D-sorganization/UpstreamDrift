"""A raw ``simulate_shot`` result must reach the metrics layer (issue #8702).

This is the acceptance test for the composition defect: before it, the two
halves of the package did not meet. The metrics layer locates the divot by
interpolating the two ``depth = 0`` crossings of the sole, so it needs samples
on the *far side* of both -- free flight before entry, and a sample with the
sole back above the surface after exit. ``simulate_shot`` supplied neither: it
dropped the head so its lowest element started exactly on the surface, and it
stopped the moment nothing was engaged, which happens while the sole is still
geometrically in the divot.

Every one of the demo sweep's 77 design points failed until its trace was
padded by hand with synthetic free-flight samples. That padding is what these
tests forbid: a shot run with **default settings** goes straight into the
metrics layer, and the trace the metrics see is the trace the solver recorded,
sample for sample.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.geometry import (
    build_wedge_mesh,
    compute_mass_properties,
    get_preset,
    preset_names,
    shaft_axis,
)
from bunkershot3d.metrics import (
    HeadModel,
    StrikeScene,
    StrikeTrace,
    dig_vs_skid,
    divot_metrics,
    head_load_metrics,
    sole_depth_profile,
    submerged_interval,
)
from bunkershot3d.sand import PlayingCondition, playing_condition
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
_DELIVERY_SPEED_M_S = 25.0
_ATTACK_ANGLE_DEG = 6.0

#: Nominal cutting width of a wedge sole [m]; the prismatic divot model's one
#: free parameter, and a caller input rather than anything the solver knows.
_SOLE_WIDTH_M = 0.020

#: Measured bulk density of Covia Signature 500 bunker sand (research addendum).
_BULK_DENSITY_KG_M3 = 1550.0

#: Ball 30 mm ahead of where the sole enters, inside the 25-150 mm band Wivou
#: et al. (2016) report for entry distance behind the ball.
_BALL_AHEAD_M = 0.030


@pytest.fixture(scope="module")
def wedge() -> SurfaceElements:
    """A real lofted wedge head, discretised. Module-scoped: lofting is slow."""
    preset = get_preset(preset_names()[0])
    mesh = build_wedge_mesh(preset.geometry, n_profile_points=24, n_stations=11)
    return SurfaceElements.from_mesh(mesh)


@pytest.fixture(scope="module")
def head(wedge: SurfaceElements) -> HeadModel:
    """The designer's description of the same head, for the metrics layer.

    The sole reference is deliberately *not* re-derived here: it is read off
    the shot, so the point the metrics measure the divot on is the point the
    solver marched to its exit crossing.
    """
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
    """A flat firm lie with the ball 30 mm down the travel axis."""
    return StrikeScene(
        sand_surface_height_m=0.0,
        ball_position_m=(-_BALL_AHEAD_M, 0.0, 0.0),
        travel_axis=(-1.0, 0.0, 0.0),
    )


@pytest.fixture(scope="module")
def shot(wedge: SurfaceElements) -> ShotResult:
    """One nominal bunker shot, run with **default** settings.

    25 m/s at -6 deg into firm USGA sand: the condition the demo swept, and
    the one the 10 ms default window used to stop 0.75 ms short of.
    """
    solver = DRFTSolver(
        material=MaterialResponse.from_sand_state(
            playing_condition(PlayingCondition.FIRM)
        ),
        refusal_policy=RefusalPolicy.REPORT,
    )
    angle = math.radians(_ATTACK_ANGLE_DEG)
    velocity = _DELIVERY_SPEED_M_S * np.array([-math.cos(angle), 0.0, -math.sin(angle)])
    return simulate_shot(
        solver,
        wedge,
        head_mass_kg=_HEAD_MASS_KG,
        kinematics=HeadKinematics(velocity_m_s=velocity),
    )


class TestTheShotIsTheTraceTheMetricsMeasure:
    """No padding, no resampling, no synthetic samples: one trace."""

    def test_the_metrics_trace_is_the_shot_sample_for_sample(
        self, shot: ShotResult
    ) -> None:
        trace = StrikeTrace.from_shot(shot)
        assert trace.n_samples == shot.n_steps
        np.testing.assert_array_equal(trace.time_s, shot.times_s)
        np.testing.assert_array_equal(trace.head_position_m, shot.positions_m)
        np.testing.assert_array_equal(trace.sand_force_N, shot.forces_n)
        np.testing.assert_array_equal(trace.sand_moment_Nm, shot.torques_n_m)

    def test_the_shot_brackets_both_crossings(
        self, shot: ShotResult, head: HeadModel, scene: StrikeScene
    ) -> None:
        """Free flight before entry, and the sole back out after exit."""
        profile = sole_depth_profile(StrikeTrace.from_shot(shot), head, scene)
        assert profile.depth_m[0] < 0.0, "the record must start above the sand"
        assert profile.depth_m[-1] <= 0.0, "the record must end above the sand"
        assert profile.depth_m.max() > 0.0, "the sole must actually enter"


class TestDivotMetricsFromARawShot:
    """The acceptance criterion of #8702."""

    def test_divot_metrics_accepts_the_shot_with_zero_massaging(
        self, shot: ShotResult, head: HeadModel, scene: StrikeScene
    ) -> None:
        divot = divot_metrics(
            StrikeTrace.from_shot(shot),
            head,
            scene,
            width_m=_SOLE_WIDTH_M,
            bulk_density_kg_m3=_BULK_DENSITY_KG_M3,
        )
        assert 0.002 < divot.max_depth_m < 0.060
        assert divot.length_m > 0.0
        assert divot.section_area_m2 > 0.0
        assert divot.volume_m3 == pytest.approx(divot.section_area_m2 * _SOLE_WIDTH_M)
        assert divot.mass_kg > 0.0
        assert divot.submerged_duration_s > 0.0

    def test_the_entry_lands_behind_the_ball(
        self, shot: ShotResult, head: HeadModel, scene: StrikeScene
    ) -> None:
        """Wivou et al. (2016) report 25-150 mm; this scene puts it in band."""
        divot = divot_metrics(
            StrikeTrace.from_shot(shot),
            head,
            scene,
            width_m=_SOLE_WIDTH_M,
            bulk_density_kg_m3=_BULK_DENSITY_KG_M3,
        )
        assert 0.025 <= divot.entry_distance_behind_ball_m <= 0.150
        assert divot.exit_distance_past_ball_m > -_BALL_AHEAD_M

    def test_the_submerged_window_is_inside_the_record(
        self, shot: ShotResult, head: HeadModel, scene: StrikeScene
    ) -> None:
        interval = submerged_interval(StrikeTrace.from_shot(shot), head, scene)
        assert 0 < interval.entry_index < interval.exit_index < shot.n_steps - 1
        assert interval.duration_s > 0.0


class TestDigSkidFromARawShot:
    """The discriminator needs two free-flight samples; the shot records them."""

    def test_dig_vs_skid_accepts_the_shot_with_zero_massaging(
        self, shot: ShotResult, head: HeadModel, scene: StrikeScene
    ) -> None:
        result = dig_vs_skid(StrikeTrace.from_shot(shot), head, scene)
        assert result.incoming_path_slope == pytest.approx(
            math.tan(math.radians(_ATTACK_ANGLE_DEG)), rel=1e-6
        )
        assert math.degrees(result.entry_attack_angle_rad) == pytest.approx(
            -_ATTACK_ANGLE_DEG, abs=1e-6
        )
        assert result.verdict is not None

    def test_the_descent_return_ratio_comes_off_the_same_two_crossings(
        self, shot: ShotResult, head: HeadModel, scene: StrikeScene
    ) -> None:
        """The discriminant needs both speeds, so it needs both crossings.

        A shot truncated before the sole came back out would have no exit
        climb to divide by the entry descent, which is why issue #8702 had to
        land before #8703 could be answered on a real record.
        """
        result = dig_vs_skid(StrikeTrace.from_shot(shot), head, scene)
        assert result.entry_descent_speed_mps > 0.0
        assert result.descent_return_ratio == pytest.approx(
            result.exit_climb_speed_mps / result.entry_descent_speed_mps, rel=1e-12
        )

    def test_head_loads_come_off_the_same_trace(
        self, shot: ShotResult, head: HeadModel
    ) -> None:
        loads = head_load_metrics(StrikeTrace.from_shot(shot), head)
        assert loads.peak_resultant_force_N == pytest.approx(
            shot.peak_force_n, rel=1e-9
        )
