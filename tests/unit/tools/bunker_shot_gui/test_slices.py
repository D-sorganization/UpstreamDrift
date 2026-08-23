"""Cutting planes through a sand field (issue #8711), headless.

Three things are load-bearing here and none of them is a shape check.

A heel-to-toe series through a plane-strain field shows the *same*
picture at every station, because plane strain has no heel-to-toe
direction. The tests assert that equality rather than hiding it, and
assert that the frames are labelled ``EXTRUDED`` so nobody reads five
identical panels as five solves.

Velocity keeps its direction. Sand pushed ahead of the sole and sand
riding up the face can carry the same speed, so a slice that only
reported a magnitude would answer the wrong question.

The colour scale is built over every frame of every compared field and
injected. Issue #8728 fixed a real per-grid auto-scaling bug; a
cross-section view is the worst place to reintroduce it, because the
whole question is whether a magnitude changes through impact.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.fields.schema import (
    FieldLayout,
    GridGeometry,
    SandFieldSeries,
)
from bunkershot3d.fields.standing import (
    FieldProvenance,
    OccupancyRule,
    RetentionPolicy,
    RetentionRecord,
)
from bunkershot3d.solvers.envelope import EnvelopeStatus
from bunkershot3d.solvers.protocol import FidelityTier
from src.tools.bunker_shot_gui.slices import (
    CursorMap,
    CuttingPlane,
    PlanePreset,
    SliceFidelity,
    SliceScale,
    body_focus_bounds_m,
    face_normal_plane,
    heel_to_toe_series,
    preset_planes,
    sample_plane,
    slice_scale,
    swing_plane,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

CELL_M = 0.002
EFFECTIVE_WIDTH_M = 0.030
BULK_DENSITY = 1712.0


def analytic_field(
    n_frames: int = 4,
    *,
    peak_m_s: float = 20.0,
    with_body: bool = True,
    free_surface_height_m: float = 0.0,
) -> SandFieldSeries:
    """A field whose answer is known, so a slice can be checked against it.

    The velocity is a analytic ramp: horizontal flow grows with ``x`` and
    upward flow grows with ``z``, so a cut that mixes the two axes has a
    checkable answer rather than only a plausible picture.
    """
    shape = (21, 15)
    geometry = GridGeometry(
        origin_m=np.array([-0.020, -0.014 + free_surface_height_m]),
        cell_size_m=CELL_M,
        shape=shape,
        axis_names=("x", "z"),
    )
    positions = geometry.sample_positions_m()
    x = positions[:, 0]
    z = positions[:, 1]
    velocity = np.zeros((n_frames, geometry.n_samples, 2))
    density = np.zeros((n_frames, geometry.n_samples))
    shear = np.zeros((n_frames, geometry.n_samples))
    for index in range(n_frames):
        growth = (index + 1) / n_frames
        velocity[index, :, 0] = peak_m_s * growth * (x - x.min()) / (np.ptp(x) or 1.0)
        velocity[index, :, 1] = peak_m_s * growth * (z - z.min()) / (np.ptp(z) or 1.0)
        # Sand below z = 0, air above: a free surface the mask can find.
        density[index] = np.where(z <= free_surface_height_m, BULK_DENSITY, 0.0)
        shear[index] = 100.0 * growth * np.where(z <= free_surface_height_m, 1.0, 0.0)
    outline = None
    if with_body:
        base = free_surface_height_m
        square = np.array(
            [
                [-0.006, base],
                [0.006, base],
                [0.006, base + 0.006],
                [-0.006, base + 0.006],
            ]
        )
        outline = np.stack([square + [0.002 * step, 0.0] for step in range(n_frames)])
    return SandFieldSeries(
        time_s=np.linspace(0.0, 1.0e-3, n_frames),
        velocity_m_s=velocity,
        density_kg_m3=density,
        shear_rate_1_s=shear,
        positions_m=None,
        layout=FieldLayout.GRID,
        geometry=geometry,
        provenance=FieldProvenance(
            fidelity_tier=FidelityTier.F1,
            envelope_status=EnvelopeStatus.BEYOND_VALIDATION,
            solver_name="bunkershot3d.solvers.mpm.solver.PlaneStrainMPMSolver",
            kinematics="declared straight-line approach",
            peak_speed_m_s=25.0,
            refused=("club_force", "out_of_plane"),
            settings={
                "effective_width_m": EFFECTIVE_WIDTH_M,
                "cell_size_m": CELL_M,
                "free_surface_height_m": free_surface_height_m,
            },
        ),
        retention=RetentionRecord(
            policy=RetentionPolicy(),
            steps_marched=n_frames,
            time_stride=1,
            frames_kept=n_frames,
            time_step_s=1.0e-5,
            samples_in_domain=geometry.n_samples,
            samples_kept=geometry.n_samples,
        ),
        occupancy=OccupancyRule(
            reference_density_kg_m3=BULK_DENSITY,
            max_admissible_density_kg_m3=1747.0,
        ),
        body_outline_m=outline,
    )


class TestNamedPresets:
    """The cuts a wedge shot is actually discussed in terms of."""

    def test_the_swing_plane_is_the_solved_plane(self) -> None:
        sample = sample_plane(analytic_field(), 0, swing_plane())
        assert sample.fidelity is SliceFidelity.SOLVED

    def test_a_square_face_normal_plane_is_the_swing_plane(self) -> None:
        sample = sample_plane(analytic_field(), 0, face_normal_plane())
        assert sample.fidelity is SliceFidelity.SOLVED
        assert sample.plane.obliquity_deg == pytest.approx(0.0)

    def test_an_open_face_normal_plane_is_a_projection(self) -> None:
        plane = face_normal_plane(face_open_deg=12.0)
        assert plane.obliquity_deg == pytest.approx(12.0)
        sample = sample_plane(analytic_field(), 0, plane)
        assert sample.fidelity is SliceFidelity.PROJECTED

    def test_the_heel_to_toe_series_spans_the_declared_width(self) -> None:
        planes = heel_to_toe_series(width_m=EFFECTIVE_WIDTH_M, n_stations=5)
        offsets = [plane.offset_m for plane in planes]
        assert offsets[0] == pytest.approx(-EFFECTIVE_WIDTH_M / 2)
        assert offsets[-1] == pytest.approx(EFFECTIVE_WIDTH_M / 2)
        assert len(planes) == 5

    def test_the_presets_size_themselves_from_the_field(self) -> None:
        planes = preset_planes(analytic_field(), n_stations=3)
        assert planes[0].preset is PlanePreset.SWING_PLANE
        assert planes[1].preset is PlanePreset.FACE_NORMAL
        assert [plane.preset for plane in planes[2:]] == [PlanePreset.HEEL_TO_TOE] * 3
        assert planes[-1].offset_m == pytest.approx(EFFECTIVE_WIDTH_M / 2)

    def test_an_arbitrary_plane_is_accepted(self) -> None:
        """Interactive placement, not only the named cuts."""
        plane = CuttingPlane(
            name="arbitrary",
            origin_m=np.array([0.001, 0.004, -0.002]),
            along=np.array([1.0, 0.3, 0.0]),
            up=np.array([0.0, 0.0, 1.0]),
        )
        assert plane.preset is None
        sample = sample_plane(analytic_field(), 0, plane)
        assert sample.fidelity is SliceFidelity.PROJECTED

    def test_every_preset_names_itself_and_says_what_it_shows(self) -> None:
        for preset in PlanePreset:
            assert preset.label
            assert preset.description


class TestPlaneStrainIsNotHidden:
    """Five identical panels must not read as five solves."""

    def test_heel_to_toe_stations_are_bit_for_bit_identical(self) -> None:
        """This is what plane strain means, asserted rather than glossed."""
        series = analytic_field()
        planes = heel_to_toe_series(width_m=EFFECTIVE_WIDTH_M, n_stations=5)
        samples = [sample_plane(series, 1, plane) for plane in planes]
        for sample in samples[1:]:
            np.testing.assert_array_equal(
                sample.velocity_along_m_s, samples[0].velocity_along_m_s
            )
            np.testing.assert_array_equal(
                sample.density_kg_m3, samples[0].density_kg_m3
            )

    def test_an_offset_station_is_labelled_extruded(self) -> None:
        series = analytic_field()
        planes = heel_to_toe_series(width_m=EFFECTIVE_WIDTH_M, n_stations=5)
        assert sample_plane(series, 1, planes[0]).fidelity is SliceFidelity.EXTRUDED
        assert sample_plane(series, 1, planes[2]).fidelity is SliceFidelity.SOLVED

    def test_a_station_outside_the_declared_width_is_refused(self) -> None:
        """The width is an assumption; there is nothing beyond it to extrude."""
        series = analytic_field()
        with pytest.raises(ValueError, match="effective width"):
            sample_plane(series, 0, swing_plane(offset_m=EFFECTIVE_WIDTH_M))

    def test_a_parallel_cut_reports_no_through_plane_velocity(self) -> None:
        """Absent, not measured as zero: different claims."""
        sample = sample_plane(analytic_field(), 1, swing_plane())
        assert sample.velocity_through_m_s is None
        assert "absent" in sample.through_plane_note

    def test_an_oblique_cut_says_its_through_component_is_resolved(self) -> None:
        sample = sample_plane(
            analytic_field(), 1, face_normal_plane(face_open_deg=20.0)
        )
        assert sample.velocity_through_m_s is not None
        assert "NOT measured heel-to-toe" in sample.through_plane_note

    def test_an_edge_on_cut_is_refused(self) -> None:
        """A cut across the solved plane meets it in a line, not an area."""
        plane = CuttingPlane(
            name="edge on",
            origin_m=np.zeros(3),
            along=np.array([0.0, 1.0, 0.0]),
            up=np.array([0.0, 0.0, 1.0]),
        )
        with pytest.raises(ValueError, match="edge-on"):
            sample_plane(analytic_field(), 0, plane)

    def test_a_particle_field_is_refused_rather_than_scattered(self) -> None:
        series = analytic_field()
        particles = SandFieldSeries(
            time_s=series.time_s,
            velocity_m_s=series.velocity_m_s,
            density_kg_m3=series.density_kg_m3,
            shear_rate_1_s=series.shear_rate_1_s,
            positions_m=np.broadcast_to(
                series.geometry.sample_positions_m(),  # type: ignore[union-attr]
                series.velocity_m_s.shape,
            ).copy(),
            layout=FieldLayout.PARTICLE,
            geometry=None,
            provenance=series.provenance,
            retention=series.retention,
            occupancy=series.occupancy,
        )
        with pytest.raises(ValueError, match="no lattice"):
            sample_plane(particles, 0, swing_plane())


class TestAnEdgeOnCutCannotProduceABlankPicture:
    """A blank cut nobody was told about is the worst failure here."""

    def test_the_focus_bounds_refuse_an_edge_on_plane(self) -> None:
        """This runs before any sampling, so it needs its own guard."""
        plane = CuttingPlane(
            name="edge on",
            origin_m=np.zeros(3),
            along=np.array([0.0, 1.0, 0.0]),
            up=np.array([0.0, 0.0, 1.0]),
        )
        with pytest.raises(ValueError, match="edge-on"):
            body_focus_bounds_m(analytic_field(), plane)

    def test_the_focus_bounds_are_finite_for_an_oblique_plane(self) -> None:
        bounds = body_focus_bounds_m(
            analytic_field(), face_normal_plane(face_open_deg=60.0)
        )
        assert bounds is not None
        assert all(np.isfinite(value) for pair in bounds for value in pair)

    def test_an_infinite_window_is_refused_rather_than_drawn(self) -> None:
        """Infinite bounds pass an increasing check and make every sample nan."""
        with pytest.raises(ValueError, match="must be finite"):
            sample_plane(
                analytic_field(),
                0,
                swing_plane(),
                along_bounds_m=(-np.inf, np.inf),
            )


class TestThePresetsSitOnTheFieldsOwnSurface:
    """The axis says "above the free surface", so it had better be."""

    def test_the_presets_take_the_recorded_free_surface(self) -> None:
        field = analytic_field(free_surface_height_m=0.012)
        for plane in preset_planes(field):
            assert plane.origin_m[2] == pytest.approx(0.012)

    def test_a_caller_can_still_state_a_height(self) -> None:
        field = analytic_field(free_surface_height_m=0.012)
        for plane in preset_planes(field, height_m=0.0):
            assert plane.origin_m[2] == pytest.approx(0.0)

    def test_a_raised_bed_puts_its_surface_at_h_equals_zero(self) -> None:
        """Otherwise the sand, the club and the axis label all disagree."""
        raised = analytic_field(free_surface_height_m=0.012)
        sample = sample_plane(raised, 0, preset_planes(raised)[0])
        assert sample.body_outline_m is not None
        # The body rests on the surface, so the cut reads it at h = 0.
        assert float(sample.body_outline_m[:, 1].min()) == pytest.approx(0.0)

    def test_ignoring_the_surface_would_offset_everything_by_it(self) -> None:
        """The defect, demonstrated: a world-zero datum on a raised bed."""
        raised = analytic_field(free_surface_height_m=0.012)
        sample = sample_plane(raised, 0, swing_plane(height_m=0.0))
        assert sample.body_outline_m is not None
        assert float(sample.body_outline_m[:, 1].min()) == pytest.approx(0.012)


class TestVelocityKeepsItsDirection:
    """Magnitude alone hides the distinction the question is about."""

    def test_the_solved_cut_recovers_the_field_components(self) -> None:
        series = analytic_field(n_frames=1, peak_m_s=20.0)
        sample = sample_plane(series, 0, swing_plane(), n_along=21, n_up=15)
        # The analytic ramp is linear, so bilinear resampling is exact.
        assert float(np.nanmax(sample.velocity_along_m_s)) == pytest.approx(
            20.0, rel=1e-9
        )
        assert float(np.nanmax(sample.velocity_up_m_s)) == pytest.approx(20.0, rel=1e-9)

    def test_the_along_component_is_projected_by_cos_obliquity(self) -> None:
        series = analytic_field(n_frames=1, peak_m_s=20.0)
        straight = sample_plane(series, 0, swing_plane())
        oblique = sample_plane(series, 0, face_normal_plane(face_open_deg=60.0))
        assert float(np.nanmax(oblique.velocity_along_m_s)) == pytest.approx(
            0.5 * float(np.nanmax(straight.velocity_along_m_s)), rel=1e-6
        )

    def test_the_up_component_is_unchanged_by_obliquity(self) -> None:
        """The cut's vertical axis is the world vertical either way."""
        series = analytic_field(n_frames=1)
        straight = sample_plane(series, 0, swing_plane())
        oblique = sample_plane(series, 0, face_normal_plane(face_open_deg=60.0))
        assert float(np.nanmax(oblique.velocity_up_m_s)) == pytest.approx(
            float(np.nanmax(straight.velocity_up_m_s)), rel=1e-6
        )

    def test_speed_is_masked_where_there_is_no_sand(self) -> None:
        sample = sample_plane(analytic_field(), 2, swing_plane())
        assert bool(np.isnan(sample.speed_m_s[~sample.occupied]).all())
        assert not bool(np.isnan(sample.speed_m_s[sample.occupied]).any())

    def test_off_grid_samples_are_nan_not_clamped(self) -> None:
        """Extending the edge outward would draw sand the solve had none of."""
        sample = sample_plane(
            analytic_field(),
            0,
            swing_plane(),
            along_bounds_m=(-0.5, 0.5),
            up_bounds_m=(-0.5, 0.5),
        )
        assert bool(np.isnan(sample.density_kg_m3).any())

    def test_the_peak_grows_through_the_shot(self) -> None:
        """The motivating question, on a field whose answer is known."""
        series = analytic_field(n_frames=4)
        peaks = [
            sample_plane(series, frame, swing_plane()).peak_speed_m_s
            for frame in range(series.n_frames)
        ]
        assert peaks == sorted(peaks)
        assert peaks[-1] > peaks[0]


class TestTheBodyIsInTheFrame:
    """Without the club, "ahead of the sole" is a guess."""

    def test_the_outline_is_returned_in_cut_coordinates(self) -> None:
        sample = sample_plane(analytic_field(), 0, swing_plane())
        assert sample.body_outline_m is not None
        assert sample.body_outline_m.shape[1] == 2

    def test_the_outline_moves_with_the_frame(self) -> None:
        series = analytic_field(n_frames=4)
        first = sample_plane(series, 0, swing_plane()).body_outline_m
        last = sample_plane(series, 3, swing_plane()).body_outline_m
        assert first is not None
        assert last is not None
        assert float(last[:, 0].mean()) > float(first[:, 0].mean())

    def test_an_oblique_cut_stretches_the_outline_with_the_axis(self) -> None:
        """Body and sand must be projected the same way or they part."""
        series = analytic_field(n_frames=1)
        straight = sample_plane(series, 0, swing_plane()).body_outline_m
        oblique = sample_plane(
            series, 0, face_normal_plane(face_open_deg=60.0)
        ).body_outline_m
        assert straight is not None
        assert oblique is not None
        span_straight = float(straight[:, 0].max() - straight[:, 0].min())
        span_oblique = float(oblique[:, 0].max() - oblique[:, 0].min())
        assert span_oblique == pytest.approx(2.0 * span_straight, rel=1e-6)

    def test_a_field_without_a_body_still_slices(self) -> None:
        sample = sample_plane(analytic_field(with_body=False), 0, swing_plane())
        assert sample.body_outline_m is None


class TestScalesAreInjectedNotInferred:
    """#8728 again, in the place it would do the most damage."""

    def test_the_scale_covers_every_frame(self) -> None:
        series = analytic_field(n_frames=4, peak_m_s=20.0)
        scale = slice_scale([series])
        assert scale.speed_m_s[1] == pytest.approx(
            float(np.nanmax(series.occupied_speed_m_s()))
        )

    def test_the_scale_merges_across_compared_designs(self) -> None:
        quiet = analytic_field(peak_m_s=5.0)
        loud = analytic_field(peak_m_s=25.0)
        shared = slice_scale([quiet, loud])
        assert shared.speed_m_s == slice_scale([loud, quiet]).speed_m_s
        assert shared.speed_m_s[1] >= slice_scale([loud]).speed_m_s[1]
        assert shared.speed_m_s[1] > slice_scale([quiet]).speed_m_s[1]

    def test_the_shared_scale_makes_a_quiet_design_look_quiet(self) -> None:
        """The bug #8728 fixed: each design auto-scaled to its own maximum."""
        quiet = analytic_field(peak_m_s=5.0)
        loud = analytic_field(peak_m_s=25.0)
        shared = slice_scale([quiet, loud])
        alone = slice_scale([quiet])
        assert alone.speed_m_s[1] < shared.speed_m_s[1]

    def test_the_scale_ignores_the_stencil_tail(self) -> None:
        """Limits come from occupied samples only, or noise sets the ramp."""
        series = analytic_field(n_frames=1, peak_m_s=10.0)
        velocity = np.array(series.velocity_m_s)
        empty = ~series.occupied()[0]
        velocity[0][empty] = 500.0
        noisy = SandFieldSeries(
            time_s=series.time_s,
            velocity_m_s=velocity,
            density_kg_m3=series.density_kg_m3,
            shear_rate_1_s=series.shear_rate_1_s,
            positions_m=None,
            layout=series.layout,
            geometry=series.geometry,
            provenance=series.provenance,
            retention=series.retention,
            occupancy=series.occupancy,
        )
        assert slice_scale([noisy]).speed_m_s[1] < 50.0

    def test_an_empty_covering_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one field"):
            slice_scale([])

    def test_a_scale_carries_its_units(self) -> None:
        scale = slice_scale([analytic_field()])
        assert scale.speed_unit == "m/s"
        assert scale.density_unit == "kg/m^3"
        assert scale.shear_unit == "1/s"

    def test_merging_is_symmetric_and_widening(self) -> None:
        left = SliceScale((0.0, 3.0), (0.0, 1500.0), (0.0, 20.0))
        right = SliceScale((0.0, 7.0), (0.0, 1800.0), (0.0, 5.0))
        assert left.merged(right) == right.merged(left)
        assert left.merged(right).speed_m_s == (0.0, 7.0)
        assert left.merged(right).shear_rate_1_s == (0.0, 20.0)


class TestCursorMapping:
    """One transport, two time bases, and no pretence otherwise."""

    def test_a_matching_length_maps_one_to_one(self) -> None:
        cursor = CursorMap(n_transport=53, n_field=53)
        assert cursor.is_one_to_one
        assert [cursor.field_frame(index) for index in (0, 26, 52)] == [0, 26, 52]

    def test_a_different_length_maps_by_progress(self) -> None:
        cursor = CursorMap(n_transport=53, n_field=97)
        assert cursor.field_frame(0) == 0
        assert cursor.field_frame(52) == 96
        assert cursor.field_frame(26) == 48  # halfway through both records

    def test_the_mapping_says_the_time_bases_differ(self) -> None:
        """A declared approach is not the shot's clock, and the frame says so."""
        described = CursorMap(n_transport=53, n_field=97).describe()
        assert "different time bases" in described
        assert "53" in described
        assert "97" in described

    def test_a_matching_length_says_it_is_one_to_one(self) -> None:
        assert "1:1" in CursorMap(n_transport=8, n_field=8).describe()

    def test_an_out_of_range_frame_is_refused_not_clamped(self) -> None:
        cursor = CursorMap(n_transport=53, n_field=97)
        with pytest.raises(ValueError, match="outside the shot"):
            cursor.field_frame(53)

    def test_a_single_frame_field_collapses_to_its_only_frame(self) -> None:
        cursor = CursorMap(n_transport=10, n_field=1)
        assert {cursor.field_frame(index) for index in range(10)} == {0}


class TestPlaneGeometry:
    """The frame the picture's axes are stated in."""

    def test_a_plane_must_have_perpendicular_axes(self) -> None:
        with pytest.raises(ValueError, match="perpendicular"):
            CuttingPlane(
                name="skew",
                origin_m=np.zeros(3),
                along=np.array([1.0, 0.0, 0.0]),
                up=np.array([1.0, 0.0, 1.0]),
            )

    def test_a_plane_must_be_named(self) -> None:
        with pytest.raises(ValueError, match="named"):
            CuttingPlane(
                name="  ",
                origin_m=np.zeros(3),
                along=np.array([1.0, 0.0, 0.0]),
                up=np.array([0.0, 0.0, 1.0]),
            )

    def test_the_normal_is_the_cross_product(self) -> None:
        np.testing.assert_allclose(swing_plane().normal, [0.0, -1.0, 0.0], atol=1e-12)

    def test_the_description_names_offset_and_obliquity(self) -> None:
        described = face_normal_plane(face_open_deg=8.0, offset_m=0.004).describe()
        assert "+4" in described
        assert "8" in described
        assert "mm" in described
        assert "deg" in described

    def test_a_zero_direction_is_refused(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            CuttingPlane(
                name="nowhere",
                origin_m=np.zeros(3),
                along=np.zeros(3),
                up=np.array([0.0, 0.0, 1.0]),
            )

    def test_a_degenerate_series_is_refused(self) -> None:
        with pytest.raises(ValueError, match="width_m"):
            heel_to_toe_series(width_m=0.0)
        with pytest.raises(ValueError, match="n_stations"):
            heel_to_toe_series(width_m=0.03, n_stations=0)

    def test_a_single_station_sits_on_the_solved_plane(self) -> None:
        (only,) = heel_to_toe_series(width_m=0.03, n_stations=1)
        assert only.offset_m == pytest.approx(0.0)

    def test_a_frame_outside_the_field_is_refused(self) -> None:
        series = analytic_field()
        with pytest.raises(ValueError, match="outside the field"):
            sample_plane(series, series.n_frames, swing_plane())

    def test_a_one_sample_axis_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 2 samples"):
            sample_plane(analytic_field(), 0, swing_plane(), n_along=1)

    def test_the_sample_describes_itself(self) -> None:
        described = sample_plane(analytic_field(), 1, swing_plane()).describe()
        assert "Swing plane" in described
        assert "solved plane" in described
        assert "ms" in described
