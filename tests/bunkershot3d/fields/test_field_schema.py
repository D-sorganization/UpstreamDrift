"""The sand-field schema: tier as data, more than one tier, honest retention.

Issue #8710 makes one requirement non-negotiable: the stored field must
carry its tier and validity status *as data*, so an illustrative field
cannot be relabelled by copying a file.  Most of this module is that one
requirement, taken apart into the ways it can be broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.exceptions import BunkerShot3DValueError
from bunkershot3d.fields.schema import (
    DEFAULT_OCCUPANCY_FLOOR_FRACTION,
    DENSITY_UNIT,
    SHEAR_RATE_UNIT,
    VELOCITY_UNIT,
    FieldLayout,
    FieldProvenance,
    FieldQuantity,
    GridGeometry,
    OccupancyRule,
    RetentionPolicy,
    RetentionRecord,
    SandFieldSeries,
    series_digest,
)
from bunkershot3d.provenance.rng import root_seed_sequence, seed_record
from bunkershot3d.solvers.envelope import MAX_VALIDATED_SPEED_M_S, EnvelopeStatus
from bunkershot3d.solvers.protocol import FidelityTier

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def grid_geometry() -> GridGeometry:
    """A 4 x 3 plane-strain lattice at 2 mm."""
    return GridGeometry(
        origin_m=np.array([-0.004, -0.002]),
        cell_size_m=0.002,
        shape=(4, 3),
        axis_names=("x", "z"),
    )


def provenance(**overrides: object) -> FieldProvenance:
    """An F1 provenance record, overridable field by field."""
    defaults: dict[str, object] = {
        "fidelity_tier": FidelityTier.F1,
        "envelope_status": EnvelopeStatus.BEYOND_VALIDATION,
        "solver_name": "bunkershot3d.solvers.mpm.solver.PlaneStrainMPMSolver",
        "kinematics": "declared straight-line approach",
        "peak_speed_m_s": 25.0,
        "caveats": ("plane_strain_no_out_of_plane",),
        "reasons": ("no published measurement exists",),
        "refused": ("club_force", "out_of_plane"),
        "settings": {"cell_size_m": 0.002, "effective_width_m": 0.03},
    }
    defaults.update(overrides)
    return FieldProvenance(**defaults)  # type: ignore[arg-type]


def retention(**overrides: object) -> RetentionRecord:
    """A retention record for a 12-frame keep out of 120 steps."""
    defaults: dict[str, object] = {
        "policy": RetentionPolicy(target_frames=12),
        "steps_marched": 120,
        "time_stride": 10,
        "frames_kept": 12,
        "time_step_s": 1.0e-5,
        "samples_in_domain": 12,
        "samples_kept": 12,
        "dropped": ("kept every 10th step",),
    }
    defaults.update(overrides)
    return RetentionRecord(**defaults)  # type: ignore[arg-type]


def grid_series(n_frames: int = 3, **overrides: object) -> SandFieldSeries:
    """A small GRID series with every quantity present."""
    geometry = grid_geometry()
    samples = geometry.n_samples
    rng = np.random.default_rng(20260816)
    defaults: dict[str, object] = {
        "time_s": np.linspace(0.0, 1.0e-4, n_frames),
        "velocity_m_s": rng.normal(size=(n_frames, samples, 2)),
        "density_kg_m3": rng.uniform(1500.0, 1700.0, size=(n_frames, samples)),
        "shear_rate_1_s": rng.uniform(0.0, 500.0, size=(n_frames, samples)),
        "positions_m": None,
        "layout": FieldLayout.GRID,
        "geometry": geometry,
        "provenance": provenance(),
        "retention": retention(
            frames_kept=n_frames, samples_in_domain=samples, samples_kept=samples
        ),
        "occupancy": OccupancyRule(reference_density_kg_m3=1712.0),
    }
    defaults.update(overrides)
    return SandFieldSeries(**defaults)  # type: ignore[arg-type]


def particle_series(n_frames: int = 3, n_samples: int = 7) -> SandFieldSeries:
    """A PARTICLE series in 3-D, to prove the schema is not F1-shaped."""
    rng = np.random.default_rng(11)
    return SandFieldSeries(
        time_s=np.linspace(0.0, 2.0e-4, n_frames),
        velocity_m_s=rng.normal(size=(n_frames, n_samples, 3)),
        density_kg_m3=rng.uniform(1400.0, 1800.0, size=(n_frames, n_samples)),
        shear_rate_1_s=None,
        positions_m=rng.normal(size=(n_frames, n_samples, 3)),
        layout=FieldLayout.PARTICLE,
        geometry=None,
        provenance=provenance(fidelity_tier=FidelityTier.F3, peak_speed_m_s=25.0),
        retention=retention(
            frames_kept=n_frames, samples_in_domain=n_samples, samples_kept=n_samples
        ),
        occupancy=OccupancyRule(reference_density_kg_m3=1712.0),
    )


class TestTierAndValidityAreData:
    """The non-negotiable: standing lives inside the object, not the name."""

    def test_a_series_cannot_exist_without_a_tier_and_a_status(self) -> None:
        with pytest.raises(TypeError):
            SandFieldSeries(  # type: ignore[call-arg]
                time_s=np.zeros(1),
                velocity_m_s=np.zeros((1, 1, 2)),
                density_kg_m3=np.zeros((1, 1)),
                shear_rate_1_s=None,
                positions_m=None,
                layout=FieldLayout.GRID,
                geometry=grid_geometry(),
            )

    def test_the_digest_covers_the_declared_tier(self) -> None:
        """Editing the tier changes the digest, so a relabel is detectable."""
        series = grid_series()
        relabelled = grid_series(
            provenance=provenance(fidelity_tier=FidelityTier.F2),
            retention=series.retention,
        )
        np.testing.assert_array_equal(series.velocity_m_s, relabelled.velocity_m_s)
        assert series_digest(series) != series_digest(relabelled)

    def test_the_digest_covers_the_declared_status(self) -> None:
        series = grid_series()
        promoted = grid_series(
            provenance=provenance(envelope_status=EnvelopeStatus.WITHIN),
            retention=series.retention,
        )
        assert series_digest(series) != series_digest(promoted)

    def test_the_digest_covers_the_arrays(self) -> None:
        """Swapping the numbers under an honest label is detectable too."""
        series = grid_series()
        tampered = grid_series(
            velocity_m_s=series.velocity_m_s * 2.0, retention=series.retention
        )
        assert series_digest(series) != series_digest(tampered)

    def test_the_digest_is_stable_for_the_same_content(self) -> None:
        assert series_digest(grid_series()) == series_digest(grid_series())

    def test_the_digest_ignores_memory_layout(self) -> None:
        """A Fortran-ordered copy is the same field, so the same digest."""
        series = grid_series()
        reordered = grid_series(
            velocity_m_s=np.asfortranarray(series.velocity_m_s),
            retention=series.retention,
        )
        assert series_digest(reordered) == series_digest(series)

    def test_an_anonymous_solver_is_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="solver_name"):
            provenance(solver_name="  ")

    def test_unstated_kinematics_are_refused(self) -> None:
        """A declared approach and a marched shot animate identically."""
        with pytest.raises(BunkerShot3DValueError, match="kinematics"):
            provenance(kinematics="")


class TestSpeedStandingIsCarried:
    """1.44 m/s is the published corpus limit; a shot is far outside it."""

    def test_the_speed_ratio_is_against_the_published_limit(self) -> None:
        record = provenance(peak_speed_m_s=25.0)
        assert record.speed_ratio == pytest.approx(25.0 / MAX_VALIDATED_SPEED_M_S)
        assert record.speed_ratio > 17.0

    def test_a_shot_speed_is_outside_the_published_corpus(self) -> None:
        assert not provenance(peak_speed_m_s=25.0).is_within_published_speed

    def test_the_headline_names_the_limit_and_the_multiple(self) -> None:
        headline = provenance(peak_speed_m_s=25.0).headline()
        assert "1.44" in headline
        assert "17x" in headline
        assert "F1" in headline
        assert "BEYOND VALIDATION" in headline

    def test_a_slow_query_is_reported_as_inside(self) -> None:
        record = provenance(peak_speed_m_s=1.0)
        assert record.is_within_published_speed
        assert "within" in record.speed_headline()

    def test_a_negative_speed_is_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="peak_speed_m_s"):
            provenance(peak_speed_m_s=-1.0)


class TestMoreThanOneTierIsRepresentable:
    """Switching tiers must not invalidate stored results."""

    def test_a_particle_layout_round_trips_its_shapes(self) -> None:
        series = particle_series()
        assert series.layout is FieldLayout.PARTICLE
        assert series.dimension == 3
        assert series.geometry is None
        np.testing.assert_array_equal(
            series.sample_positions_m(1), series.positions_m[1]
        )

    def test_a_grid_layout_implies_its_positions(self) -> None:
        series = grid_series()
        assert series.positions_m is None
        positions = series.sample_positions_m(0)
        assert positions.shape == (series.n_samples, 2)
        np.testing.assert_allclose(positions[0], [-0.004, -0.002])

    def test_both_layouts_answer_the_same_questions(self) -> None:
        """A view written against the schema does not branch on tier."""
        for series in (grid_series(), particle_series()):
            frame = series.frame(0)
            assert frame.positions_m.shape == (series.n_samples, series.dimension)
            assert frame.speed_m_s.shape == (series.n_samples,)
            assert FieldQuantity.VELOCITY in series.quantities
            assert FieldQuantity.DENSITY in series.quantities

    def test_shear_rate_is_optional_and_declared(self) -> None:
        assert FieldQuantity.SHEAR_RATE in grid_series().quantities
        assert FieldQuantity.SHEAR_RATE not in particle_series().quantities

    def test_a_grid_series_without_geometry_is_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="geometry"):
            grid_series(geometry=None)

    def test_a_particle_series_without_positions_is_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="positions"):
            grid_series(layout=FieldLayout.PARTICLE, geometry=None)

    def test_a_geometry_that_disagrees_with_the_arrays_is_refused(self) -> None:
        wrong = GridGeometry(
            origin_m=np.zeros(2), cell_size_m=0.002, shape=(5, 5), axis_names=("x", "z")
        )
        with pytest.raises(BunkerShot3DValueError, match="samples"):
            grid_series(geometry=wrong)


class TestOccupancyIsDeclaredNotChosenByTheViewer:
    """A nodal velocity is momentum over mass; the tail of a stencil lies."""

    def test_the_floor_is_a_fraction_of_the_reference_density(self) -> None:
        rule = OccupancyRule(reference_density_kg_m3=1700.0, floor_fraction=0.1)
        assert rule.floor_kg_m3 == pytest.approx(170.0)

    def test_the_default_floor_is_the_measured_one(self) -> None:
        """10% is where the reported peak stops moving on the 2 mm capture."""
        assert OccupancyRule(1700.0).floor_fraction == (
            DEFAULT_OCCUPANCY_FLOOR_FRACTION
        )
        assert DEFAULT_OCCUPANCY_FLOOR_FRACTION == 0.10

    def test_a_near_empty_sample_is_not_reportable_sand(self) -> None:
        rule = OccupancyRule(reference_density_kg_m3=1712.0)
        densities = np.array([0.0, 0.0128, 1.2, 171.3, 1700.0])
        np.testing.assert_array_equal(
            rule.occupied(densities), [False, False, False, True, True]
        )

    def test_the_masked_peak_excludes_the_stencil_tail(self) -> None:
        """The whole reason this rule exists, in one assertion."""
        series = grid_series(n_frames=2)
        density = np.array(series.density_kg_m3)
        velocity = np.array(series.velocity_m_s)
        density[0, 0] = 1.0e-3  # a stencil tail: essentially no sand
        velocity[0, 0] = [900.0, 0.0]  # round-off divided by that mass
        loud = grid_series(
            n_frames=2,
            density_kg_m3=density,
            velocity_m_s=velocity,
            retention=series.retention,
        )
        assert float(loud.speed_m_s().max()) > 800.0
        assert loud.peak_speed_m_s() < 10.0
        assert bool(np.isnan(loud.occupied_speed_m_s()[0, 0]))

    def test_a_field_with_no_reportable_sand_reports_zero_not_nan(self) -> None:
        series = grid_series(n_frames=2)
        empty = grid_series(
            n_frames=2,
            density_kg_m3=np.zeros_like(series.density_kg_m3),
            retention=series.retention,
        )
        assert empty.peak_speed_m_s() == 0.0

    def test_the_rule_describes_itself_in_both_forms(self) -> None:
        described = OccupancyRule(1712.0).describe()
        assert "10%" in described
        assert DENSITY_UNIT in described

    def test_a_density_above_the_packing_limit_is_counted_not_clipped(self) -> None:
        """Sand cannot be denser than its own densest packing."""
        rule = OccupancyRule(1712.0, max_admissible_density_kg_m3=1747.0)
        densities = np.array([1600.0, 1740.0, 1900.0, 2914.0])
        np.testing.assert_array_equal(
            rule.over_packing_limit(densities), [False, False, True, True]
        )
        note = rule.packing_note(densities)
        assert "2 of 4" in note
        assert "2914" in note
        assert "transfer artefact" in note

    def test_a_field_inside_the_limit_gets_no_note(self) -> None:
        rule = OccupancyRule(1712.0, max_admissible_density_kg_m3=1747.0)
        assert rule.packing_note(np.array([1600.0, 1700.0])) == ""

    def test_a_rule_with_no_stated_limit_makes_no_claim(self) -> None:
        rule = OccupancyRule(1712.0)
        assert rule.packing_note(np.array([9000.0])) == ""
        assert not bool(rule.over_packing_limit(np.array([9000.0])).any())

    def test_a_limit_below_the_bulk_density_is_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="max_admissible"):
            OccupancyRule(1712.0, max_admissible_density_kg_m3=1000.0)

    def test_the_limit_is_named_in_the_description(self) -> None:
        described = OccupancyRule(
            1712.0, max_admissible_density_kg_m3=1747.0
        ).describe()
        assert "densest admissible packing" in described
        assert "1747" in described

    def test_an_impossible_floor_is_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="floor_fraction"):
            OccupancyRule(1700.0, floor_fraction=1.0)

    def test_a_field_cannot_exist_without_an_occupancy_rule(self) -> None:
        with pytest.raises(TypeError):
            SandFieldSeries(  # type: ignore[call-arg]
                time_s=np.zeros(1),
                velocity_m_s=np.zeros((1, 12, 2)),
                density_kg_m3=np.zeros((1, 12)),
                shear_rate_1_s=None,
                positions_m=None,
                layout=FieldLayout.GRID,
                geometry=grid_geometry(),
                provenance=provenance(),
                retention=retention(frames_kept=1),
            )

    def test_the_digest_covers_the_occupancy_rule(self) -> None:
        """Re-thresholding a stored field is a different claim about it."""
        series = grid_series()
        loosened = grid_series(
            occupancy=OccupancyRule(1712.0, floor_fraction=0.0),
            retention=series.retention,
        )
        assert series_digest(series) != series_digest(loosened)


class TestUnitsAreOnTheData:
    """Every quantity names its unit, so no view has to remember one."""

    def test_each_quantity_carries_its_si_unit(self) -> None:
        assert FieldQuantity.VELOCITY.unit == VELOCITY_UNIT
        assert FieldQuantity.DENSITY.unit == DENSITY_UNIT
        assert FieldQuantity.SHEAR_RATE.unit == SHEAR_RATE_UNIT

    def test_each_label_includes_the_unit(self) -> None:
        for quantity in FieldQuantity:
            assert quantity.unit in quantity.label

    def test_the_metadata_records_every_unit(self) -> None:
        units = grid_series().metadata()["units"]
        assert units == {
            "time": "s",
            "velocity": VELOCITY_UNIT,
            "density": DENSITY_UNIT,
            "shear_rate": SHEAR_RATE_UNIT,
            "length": "m",
        }


class TestRetentionIsRecordedNotAssumed:
    """What was dropped is data, because "missing" and "absent" differ."""

    def test_the_stride_lands_on_or_under_the_target(self) -> None:
        policy = RetentionPolicy(target_frames=100)
        for steps in (1, 99, 100, 101, 365, 20000):
            stride = policy.stride_for(steps)
            assert stride >= 1
            assert -(-steps // stride) <= policy.target_frames

    def test_a_short_run_is_not_padded(self) -> None:
        assert RetentionPolicy(target_frames=100).stride_for(10) == 1

    def test_the_record_reports_both_fractions(self) -> None:
        record = retention()
        assert record.temporal_fraction_kept == pytest.approx(0.1)
        assert record.spatial_fraction_kept == pytest.approx(1.0)

    def test_the_sample_interval_follows_from_the_stride(self) -> None:
        assert retention().sample_interval_s == pytest.approx(1.0e-4)

    def test_the_description_names_stride_crop_and_precision(self) -> None:
        described = retention().describe()
        assert "every 10" in described
        assert "12 of 12 samples" in described
        assert "float32" in described
        assert "gzip" in described

    def test_keeping_more_samples_than_exist_is_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="samples_kept"):
            retention(samples_in_domain=4, samples_kept=8)

    def test_the_stored_precision_is_reported(self) -> None:
        assert RetentionPolicy(store_dtype="float32").relative_precision > 1e-8
        assert RetentionPolicy(store_dtype="float64").relative_precision < 1e-15

    def test_an_unsupported_dtype_is_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="store_dtype"):
            RetentionPolicy(store_dtype="float16")

    def test_a_backwards_crop_is_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="region_m"):
            RetentionPolicy(region_m=((0.0, 0.0), (-1.0, 1.0)))


class TestSeriesInvariants:
    """The refusals that stop a broken field from reading as a result."""

    def test_an_empty_series_is_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="no frames"):
            grid_series(
                time_s=np.zeros(0),
                velocity_m_s=np.zeros((0, 12, 2)),
                density_kg_m3=np.zeros((0, 12)),
                shear_rate_1_s=None,
            )

    def test_backwards_time_is_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="non-decreasing"):
            grid_series(time_s=np.array([0.0, 2.0e-5, 1.0e-5]))

    def test_a_mismatched_density_shape_is_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="density_kg_m3"):
            grid_series(density_kg_m3=np.zeros((3, 5)))

    def test_a_frame_outside_the_series_is_refused(self) -> None:
        series = grid_series()
        with pytest.raises(BunkerShot3DValueError, match="outside the field"):
            series.frame(series.n_frames)

    def test_the_series_reports_its_own_span(self) -> None:
        series = grid_series(n_frames=5)
        assert series.n_frames == 5
        assert series.duration_s == pytest.approx(1.0e-4)
        assert series.speed_m_s().shape == (5, series.n_samples)


class TestGeometry:
    """The lattice that lets GRID skip storing its positions."""

    def test_axis_coordinates_step_by_the_cell_size(self) -> None:
        geometry = grid_geometry()
        np.testing.assert_allclose(
            geometry.axis_coordinates_m(0), [-0.004, -0.002, 0.0, 0.002]
        )
        np.testing.assert_allclose(geometry.axis_coordinates_m(1), [-0.002, 0.0, 0.002])

    def test_bounds_span_the_sampled_region(self) -> None:
        lower, upper = grid_geometry().bounds_m()
        np.testing.assert_allclose(lower, [-0.004, -0.002])
        np.testing.assert_allclose(upper, [0.002, 0.002])

    def test_positions_are_raveled_in_c_order(self) -> None:
        geometry = grid_geometry()
        positions = geometry.sample_positions_m()
        assert positions.shape == (12, 2)
        # C order over (x, z): z varies fastest.
        np.testing.assert_allclose(positions[1], [-0.004, 0.0])
        np.testing.assert_allclose(positions[3], [-0.002, -0.002])

    def test_a_one_dimensional_grid_is_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="2-D or 3-D"):
            GridGeometry(
                origin_m=np.zeros(1),
                cell_size_m=0.002,
                shape=(4,),
                axis_names=("x",),
            )

    def test_an_out_of_range_axis_is_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="axis"):
            grid_geometry().axis_coordinates_m(2)

    def test_geometry_round_trips_through_a_mapping(self) -> None:
        geometry = grid_geometry()
        rebuilt = GridGeometry.from_dict(geometry.to_dict())
        np.testing.assert_allclose(rebuilt.origin_m, geometry.origin_m)
        assert rebuilt.shape == geometry.shape
        assert rebuilt.axis_names == geometry.axis_names


class TestMappingRoundTrips:
    """Everything that reaches disk survives a dict round trip."""

    def test_provenance_round_trips_including_seeds(self) -> None:
        record = seed_record(root_seed_sequence(12345), "field")
        original = provenance(seeds=(record,))
        rebuilt = FieldProvenance.from_dict(original.to_dict())
        assert rebuilt.fidelity_tier is original.fidelity_tier
        assert rebuilt.envelope_status is original.envelope_status
        assert rebuilt.kinematics == original.kinematics
        assert rebuilt.refused == original.refused
        assert rebuilt.seeds[0].entropy == record.entropy

    def test_retention_round_trips_including_the_policy(self) -> None:
        original = retention(
            policy=RetentionPolicy(
                target_frames=64,
                store_dtype="float64",
                region_m=((-0.05, -0.02), (0.05, 0.02)),
            )
        )
        rebuilt = RetentionRecord.from_dict(original.to_dict())
        assert rebuilt.policy.region_m == ((-0.05, -0.02), (0.05, 0.02))
        assert rebuilt.policy.store_dtype == "float64"
        assert rebuilt.dropped == original.dropped

    def test_metadata_is_canonical_json_safe(self) -> None:
        from bunkershot3d.provenance.hashing import canonical_json

        assert canonical_json(grid_series().metadata()).startswith("{")
