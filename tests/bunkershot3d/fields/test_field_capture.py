"""Extracting the sand field from an F1 march (issue #8710).

Two things are being checked here that a shape assertion would miss.

The first is that capture does not change the answer: a march driven in
stride-sized blocks so the field can be sampled between them has to end
where the same march taken in one go ends, bit for bit, or the field
being stored is not the field the solve had.

The second is that "empty" and "at rest" stay distinguishable.  A node
with no sand gets density zero and shear rate ``nan``; a shear rate of
zero there would assert that the sand at the free surface is not
shearing, which is a different and false claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.exceptions import BunkerShot3DValueError
from bunkershot3d.fields.capture import (
    F1_KINEMATICS_NOTE,
    capture_f1_field,
    sample_grid_field,
)
from bunkershot3d.fields.schema import FieldLayout
from bunkershot3d.fields.standing import RetentionPolicy
from bunkershot3d.sand import playing_condition
from bunkershot3d.sand.presets import PlayingCondition
from bunkershot3d.solvers import (
    DRFTSolver,
    IntrusionState,
    MaterialResponse,
    RefusalPolicy,
    SurfaceElements,
)
from bunkershot3d.solvers.envelope import MAX_VALIDATED_SPEED_M_S, EnvelopeStatus
from bunkershot3d.solvers.mpm.constitutive import SandContinuum
from bunkershot3d.solvers.mpm.solver import PlaneStrainMPMSolver
from bunkershot3d.solvers.mpm.state import settled_bed
from bunkershot3d.solvers.protocol import FidelityTier

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

SHOT_SPEED_M_S = 25.0
"""A real bunker shot, and 17x the published corpus limit of 1.44 m/s."""


@pytest.fixture(scope="module")
def material() -> SandContinuum:
    """Firm bunker sand as a continuum."""
    return SandContinuum.from_sand_state(playing_condition(PlayingCondition.FIRM))


@pytest.fixture(scope="module")
def solver(material: SandContinuum) -> PlaneStrainMPMSolver:
    """A deliberately coarse F1 solver, so a whole march fits in a test."""
    return PlaneStrainMPMSolver(
        material=material,
        cell_size_m=0.008,
        effective_width_m=0.030,
        bed_depth_m=0.03,
        refusal_policy=RefusalPolicy.REPORT,
        max_steps=4000,
    )


def sole_state(speed_m_s: float = SHOT_SPEED_M_S) -> IntrusionState:
    """A 40 x 16 mm sole section entering at 20 degrees."""
    corners = np.array(
        [
            [-0.020, 0.0, -0.008],
            [0.020, 0.0, -0.008],
            [0.020, 0.0, 0.008],
            [-0.020, 0.0, 0.008],
        ]
    )
    angle = math.radians(20.0)
    return IntrusionState(
        SurfaceElements(
            corners,
            np.tile([0.0, 0.0, -1.0], (corners.shape[0], 1)),
            np.full(corners.shape[0], 4.0e-4),
        ),
        (speed_m_s * math.cos(angle), 0.0, -speed_m_s * math.sin(angle)),
        free_surface_height_m=0.0,
    )


@pytest.fixture(scope="module")
def captured(
    solver: PlaneStrainMPMSolver,
) -> tuple[object, object]:
    """One captured march, shared by every test that only reads it."""
    return capture_f1_field(
        solver, sole_state(), policy=RetentionPolicy(target_frames=12)
    )


class TestCaptureDoesNotChangeTheAnswer:
    """A strided march must be the same march."""

    def test_the_strided_march_ends_where_a_single_march_ends(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        state = sole_state()
        reference = solver.run(state)
        _, captured_run = capture_f1_field(
            solver, state, policy=RetentionPolicy(target_frames=7)
        )
        np.testing.assert_array_equal(
            captured_run.particles.position_m, reference.particles.position_m
        )
        np.testing.assert_array_equal(
            captured_run.particles.velocity_m_s, reference.particles.velocity_m_s
        )

    def test_the_step_count_and_duration_agree(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        state = sole_state()
        reference = solver.run(state)
        _, captured_run = capture_f1_field(
            solver, state, policy=RetentionPolicy(target_frames=7)
        )
        assert captured_run.n_steps == reference.n_steps
        assert captured_run.duration_s == pytest.approx(reference.duration_s)

    def test_the_wrench_history_is_re_timed_onto_one_clock(
        self, captured: tuple[object, object]
    ) -> None:
        """``march`` restarts its clock per call; a capture must not."""
        _, run = captured
        times = run.time_history_s()  # type: ignore[attr-defined]
        assert np.all(np.diff(times) > 0.0)
        assert times[-1] == pytest.approx(
            run.n_steps * run.time_step_s  # type: ignore[attr-defined]
        )


class TestTheFieldIsTheSolversOwn:
    """Every array comes from the solver's own transfer operators."""

    def test_density_is_nodal_mass_over_cell_area(
        self, solver: PlaneStrainMPMSolver, material: SandContinuum
    ) -> None:
        setup = solver.prepare(sole_state())
        sample = sample_grid_field(setup.grid, setup.particles)
        interior = sample.density_kg_m3[sample.density_kg_m3 > 0.0]
        # Deep in the bed the scatter is a partition of unity, so the nodal
        # density recovers the bulk density it was built from.
        assert float(interior.max()) == pytest.approx(material.density_kg_m3, rel=0.02)

    def test_a_settled_bed_is_at_rest(
        self, solver: PlaneStrainMPMSolver, material: SandContinuum
    ) -> None:
        bed = settled_bed(
            material,
            x_bounds_m=(-0.05, 0.05),
            free_surface_height_m=0.0,
            depth_m=0.03,
            cell_size_m=0.008,
        )
        setup = solver.prepare(sole_state())
        sample = sample_grid_field(setup.grid, bed)
        assert float(np.abs(sample.velocity_m_s).max()) == 0.0

    def test_empty_nodes_are_nan_shear_not_zero_shear(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        setup = solver.prepare(sole_state())
        sample = sample_grid_field(setup.grid, setup.particles)
        assert sample.shear_rate_1_s is not None
        empty = sample.density_kg_m3 <= 0.0
        assert bool(empty.any()), "the domain must have air above the bed"
        assert bool(np.isnan(sample.shear_rate_1_s[empty]).all())
        assert not bool(np.isnan(sample.shear_rate_1_s[~empty]).any())

    def test_shear_rate_can_be_declined(self, solver: PlaneStrainMPMSolver) -> None:
        setup = solver.prepare(sole_state())
        assert (
            sample_grid_field(
                setup.grid, setup.particles, include_shear_rate=False
            ).shear_rate_1_s
            is None
        )

    def test_a_sheared_bed_reports_a_positive_shear_rate(
        self, captured: tuple[object, object]
    ) -> None:
        series, _ = captured
        shear = series.shear_rate_1_s  # type: ignore[attr-defined]
        assert shear is not None
        assert float(np.nanmax(shear)) > 0.0

    def test_the_sand_moves_once_the_club_arrives(
        self, captured: tuple[object, object]
    ) -> None:
        """The whole point: sand velocity exists and changes through impact."""
        series, _ = captured
        speed = series.speed_m_s()  # type: ignore[attr-defined]
        assert float(speed[0].max()) == 0.0
        assert float(speed[-1].max()) > 0.1


class TestOccupancyIsCarriedByTheCapture:
    """The stencil-tail finding, pinned against a real march."""

    def test_the_reference_density_is_the_material_it_was_solved_in(
        self, captured: tuple[object, object], material: SandContinuum
    ) -> None:
        series, _ = captured
        assert series.occupancy.reference_density_kg_m3 == pytest.approx(  # type: ignore[attr-defined]
            material.density_kg_m3
        )

    def test_the_masked_peak_is_below_the_unmasked_one(
        self, captured: tuple[object, object]
    ) -> None:
        """A stencil tail divides round-off by a millionth of a cell's sand."""
        series, _ = captured
        unmasked = float(series.speed_m_s().max())  # type: ignore[attr-defined]
        masked = series.peak_speed_m_s()  # type: ignore[attr-defined]
        assert masked < unmasked
        assert masked > 0.0

    def test_the_masked_peak_is_a_believable_multiple_of_head_speed(
        self, captured: tuple[object, object]
    ) -> None:
        """Splash outruns the sole; it does not outrun it by 87 per cent."""
        series, _ = captured
        assert series.peak_speed_m_s() < 2.0 * SHOT_SPEED_M_S  # type: ignore[attr-defined]

    def test_the_packing_limit_comes_from_the_materials_own_cap(
        self, captured: tuple[object, object], material: SandContinuum
    ) -> None:
        """No new constant: the cap the sand package already carries."""
        series, _ = captured
        ceiling = series.occupancy.max_admissible_density_kg_m3  # type: ignore[attr-defined]
        assert ceiling is not None
        assert ceiling == pytest.approx(
            material.density_kg_m3 * math.exp(-material.cap_volumetric_strain)
        )
        assert ceiling > material.density_kg_m3

    def test_the_masked_speed_is_nan_where_there_is_no_sand(
        self, captured: tuple[object, object]
    ) -> None:
        series, _ = captured
        masked = series.occupied_speed_m_s()  # type: ignore[attr-defined]
        occupied = series.occupied()  # type: ignore[attr-defined]
        assert bool(np.isnan(masked[~occupied]).all())
        assert not bool(np.isnan(masked[occupied]).any())


class TestRetentionIsRecorded:
    """What was dropped is written down, not inferred."""

    def test_the_frame_count_honours_the_target(
        self, captured: tuple[object, object]
    ) -> None:
        series, _ = captured
        # The undisturbed bed is kept as frame 0 on top of the target.
        assert series.n_frames <= 13  # type: ignore[attr-defined]
        assert series.n_frames >= 2  # type: ignore[attr-defined]

    def test_the_tail_of_the_shot_is_never_truncated(
        self, captured: tuple[object, object]
    ) -> None:
        """Striding, not truncating: the last frame is the last step."""
        series, run = captured
        assert series.time_s[-1] == pytest.approx(  # type: ignore[attr-defined]
            run.duration_s,
            rel=1e-9,  # type: ignore[attr-defined]
        )

    def test_the_record_names_the_stride_and_the_precision(
        self, captured: tuple[object, object]
    ) -> None:
        series, _ = captured
        dropped = " ".join(series.retention.dropped)  # type: ignore[attr-defined]
        assert "temporal:" in dropped
        assert "precision:" in dropped
        assert "float32" in dropped

    def test_a_full_rate_float64_capture_drops_nothing(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        series, run = capture_f1_field(
            solver,
            sole_state(),
            policy=RetentionPolicy(
                target_frames=10_000, store_dtype="float64", compression=""
            ),
        )
        assert series.retention.dropped == ()
        assert series.retention.time_stride == 1
        assert series.n_frames == run.n_steps + 1

    def test_a_crop_keeps_a_lattice_and_says_what_it_lost(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        series, _ = capture_f1_field(
            solver,
            sole_state(),
            policy=RetentionPolicy(
                target_frames=4, region_m=((-0.04, -0.02), (0.04, 0.01))
            ),
        )
        assert series.geometry is not None
        assert series.geometry.n_samples == series.n_samples
        assert series.retention.samples_kept < series.retention.samples_in_domain
        assert any("spatial:" in line for line in series.retention.dropped)

    def test_a_crop_that_keeps_nothing_is_refused(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        with pytest.raises(BunkerShot3DValueError, match="keeps no node"):
            capture_f1_field(
                solver,
                sole_state(),
                policy=RetentionPolicy(region_m=((9.0, 9.0), (10.0, 10.0))),
            )

    def test_declining_the_shear_rate_is_recorded(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        series, _ = capture_f1_field(
            solver,
            sole_state(),
            policy=RetentionPolicy(target_frames=3, include_shear_rate=False),
        )
        assert series.shear_rate_1_s is None
        assert any("shear rate:" in line for line in series.retention.dropped)


class TestProvenanceTravelsWithTheField:
    """Tier, status, kinematics and settings, on every capture."""

    def test_the_tier_is_f1(self, captured: tuple[object, object]) -> None:
        series, _ = captured
        assert series.provenance.fidelity_tier is FidelityTier.F1  # type: ignore[attr-defined]

    def test_the_status_can_never_be_better_than_beyond_validation(
        self, captured: tuple[object, object]
    ) -> None:
        series, _ = captured
        assert series.provenance.envelope_status in {  # type: ignore[attr-defined]
            EnvelopeStatus.BEYOND_VALIDATION,
            EnvelopeStatus.REFUSED,
        }

    def test_the_declared_approach_is_recorded_as_the_kinematics(
        self, captured: tuple[object, object]
    ) -> None:
        """An approach and a swing animate identically; only the note differs."""
        series, _ = captured
        assert series.provenance.kinematics == F1_KINEMATICS_NOTE  # type: ignore[attr-defined]
        assert "#8733" in F1_KINEMATICS_NOTE

    def test_the_shot_speed_is_far_outside_the_published_corpus(
        self, captured: tuple[object, object]
    ) -> None:
        series, _ = captured
        record = series.provenance  # type: ignore[attr-defined]
        assert record.peak_speed_m_s == pytest.approx(SHOT_SPEED_M_S)
        assert not record.is_within_published_speed
        assert record.speed_ratio == pytest.approx(
            SHOT_SPEED_M_S / MAX_VALIDATED_SPEED_M_S
        )

    def test_the_refused_quantities_travel_with_the_field(
        self, captured: tuple[object, object]
    ) -> None:
        series, _ = captured
        assert set(series.provenance.refused) >= {  # type: ignore[attr-defined]
            "club_force",
            "out_of_plane",
        }

    def test_the_settings_describe_the_run(
        self, captured: tuple[object, object]
    ) -> None:
        series, _ = captured
        settings = series.provenance.settings  # type: ignore[attr-defined]
        assert settings["cell_size_m"] == pytest.approx(0.008)
        assert settings["effective_width_m"] == pytest.approx(0.030)
        assert settings["n_steps"] > 0
        assert settings["sand_grain_diameter_m"] > 0.0

    def test_the_intruder_outline_travels_with_the_field(
        self, captured: tuple[object, object]
    ) -> None:
        """Without the body, a velocity picture cannot locate the face."""
        series, _ = captured
        outline = series.body_outline_m  # type: ignore[attr-defined]
        assert outline is not None
        assert outline.shape[0] == series.n_frames  # type: ignore[attr-defined]
        assert outline.shape[2] == 2
        assert outline.shape[1] >= 3

    def test_the_outline_advances_along_the_approach(
        self, captured: tuple[object, object]
    ) -> None:
        series, _ = captured
        outline = series.body_outline_m  # type: ignore[attr-defined]
        assert float(outline[-1, :, 0].mean()) > float(outline[0, :, 0].mean())
        assert float(outline[-1, :, 1].mean()) < float(outline[0, :, 1].mean())

    def test_the_outline_round_trips_through_the_store(
        self, captured: tuple[object, object], tmp_path: object
    ) -> None:
        from bunkershot3d.fields.store import load_field, save_field

        series, _ = captured
        path = save_field(tmp_path / "outline.h5", series)  # type: ignore[operator]
        loaded = load_field(path)
        assert loaded.body_outline_m is not None
        np.testing.assert_allclose(
            loaded.body_outline_m,
            series.body_outline_m,  # type: ignore[attr-defined]
            rtol=1e-6,
            atol=1e-9,
        )

    def test_the_layout_is_a_grid_with_named_axes(
        self, captured: tuple[object, object]
    ) -> None:
        series, _ = captured
        assert series.layout is FieldLayout.GRID  # type: ignore[attr-defined]
        assert series.geometry is not None  # type: ignore[attr-defined]
        assert series.geometry.axis_names == ("x", "z")  # type: ignore[attr-defined]


class TestCaptureRefusals:
    """What capture will not do."""

    def test_another_tier_is_refused(self) -> None:
        f0 = DRFTSolver(
            material=MaterialResponse.from_sand_state(
                playing_condition(PlayingCondition.FIRM)
            ),
            refusal_policy=RefusalPolicy.REPORT,
        )
        with pytest.raises(BunkerShot3DValueError, match="PlaneStrainMPMSolver"):
            capture_f1_field(f0, sole_state())  # type: ignore[arg-type]

    def test_a_static_query_is_refused_rather_than_invented(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        """F1 has no instantaneous answer, and capture does not invent one."""
        from bunkershot3d.solvers.exceptions import SolverInputError

        static = IntrusionState(
            sole_state().elements, (0.0, 0.0, 0.0), free_surface_height_m=0.0
        )
        with pytest.raises(SolverInputError, match="no in-plane velocity"):
            capture_f1_field(solver, static)
