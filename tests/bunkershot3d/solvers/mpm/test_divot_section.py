"""The F1 divot as a *section*, not only a depth (issue #8713).

:meth:`~bunkershot3d.solvers.mpm.solver.MPMRun.divot_depth_m` answers how
deep the free surface was pushed. The cross-tier comparison needs the other
half of the same measurement -- how much sand was moved -- because F0's
divot is reported as an area (``DivotMetrics.section_area_m2``) and a mass,
and a depth alone cannot be compared against either.

The empty-bin case is the whole reason this is a value object rather than a
float: a bin the sand has left entirely has no surface height at all, so it
cannot contribute a depth to an integral. Skipping it under-reports the
area, and filling it in invents one. The count travels with the number so a
caller can say which it is looking at.
"""

from __future__ import annotations

import pytest

from bunkershot3d.sand import PlayingCondition, playing_condition
from bunkershot3d.solvers.exceptions import SolverInputError
from bunkershot3d.solvers.mpm.constitutive import SandContinuum
from bunkershot3d.solvers.mpm.state import (
    ParticleState,
    SurfaceDepression,
    settled_bed,
    surface_depression,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def material() -> SandContinuum:
    return SandContinuum.from_sand_state(playing_condition(PlayingCondition.FIRM))


def flat_bed(material: SandContinuum) -> ParticleState:
    """An undisturbed bed: the surface is exactly where it was seeded."""
    return settled_bed(
        material,
        x_bounds_m=(0.0, 0.080),
        free_surface_height_m=0.0,
        depth_m=0.040,
        cell_size_m=0.004,
        particles_per_cell_axis=2,
    )


class TestAnUndisturbedBedHasNoDivot:
    """The zero the measurement has to return before any other number counts."""

    def test_the_section_area_is_zero(self, material: SandContinuum) -> None:
        depression = surface_depression(
            flat_bed(material),
            free_surface_height_m=0.0,
            x_bounds_m=(0.0, 0.080),
            n_bins=16,
        )
        # The topmost particle sits half a particle spacing below the seeded
        # surface, so the profile is a constant offset rather than exactly
        # zero. What must be zero is the *variation*: a flat bed has no
        # section, whatever datum it is measured from.
        assert depression.section_area_m2 == pytest.approx(
            depression.max_depth_m * 0.080, rel=1e-9
        )

    def test_no_bin_is_empty(self, material: SandContinuum) -> None:
        depression = surface_depression(
            flat_bed(material),
            free_surface_height_m=0.0,
            x_bounds_m=(0.0, 0.080),
            n_bins=16,
        )
        assert depression.n_empty_bins == 0
        assert depression.fully_resolved


class TestAScoopedBedReportsWhatWasRemoved:
    """A depression cut by hand, against its own closed-form area."""

    def test_a_rectangular_scoop_gives_its_own_area(
        self, material: SandContinuum
    ) -> None:
        """Remove every particle above -20 mm over the middle 40 mm.

        The resulting section is a 40 mm x 20 mm rectangle, so the area is
        8.0e-4 m^2 up to the half-spacing offset the seeding leaves on the
        undisturbed shoulders.
        """
        particles = flat_bed(material)
        x = particles.position_m[:, 0]
        z = particles.position_m[:, 1]
        keep = ~((x > 0.020) & (x < 0.060) & (z > -0.020))
        scooped = ParticleState(
            position_m=particles.position_m[keep],
            velocity_m_s=particles.velocity_m_s[keep],
            mass_kg=particles.mass_kg[keep],
            initial_volume_m2=particles.initial_volume_m2[keep],
            deformation_gradient=particles.deformation_gradient[keep],
            affine=particles.affine[keep],
        )
        flat = surface_depression(
            particles,
            free_surface_height_m=0.0,
            x_bounds_m=(0.0, 0.080),
            n_bins=40,
        )
        cut = surface_depression(
            scooped,
            free_surface_height_m=0.0,
            x_bounds_m=(0.0, 0.080),
            n_bins=40,
        )
        removed = cut.section_area_m2 - flat.section_area_m2
        # 40 mm wide, ~20 mm deep. The tolerance is the particle lattice: the
        # scoop wall lands between particle rows, so the discrete cut is a
        # spacing wider or narrower than the 40 mm the mask asked for.
        assert removed == pytest.approx(8.0e-4, rel=0.12), (
            f"{cut.summary()} against a flat {flat.summary()}"
        )

    def test_the_deepest_point_is_the_scoop_floor(
        self, material: SandContinuum
    ) -> None:
        particles = flat_bed(material)
        x = particles.position_m[:, 0]
        z = particles.position_m[:, 1]
        keep = ~((x > 0.020) & (x < 0.060) & (z > -0.020))
        scooped = ParticleState(
            position_m=particles.position_m[keep],
            velocity_m_s=particles.velocity_m_s[keep],
            mass_kg=particles.mass_kg[keep],
            initial_volume_m2=particles.initial_volume_m2[keep],
            deformation_gradient=particles.deformation_gradient[keep],
            affine=particles.affine[keep],
        )
        cut = surface_depression(
            scooped,
            free_surface_height_m=0.0,
            x_bounds_m=(0.0, 0.080),
            n_bins=40,
        )
        assert cut.max_depth_m == pytest.approx(0.020, abs=0.002)


class TestAnEvacuatedBinIsCountedRatherThanGuessed:
    """The failure mode the value object exists to make visible."""

    def test_an_emptied_column_is_counted(self, material: SandContinuum) -> None:
        particles = flat_bed(material)
        x = particles.position_m[:, 0]
        keep = ~((x > 0.030) & (x < 0.050))
        pierced = ParticleState(
            position_m=particles.position_m[keep],
            velocity_m_s=particles.velocity_m_s[keep],
            mass_kg=particles.mass_kg[keep],
            initial_volume_m2=particles.initial_volume_m2[keep],
            deformation_gradient=particles.deformation_gradient[keep],
            affine=particles.affine[keep],
        )
        depression = surface_depression(
            pierced,
            free_surface_height_m=0.0,
            x_bounds_m=(0.0, 0.080),
            n_bins=40,
        )
        assert depression.n_empty_bins > 0
        assert not depression.fully_resolved
        assert "bin" in depression.summary()

    def test_a_window_with_no_particles_at_all_is_refused(
        self, material: SandContinuum
    ) -> None:
        """Not a zero divot: nothing was measured, and zero would read as one."""
        with pytest.raises(SolverInputError, match="no particle"):
            surface_depression(
                flat_bed(material),
                free_surface_height_m=0.0,
                x_bounds_m=(0.500, 0.580),
                n_bins=8,
            )


class TestTheValueObjectRefusesNonsense:
    """``raise``, never ``assert`` -- ``python -O`` must not switch these off."""

    def test_a_negative_area_is_refused(self) -> None:
        with pytest.raises(SolverInputError, match="non-negative"):
            SurfaceDepression(
                section_area_m2=-1.0,
                max_depth_m=0.0,
                n_bins=8,
                n_empty_bins=0,
                bed_width_m=0.08,
            )

    def test_more_empty_bins_than_bins_is_refused(self) -> None:
        with pytest.raises(SolverInputError, match="empty"):
            SurfaceDepression(
                section_area_m2=0.0,
                max_depth_m=0.0,
                n_bins=8,
                n_empty_bins=9,
                bed_width_m=0.08,
            )

    def test_the_displaced_mass_needs_a_width_and_a_density(self) -> None:
        depression = SurfaceDepression(
            section_area_m2=8.0e-4,
            max_depth_m=0.02,
            n_bins=40,
            n_empty_bins=0,
            bed_width_m=0.08,
        )
        assert depression.displaced_mass_kg(
            width_m=0.030, bulk_density_kg_m3=1550.0
        ) == pytest.approx(8.0e-4 * 0.030 * 1550.0)
        with pytest.raises(SolverInputError, match="width_m"):
            depression.displaced_mass_kg(width_m=0.0, bulk_density_kg_m3=1550.0)
        with pytest.raises(SolverInputError, match="bulk_density"):
            depression.displaced_mass_kg(width_m=0.03, bulk_density_kg_m3=-1.0)
