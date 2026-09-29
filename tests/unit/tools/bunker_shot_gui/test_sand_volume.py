"""Extruding a plane-strain sand field into a volume (issue #8729), headless.

Four things are load-bearing here, and none of them is a shape check.

**A 3-D picture of a 2-D solve is an extrusion.** F1 solves one
plane-strain section. Sweeping it across the declared effective width
produces a volume that *looks* solved and is not, so the volume carries
:class:`~.slices.SliceFidelity.EXTRUDED` and every sheet across the width
is bit-identical. The tests assert that identity rather than hiding it:
if the sheets ever differ, something has invented out-of-plane structure
the model does not have.

**Velocity keeps its direction.** Sand pushed ahead of the sole and sand
riding up the face reach the same speed. A volume that stored only a
magnitude could not tell them apart, so the in-plane components survive
into the volume and the tests check a known ramp against them.

**The colour scale is injected, never inferred per frame.** Issue #8728
fixed a real per-grid auto-scaling defect. A volume is the worst place to
reintroduce it, so the scale merges across frames and across designs.

**Tier travels with the sand.** An F1 field drawn over an F0 shot would
be the most persuasive lie this epic could tell, so the tier is stored as
data on the volume and checked against the scene's own.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.fields.schema import FieldQuantity
from bunkershot3d.solvers.envelope import EnvelopeStatus
from bunkershot3d.solvers.protocol import FidelityTier
from src.tools.bunker_shot_gui.sandvolume import (
    SandVolume,
    SandVolumeScale,
    sand_volume,
    sand_volume_scale,
)
from src.tools.bunker_shot_gui.slices import SliceFidelity

from .test_slices import BULK_DENSITY, EFFECTIVE_WIDTH_M, analytic_field

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def field_with_ejecta():  # type: ignore[no-untyped-def]
    """The analytic field plus a detached patch of sand up in the air.

    A real F1 bed throws ejecta clear of the free surface, so the lattice
    lines between the bed and the plume hold air that the crop must keep:
    cropping to the *span* of occupied lines rather than to the set of
    them is what stops the lattice going ragged.
    """
    field = analytic_field()
    density = np.array(field.density_kg_m3)
    shape = field.geometry.shape
    block = density.reshape(field.n_frames, shape[0], shape[1])
    block[:, 2:5, -2:] = BULK_DENSITY
    object.__setattr__(field, "density_kg_m3", density)
    return field


class TestTheExtrusionAnnouncesItself:
    """A volume built from a plane-strain solve must say so."""

    def test_the_volume_is_labelled_extruded(self) -> None:
        volume = sand_volume(analytic_field())
        assert volume.fidelity is SliceFidelity.EXTRUDED

    def test_the_word_extruded_reaches_the_caption(self) -> None:
        text = sand_volume(analytic_field()).describe()
        assert "extrud" in text.lower()

    def test_the_caption_names_the_tier_that_solved_the_sand(self) -> None:
        text = sand_volume(analytic_field()).describe()
        assert "F1" in text

    def test_the_caption_denies_out_of_plane_flow(self) -> None:
        """The one thing a viewer will read into a volume that is not there."""
        text = sand_volume(analytic_field()).describe().lower()
        assert "heel-to-toe" in text or "out-of-plane" in text

    def test_every_sheet_across_the_width_is_identical(self) -> None:
        """Plane strain has no across-width variation, so neither may this."""
        volume = sand_volume(analytic_field(), n_sheets=5)
        sheets = volume.sheet_speed_m_s(0)
        assert sheets.shape[0] == 5
        for index in range(1, sheets.shape[0]):
            np.testing.assert_array_equal(sheets[0], sheets[index])

    def test_the_sheets_span_the_declared_effective_width(self) -> None:
        volume = sand_volume(analytic_field(), n_sheets=5)
        assert volume.across_m[0] == pytest.approx(-EFFECTIVE_WIDTH_M / 2.0)
        assert volume.across_m[-1] == pytest.approx(EFFECTIVE_WIDTH_M / 2.0)

    def test_a_single_sheet_is_refused(self) -> None:
        """One sheet is a slice; the 2-D view already draws that honestly."""
        with pytest.raises(ValueError, match="sheet"):
            sand_volume(analytic_field(), n_sheets=1)


class TestVelocityKeepsItsDirection:
    """Magnitude alone cannot separate sand ahead of the sole from sand up the face."""

    def test_the_in_plane_components_survive(self) -> None:
        volume = sand_volume(analytic_field())
        assert volume.velocity_along_m_s.shape == volume.density_kg_m3.shape
        assert volume.velocity_up_m_s.shape == volume.density_kg_m3.shape

    def test_the_analytic_ramp_comes_back_out(self) -> None:
        """The fixture's flow grows with x along and with z up."""
        volume = sand_volume(analytic_field(), max_cells=100_000)
        last = volume.n_frames - 1
        along = volume.velocity_along_m_s[last]
        up = volume.velocity_up_m_s[last]
        # Along-flow grows with x (axis 0) and is flat in z.
        assert np.all(np.diff(along, axis=0) >= -1e-9)
        np.testing.assert_allclose(np.diff(along, axis=1), 0.0, atol=1e-9)
        # Up-flow grows with z (axis 1) and is flat in x.
        assert np.all(np.diff(up, axis=1) >= -1e-9)
        np.testing.assert_allclose(np.diff(up, axis=0), 0.0, atol=1e-9)

    def test_speed_is_derived_not_stored(self) -> None:
        volume = sand_volume(analytic_field(), max_cells=100_000)
        speed = volume.speed_m_s
        expected = np.hypot(volume.velocity_along_m_s, volume.velocity_up_m_s)
        np.testing.assert_allclose(speed[volume.occupied], expected[volume.occupied])

    def test_empty_cells_are_nan_not_zero(self) -> None:
        """Zero speed asserts still sand; there is no sand there at all."""
        volume = sand_volume(field_with_ejecta(), max_cells=100_000)
        assert np.any(~volume.occupied)
        assert np.all(np.isnan(volume.speed_m_s[~volume.occupied]))

    def test_occupied_cells_always_carry_a_number(self) -> None:
        volume = sand_volume(field_with_ejecta(), max_cells=100_000)
        assert np.all(np.isfinite(volume.speed_m_s[volume.occupied]))

    def test_arrow_samples_are_a_decimated_lattice(self) -> None:
        """Direction is drawn from a coarser lattice than the colour is."""
        volume = sand_volume(analytic_field(), max_cells=100_000)
        arrows = volume.arrows(0, n_along=6, n_up=4)
        assert arrows.along_m.size <= 6
        assert arrows.up_m.size <= 4
        assert arrows.velocity_along_m_s.shape == (
            arrows.along_m.size,
            arrows.up_m.size,
        )

    def test_arrows_are_finite_so_a_quiver_can_be_updated(self) -> None:
        """NaN in a quiver refuses to update; unoccupied cells must read zero."""
        arrows = sand_volume(analytic_field()).arrows(0)
        assert np.all(np.isfinite(arrows.velocity_along_m_s))
        assert np.all(np.isfinite(arrows.velocity_up_m_s))


class TestNothingAutoScales:
    """Issue #8728, in three dimensions."""

    def test_the_scale_covers_every_frame(self) -> None:
        volume = sand_volume(analytic_field(), max_cells=100_000)
        scale = sand_volume_scale((volume,))
        assert scale.speed_m_s[1] >= volume.peak_speed_m_s - 1e-9

    def test_two_designs_share_one_scale(self) -> None:
        slow = sand_volume(analytic_field(peak_m_s=5.0))
        fast = sand_volume(analytic_field(peak_m_s=25.0))
        shared = sand_volume_scale((slow, fast))
        assert shared.speed_m_s[1] >= fast.peak_speed_m_s - 1e-9
        assert shared == sand_volume_scale((slow,)).merged(sand_volume_scale((fast,)))

    def test_a_merged_scale_is_not_either_half(self) -> None:
        """The bug #8728 fixed: two designs each normalised to its own peak."""
        slow = sand_volume(analytic_field(peak_m_s=5.0))
        fast = sand_volume(analytic_field(peak_m_s=25.0))
        assert sand_volume_scale((slow,)) != sand_volume_scale((slow, fast))

    def test_normalising_is_clamped_to_the_unit_interval(self) -> None:
        scale = sand_volume_scale((sand_volume(analytic_field()),))
        values = np.array([-1.0, 0.0, 1.0e6])
        normalised = scale.normalise(FieldQuantity.VELOCITY, values)
        assert np.nanmin(normalised) >= 0.0
        assert np.nanmax(normalised) <= 1.0

    def test_normalising_keeps_nan_as_nan(self) -> None:
        """An empty cell must stay transparent, not clamp to the ramp's floor."""
        scale = sand_volume_scale((sand_volume(analytic_field()),))
        out = scale.normalise(FieldQuantity.VELOCITY, np.array([np.nan]))
        assert np.isnan(out[0])

    def test_an_empty_comparison_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            sand_volume_scale(())

    def test_the_density_ramp_differs_from_the_speed_ramp(self) -> None:
        scale = sand_volume_scale((sand_volume(analytic_field()),))
        assert scale.colormap_name(FieldQuantity.VELOCITY) != scale.colormap_name(
            FieldQuantity.DENSITY
        )

    def test_shear_is_not_a_volume_channel(self) -> None:
        scale = sand_volume_scale((sand_volume(analytic_field()),))
        with pytest.raises(ValueError, match="shear"):
            scale.colormap_name(FieldQuantity.SHEAR_RATE)


class TestTierTravelsWithTheSand:
    """An F1 field over an F0 shot is the lie this epic must not ship."""

    def test_the_tier_is_stored_as_data(self) -> None:
        volume = sand_volume(analytic_field())
        assert volume.fidelity_tier is FidelityTier.F1

    def test_the_validity_status_is_stored_as_data(self) -> None:
        volume = sand_volume(analytic_field())
        assert volume.envelope_status is EnvelopeStatus.BEYOND_VALIDATION

    def test_the_source_digest_is_carried(self) -> None:
        """Provenance covers the arrays, so a redrawn volume is traceable."""
        volume = sand_volume(analytic_field())
        assert len(volume.source_digest) == 64

    def test_a_field_with_no_grains_is_refused(self) -> None:
        """F0 resolves no grains, so there is no volume to build from one."""
        field = analytic_field()
        object.__setattr__(field.provenance, "fidelity_tier", FidelityTier.F0)
        with pytest.raises(ValueError, match="F0"):
            sand_volume(field)

    def test_the_speed_headline_survives_into_the_volume(self) -> None:
        volume = sand_volume(analytic_field())
        assert "1.44" in volume.speed_headline or "m/s" in volume.speed_headline


class TestTheLatticeIsWorldSpace:
    """A volume drawn in the wrong axes is a volume in the wrong place."""

    def test_the_along_axis_is_world_x(self) -> None:
        field = analytic_field()
        volume = sand_volume(field, max_cells=100_000)
        expected = field.geometry.axis_coordinates_m(0)
        np.testing.assert_allclose(volume.along_m, expected)

    def test_lattice_lines_empty_for_the_whole_record_are_dropped(self) -> None:
        """An F1 bed's run-in and ejecta headroom are mostly air.

        The first render of a real capture framed the scene around the
        whole bed and shrank the impact zone to a smudge.
        """
        field = analytic_field()
        volume = sand_volume(field, max_cells=100_000)
        assert volume.n_up < field.geometry.shape[1]

    def test_the_kept_lines_stay_a_uniform_lattice(self) -> None:
        """A ragged crop forces every view into scattered interpolation."""
        volume = sand_volume(field_with_ejecta(), max_cells=100_000)
        np.testing.assert_allclose(
            np.diff(volume.up_m), np.diff(volume.up_m)[0], atol=1e-12
        )

    def test_the_crop_is_reported(self) -> None:
        volume = sand_volume(analytic_field(), max_cells=100_000)
        assert "hold sand" in volume.decimation_note

    def test_a_field_with_no_sand_anywhere_keeps_its_whole_lattice(self) -> None:
        """An empty field is a real answer, not a crash."""
        field = analytic_field()
        object.__setattr__(field, "density_kg_m3", np.zeros_like(field.density_kg_m3))
        volume = sand_volume(field, max_cells=100_000)
        assert volume.n_up == field.geometry.shape[1]

    def test_the_up_axis_is_world_z(self) -> None:
        field = field_with_ejecta()
        volume = sand_volume(field, max_cells=100_000)
        expected = field.geometry.axis_coordinates_m(1)
        np.testing.assert_allclose(volume.up_m, expected)

    def test_decimation_keeps_the_lattice_uniform(self) -> None:
        volume = sand_volume(analytic_field(), max_cells=40)
        assert volume.n_along * volume.n_up <= 40
        np.testing.assert_allclose(
            np.diff(volume.along_m), np.diff(volume.along_m)[0], atol=1e-12
        )

    def test_decimation_says_how_much_it_dropped(self) -> None:
        volume = sand_volume(analytic_field(), max_cells=40)
        assert "of" in volume.decimation_note

    def test_the_body_outline_is_carried_for_every_frame(self) -> None:
        volume = sand_volume(analytic_field())
        outline = volume.body_outline_m
        assert outline is not None
        assert outline.shape[0] == volume.n_frames

    def test_a_frame_outside_the_record_is_refused(self) -> None:
        volume = sand_volume(analytic_field())
        with pytest.raises(ValueError, match="outside"):
            volume.arrows(volume.n_frames)


class TestValueObjectRefusals:
    """The volume validates itself rather than being drawn wrong."""

    def test_a_ragged_volume_is_refused(self) -> None:
        volume = sand_volume(analytic_field())
        with pytest.raises(ValueError, match="shape"):
            SandVolume(
                time_s=volume.time_s,
                along_m=volume.along_m,
                up_m=volume.up_m,
                across_m=volume.across_m,
                velocity_along_m_s=volume.velocity_along_m_s[:, :-1],
                velocity_up_m_s=volume.velocity_up_m_s,
                density_kg_m3=volume.density_kg_m3,
                occupied=volume.occupied,
                body_outline_m=volume.body_outline_m,
                fidelity=volume.fidelity,
                fidelity_tier=volume.fidelity_tier,
                envelope_status=volume.envelope_status,
                source_digest=volume.source_digest,
                kinematics=volume.kinematics,
                speed_headline=volume.speed_headline,
                effective_width_m=volume.effective_width_m,
                decimation_note=volume.decimation_note,
            )

    def test_a_degenerate_scale_is_refused(self) -> None:
        with pytest.raises(ValueError, match="increase"):
            SandVolumeScale(speed_m_s=(1.0, 1.0), density_kg_m3=(0.0, 1.0))
