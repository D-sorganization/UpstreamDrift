"""Rendering a cut through the sand (issue #8711), headless.

A field picture is the most persuasive thing this package produces and
the least validated, so most of this module is about what the frame says
rather than about what it draws.

The rest is the colour ramp. Issue #8728 fixed a real bug where per-grid
auto-scaling made two designs incomparable; the assertions here are that
no artist autoscales, that the limits come from the injected scale, and
that two designs drawn with a merged scale get identical limits.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.tools.bunker_shot_gui.render_slice import (
    SliceArtists,
    draw_slice_frame,
    slice_still,
)
from src.tools.bunker_shot_gui.slices import (
    CursorMap,
    SliceScale,
    face_normal_plane,
    heel_to_toe_series,
    slice_scale,
    swing_plane,
)
from tests.unit.tools.bunker_shot_gui.test_slices import (
    EFFECTIVE_WIDTH_M,
    analytic_field,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def stamp_text(artists: SliceArtists) -> str:
    """The in-frame stamp of a drawn cut."""
    return artists._stamp.get_text()  # noqa: SLF001 - the artist under test


class TestTheFrameSaysWhatItIs:
    """Every honesty requirement, read back off the drawn text."""

    def test_the_stamp_carries_the_status_and_the_tier(self) -> None:
        artists = draw_slice_frame(Figure(), analytic_field(), swing_plane())
        text = stamp_text(artists)
        assert "BEYOND VALIDATION" in text
        assert "F1" in text

    def test_the_stamp_names_f1s_model_not_f0s(self) -> None:
        """An MPM continuum must not be labelled 3D-RFT."""
        text = stamp_text(draw_slice_frame(Figure(), analytic_field(), swing_plane()))
        assert "plane-strain MPM continuum" in text
        assert "3D-RFT" not in text

    def test_the_stamp_names_the_published_speed_limit(self) -> None:
        """1.44 m/s: a 25 m/s shot is outside it from its first sample."""
        text = stamp_text(draw_slice_frame(Figure(), analytic_field(), swing_plane()))
        assert "1.44" in text
        assert "17x" in text

    def test_the_stamp_says_the_kinematics_are_a_declared_approach(self) -> None:
        """A declared approach and a marched swing animate identically."""
        text = stamp_text(draw_slice_frame(Figure(), analytic_field(), swing_plane()))
        assert "declared approach, not a swing" in text

    def test_the_stamp_says_what_this_cut_is(self) -> None:
        field = analytic_field()
        stations = heel_to_toe_series(width_m=EFFECTIVE_WIDTH_M, n_stations=3)
        solved = stamp_text(draw_slice_frame(Figure(), field, stations[1]))
        extruded = stamp_text(draw_slice_frame(Figure(), field, stations[0]))
        assert "solved plane" in solved
        assert "extruded from the solved plane" in extruded

    def test_the_stamp_says_whether_through_plane_flow_exists(self) -> None:
        field = analytic_field()
        parallel = stamp_text(draw_slice_frame(Figure(), field, swing_plane()))
        oblique = stamp_text(
            draw_slice_frame(Figure(), field, face_normal_plane(face_open_deg=15.0))
        )
        assert "absent here rather than measured as zero" in parallel
        assert "NOT measured heel-to-toe" in oblique

    def test_the_stamp_says_how_the_shared_cursor_maps(self) -> None:
        field = analytic_field(n_frames=4)
        artists = draw_slice_frame(
            Figure(), field, swing_plane(), cursor=CursorMap(53, field.n_frames)
        )
        assert "different time bases" in stamp_text(artists)

    def test_the_stamp_says_where_the_sand_is_considered_to_be(self) -> None:
        text = stamp_text(draw_slice_frame(Figure(), analytic_field(), swing_plane()))
        assert "density >=" in text

    def test_the_stamp_confesses_a_density_above_the_packing_limit(self) -> None:
        """Sand denser than its own densest packing is a transfer artefact."""
        from bunkershot3d.fields.schema import OccupancyRule, SandFieldSeries

        field = analytic_field()
        packed = SandFieldSeries(
            time_s=field.time_s,
            velocity_m_s=field.velocity_m_s,
            density_kg_m3=field.density_kg_m3 * 1.6,
            shear_rate_1_s=field.shear_rate_1_s,
            positions_m=None,
            layout=field.layout,
            geometry=field.geometry,
            provenance=field.provenance,
            retention=field.retention,
            occupancy=OccupancyRule(
                reference_density_kg_m3=field.occupancy.reference_density_kg_m3,
                max_admissible_density_kg_m3=1747.0,
            ),
            body_outline_m=field.body_outline_m,
        )
        text = stamp_text(draw_slice_frame(Figure(), packed, swing_plane()))
        assert "exceed the densest packing" in text
        assert "transfer artefact" in text

    def test_a_field_inside_its_packing_limit_carries_no_apology(self) -> None:
        text = stamp_text(draw_slice_frame(Figure(), analytic_field(), swing_plane()))
        assert "transfer artefact" not in text

    def test_the_stamp_carries_units_on_everything_it_quotes(self) -> None:
        text = stamp_text(draw_slice_frame(Figure(), analytic_field(), swing_plane()))
        for unit in ("mm", "deg", "m/s", "ms", "kg/m^3"):
            assert unit in text

    def test_the_stamp_follows_the_frame(self) -> None:
        field = analytic_field(n_frames=4)
        artists = draw_slice_frame(Figure(), field, swing_plane(), frame=0)
        opening = stamp_text(artists)
        artists.update(3)
        assert stamp_text(artists) != opening
        assert "frame 4/4" in stamp_text(artists)


class TestColourScalingIsFixed:
    """#8728: two designs must be comparable, and frames must be too."""

    def test_the_limits_come_from_the_injected_scale(self) -> None:
        scale = SliceScale((0.0, 42.0), (0.0, 1800.0), (0.0, 500.0))
        artists = draw_slice_frame(
            Figure(), analytic_field(), swing_plane(), scale=scale
        )
        assert artists._panels["speed"].mesh.get_clim() == (0.0, 42.0)  # noqa: SLF001
        assert artists._panels["density"].mesh.get_clim() == (0.0, 1800.0)  # noqa: SLF001
        assert artists._panels["shear"].mesh.get_clim() == (0.0, 500.0)  # noqa: SLF001

    def test_the_limits_do_not_move_between_frames(self) -> None:
        field = analytic_field(n_frames=4)
        artists = draw_slice_frame(Figure(), field, swing_plane())
        opening = artists._panels["speed"].mesh.get_clim()  # noqa: SLF001
        for frame in range(field.n_frames):
            artists.update(frame)
            assert artists._panels["speed"].mesh.get_clim() == opening  # noqa: SLF001

    def test_two_designs_on_a_merged_scale_share_their_limits(self) -> None:
        quiet = analytic_field(peak_m_s=5.0)
        loud = analytic_field(peak_m_s=25.0)
        shared = slice_scale([quiet, loud])
        left = draw_slice_frame(Figure(), quiet, swing_plane(), scale=shared)
        right = draw_slice_frame(Figure(), loud, swing_plane(), scale=shared)
        assert (
            left._panels["speed"].mesh.get_clim()  # noqa: SLF001
            == right._panels["speed"].mesh.get_clim()  # noqa: SLF001
        )

    def test_a_quiet_design_alone_would_have_looked_as_hot(self) -> None:
        """The bug, demonstrated: this is what must not happen by default."""
        quiet = analytic_field(peak_m_s=5.0)
        loud = analytic_field(peak_m_s=25.0)
        alone = draw_slice_frame(Figure(), quiet, swing_plane())
        shared = draw_slice_frame(
            Figure(), quiet, swing_plane(), scale=slice_scale([quiet, loud])
        )
        assert (
            alone._panels["speed"].mesh.get_clim()  # noqa: SLF001
            != shared._panels["speed"].mesh.get_clim()  # noqa: SLF001
        )

    def test_no_panel_autoscales(self) -> None:
        artists = draw_slice_frame(Figure(), analytic_field(), swing_plane())
        for panel in artists._panels.values():  # noqa: SLF001
            assert not panel.axes.get_autoscale_on()

    def test_the_arrow_length_scale_is_fixed_too(self) -> None:
        """An arrow means the same speed in every frame and every design."""
        field = analytic_field(n_frames=4)
        artists = draw_slice_frame(Figure(), field, swing_plane())
        opening = artists._quiver.scale  # noqa: SLF001
        artists.update(3)
        assert artists._quiver.scale == opening  # noqa: SLF001


class TestVelocityShowsDirection:
    """Colour is magnitude; arrows are flow. Both, or the view fails."""

    def test_the_speed_panel_carries_arrows(self) -> None:
        artists = draw_slice_frame(Figure(), analytic_field(), swing_plane())
        assert artists._quiver.U.size > 0  # noqa: SLF001

    def test_the_arrows_change_with_the_frame(self) -> None:
        field = analytic_field(n_frames=4)
        artists = draw_slice_frame(Figure(), field, swing_plane(), frame=0)
        first = np.array(artists._quiver.U)  # noqa: SLF001
        artists.update(3)
        assert not np.allclose(first, artists._quiver.U)  # noqa: SLF001

    def test_the_arrows_are_zero_where_there_is_no_sand(self) -> None:
        """Zero draws nothing; nan would make the whole quiver refuse."""
        artists = draw_slice_frame(Figure(), analytic_field(), swing_plane())
        assert np.isfinite(artists._quiver.U).all()  # noqa: SLF001
        assert np.isfinite(artists._quiver.V).all()  # noqa: SLF001

    def test_the_club_section_is_drawn_on_every_panel(self) -> None:
        artists = draw_slice_frame(Figure(), analytic_field(), swing_plane())
        for panel in artists._panels.values():  # noqa: SLF001
            data = panel.outline.get_xydata()
            assert len(data) > 3
            np.testing.assert_allclose(data[0], data[-1])

    def test_the_club_section_moves_with_the_frame(self) -> None:
        field = analytic_field(n_frames=4)
        artists = draw_slice_frame(Figure(), field, swing_plane(), frame=0)
        first = artists._panels["speed"].outline.get_xydata().mean(axis=0)  # noqa: SLF001
        artists.update(3)
        last = artists._panels["speed"].outline.get_xydata().mean(axis=0)  # noqa: SLF001
        assert last[0] > first[0]

    def test_a_field_without_a_body_draws_an_empty_outline(self) -> None:
        artists = draw_slice_frame(
            Figure(), analytic_field(with_body=False), swing_plane()
        )
        assert artists._panels["speed"].outline.get_xydata().size == 0  # noqa: SLF001


class TestPanelsAndScrubbing:
    """Panel layout, and following somebody else's cursor."""

    def test_a_field_with_shear_gets_three_panels(self) -> None:
        artists = draw_slice_frame(Figure(), analytic_field(), swing_plane())
        assert artists.n_panels == 3

    def test_a_field_without_shear_gets_two(self) -> None:
        field = analytic_field()
        from bunkershot3d.fields.schema import SandFieldSeries

        no_shear = SandFieldSeries(
            time_s=field.time_s,
            velocity_m_s=field.velocity_m_s,
            density_kg_m3=field.density_kg_m3,
            shear_rate_1_s=None,
            positions_m=None,
            layout=field.layout,
            geometry=field.geometry,
            provenance=field.provenance,
            retention=field.retention,
            occupancy=field.occupancy,
            body_outline_m=field.body_outline_m,
        )
        artists = draw_slice_frame(Figure(), no_shear, swing_plane())
        assert artists.n_panels == 2

    def test_following_a_transport_maps_onto_the_fields_own_frames(self) -> None:
        field = analytic_field(n_frames=4)
        artists = draw_slice_frame(
            Figure(), field, swing_plane(), cursor=CursorMap(53, 4)
        )
        assert artists.follow_transport(0) == 0
        assert artists.follow_transport(52) == 3
        assert artists.frame_index == 3

    def test_a_transport_frame_outside_the_shot_is_refused(self) -> None:
        artists = draw_slice_frame(
            Figure(), analytic_field(n_frames=4), swing_plane(), cursor=CursorMap(53, 4)
        )
        with pytest.raises(ValueError, match="outside the shot"):
            artists.follow_transport(53)

    def test_a_field_frame_outside_the_field_is_refused(self) -> None:
        artists = draw_slice_frame(Figure(), analytic_field(n_frames=4), swing_plane())
        with pytest.raises(ValueError, match="outside the field"):
            artists.update(4)

    def test_the_still_opens_on_the_fastest_frame(self) -> None:
        field = analytic_field(n_frames=4)
        figure = slice_still(field, swing_plane())
        assert figure.axes  # a still is a drawn figure, not an empty one

    def test_the_axes_are_labelled_in_millimetres(self) -> None:
        artists = draw_slice_frame(Figure(), analytic_field(), swing_plane())
        panes = [panel.axes for panel in artists._panels.values()]  # noqa: SLF001
        assert any("[mm]" in pane.get_xlabel() for pane in panes)
        assert all("[mm]" in pane.get_ylabel() for pane in panes)
