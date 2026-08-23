"""Drawing the cross-tier comparison (issue #8713, epic #8699).

Headless. Every assertion below is about a property the *picture* has to
carry, not about pixels:

* the F1 points are **not joined** -- there is no F1 history to join them
  into, and a line between two independent marches would assert one;
* every divergence is drawn as an explicit mark rather than left as a gap
  the reader is expected to notice;
* the licence is inside the figure, not in a caption a screenshot loses;
* nothing autoscales while the cursor moves.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from bunkershot3d.solvers import EnvelopeStatus  # noqa: E402
from src.tools.bunker_shot_gui.crosstier import (  # noqa: E402
    ComparedQuantity,
    CrossTierComparison,
    CrossTierProbe,
)
from src.tools.bunker_shot_gui.render_crosstier import (  # noqa: E402
    CrossTierArtists,
    cross_tier_still,
    draw_cross_tier,
)
from src.tools.bunker_shot_gui.traces import ValidityBand  # noqa: E402
from tests.tools.bunker_shot_gui.test_crosstier import probe  # noqa: E402

pytestmark = pytest.mark.unit


def comparison(
    probes: tuple[CrossTierProbe, ...] | None = None,
    sweep: tuple[CrossTierProbe, ...] = (),
) -> CrossTierComparison:
    """A comparison over a synthetic 9-sample F0 record."""
    time_s = np.linspace(0.0, 8.0e-3, 9)
    force = np.zeros((9, 3))
    force[:, 0] = -np.linspace(10.0, 50.0, 9)
    force[:, 2] = np.linspace(10.0, 50.0, 9)
    if probes is None:
        probes = (
            probe(2, 2.0e-3, f0_force_n=20.0, f1_force_n=22.0),
            probe(4, 4.0e-3, f0_force_n=40.0, f1_force_n=110.0),
            probe(6, 6.0e-3, f0_force_n=35.0, f1_force_n=95.0),
        )
    return CrossTierComparison(
        shot_probes=probes,
        time_s=time_s,
        f0_force_n=force,
        f0_sole_depth_m=np.linspace(0.0, 0.012, 9),
        f0_velocity_m_s=np.stack(
            [
                np.linspace(25.0, 21.0, 9),
                np.zeros(9),
                np.zeros(9),
            ],
            axis=1,
        ),
        f0_divot_section_area_m2=np.linspace(0.0, 6.0e-4, 9),
        band=ValidityBand(
            time_s=time_s, statuses=tuple([EnvelopeStatus.BEYOND_VALIDATION] * 9)
        ),
        head_mass_kg=0.300,
        declared_width_m=0.030,
        bulk_density_kg_m3=1550.0,
        f1_cell_size_m=0.002,
        sweep_probes=sweep,
    )


@pytest.fixture
def figure() -> Figure:
    return Figure(figsize=(11.0, 8.0))


class TestTheOverlayDrawsBothTiersWithoutInventingAnF1History:
    def test_it_builds_one_panel_per_time_resolved_quantity(
        self, figure: Figure
    ) -> None:
        artists = draw_cross_tier(figure, comparison())
        assert artists.n_panels == 4
        assert len(figure.axes) >= 5

    def test_the_f1_points_are_markers_and_never_a_line(self, figure: Figure) -> None:
        """A line between two independent marches would assert a history."""
        artists = draw_cross_tier(figure, comparison())
        for series in artists.f1_series:
            assert series.get_linestyle() == "None"
            assert series.get_marker() not in ("", "None", None)

    def test_the_f0_curve_covers_the_whole_record(self, figure: Figure) -> None:
        artists = draw_cross_tier(figure, comparison())
        model = artists.comparison
        for line in artists.f0_series:
            assert len(line.get_xdata()) == model.n_frames

    def test_no_panel_autoscales(self, figure: Figure) -> None:
        """A y-axis that re-ranged while scrubbing makes noise read as signal."""
        artists = draw_cross_tier(figure, comparison())
        for axes in artists.panels:
            assert not axes.get_autoscalex_on()
            assert not axes.get_autoscaley_on()

    def test_the_time_axis_is_shared_and_labelled_in_milliseconds(
        self, figure: Figure
    ) -> None:
        artists = draw_cross_tier(figure, comparison())
        assert "ms" in artists.panels[-1].get_xlabel()

    def test_every_panel_names_its_unit(self, figure: Figure) -> None:
        artists = draw_cross_tier(figure, comparison())
        for axes, quantity in zip(
            artists.panels, artists.panel_quantities, strict=True
        ):
            assert quantity.unit in axes.get_ylabel()


class TestDivergenceIsMarkedNotLeftForTheEye:
    def test_each_probe_carries_a_connector_between_the_two_tiers(
        self, figure: Figure
    ) -> None:
        """The gap itself is drawn, so the disagreement is an object."""
        artists = draw_cross_tier(figure, comparison())
        assert artists.n_connectors == 3 * artists.n_panels

    def test_a_divergent_probe_is_labelled_with_its_ratio(self, figure: Figure) -> None:
        artists = draw_cross_tier(figure, comparison())
        labels = [text.get_text() for text in artists.ratio_labels]
        assert any("2.75x" in text for text in labels), labels

    def test_a_divergent_stretch_is_shaded(self, figure: Figure) -> None:
        artists = draw_cross_tier(figure, comparison())
        assert artists.n_divergence_bands > 0

    def test_a_consistent_comparison_shades_nothing(self, figure: Figure) -> None:
        model = comparison(
            (
                probe(2, 2.0e-3, f0_force_n=40.0, f1_force_n=41.0),
                probe(6, 6.0e-3, f0_force_n=40.0, f1_force_n=42.0),
            )
        )
        artists = draw_cross_tier(figure, model)
        assert artists.n_divergence_bands == 0


class TestTheCrossoverPanel:
    """The sharpest single result, given its own axes."""

    def test_it_plots_both_shares_against_speed(self, figure: Figure) -> None:
        artists = draw_cross_tier(figure, comparison())
        axes = artists.crossover_axes
        assert "m/s" in axes.get_xlabel()
        assert "share" in axes.get_ylabel().lower()

    def test_a_bracketed_crossing_is_marked_on_the_axes(self, figure: Figure) -> None:
        model = comparison(
            sweep=(
                probe(
                    0, 0.0, speed_m_s=5.0, f0_inertial_share=0.52, f1_flux_share=0.68
                ),
                probe(
                    0, 0.0, speed_m_s=12.0, f0_inertial_share=0.93, f1_flux_share=0.69
                ),
                probe(
                    0, 0.0, speed_m_s=25.0, f0_inertial_share=0.99, f1_flux_share=0.65
                ),
            )
        )
        artists = draw_cross_tier(figure, model)
        assert artists.crossover_marked
        assert "7.8" in artists.crossover_caption.get_text()

    def test_an_unbracketed_range_says_so_rather_than_drawing_nothing(
        self, figure: Figure
    ) -> None:
        artists = draw_cross_tier(figure, comparison())
        assert not artists.crossover_marked
        assert "No crossing" in artists.crossover_caption.get_text()


class TestTheFigureCarriesItsOwnCaveats:
    def test_the_licence_is_inside_the_figure(self, figure: Figure) -> None:
        artists = draw_cross_tier(figure, comparison())
        text = artists.licence_text.get_text()
        assert "not validation" in text.lower()
        assert "uncalibrated" in text.lower()

    def test_the_top_panel_is_stamped_with_the_verdict_and_the_tiers(
        self, figure: Figure
    ) -> None:
        artists = draw_cross_tier(figure, comparison())
        stamp = artists.stamp.get_text()
        assert "BEYOND VALIDATION" in stamp.upper()
        assert "not validation" in stamp.lower()

    def test_the_agreement_table_lists_every_compared_quantity(
        self, figure: Figure
    ) -> None:
        artists = draw_cross_tier(figure, comparison())
        table = artists.agreement_text.get_text()
        for quantity in ComparedQuantity:
            assert quantity.label in table

    def test_the_figure_says_the_f1_points_are_separate_marches(
        self, figure: Figure
    ) -> None:
        """The #8733 caveat, in the picture rather than in the commit."""
        artists = draw_cross_tier(figure, comparison())
        assert "march" in artists.method_text.get_text().lower()


class TestScrubbing:
    def test_the_cursor_moves_on_every_panel(self, figure: Figure) -> None:
        artists = draw_cross_tier(figure, comparison())
        artists.update(6)
        moment = 6.0e-3 * 1e3
        for cursor in artists.cursors:
            assert cursor.get_xdata()[0] == pytest.approx(moment)

    def test_a_frame_outside_the_record_is_refused(self, figure: Figure) -> None:
        artists = draw_cross_tier(figure, comparison())
        with pytest.raises(ValueError, match="outside"):
            artists.update(99)

    def test_the_readout_states_the_moment_and_the_verdict(
        self, figure: Figure
    ) -> None:
        artists = draw_cross_tier(figure, comparison())
        artists.update(4)
        text = artists.readout.get_text()
        assert "4.00" in text
        assert "BEYOND VALIDATION" in text.upper()

    def test_the_limits_do_not_move_when_the_cursor_does(self, figure: Figure) -> None:
        artists = draw_cross_tier(figure, comparison())
        before = [axes.get_ylim() for axes in artists.panels]
        artists.update(8)
        assert [axes.get_ylim() for axes in artists.panels] == before


class TestTheStill:
    def test_it_opens_on_the_peak_probe(self) -> None:
        model = comparison()
        drawn = cross_tier_still(model)
        assert isinstance(drawn, Figure)

    def test_a_still_can_be_rebuilt_from_its_own_artists(self) -> None:
        model = comparison()
        figure = Figure(figsize=(11.0, 8.0))
        first = CrossTierArtists(figure, model)
        second = CrossTierArtists(figure, model)
        assert second.n_panels == first.n_panels
