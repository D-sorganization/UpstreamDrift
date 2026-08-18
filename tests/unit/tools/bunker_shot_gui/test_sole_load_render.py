"""Drawing the sole load field and the contact patch (issues #8705, #8707).

The drawing is tested without Qt. ADR-0027's viewport layer offers MeshCat,
Rerun and VTK; none is installed here, so the workbench takes the documented
degraded path and draws a matplotlib figure instead of a 3-D scene. That
fallback is a *stated* condition, checked below, not a silent substitution.

The honesty properties are the load-bearing ones:

* the validity status and the fidelity tier are drawn **inside every axes**,
  so a frame that is screenshotted, exported or pasted carries its own
  disclaimer;
* colour limits come from a scale fixed over the whole shot, so two frames --
  and two grinds -- are directly comparable;
* every axis and every colour bar states its unit.
"""

from __future__ import annotations

import numpy as np
import pytest

from matplotlib.figure import Figure

from src.tools.bunker_shot_gui.field import (
    ContactPatch,
    LoadComponent,
    SoleLoadField,
    contact_patch,
)
from src.tools.bunker_shot_gui.render import (
    ViewportFallback,
    draw_shot_frame,
    field_scales,
    frame_stamp,
    sole_load_still,
    viewport_fallback,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="module")
def field(nominal_shot) -> SoleLoadField:  # type: ignore[no-untyped-def]
    """The per-element load field of the nominal shot."""
    assert nominal_shot.sole_field is not None
    return nominal_shot.sole_field


@pytest.fixture(scope="module")
def patch(field: SoleLoadField) -> ContactPatch:
    """Its contact-patch series."""
    return contact_patch(field)


def _texts(figure: Figure) -> list[str]:
    """Every string drawn inside an axes of ``figure``."""
    return [text.get_text() for axes in figure.axes for text in axes.texts]


class TestTheViewportDegradesOutLoud:
    """ADR-0027 picks the renderer; the workbench states what it got."""

    def test_no_three_dimensional_provider_is_installed_here(self) -> None:
        fallback = viewport_fallback()
        assert isinstance(fallback, ViewportFallback)
        assert fallback.provider is None
        assert fallback.degraded

    def test_the_reason_names_what_is_missing(self) -> None:
        reason = viewport_fallback().reason
        assert reason
        assert "meshcat" in reason.lower()

    def test_the_fallback_names_the_renderer_actually_used(self) -> None:
        assert viewport_fallback().renderer == "matplotlib"


class TestTheStampIsInTheFrame:
    """A caption beside a figure does not travel with a screenshot."""

    def test_the_stamp_states_the_validity_status(self, field: SoleLoadField) -> None:
        assert "BEYOND VALIDATION" in frame_stamp(field)

    def test_the_stamp_states_the_fidelity_tier(self, field: SoleLoadField) -> None:
        assert "F0" in frame_stamp(field)

    def test_every_axes_carries_the_stamp(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        figure = sole_load_still(field, patch)
        stamp = frame_stamp(field)
        drawn = [
            axes
            for axes in figure.axes
            if any(t.get_text() == stamp for t in axes.texts)
        ]
        assert len(drawn) >= 4

    def test_the_stamp_is_not_only_a_title(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        figure = sole_load_still(field, patch)
        assert frame_stamp(field) in _texts(figure)


class TestTheColourScaleIsSharedAcrossFrames:
    """Auto-scaling per frame would make every frame look like peak load."""

    def _limits(self, figure: Figure) -> list[tuple[float, float]]:
        return [
            collection.get_clim()
            for axes in figure.axes
            for collection in axes.collections
            if collection.get_array() is not None
        ]

    def test_two_frames_are_drawn_on_the_same_limits(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        scales = field_scales((field,))
        early = sole_load_still(field, patch, frame=patch.initial_frame, scales=scales)
        late = sole_load_still(
            field, patch, frame=int(patch.area_m2.argmax()), scales=scales
        )
        assert self._limits(early) == self._limits(late)

    def test_the_limits_are_the_scale_not_the_frame_maximum(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        scales = field_scales((field,))
        figure = sole_load_still(field, patch, frame=patch.initial_frame, scales=scales)
        assert scales[LoadComponent.INERTIAL].limits_pa in self._limits(figure)

    def test_scales_cover_every_supplied_design(self, field: SoleLoadField) -> None:
        scales = field_scales((field, field))
        assert scales[LoadComponent.DEPTH].peak_pa == pytest.approx(
            field.scale(LoadComponent.DEPTH).peak_pa
        )


class TestUnitsAreOnEverything:
    def test_the_sole_axes_are_in_millimetres(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        figure = sole_load_still(field, patch)
        labels = [axes.get_xlabel() + axes.get_ylabel() for axes in figure.axes]
        assert any("mm" in label for label in labels)

    def test_the_colour_bars_are_in_pascals(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        figure = sole_load_still(field, patch)
        labels = [axes.get_ylabel() for axes in figure.axes]
        assert any("Pa" in label for label in labels)

    def test_the_patch_area_is_in_square_centimetres(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        figure = sole_load_still(field, patch)
        labels = [axes.get_ylabel() for axes in figure.axes]
        assert any("cm" in label for label in labels)

    def test_the_time_axis_is_in_milliseconds(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        figure = sole_load_still(field, patch)
        labels = [axes.get_xlabel() for axes in figure.axes]
        assert any("ms" in label for label in labels)


class TestTheFigureShowsBothIssues:
    def test_the_two_terms_get_a_panel_each(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        figure = sole_load_still(field, patch)
        titles = " ".join(axes.get_title() for axes in figure.axes)
        assert LoadComponent.DEPTH.label in titles
        assert LoadComponent.INERTIAL.label in titles

    def test_the_contact_patch_gets_its_own_panel(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        titles = " ".join(
            axes.get_title() for axes in sole_load_still(field, patch).axes
        )
        assert "patch" in titles.lower()

    def test_the_patch_area_time_series_is_drawn(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        figure = sole_load_still(field, patch)
        drawn = [
            line
            for axes in figure.axes
            for line in axes.lines
            if np.asarray(line.get_ydata()).size == patch.n_frames
        ]
        assert drawn

    def test_the_leading_edge_is_marked(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        figure = sole_load_still(field, patch)
        assert any("leading edge" in text.lower() for text in _texts(figure)) or any(
            "leading edge" in (line.get_label() or "").lower()
            for axes in figure.axes
            for line in axes.lines
        )


class TestDrawingIsRepeatableAndBounded:
    def test_redrawing_into_one_figure_does_not_accumulate_axes(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        figure = Figure(figsize=(8.0, 6.0))
        draw_shot_frame(figure, field, patch, frame=0)
        first = len(figure.axes)
        draw_shot_frame(figure, field, patch, frame=1)
        assert len(figure.axes) == first

    def test_a_frame_outside_the_shot_is_refused(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        with pytest.raises(ValueError, match="frame"):
            sole_load_still(field, patch, frame=field.n_frames)

    def test_a_patch_from_another_shot_is_refused(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        shorter = ContactPatch(
            time_s=patch.time_s[:-1],
            engaged=patch.engaged[:-1],
            element_centroid_body_m=patch.element_centroid_body_m,
            element_area_m2=patch.element_area_m2,
        )
        with pytest.raises(ValueError, match="same shot"):
            sole_load_still(field, shorter)

    def test_the_still_needs_no_patch_to_draw_the_field(
        self, field: SoleLoadField
    ) -> None:
        figure = sole_load_still(field)
        assert len(figure.axes) >= 2

    def test_every_frame_of_the_shot_can_be_drawn(
        self, field: SoleLoadField, patch: ContactPatch
    ) -> None:
        figure = Figure(figsize=(8.0, 6.0))
        scales = field_scales((field,))
        for frame in range(field.n_frames):
            draw_shot_frame(figure, field, patch, frame=frame, scales=scales)
        assert np.isfinite(patch.area_m2).all()
