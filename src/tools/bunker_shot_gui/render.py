"""Drawing the sole load field and the contact patch (issues #8705, #8707).

Headless. This module imports matplotlib and no GUI toolkit, so the same
figure can be produced in a test, written to a file by a batch sweep, or
embedded in the Qt workbench by :mod:`src.tools.bunker_shot_gui.widgets`.

Why matplotlib and not a 3-D scene
----------------------------------

ADR-0027 put the choice of 3-D viewport behind
:mod:`src.shared.python.visualization.viewport`, which evaluates MeshCat,
Rerun and VTK/PyVista. None of the three is installed in this environment, so
the selection *degrades*, and the workbench draws the sole in plan instead of
in perspective. That is a stated fallback, reported by :func:`viewport_fallback`
and surfaced in the workbench, not a silent substitution -- the sole is a
nearly planar surface seen from below, so a plan view loses little, but the
reader is told which renderer produced what they are looking at.

The honesty rules this module implements
----------------------------------------

* **The stamp is inside the axes.** Every panel carries the validity status
  and the fidelity tier, drawn in the data area rather than in a caption
  beneath the figure, because a screenshot of a panel keeps its contents and
  loses its surroundings.
* **The colour limits are fixed.** They come from a :class:`~.field.LoadScale`
  covering the whole shot -- and, in an A/B comparison, both designs -- so a
  frame that looks hotter *is* hotter. Nothing here calls a per-frame
  ``autoscale``.
* **Units are on everything**: mm on the sole axes, Pa on the colour bars,
  cm^2 on the patch area, ms on the time axis, N in the resultant legend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from numpy.typing import NDArray

from src.shared.python.visualization.viewport import select_viewport_provider

from .field import ContactPatch, LoadComponent, LoadScale, SoleLoadField
from .report import status_colour

__all__ = [
    "RENDERER",
    "ShotFrameArtists",
    "ViewportFallback",
    "draw_shot_frame",
    "field_scales",
    "frame_stamp",
    "sole_load_still",
    "viewport_fallback",
]

RENDERER = "matplotlib"
"""What actually draws the frame once the 3-D providers have degraded."""

_MARKER_AREA_PT2 = 4.2e6
"""Points^2 per m^2 of element area. Sized so a 12x5 sole mesh tiles."""

_PALE = "#e9e4da"
"""Colour of a sole element that carried nothing at this sample."""

_EDGE = "#3a3a3a"
_STAMP_ALPHA = 0.88


@dataclass(frozen=True)
class ViewportFallback:
    """Which renderer the ADR-0027 viewport layer left the workbench with.

    Attributes:
        provider: The selected 3-D provider, or ``None`` when none is
            installed.
        reason: Why the selection degraded; empty when it did not.
        renderer: What draws the frame in the end.
    """

    provider: str | None
    reason: str
    renderer: str = RENDERER

    @property
    def degraded(self) -> bool:
        """Whether no 3-D provider was available."""
        return self.provider is None

    def describe(self) -> str:
        """One line naming the renderer and, when degraded, why.

        Returns:
            The sentence shown beside the field view.
        """
        if not self.degraded:
            return f"3-D viewport: {self.provider} (ADR-0027)"
        return (
            f"3-D viewport unavailable, drawn as a {self.renderer} plan view of "
            f"the sole. {self.reason}"
        )


def viewport_fallback() -> ViewportFallback:
    """Ask the ADR-0027 viewport layer what it can offer, and report it.

    Returns:
        The selection, degraded to :data:`RENDERER` when no 3-D provider is
        import-discoverable.
    """
    selection = select_viewport_provider()
    selected = selection.selected
    if selected is not None:
        return ViewportFallback(provider=selected.metadata.display_name, reason="")
    # Name every provider that was tried and what it wanted, rather than the
    # layer's one-line "no provider is available": the point of stating the
    # fallback is that a reader can undo it.
    missing = "; ".join(
        reason
        for status in selection.statuses
        if (reason := status.degradation_reason) is not None
    )
    return ViewportFallback(
        provider=None, reason=missing or (selection.reason or "no provider available")
    )


def frame_stamp(field: SoleLoadField) -> str:
    """Return the validity line drawn inside every panel.

    Args:
        field: The load field being drawn.

    Returns:
        A one-line stamp: the status, the tier, and the reminder that neither
        is a measurement. Kept short enough to sit inside an axes without
        covering the data it qualifies.
    """
    status, tier = field.status, field.fidelity_tier
    return (
        f"{status.value.replace('_', ' ').upper()} - "
        f"{tier.value.upper()} dynamic 3D-RFT\n"
        "not calibrated for bunker sand"
    )


def field_scales(
    fields: tuple[SoleLoadField, ...],
) -> dict[LoadComponent, LoadScale]:
    """Build the fixed colour scales one or more designs share.

    Args:
        fields: Every load field that will be drawn on these scales. Passing
            both halves of an A/B comparison is what makes the two panels
            comparable; passing one design gives a scale fixed across its own
            frames.

    Returns:
        One scale per component.

    Raises:
        ValueError: If no field was supplied.
    """
    return {
        component: LoadScale.covering(component, fields) for component in LoadComponent
    }


def _check_frame(field: SoleLoadField, frame: int) -> int:
    """Validate a frame index against the shot.

    Args:
        field: The load field.
        frame: The requested sample index.

    Returns:
        The index.

    Raises:
        ValueError: If the index is outside the recorded shot. A ``raise``,
            not an ``assert``: a silently wrapped index would draw a
            different moment from the one the transport says it is showing.
    """
    if not 0 <= int(frame) < field.n_frames:
        raise ValueError(
            f"frame {frame} is outside the recorded shot, which has "
            f"{field.n_frames} samples"
        )
    return int(frame)


def _check_patch(field: SoleLoadField, patch: ContactPatch | None) -> None:
    """Validate that a patch series describes the field it is drawn over.

    Args:
        field: The load field.
        patch: The patch series, or ``None``.

    Raises:
        ValueError: If the two do not come from one shot.
    """
    if patch is None:
        return
    if patch.n_frames != field.n_frames or patch.n_elements != field.n_elements:
        raise ValueError(
            "the load field and the contact patch must come from the same shot; "
            f"got {field.n_frames}x{field.n_elements} against "
            f"{patch.n_frames}x{patch.n_elements}"
        )


def _stamp(axes: Axes, field: SoleLoadField) -> None:
    """Draw the validity stamp inside one axes."""
    axes.text(
        0.02,
        0.98,
        frame_stamp(field),
        transform=axes.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        color="white",
        bbox={
            "facecolor": status_colour(field.status),
            "edgecolor": "none",
            "alpha": _STAMP_ALPHA,
            "boxstyle": "round,pad=0.25",
        },
        zorder=10,
    )


def _marker_sizes(field: SoleLoadField) -> NDArray[np.float64]:
    """Marker areas [pt^2] proportional to the elements' own areas."""
    return np.clip(field.element_area_m2 * _MARKER_AREA_PT2, 4.0, 400.0)


def _build_component(
    axes: Axes,
    figure: Figure,
    field: SoleLoadField,
    component: LoadComponent,
    scale: LoadScale,
) -> PathCollection:
    """Build one component's panel, everything about it that never changes."""
    low, high = scale.limits_pa
    scatter = axes.scatter(
        field.element_centroid_body_m[:, 1] * 1e3,
        field.element_centroid_body_m[:, 0] * 1e3,
        c=np.zeros(field.n_elements, dtype=np.float64),
        s=_marker_sizes(field),
        cmap=scale.colormap_name,
        vmin=low,
        vmax=high,
        marker="s",
        linewidths=0.0,
    )
    bar = figure.colorbar(scatter, ax=axes)
    bar.ax.set_ylabel(f"{component.label} [{scale.unit}]", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    _label_sole_axes(axes)
    _stamp(axes, field)
    _note(
        axes,
        f"scale peak {scale.peak_pa:.3g} {scale.unit}\n"
        f"term peaks at {field.peak_time_s(component) * 1e3:.2f} ms",
    )
    return scatter


def _note(axes: Axes, text: str) -> Text:
    """Draw a small qualifier under the stamp, inside the axes.

    Args:
        axes: The panel to draw on.
        text: The qualifier.

    Returns:
        The text artist, so a caller that has to rewrite it per frame can hold
        it rather than search the axes for it.
    """
    return axes.text(
        0.02,
        0.02,
        text,
        transform=axes.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
        color="#333333",
        zorder=9,
    )


def _label_sole_axes(axes: Axes) -> None:
    """Label a sole plan in millimetres, in the report's own orientation.

    Heel to toe across and leading edge to trailing edge *down*, matching the
    shaded plan :func:`~.report.sole_map_text` already prints, so the picture
    and the text describe the sole the same way round. It is also the shape of
    the object: a wedge sole is roughly 20 mm front to back and 70 mm heel to
    toe, so this is the orientation that does not waste the panel.
    """
    axes.set_xlabel("body y, heel -> toe [mm]", fontsize=7)
    axes.set_ylabel("body x, leading -> trailing edge [mm]", fontsize=7)
    axes.tick_params(labelsize=6)
    axes.set_aspect("equal", adjustable="box")
    if not axes.yaxis_inverted():
        axes.invert_yaxis()


def _build_patch(
    axes: Axes, field: SoleLoadField, patch: ContactPatch
) -> tuple[PathCollection, Text]:
    """Build the patch panel; only the element colours change per frame.

    One scatter over *every* element, recoloured per frame, rather than a
    static pale layer plus a per-frame engaged layer: the engaged set changes
    size every sample, so a second scatter would have to be rebuilt each time,
    which is what made the animation cost a quarter of a second per frame.
    """
    stations = patch.element_centroid_body_m[:, 0] * 1e3
    across = patch.element_centroid_body_m[:, 1] * 1e3
    sizes = np.clip(patch.element_area_m2 * _MARKER_AREA_PT2, 4.0, 400.0)
    scatter = axes.scatter(
        across, stations, s=sizes, c=_PALE, marker="s", linewidths=0.0, zorder=1
    )
    axes.axhline(
        patch.leading_edge_m * 1e3,
        color=_EDGE,
        linestyle="--",
        linewidth=1.0,
        label="leading edge",
        zorder=3,
    )
    _label_sole_axes(axes)
    axes.legend(loc="lower right", fontsize=6, framealpha=0.6)
    _stamp(axes, field)
    return scatter, _note(axes, "")


def _build_time_series(axes: Axes, field: SoleLoadField, patch: ContactPatch) -> Line2D:
    """Build the time-series panel; only the frame cursor moves."""
    time_ms = patch.time_s * 1e3
    axes.plot(
        time_ms,
        patch.area_m2 * 1e4,
        color="#8a5a1e",
        linewidth=1.6,
        label="contact patch area",
    )
    axes.set_xlabel("time from the start of the record [ms]", fontsize=7)
    axes.set_ylabel("contact patch area [cm^2]", fontsize=7)
    axes.tick_params(labelsize=6)
    cursor = axes.axvline(time_ms[0], color=_EDGE, linewidth=1.0, linestyle=":")

    # The two terms differ by three orders of magnitude at greenside speed, so
    # plotting both in newtons on one axis hides the depth term entirely.
    # Each is shown as a fraction of its own peak, with that peak stated in
    # the legend in newtons, which is what makes "when did each peak" legible
    # without implying the two are the same size.
    shares = axes.twinx()
    for component, colour in (
        (LoadComponent.DEPTH, "#2f6f9f"),
        (LoadComponent.INERTIAL, "#9b1c1c"),
    ):
        resultant = field.resultant_force_N(component)
        peak = float(resultant.max())
        if peak <= 0.0:
            continue
        shares.plot(
            field.time_s * 1e3,
            resultant / peak,
            color=colour,
            linewidth=1.0,
            linestyle="--",
            label=f"{component.label} (peak {peak:.4g} N)",
        )
    shares.set_ylabel("resultant, fraction of that term's own peak", fontsize=7)
    shares.tick_params(labelsize=6)
    handles, labels = axes.get_legend_handles_labels()
    extra = shares.get_legend_handles_labels()
    axes.legend(
        handles + extra[0],
        labels + extra[1],
        loc="lower right",
        fontsize=6,
        framealpha=0.6,
    )
    axes.set_title("Contact patch area and term resultants", fontsize=8)
    _stamp(axes, field)
    _note(
        axes,
        f"patch {patch.initial_area_m2 * 1e4:.2f} cm^2 at first contact\n"
        f"peak {patch.peak_area_m2 * 1e4:.2f} cm^2 at "
        f"{patch.peak_area_time_s * 1e3:.2f} ms",
    )
    return cursor


class ShotFrameArtists:
    """Axes built once for one shot, and the artists a frame change touches.

    Rebuilding the whole figure per frame costs about 250 ms at the shipped
    discretization -- the colour bars and the layout pass, not the 500-point
    scatters -- which is slower than the transport interval and makes playback
    queue rather than animate. Everything that does not depend on the sample is
    therefore built once and only the frame-varying artists are mutated:
    two colour arrays, one set of face colours, one cursor line and four
    titles.

    The colour limits live on the scatters, set at build time from the fixed
    :class:`~.field.LoadScale`, so no update path can reintroduce per-frame
    autoscaling.
    """

    def __init__(
        self,
        figure: Figure,
        field: SoleLoadField,
        patch: ContactPatch | None,
        scales: dict[LoadComponent, LoadScale],
    ) -> None:
        """Build the axes for one shot.

        Args:
            figure: The figure to build into; cleared first.
            field: The per-element load field.
            patch: The contact-patch series, when there is one.
            scales: The fixed colour scales to pin the panels to.
        """
        self._field = field
        self._patch = patch
        figure.clear()
        rows = 2 if patch is not None else 1
        axes = figure.subplots(rows, 2, squeeze=False)
        self._component_axes = {
            LoadComponent.DEPTH: axes[0][0],
            LoadComponent.INERTIAL: axes[0][1],
        }
        self._component_art = {
            component: _build_component(
                panel, figure, field, component, scales[component]
            )
            for component, panel in self._component_axes.items()
        }
        self._patch_axes: Axes | None = None
        self._patch_art: PathCollection | None = None
        self._patch_note: Text | None = None
        self._cursor: Line2D | None = None
        if patch is not None:
            self._patch_axes = axes[1][0]
            self._patch_art, self._patch_note = _build_patch(axes[1][0], field, patch)
            self._cursor = _build_time_series(axes[1][1], field, patch)
        figure.tight_layout()

    def update(self, frame: int) -> None:
        """Show one sample.

        Args:
            frame: The sample index.

        Raises:
            ValueError: If the index is outside the recorded shot.
        """
        index = _check_frame(self._field, frame)
        field = self._field
        for component, art in self._component_art.items():
            art.set_array(field.component_pressure_pa(component)[index])
            resultant = field.resultant_force_N(component)[index]
            self._component_axes[component].set_title(
                f"{component.label}: {resultant:.4g} N", fontsize=8
            )
        patch = self._patch
        if patch is None or self._patch_art is None or self._patch_axes is None:
            return
        engaged = patch.engaged[index]
        loaded = status_colour(field.status)
        self._patch_art.set_facecolor([loaded if flag else _PALE for flag in engaged])
        self._patch_axes.set_title(
            f"Contact patch: {patch.area_m2[index] * 1e4:.2f} cm^2", fontsize=8
        )
        reach_mm = patch.reach_m[index] * 1e3
        reach = "no contact" if not np.isfinite(reach_mm) else f"{reach_mm:.2f} mm"
        if self._patch_note is not None:
            self._patch_note.set_text(
                f"{patch.time_s[index] * 1e3:.2f} ms\n"
                f"nearest load {reach} behind the leading edge"
            )
        if self._cursor is not None:
            moment = patch.time_s[index] * 1e3
            self._cursor.set_xdata([moment, moment])


def draw_shot_frame(
    figure: Figure,
    field: SoleLoadField,
    patch: ContactPatch | None = None,
    *,
    frame: int = 0,
    scales: dict[LoadComponent, LoadScale] | None = None,
) -> ShotFrameArtists:
    """Draw one sample of one shot into an existing figure.

    The figure is cleared and rebuilt, so this is the right call for a still
    and the wrong one for an animation: hold the returned
    :class:`ShotFrameArtists` and call :meth:`~ShotFrameArtists.update`
    instead, which is what the workbench view does.

    Args:
        figure: The figure to draw into.
        field: The per-element load field.
        patch: The contact-patch series; when omitted only the two load
            panels are drawn.
        frame: Which sample to show.
        scales: Fixed colour scales, from :func:`field_scales`. Defaults to
            this field's own, which is correct for a single design and
            **wrong** for a comparison -- pass the merged scales there.

    Returns:
        The built artists, ready to be updated to another frame.

    Raises:
        ValueError: If the frame is outside the shot, or the patch does not
            describe the same shot as the field.
    """
    _check_frame(field, frame)
    _check_patch(field, patch)
    limits = field_scales((field,)) if scales is None else scales
    artists = ShotFrameArtists(figure, field, patch, limits)
    artists.update(frame)
    return artists


def sole_load_still(
    field: SoleLoadField,
    patch: ContactPatch | None = None,
    *,
    frame: int | None = None,
    scales: dict[LoadComponent, LoadScale] | None = None,
    figsize: tuple[float, float] = (11.0, 7.0),
) -> Figure:
    """Render one frame as a standalone figure -- the ADR-0027 fallback.

    Args:
        field: The per-element load field.
        patch: The contact-patch series, when there is one.
        frame: Which sample to show; defaults to the moment the sole's
            compressive resultant peaked, which is the single most
            informative still.
        scales: Fixed colour scales; see :func:`draw_shot_frame`.
        figsize: Figure size in inches.

    Returns:
        The figure.

    Raises:
        ValueError: If the frame is outside the shot, or the patch does not
            describe the same shot as the field.
    """
    chosen = (
        int(field.resultant_force_N(LoadComponent.TOTAL).argmax())
        if frame is None
        else frame
    )
    figure = Figure(figsize=figsize)
    draw_shot_frame(figure, field, patch, frame=chosen, scales=scales)
    return figure
