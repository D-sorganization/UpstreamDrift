"""Drawing the 3-D shot scene (issue #8706, epic #8699).

Headless. This module imports matplotlib and no GUI toolkit, so the same
frame can be produced in a test, written to a file by a batch sweep, or
embedded in the Qt workbench by
:mod:`src.tools.bunker_shot_gui.viewport_widgets`.

Why this is not a second renderer
---------------------------------

ADR-0027 put the choice of 3-D viewport behind
:mod:`src.shared.python.visualization.viewport`, which evaluates MeshCat,
Rerun and VTK/PyVista and returns an explicit degradation reason when none of
them is import-discoverable. None of the three is installed here, so the
selection degrades and this module draws the frame instead.

It is deliberately not an independent implementation. Every frame is built
from the **backend-neutral**
:class:`~src.shared.python.visualization.viewport.ViewportOverlayPayload`
that :func:`~.shot3d.viewport_payload` produces -- the same object a MeshCat
or Rerun provider would consume -- so the fallback cannot quietly start
showing something a real 3-D backend would not. When a provider does appear,
what changes is who draws the payload, not what the payload says.
:func:`~.render.viewport_fallback` reports which of the two happened, and the
workbench surfaces it.

Nothing here auto-scales
------------------------

Issue #8728 fixed a real defect in the sole load field: per-grid
auto-scaling meant two grinds each normalised to their own peak looked
identical however far apart they were. The three-dimensional form of that
defect is an auto-ranged world box -- two designs each framed to their own
divot look like the same divot. :class:`SceneScale` is therefore fixed over
the whole shot and merges across designs, exactly as
:class:`~.field.LoadScale` does, and it is injected rather than inferred.

What the frame is allowed to imply
----------------------------------

The sand plane is the model's free-surface height and the section under it
is the head's swept envelope. Neither is resolved sand, and both say so, in
the axes rather than in a caption -- :class:`~.shot3d.SandSurface` and
:class:`~.shot3d.DivotSection` compose those sentences and this module
draws them.

Two defects fixed here, both visible only in a rendered frame (issue #8706)
------------------------------------------------------------------------

The head used to be drawn as a scatter of the solver's own element
centroids -- a cloud of dots standing in for a solid wedge. It is now the
lofted, watertight mesh itself, posed every frame and drawn as a single
translucent :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`, built
once and mutated through the public ``set_verts`` the same way every other
artist here is mutated rather than rebuilt.

The footer caption used to run unwrapped off the right edge of the figure,
clipped mid-word. It is now word-wrapped to how wide it actually renders in
*this* figure's own font -- measured, not guessed at a character count --
and the world box itself now fills most of the canvas rather than a
fraction of it, within the one hard constraint mplot3d imposes on a 3-D
panel: :meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.apply_aspect` always
shrinks it to a square in physical inches, so a landscape figure keeps some
white margin either side of it regardless.
"""

from __future__ import annotations

from typing import cast

from dataclasses import dataclass

import numpy as np
from matplotlib.backend_bases import RendererBase
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.typing import NDArray

from src.shared.python.visualization.viewport import ViewportOverlayPayload

from .render import (
    ViewportFallback,
    stamp_axes,
    validity_stamp,
    viewport_fallback,
)
from .shot3d import CameraPreset, ShotScene, viewport_payload
from .traces import ValidityBand

__all__ = [
    "SceneScale",
    "ShotSceneArtists",
    "draw_scene_frame",
    "scene_scale",
    "shot_scene_still",
]

_MM_PER_M = 1e3

_SURFACE_COLOUR = "#d9c9a3"
_SURFACE_ALPHA = 0.35
_HEAD_COLOUR = "#5a5a5a"
_HEAD_MESH_EDGE_COLOUR = "#f0f0f0"
_HEAD_MESH_ALPHA = 0.62
"""Translucent, not opaque: the divot floor behind the head must stay
visible through it (issue #8706 defect 1), which an opaque solid would
hide from every camera that looks along the swing."""
_HEAD_MESH_LINEWIDTH = 0.15
_SOLE_COLOUR = "#1f4e79"
_SOLE_ALPHA = 0.55
_PATH_COLOUR = "#9b1c1c"
_FLOOR_COLOUR = "#6b4a1e"

_SOLE_MARKER_PT = 1.6
"""De-emphasised now that the head itself is the solid mesh: this cloud
is kept only to mark which faces the solver classed as sole, not to stand
in for the head's shape the way it used to (issue #8706 defect 1)."""

_PADDING = 0.08
"""Slack added around the world box, as a fraction of its own span."""

_TITLE_FONTSIZE = 8.0
_LABEL_FONTSIZE = 7.0
_TICK_FONTSIZE = 6.0
_NOTE_FONTSIZE = 6.0
_LABELPAD = 9.0
_LABELPAD_Y = 34.0
"""Extra padding the across-track label alone needs.

At :attr:`~.shot3d.CameraPreset.SOLE_LEVEL` the eye sights along this axis,
foreshortening it to nearly a point, so its ticks bunch up tightly and the
common labelpad is not enough clearance for the label to sit clear of them
-- the exact collision issue #8706 defect 2 named. The other two labels
are never this foreshortened at any of the three presets, so they keep the
common value.
"""
_TICK_PAD = 2.0
_MAX_TICKS = 3
"""Ticks per axis. Fewer numbers is fewer chances for two of them to
collide when a camera foreshortens an axis toward a point (issue #8706
defect 2, worst at :attr:`~.shot3d.CameraPreset.SOLE_LEVEL`)."""

_BOX_ZOOM = 1.35
"""How much of its allotted rectangle the 3-D box itself fills.

mplot3d's own default (``zoom=1``) leaves the box well short of the axes
rectangle it is drawn in, which is most of where the "60% empty canvas"
half of defect 2 came from -- the rectangle was already close to the
figure's own bounds; the box inside it was not.
"""

_FIGURE_MARGINS: dict[str, float] = {
    "left": 0.02,
    "right": 0.98,
    "top": 0.95,
    "bottom": 0.21,
}
"""Figure-fraction margins the whole frame is drawn within (issue #8706
defect 2). Tight on three sides so the scene fills the canvas rather than
sitting in a wide white border; generous on the bottom, which is not
white border but the caption's own reserved band -- see
:data:`_NOTE_TOP_MARGIN_FIG_FRACTION`.
"""

_NOTE_WIDTH_FRACTION = 0.97
"""Fraction of the axes width the wrapped footer caption may use."""

_NOTE_TOP_MARGIN_FIG_FRACTION = 0.02
"""How far the caption's *last* line sits above the physical bottom of the
figure, in figure fraction. The caption is anchored here and grows
*upward* into the band :data:`_FIGURE_MARGINS`'s ``bottom`` reserves for
it, rather than being anchored at the top of that band and risking a long
wrap climbing back up into the tick labels above it.
"""


def _check_range(name: str, bounds: tuple[float, float]) -> tuple[float, float]:
    """Validate one axis of a scale.

    Args:
        name: Which axis, for the message.
        bounds: ``(low, high)``.

    Returns:
        The bounds as floats.

    Raises:
        ValueError: If a bound is not finite or the pair does not increase.
            A ``raise``: a degenerate axis would collapse the frame under
            ``python -O`` rather than being rejected.
    """
    low, high = float(bounds[0]), float(bounds[1])
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(f"{name} must be finite, got {bounds!r}")
    if not low < high:
        raise ValueError(f"{name} must increase, got {low} to {high}")
    return (low, high)


@dataclass(frozen=True)
class SceneScale:
    """A fixed world box and depth ramp, shared across frames and designs.

    Attributes:
        x_m: ``(low, high)`` world ``x`` the frame spans [m].
        y_m: ``(low, high)`` world ``y`` [m].
        z_m: ``(low, high)`` world ``z`` [m].
        depth_m: ``(low, high)`` divot depth the colour ramp covers [m];
            always starts at zero so untouched surface reads as untouched.
    """

    x_m: tuple[float, float]
    y_m: tuple[float, float]
    z_m: tuple[float, float]
    depth_m: tuple[float, float]

    def __post_init__(self) -> None:
        """Validate the scale.

        Raises:
            ValueError: If any axis is not finite and increasing.
        """
        for name in ("x_m", "y_m", "z_m", "depth_m"):
            object.__setattr__(self, name, _check_range(name, getattr(self, name)))

    @property
    def colormap_name(self) -> str:
        """The ramp the divot depth is painted on."""
        return "YlOrBr"

    def merged(self, other: SceneScale) -> SceneScale:
        """Return the scale covering both this one and ``other``.

        Args:
            other: The scale to merge with.

        Returns:
            The covering scale, which is what makes two designs directly
            comparable rather than each framed to its own extent.
        """
        return SceneScale(
            x_m=(min(self.x_m[0], other.x_m[0]), max(self.x_m[1], other.x_m[1])),
            y_m=(min(self.y_m[0], other.y_m[0]), max(self.y_m[1], other.y_m[1])),
            z_m=(min(self.z_m[0], other.z_m[0]), max(self.z_m[1], other.z_m[1])),
            depth_m=(
                min(self.depth_m[0], other.depth_m[0]),
                max(self.depth_m[1], other.depth_m[1]),
            ),
        )


def _padded(low: float, high: float) -> tuple[float, float]:
    """Return a range with slack, never degenerate."""
    span = high - low
    if span <= 0.0:
        span = max(abs(high), 1e-3)
    pad = span * _PADDING
    return (low - pad, high + pad)


def _scale_for(scene: ShotScene) -> SceneScale:
    """Build the fixed box one scene needs, over its whole record."""
    corners = np.concatenate(
        [scene.head_world_m(frame) for frame in range(scene.n_frames)], axis=0
    )
    surface = scene.surface
    along = surface.along_extent_m
    across = surface.across_extent_m
    return SceneScale(
        x_m=_padded(
            min(along[0], float(corners[:, 0].min())),
            max(along[1], float(corners[:, 0].max())),
        ),
        y_m=_padded(
            min(across[0], float(corners[:, 1].min())),
            max(across[1], float(corners[:, 1].max())),
        ),
        z_m=_padded(
            float(corners[:, 2].min()),
            max(float(corners[:, 2].max()), surface.height_m),
        ),
        depth_m=(0.0, max(scene.divot.max_depth_m, 1e-4)),
    )


def scene_scale(scenes: tuple[ShotScene, ...]) -> SceneScale:
    """Build the one world box two or more designs are drawn in.

    Args:
        scenes: Every scene that will be drawn on this scale. Passing both
            halves of an A/B comparison is what makes the two views readable
            against each other; passing one gives a box fixed across its own
            frames.

    Returns:
        The covering scale.

    Raises:
        ValueError: If no scene was supplied; there is nothing to frame, and
            an empty comparison silently framed to nothing is the failure
            this refuses.
    """
    scales = [_scale_for(scene) for scene in scenes]
    if not scales:
        raise ValueError(
            "a shared scene scale needs at least one scene to cover; drawing "
            "two designs each framed to its own extent is what this prevents"
        )
    merged = scales[0]
    for scale in scales[1:]:
        merged = merged.merged(scale)
    return merged


def _figure_renderer(figure: Figure) -> RendererBase:
    """Return a renderer that can measure text extents on ``figure``.

    A figure a caller has already embedded -- the Qt workbench, through its
    own ``FigureCanvasQTAgg`` -- carries a canvas that already supports
    ``get_renderer``, and reusing it measures against the exact font this
    figure will actually draw with. A bare :class:`~matplotlib.figure.Figure`
    (a still, or a test) starts with matplotlib's base canvas, which cannot
    measure anything; attaching an Agg canvas to it is exactly what
    :meth:`~matplotlib.figure.Figure.savefig` would do to it regardless, so
    doing it here first costs nothing that was not already going to happen.

    Args:
        figure: The figure the caption is being wrapped for.

    Returns:
        A renderer :meth:`~matplotlib.text.Text.get_window_extent` accepts.
    """
    get_renderer = getattr(figure.canvas, "get_renderer", None)
    if get_renderer is not None:
        return get_renderer()
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    return FigureCanvasAgg(figure).get_renderer()


def _wrap_to_pixels(
    probe: Text, renderer: RendererBase, line: str, max_width_px: float
) -> list[str]:
    """Word-wrap one line to a pixel budget, measured in ``probe``'s font.

    The #8706 defect this fixes clipped a caption mid-word ("...not where
    sand h"), so wrapping only ever breaks at a space -- a single word wider
    than the budget is left whole on its own row rather than sliced.

    Args:
        probe: A ``Text`` artist already attached to the figure being
            measured, so its font, size and family are what will actually be
            drawn. Its content is overwritten by every candidate row this
            probes; the caller is responsible for setting the text it
            actually wants shown once wrapping is done.
        renderer: A renderer ``probe`` can measure against, from
            :func:`_figure_renderer`.
        max_width_px: The budget one rendered row may not exceed.

    Returns:
        One or more rows, each within the budget except a single
        unsplittable word.
    """
    words = line.split()
    if not words:
        return [""]
    rows: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        probe.set_text(candidate)
        width = probe.get_window_extent(renderer=renderer).width
        if width <= max_width_px:
            current = candidate
        else:
            rows.append(current)
            current = word
    rows.append(current)
    return rows


class ShotSceneArtists:
    """Axes built once for one scene, and the artists a frame change touches.

    The same pattern :class:`~.render.ShotFrameArtists` established: building
    a 3-D axes, its surface patch and its labels costs far more than the
    transport interval, so everything that does not depend on the sample is
    built once and only the frame-varying artists are mutated -- the head
    mesh, the sole point cloud, one trail, one divot profile, one stamp and
    one title.

    The head is a :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`
    built once from ``scene.head_mesh_body.faces`` and re-posed every frame
    through the public ``set_verts`` (issue #8706 defect 1) -- never
    rebuilt, so a whole shot's worth of frames costs one collection, not
    one per sample. Everything else here is a
    :class:`~matplotlib.lines.Line2D` in 3-D, updated through the public
    ``set_data_3d``. Nothing here reaches into a collection's private
    offsets.

    The axis limits are set once from the injected :class:`SceneScale`, and
    ``autoscale`` is switched off, so no update path can reintroduce the
    per-frame reframing issue #8728 removed from the sole field.
    """

    def __init__(
        self,
        figure: Figure,
        scene: ShotScene,
        scale: SceneScale,
        *,
        camera: CameraPreset = CameraPreset.DOWN_THE_LINE,
        band: ValidityBand | None = None,
    ) -> None:
        """Build the axes for one scene.

        Args:
            figure: The figure to build into; cleared first.
            scene: The scene to draw.
            scale: The fixed world box and depth ramp.
            camera: Which named view to open on.
            band: The per-sample validity band, when there is one. With it
                the stamp shows the verdict *at the drawn moment*; without
                it, the one verdict the whole scene carries.

        Raises:
            ValueError: If the band does not describe this scene.
        """
        if band is not None and band.n_frames != scene.n_frames:
            raise ValueError(
                "the scene and the validity band must come from one shot; got "
                f"{scene.n_frames} poses against {band.n_frames} verdicts"
            )
        self._scene = scene
        self._scale = scale
        self._band = band
        self._camera = CameraPreset(camera)
        self._payload = viewport_payload(scene)
        self._fallback = viewport_fallback()

        figure.clear()
        # Tight on three sides, generous on the fourth: the scene fills the
        # canvas instead of sitting in a wide white border, and the bottom
        # band left over is the wrapped caption's own reserved space, not
        # waste (issue #8706 defect 2).
        figure.subplots_adjust(**_FIGURE_MARGINS)
        axes = figure.add_subplot(111, projection="3d")
        self._axes: Axes3D = axes
        self._build_surface()
        self._floor = self._build_floor()
        self._trail = self._new_line(_PATH_COLOUR, 1.2, "sole reference path")
        self._head_mesh = self._build_head_mesh()
        self._sole = self._new_points(
            _SOLE_COLOUR, _SOLE_MARKER_PT, "sole elements", alpha=_SOLE_ALPHA
        )
        self._label_axes()
        self._stamp: Text = stamp_axes(
            axes,
            scene.status,
            scene.fidelity_tier,
            extra=f"renderer: {self._fallback.renderer}",
        )
        self._note = axes.text2D(
            0.02,
            0.02,
            "",
            transform=axes.transAxes,
            ha="left",
            va="bottom",
            fontsize=_NOTE_FONTSIZE,
            color="#333333",
            zorder=9,
        )
        self.set_camera(self._camera)

    # ----------------------------------------------------------- accessors

    @property
    def camera(self) -> CameraPreset:
        """Which named view is showing."""
        return self._camera

    @property
    def fallback(self) -> ViewportFallback:
        """What the ADR-0027 layer left this frame drawing with."""
        return self._fallback

    @property
    def payload(self) -> ViewportOverlayPayload:
        """The backend-neutral payload this frame is drawn from."""
        return self._payload

    @property
    def scale(self) -> SceneScale:
        """The fixed world box in force."""
        return self._scale

    # ------------------------------------------------------------- building

    def _new_line(self, colour: str, width: float, label: str) -> Line3D:
        """Add an empty 3-D polyline."""
        (line,) = self._axes.plot(
            [], [], [], color=colour, linewidth=width, label=label
        )
        # plot() on a 3-D axes returns a Line3D at runtime; the stubs type it
        # as Line2D, which lacks the public set_data_3d update path.
        return cast(Line3D, line)

    def _new_points(
        self, colour: str, size: float, label: str, *, alpha: float = 1.0
    ) -> Line3D:
        """Add an empty 3-D point cloud, drawn as a marker-only line.

        A ``Line2D`` rather than a scatter because ``set_data_3d`` is public
        and a 3-D scatter's offsets are not: an animation that mutates a
        private attribute breaks on a matplotlib upgrade, silently, in a
        picture that still looks plausible.
        """
        (line,) = self._axes.plot(
            [],
            [],
            [],
            color=colour,
            linestyle="none",
            marker=".",
            markersize=size,
            label=label,
            alpha=alpha,
        )
        # plot() on a 3-D axes returns a Line3D at runtime; the stubs type it
        # as Line2D, which lacks the public set_data_3d update path.
        return cast(Line3D, line)

    def _build_head_mesh(self) -> Poly3DCollection:
        """Draw the lofted head as a solid, once (issue #8706 defect 1).

        Built from ``scene.head_mesh_body`` -- the same watertight
        :class:`~bunkershot3d.geometry.TriangleMesh` the F0 solver
        discretised -- rather than from the element centroids a scatter used
        to stand in for. Its topology (``faces``) never changes, so the
        placeholder vertices here are replaced by the first
        :meth:`update` call and every one after only moves them, through the
        public ``set_verts``: one collection for a whole shot, not one per
        frame.

        Returns:
            The collection, added to the axes and ready for
            :meth:`~matplotlib.mplot3d.art3d.Poly3DCollection.set_verts`.
        """
        mesh = self._scene.head_mesh_body
        placeholder = np.zeros((mesh.faces.shape[0], 3, 3), dtype=np.float64)
        collection = Poly3DCollection(
            placeholder,
            facecolor=_HEAD_COLOUR,
            edgecolor=_HEAD_MESH_EDGE_COLOUR,
            linewidths=_HEAD_MESH_LINEWIDTH,
            alpha=_HEAD_MESH_ALPHA,
            zsort="max",
            label="clubhead (lofted mesh)",
        )
        self._axes.add_collection3d(collection)
        return collection

    def _build_surface(self) -> None:
        """Draw the free surface, once, as a flat translucent plane."""
        surface = self._scene.surface
        low_x, high_x = surface.along_extent_m
        low_y, high_y = surface.across_extent_m
        grid_x, grid_y = np.meshgrid(
            np.array([low_x, high_x]) * _MM_PER_M,
            np.array([low_y, high_y]) * _MM_PER_M,
        )
        self._axes.plot_surface(
            grid_x,
            grid_y,
            np.full_like(grid_x, surface.height_m * _MM_PER_M),
            color=_SURFACE_COLOUR,
            alpha=_SURFACE_ALPHA,
            shade=False,
            linewidth=0.0,
        )

    def _build_floor(self) -> Line3D:
        """Add the divot profile, drawn along the track at the sole's own y."""
        return self._new_line(
            _FLOOR_COLOUR, 1.6, "divot floor (swept envelope of the head)"
        )

    def _label_axes(self) -> None:
        """Label and bound the world box, in millimetres, once.

        ``labelpad`` and a coarser tick locator are the fix for the second
        half of issue #8706 defect 2: at
        :attr:`~.shot3d.CameraPreset.SOLE_LEVEL`, the eye sights straight
        down the across-track axis, foreshortening it toward a point, and
        the default label placement and tick density collide there into
        unreadable mush. Neither the camera's stated angles nor the world
        box itself change -- both are pinned elsewhere -- only how much room
        each axis's own label and ticks are given.
        """
        axes = self._axes
        # mplot3d's own default (``zoom=1``) leaves the box well short of
        # its allotted rectangle; the rest of that rectangle is where most
        # of defect 2's "60% empty canvas" was coming from.
        axes.set_box_aspect(None, zoom=_BOX_ZOOM)
        axes.set_xlabel(
            "world x, along the target line [mm]",
            fontsize=_LABEL_FONTSIZE,
            labelpad=_LABELPAD,
        )
        self._set_y_label()
        axes.set_zlabel(
            "world z, up [mm]", fontsize=_LABEL_FONTSIZE, labelpad=_LABELPAD
        )
        axes.tick_params(labelsize=_TICK_FONTSIZE, pad=_TICK_PAD)
        for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
            axis.set_major_locator(MaxNLocator(nbins=_MAX_TICKS, prune="both"))
        axes.set_xlim(*(value * _MM_PER_M for value in self._scale.x_m))
        axes.set_ylim(*(value * _MM_PER_M for value in self._scale.y_m))
        axes.set_zlim(*(value * _MM_PER_M for value in self._scale.z_m))
        # The limits come from the injected scale and stay there. Without
        # this, adding a frame's points would re-range the box and two
        # designs would each be framed to their own divot.
        axes.set_autoscale_on(False)
        axes.legend(loc="upper right", fontsize=5, framealpha=0.6)

    def _set_y_label(self) -> None:
        """(Re)draw the across-track label with a pad tuned to this camera.

        Only :attr:`~.shot3d.CameraPreset.SOLE_LEVEL` sights down this axis
        far enough to foreshorten it into its own tick labels, so only that
        preset needs :data:`_LABELPAD_Y`'s wider clearance. Giving every
        camera that pad moved the label into the caption's own space at the
        other two -- a second collision this fixes rather than trades for
        (issue #8706 defect 2).
        """
        axes = self._axes
        y_axis = axes.yaxis
        sighting_down_y = self._camera is CameraPreset.SOLE_LEVEL
        labelpad = _LABELPAD_Y if sighting_down_y else _LABELPAD
        axes.set_ylabel(
            "world y, across the target line [mm]",
            fontsize=_LABEL_FONTSIZE,
            labelpad=labelpad,
        )
        # A foreshortened axis crowds every one of its ticks toward the
        # same near-corner point regardless of how few bins are asked
        # for -- MaxNLocator keeps returning the same "nice" -80/0/80 no
        # matter the ``nbins`` hint, since none of them sit exactly on
        # the (padded) limits for ``prune`` to drop. Sighting down the
        # axis this hard, one reference tick at the origin says as much
        # as three crowded ones did.
        if sighting_down_y:
            y_axis.set_major_locator(FixedLocator([0.0]))
        else:
            y_axis.set_major_locator(MaxNLocator(nbins=_MAX_TICKS, prune="both"))

    # ------------------------------------------------------------ the frame

    def set_camera(self, camera: CameraPreset | str) -> None:
        """Point the view at one of the named presets.

        Args:
            camera: The preset.

        Raises:
            ValueError: If the name is not one of the three.
        """
        chosen = CameraPreset(camera)
        self._camera = chosen
        self._axes.view_init(elev=chosen.elevation_deg, azim=chosen.azimuth_deg)
        self._set_y_label()
        self._recompute_note()
        self._refresh_note()

    def _recompute_note(self) -> None:
        """Wrap the qualifier under the stamp to this figure's own width.

        Issue #8706 defect 2: the caption used to run off unwrapped and was
        clipped mid-word at the figure's right edge ("...not where sand h").
        Each source line is word-wrapped to how wide it actually renders in
        this figure's font -- not a guessed character count, which would be
        wrong the moment the font, DPI or figure size were not what was
        guessed for -- and the result is cached on ``self`` so
        :meth:`_refresh_note` can re-apply it every frame without
        re-measuring: none of the three source lines depends on which
        sample is showing, only on the camera, so this only needs to run
        when the camera changes.
        """
        scene = self._scene
        lines = (
            f"{self._camera.label} - {self._camera.description}",
            scene.surface.describe(),
            scene.divot.describe(),
        )
        axes = self._axes
        figure = axes.figure
        renderer = _figure_renderer(figure)
        # Measured, not assumed from the ``subplots_adjust`` fractions:
        # ``Axes3D.apply_aspect`` always shrinks the panel to a square in
        # physical inches (mplot3d's own rule, not something this module
        # chooses), so on a wide figure the axes are already narrower than
        # the rectangle they were given. Budgeting against the rectangle
        # instead of the real panel is exactly how the first pass at this
        # fix still let a line run off the edge unwrapped -- and
        # ``apply_aspect`` has to be forced here because matplotlib only
        # calls it as a side effect of a full draw, which has not happened
        # yet when a still is built.
        axes.apply_aspect()
        axes_bbox = axes.get_window_extent(renderer=renderer)
        budget_px = axes_bbox.width * _NOTE_WIDTH_FRACTION
        wrapped: list[str] = []
        for line in lines:
            wrapped.extend(_wrap_to_pixels(self._note, renderer, line, budget_px))
        self._note_text = "\n".join(wrapped)
        # Anchored a fixed distance above the *figure's* own bottom edge,
        # not the axes': because the panel can sit well inside the
        # rectangle subplots_adjust was given, an axes-fraction anchor
        # would put the caption back under the tick labels the reserved
        # bottom margin exists to clear.
        target_px = _NOTE_TOP_MARGIN_FIG_FRACTION * figure.bbox.height
        anchor_y = (target_px - axes_bbox.y0) / axes_bbox.height
        self._note_position = (0.02, anchor_y)

    def _refresh_note(self) -> None:
        """Re-apply the caption :meth:`_recompute_note` last wrapped."""
        self._note.set_text(self._note_text)
        self._note.set_position(self._note_position)

    def _check_frame(self, frame: int) -> int:
        """Validate a frame index against the scene.

        Args:
            frame: The requested sample index.

        Returns:
            The index.

        Raises:
            ValueError: If it is outside the recorded shot.
        """
        if not 0 <= int(frame) < self._scene.n_frames:
            raise ValueError(
                f"frame {frame} is outside the recorded shot, which has "
                f"{self._scene.n_frames} samples"
            )
        return int(frame)

    def _floor_track(self, index: int) -> tuple[NDArray[np.float64], ...]:
        """Return the divot profile at one sample, in millimetres."""
        divot = self._scene.divot
        stations = divot.station_m * _MM_PER_M
        floor = divot.floor_m[index] * _MM_PER_M
        # Drawn at the sole's own mean y, which is where the section is:
        # putting it at y = 0 would float the profile off the divot in every
        # view except face-on.
        across = np.full_like(stations, float(np.mean(self._scale.y_m)) * _MM_PER_M)
        return stations, across, floor

    def update(self, frame: int) -> None:
        """Show one sample.

        Args:
            frame: The sample index.

        Raises:
            ValueError: If the index is outside the recorded shot.
        """
        index = self._check_frame(frame)
        scene = self._scene
        mesh_vertices = scene.head_mesh_world_m(index) * _MM_PER_M
        self._head_mesh.set_verts(mesh_vertices[scene.head_mesh_body.faces])
        sole = scene.head_world_m(index)[scene.sole_index] * _MM_PER_M
        self._sole.set_data_3d(sole[:, 0], sole[:, 1], sole[:, 2])
        trail = scene.sole_reference_world_m[: index + 1] * _MM_PER_M
        self._trail.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])
        self._floor.set_data_3d(*self._floor_track(index))

        moment_ms = float(scene.time_s[index]) * 1e3
        depth_mm = float(scene.sole_depth_m[index]) * 1e3
        self._axes.set_title(
            f"{moment_ms:.2f} ms - sole {depth_mm:+.2f} mm below the free surface "
            f"- divot section {float(scene.divot.section_area_m2[index]) * 1e4:.2f} cm^2",
            fontsize=_TITLE_FONTSIZE,
            pad=4.0,
        )
        # The stamp follows the band when there is one: a shot that starts
        # inside the stated envelope and leaves it must not carry the worst
        # verdict on the frames it does not apply to.
        status = scene.status if self._band is None else self._band.status_at(index)
        self._stamp.set_text(
            f"{validity_stamp(status, scene.fidelity_tier)}\n"
            f"renderer: {self._fallback.renderer}"
        )
        self._refresh_note()


def draw_scene_frame(
    figure: Figure,
    scene: ShotScene,
    *,
    frame: int = 0,
    scale: SceneScale | None = None,
    camera: CameraPreset = CameraPreset.DOWN_THE_LINE,
    band: ValidityBand | None = None,
) -> ShotSceneArtists:
    """Draw one sample of one scene into an existing figure.

    The figure is cleared and rebuilt, so this is the right call for a still
    and the wrong one for an animation: hold the returned
    :class:`ShotSceneArtists` and call :meth:`~ShotSceneArtists.update`
    instead, which is what the workbench view does.

    Args:
        figure: The figure to draw into.
        scene: The scene.
        frame: Which sample to show.
        scale: The fixed world box, from :func:`scene_scale`. Defaults to
            this scene's own, which is correct for a single design and
            **wrong** for a comparison -- pass the merged scale there.
        camera: Which named view to open on.
        band: The per-sample validity band, when there is one.

    Returns:
        The built artists, ready to be updated to another frame.

    Raises:
        ValueError: If the frame is outside the shot, or the band does not
            describe the same shot as the scene.
    """
    limits = scene_scale((scene,)) if scale is None else scale
    artists = ShotSceneArtists(figure, scene, limits, camera=camera, band=band)
    artists.update(frame)
    return artists


def shot_scene_still(
    scene: ShotScene,
    *,
    frame: int | None = None,
    scale: SceneScale | None = None,
    camera: CameraPreset = CameraPreset.DOWN_THE_LINE,
    band: ValidityBand | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> Figure:
    """Render one frame as a standalone figure -- the ADR-0027 fallback.

    Args:
        scene: The scene.
        frame: Which sample to show; defaults to the deepest moment, which
            is the single most informative still.
        scale: The fixed world box; see :func:`draw_scene_frame`.
        camera: Which named view.
        band: The per-sample validity band, when there is one.
        figsize: Figure size in inches.

    Returns:
        The figure.

    Raises:
        ValueError: If the frame is outside the shot.
    """
    chosen = int(np.argmax(scene.sole_depth_m)) if frame is None else frame
    figure = Figure(figsize=figsize)
    draw_scene_frame(figure, scene, frame=chosen, scale=scale, camera=camera, band=band)
    return figure
