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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
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
_SOLE_COLOUR = "#1f4e79"
_PATH_COLOUR = "#9b1c1c"
_FLOOR_COLOUR = "#6b4a1e"

_HEAD_MARKER_PT = 1.2
_SOLE_MARKER_PT = 2.4

_PADDING = 0.08
"""Slack added around the world box, as a fraction of its own span."""


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


class ShotSceneArtists:
    """Axes built once for one scene, and the artists a frame change touches.

    The same pattern :class:`~.render.ShotFrameArtists` established: building
    a 3-D axes, its surface patch and its labels costs far more than the
    transport interval, so everything that does not depend on the sample is
    built once and only the frame-varying artists are mutated -- two point
    clouds, one trail, one divot profile, one stamp and one title.

    Every artist is a :class:`~matplotlib.lines.Line2D` in 3-D and is updated
    through the public ``set_data_3d``. Nothing here reaches into a
    collection's private offsets.

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
        axes = figure.add_subplot(111, projection="3d")
        self._axes: Axes3D = axes
        self._build_surface()
        self._floor = self._build_floor()
        self._trail = self._new_line(_PATH_COLOUR, 1.2, "sole reference path")
        self._head = self._new_points(_HEAD_COLOUR, _HEAD_MARKER_PT, "head surface")
        self._sole = self._new_points(_SOLE_COLOUR, _SOLE_MARKER_PT, "sole elements")
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
            fontsize=6,
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

    def _new_line(self, colour: str, width: float, label: str) -> Line2D:
        """Add an empty 3-D polyline."""
        (line,) = self._axes.plot(
            [], [], [], color=colour, linewidth=width, label=label
        )
        return line

    def _new_points(self, colour: str, size: float, label: str) -> Line2D:
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
        )
        return line

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

    def _build_floor(self) -> Line2D:
        """Add the divot profile, drawn along the track at the sole's own y."""
        return self._new_line(
            _FLOOR_COLOUR, 1.6, "divot floor (swept envelope of the head)"
        )

    def _label_axes(self) -> None:
        """Label and bound the world box, in millimetres, once."""
        axes = self._axes
        axes.set_xlabel("world x, along the target line [mm]", fontsize=7)
        axes.set_ylabel("world y, across the target line [mm]", fontsize=7)
        axes.set_zlabel("world z, up [mm]", fontsize=7)
        axes.tick_params(labelsize=6)
        axes.set_xlim(*(value * _MM_PER_M for value in self._scale.x_m))
        axes.set_ylim(*(value * _MM_PER_M for value in self._scale.y_m))
        axes.set_zlim(*(value * _MM_PER_M for value in self._scale.z_m))
        # The limits come from the injected scale and stay there. Without
        # this, adding a frame's points would re-range the box and two
        # designs would each be framed to their own divot.
        axes.set_autoscale_on(False)
        axes.legend(loc="upper right", fontsize=5, framealpha=0.6)

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
        self._refresh_note()

    def _refresh_note(self) -> None:
        """Rewrite the qualifier under the stamp."""
        scene = self._scene
        self._note.set_text(
            f"{self._camera.label} - {self._camera.description}\n"
            f"{scene.surface.describe()}\n"
            f"{scene.divot.describe()}"
        )

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
        head = scene.head_world_m(index) * _MM_PER_M
        self._head.set_data_3d(head[:, 0], head[:, 1], head[:, 2])
        sole = head[scene.sole_index]
        self._sole.set_data_3d(sole[:, 0], sole[:, 1], sole[:, 2])
        trail = scene.sole_reference_world_m[: index + 1] * _MM_PER_M
        self._trail.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])
        self._floor.set_data_3d(*self._floor_track(index))

        moment_ms = float(scene.time_s[index]) * 1e3
        depth_mm = float(scene.sole_depth_m[index]) * 1e3
        self._axes.set_title(
            f"{moment_ms:.2f} ms - sole {depth_mm:+.2f} mm below the free surface "
            f"- divot section {float(scene.divot.section_area_m2[index]) * 1e4:.2f} cm^2",
            fontsize=8,
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
