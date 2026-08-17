"""The linked follower views: 3-D scene, traces, cross-tier check, sand cut.

Issues #8706 and #8708 for the first two, #8713 for the third, and #8711
for the fourth.

No widget here owns a transport. :class:`~.widgets.SoleLoadFieldWidget`
already has one -- a slider, a timer and a play button, built for the sole
load field -- and #8708 asks for the views to be scrubbed *in lockstep*,
which is precisely the thing a second slider would break. Every class below
is therefore a :class:`~.widgets.FollowsFrame` implementation: it is handed
a sample index and repaints. The workbench wires them up with
:meth:`~.widgets.SoleLoadFieldWidget.link`.

None does physics and none draws anything of its own. The arithmetic is
:mod:`~.shot3d`'s, :mod:`~.traces`', :mod:`~.crosstier`'s and
:mod:`~.slices`'; the drawing is :mod:`~.render3d`'s,
:mod:`~.render_traces`', :mod:`~.render_crosstier`'s and
:mod:`~.render_slice`'. These classes own a canvas, a selector, and the
knowledge of when to repaint.

The sand cut is the one view whose record is not the shot's
-----------------------------------------------------------

An F0 shot is sampled on its own clock; an F1 sand field is a strided march
of a *declared approach* with its own frame count and its own time base. So
:class:`SandSliceWidget` maps the shared cursor onto its own frames with a
:class:`~.slices.CursorMap` and says in the frame that it has done so. That
is the honest way to keep one transport; a second slider would have been the
dishonest way to avoid the question.
"""

from __future__ import annotations

import numpy as np
from bunkershot3d.fields.schema import SandFieldSeries
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from .crosstier import CrossTierComparison
from .render import viewport_fallback
from .render3d import SceneScale, ShotSceneArtists, scene_scale
from .render_crosstier import CrossTierArtists
from .render_slice import SliceArtists
from .render_traces import TracePanelArtists
from .shot3d import CameraPreset, ShotScene
from .slices import (
    CursorMap,
    CuttingPlane,
    SliceScale,
    preset_planes,
    slice_scale,
)
from .traces import ShotTraces, ValidityBand
from .widgets import build_canvas_column

__all__ = [
    "CrossTierWidget",
    "SandSliceWidget",
    "ShotViewportWidget",
    "TracePanelWidget",
]

_MIN_SCENE_HEIGHT_PX = 340
_MIN_TRACE_HEIGHT_PX = 420
_MIN_CROSS_TIER_HEIGHT_PX = 460
_MIN_SLICE_HEIGHT_PX = 460
_IDLE_CROSS_TIER = (
    "The cross-tier check has not been run. It puts F1 -- the 2-D "
    "plane-strain MPM continuum -- beside F0 on the quantities both tiers "
    "produce, and it is minutes of solving rather than milliseconds: F1 has "
    "no shot history yet (#8733), so every probe is its own march to one "
    "recorded pose."
)


class ShotViewportWidget(QWidget):
    """The head moving through the sand, scrubbed by somebody else's cursor.

    The camera is the one control this view owns, because a view direction is
    not a moment in time: switching from down-the-line to sole level must not
    move the cursor, and moving the cursor must not reset the camera.

    The world box is injected, not inferred, for the same reason the sole
    field's colour scale is (issue #8728): two designs each framed to their
    own divot look like the same divot.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Build an empty view.

        Args:
            title: Heading shown above the canvas.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._title = str(title)
        self._scene: ShotScene | None = None
        self._artists: ShotSceneArtists | None = None
        self._scale: SceneScale | None = None
        self._band: ValidityBand | None = None
        self._frame = 0
        self._fallback = viewport_fallback()

        layout, self._heading, self._canvas = build_canvas_column(
            self,
            self._title,
            width_in=7.5,
            height_in=5.0,
            minimum_height_px=_MIN_SCENE_HEIGHT_PX,
        )

        controls = QHBoxLayout()
        controls.addWidget(QLabel("View:"))
        self._camera_box = QComboBox()
        for preset in CameraPreset:
            self._camera_box.addItem(preset.label, preset.value)
            self._camera_box.setItemData(
                self._camera_box.count() - 1,
                preset.description,
                Qt.ItemDataRole.ToolTipRole,
            )
        self._camera_box.currentIndexChanged.connect(self._on_camera)
        controls.addWidget(self._camera_box, stretch=1)
        layout.addLayout(controls)

        self._note = QLabel(
            f"Renderer: {self._fallback.renderer}"
            + (" (no 3-D viewport installed)" if self._fallback.degraded else "")
        )
        self._note.setToolTip(self._fallback.describe())
        self._note.setWordWrap(True)
        layout.addWidget(self._note)

    # ------------------------------------------------------------ accessors

    @property
    def title(self) -> str:
        """The heading shown above the canvas."""
        return self._title

    @property
    def has_shot(self) -> bool:
        """Whether a scene is loaded."""
        return self._scene is not None

    @property
    def n_frames(self) -> int:
        """Number of samples in the loaded scene; zero when empty."""
        return 0 if self._scene is None else self._scene.n_frames

    @property
    def frame_index(self) -> int:
        """The sample currently displayed."""
        return self._frame

    @property
    def camera(self) -> CameraPreset:
        """Which named view is showing."""
        return CameraPreset(self._camera_box.currentData())

    @property
    def scale(self) -> SceneScale | None:
        """The fixed world box in force, or ``None`` when empty."""
        return self._scale

    @property
    def renderer_note(self) -> str:
        """What the ADR-0027 viewport layer left this view drawing with."""
        return self._fallback.describe()

    # --------------------------------------------------------------- content

    def set_shot(
        self,
        scene: ShotScene,
        *,
        scale: SceneScale | None = None,
        band: ValidityBand | None = None,
    ) -> None:
        """Load one scene and open on its deepest moment.

        Args:
            scene: The 3-D scene.
            scale: The fixed world box shared with any other view this one is
                compared against. Defaults to this scene's own.
            band: The per-sample validity band, so the in-frame stamp can
                follow the regime rather than carry the worst verdict on
                every frame.

        Raises:
            ValueError: If the band does not describe the same shot.
        """
        self._scene = scene
        self._scale = scene_scale((scene,)) if scale is None else scale
        self._band = band
        self._artists = ShotSceneArtists(
            self._canvas.fig, scene, self._scale, camera=self.camera, band=band
        )
        self._frame = int(scene.sole_depth_m.argmax())
        self._redraw()

    def clear(self) -> None:
        """Drop the scene, so nothing stale stays on screen after a refusal."""
        self._scene = None
        self._artists = None
        self._scale = None
        self._band = None
        self._frame = 0
        self._canvas.fig.clear()
        self._canvas.draw_idle()

    # ------------------------------------------------------------- following

    def set_frame(self, frame: int) -> None:
        """Show one sample.

        The :class:`~.widgets.FollowsFrame` entry point. A frame arriving for
        an empty view is ignored rather than refused: the workbench clears
        views independently, so a transport tick can legitimately reach a
        view that has just been emptied.

        Args:
            frame: The sample index.

        Raises:
            ValueError: If a scene is loaded and the index is outside it. A
                clamped index would leave this view showing a different
                moment from the one driving it, which is the whole failure
                linking the views exists to prevent.
        """
        if self._scene is None:
            return
        if not 0 <= int(frame) < self._scene.n_frames:
            raise ValueError(
                f"frame {frame} is outside the shot, which has "
                f"{self._scene.n_frames} samples"
            )
        self._frame = int(frame)
        self._redraw()

    def _on_camera(self, _index: int) -> None:
        """Point the scene at the newly selected view."""
        if self._artists is None:
            return
        self._artists.set_camera(self.camera)
        self._canvas.draw_idle()

    def _redraw(self) -> None:
        """Repaint the canvas at the current frame."""
        if self._artists is None:
            return
        self._artists.update(self._frame)
        self._canvas.draw_idle()


class TracePanelWidget(QWidget):
    """The scalar traces and the validity band, on the shared cursor.

    No transport, for the reason in the module docstring: the traces exist to
    be read *against* the 3-D view and the sole field, and three sliders would
    let the three drift apart.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Build an empty panel.

        Args:
            title: Heading shown above the canvas.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._title = str(title)
        self._traces: ShotTraces | None = None
        self._artists: TracePanelArtists | None = None
        self._frame = 0

        layout, self._heading, self._canvas = build_canvas_column(
            self,
            self._title,
            width_in=7.5,
            height_in=8.0,
            minimum_height_px=_MIN_TRACE_HEIGHT_PX,
        )
        self._readout = QLabel("no shot")
        self._readout.setWordWrap(True)
        layout.addWidget(self._readout)

    # ------------------------------------------------------------ accessors

    @property
    def title(self) -> str:
        """The heading shown above the canvas."""
        return self._title

    @property
    def has_shot(self) -> bool:
        """Whether a trace set is loaded."""
        return self._traces is not None

    @property
    def n_frames(self) -> int:
        """Number of samples in the loaded set; zero when empty."""
        return 0 if self._traces is None else self._traces.n_frames

    @property
    def frame_index(self) -> int:
        """The sample currently displayed."""
        return self._frame

    @property
    def n_panels(self) -> int:
        """How many stacked panels are showing; zero when empty."""
        return 0 if self._artists is None else self._artists.n_panels

    # --------------------------------------------------------------- content

    def set_shot(self, traces: ShotTraces) -> None:
        """Load one shot's traces.

        Args:
            traces: The trace set.

        Raises:
            ValueError: If the set carries no traces to draw.
        """
        self._traces = traces
        self._artists = TracePanelArtists(self._canvas.fig, traces)
        self._frame = 0
        self._redraw()

    def clear(self) -> None:
        """Drop the traces, so nothing stale stays under a refusal banner."""
        self._traces = None
        self._artists = None
        self._frame = 0
        self._readout.setText("no shot")
        self._canvas.fig.clear()
        self._canvas.draw_idle()

    # ------------------------------------------------------------- following

    def set_frame(self, frame: int) -> None:
        """Move the cursor to one sample.

        Args:
            frame: The sample index.

        Raises:
            ValueError: If a set is loaded and the index is outside it.
        """
        if self._traces is None:
            return
        if not 0 <= int(frame) < self._traces.n_frames:
            raise ValueError(
                f"frame {frame} is outside the shot, which has "
                f"{self._traces.n_frames} samples"
            )
        self._frame = int(frame)
        self._redraw()

    def _redraw(self) -> None:
        """Repaint the canvas and restate the verdict at this moment."""
        traces = self._traces
        if traces is None or self._artists is None:
            return
        self._artists.update(self._frame)
        self._canvas.draw_idle()
        status = traces.band.status_at(self._frame)
        self._readout.setText(
            f"{traces.time_display[self._frame]:.2f} {traces.time_unit} - "
            f"{status.value.replace('_', ' ').upper()}"
        )


class CrossTierWidget(QWidget):
    """F0 against F1 on one cursor, run on demand (issue #8713).

    A follower, like the two views above, and for a stronger reason than
    tidiness: the whole point of the comparison is that both tiers are
    describing the *same* moment of the *same* shot, so a cursor of its own
    would be able to say otherwise.

    It is also **empty until asked**. A cross-tier check is minutes of F1
    marching -- F1 has no shot history (issue #8733), so each probe is a
    separate march to one recorded pose -- and putting that on the path of
    every design evaluation would make the workbench unusable. The
    workbench runs it from its own button and hands the result here.

    The status line restates the licence rather than leaving it to the
    figure. Both carry it: the figure because a screenshot keeps its
    contents and loses its surroundings, the status line because a reader
    scrubbing the cursor is looking at the widget and not at the caption.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Build an empty view.

        Args:
            title: Heading shown above the canvas.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._title = str(title)
        self._comparison: CrossTierComparison | None = None
        self._artists: CrossTierArtists | None = None
        self._frame = 0

        layout, self._heading, self._canvas = build_canvas_column(
            self,
            self._title,
            width_in=11.0,
            height_in=8.0,
            minimum_height_px=_MIN_CROSS_TIER_HEIGHT_PX,
        )
        self._readout = QLabel(_IDLE_CROSS_TIER)
        self._readout.setWordWrap(True)
        self._readout.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._readout)

    # ------------------------------------------------------------ accessors

    @property
    def title(self) -> str:
        """The heading shown above the canvas."""
        return self._title

    @property
    def has_comparison(self) -> bool:
        """Whether a comparison is loaded."""
        return self._comparison is not None

    @property
    def n_frames(self) -> int:
        """Samples in the loaded F0 record; zero when empty."""
        return 0 if self._comparison is None else self._comparison.n_frames

    @property
    def frame_index(self) -> int:
        """The sample currently under the cursor."""
        return self._frame

    @property
    def status_text(self) -> str:
        """What the status line is saying."""
        return self._readout.text()

    # --------------------------------------------------------------- content

    def set_comparison(self, comparison: CrossTierComparison) -> None:
        """Load one comparison and open on the probe F0's force peaked at.

        Args:
            comparison: The comparison to draw.
        """
        self._comparison = comparison
        self._artists = CrossTierArtists(self._canvas.fig, comparison)
        self._frame = comparison.peak_probe.frame
        self._redraw()

    def clear(self) -> None:
        """Drop the comparison, so nothing stale stays under a new shot."""
        self._comparison = None
        self._artists = None
        self._frame = 0
        self._readout.setText(_IDLE_CROSS_TIER)
        self._canvas.fig.clear()
        self._canvas.draw_idle()

    # ------------------------------------------------------------- following

    def set_frame(self, frame: int) -> None:
        """Move the cursor to one sample.

        The :class:`~.widgets.FollowsFrame` entry point. A frame arriving
        for an empty view is ignored rather than refused: the workbench
        clears views independently, so a transport tick can legitimately
        reach a view that has just been emptied.

        Args:
            frame: The sample index.

        Raises:
            ValueError: If a comparison is loaded and the index is outside
                its record. A clamped index would leave this view showing a
                different moment from the one driving it.
        """
        if self._comparison is None:
            return
        if not 0 <= int(frame) < self._comparison.n_frames:
            raise ValueError(
                f"frame {frame} is outside the shot, which has "
                f"{self._comparison.n_frames} samples"
            )
        self._frame = int(frame)
        self._redraw()

    def _redraw(self) -> None:
        """Repaint the canvas and restate the licence beside the moment."""
        comparison = self._comparison
        if comparison is None or self._artists is None:
            return
        self._artists.update(self._frame)
        self._canvas.draw_idle()
        moment = float(comparison.time_s[self._frame]) * 1e3
        status = comparison.status_at(self._frame)
        self._readout.setText(
            f"{moment:.2f} ms - {status.value.replace('_', ' ').upper()} - "
            f"{comparison.licence_stamp()}"
        )


class SandSliceWidget(QWidget):
    """A cutting plane through the sand, on the shared cursor (issue #8711).

    The plane is the one control this view owns, for the same reason the 3-D
    view owns its camera: where you cut is not when you cut. Changing the
    plane must not move the cursor, and moving the cursor must not reset the
    plane.

    The colour scale is injected, not inferred (issue #8728). Two grinds each
    scaled to their own fastest sand look equally fast, which is the one
    thing a comparison must never do.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Build an empty view.

        Args:
            title: Heading shown above the canvas.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._title = str(title)
        self._series: SandFieldSeries | None = None
        self._artists: SliceArtists | None = None
        self._scale: SliceScale | None = None
        self._cursor: CursorMap | None = None
        self._planes: tuple[CuttingPlane, ...] = ()
        self._frame = 0

        layout, self._heading, self._canvas = build_canvas_column(
            self,
            self._title,
            width_in=8.0,
            height_in=9.0,
            minimum_height_px=_MIN_SLICE_HEIGHT_PX,
        )

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Cut:"))
        self._plane_box = QComboBox()
        self._plane_box.currentIndexChanged.connect(self._on_plane)
        controls.addWidget(self._plane_box, stretch=1)
        layout.addLayout(controls)

        self._readout = QLabel("no sand field")
        self._readout.setWordWrap(True)
        layout.addWidget(self._readout)

    # ------------------------------------------------------------ accessors

    @property
    def title(self) -> str:
        """The heading shown above the canvas."""
        return self._title

    @property
    def has_shot(self) -> bool:
        """Whether a sand field is loaded."""
        return self._series is not None

    @property
    def n_frames(self) -> int:
        """Frames in the loaded field; zero when empty.

        The *field's* frame count, which is not the shot's. See
        :attr:`cursor`.
        """
        return 0 if self._series is None else self._series.n_frames

    @property
    def frame_index(self) -> int:
        """The field frame currently displayed."""
        return self._frame

    @property
    def scale(self) -> SliceScale | None:
        """The fixed colour limits in force, or ``None`` when empty."""
        return self._scale

    @property
    def cursor_map(self) -> CursorMap | None:
        """How the shared cursor maps onto this field's frames."""
        return self._cursor

    @property
    def planes(self) -> tuple[CuttingPlane, ...]:
        """The cuts offered, in the order they are listed."""
        return self._planes

    @property
    def plane(self) -> CuttingPlane | None:
        """The cut currently drawn, or ``None`` when empty."""
        index = self._plane_box.currentIndex()
        if not self._planes or not 0 <= index < len(self._planes):
            return None
        return self._planes[index]

    # --------------------------------------------------------------- content

    def set_shot(
        self,
        series: SandFieldSeries,
        *,
        scale: SliceScale | None = None,
        cursor: CursorMap | None = None,
        planes: tuple[CuttingPlane, ...] | None = None,
    ) -> None:
        """Load one sand field and open on its fastest moment.

        Args:
            series: The field, with its tier and validity status inside it.
            scale: Colour limits shared with any other field this one is
                compared against. Defaults to this field's own coverage,
                which is right alone and wrong in a comparison.
            cursor: How an external transport's frame index maps onto this
                field's frames. Defaults to the identity over this field.
            planes: The cuts to offer. Defaults to the named presets sized
                from the field's own declared effective width, so a
                heel-to-toe station cannot step outside the slab the tier
                claims.

        Raises:
            ValueError: If no cut is offered -- an empty selector would leave
                a loaded field with nothing to draw and no way to say why.
        """
        offered = preset_planes(series) if planes is None else tuple(planes)
        if not offered:
            raise ValueError("a sand cut view needs at least one plane to offer")
        self._series = series
        self._scale = slice_scale([series]) if scale is None else scale
        self._cursor = cursor or CursorMap(
            n_transport=series.n_frames, n_field=series.n_frames
        )
        self._planes = offered

        self._plane_box.blockSignals(True)
        self._plane_box.clear()
        for plane in offered:
            self._plane_box.addItem(plane.name)
            self._plane_box.setItemData(
                self._plane_box.count() - 1,
                (
                    plane.describe()
                    if plane.preset is None
                    else f"{plane.preset.description}\n{plane.describe()}"
                ),
                Qt.ItemDataRole.ToolTipRole,
            )
        self._plane_box.setCurrentIndex(0)
        self._plane_box.blockSignals(False)

        self._frame = _fastest_frame(series)
        self._rebuild()

    def clear(self) -> None:
        """Drop the field, so nothing stale stays on screen after a refusal."""
        self._series = None
        self._artists = None
        self._scale = None
        self._cursor = None
        self._planes = ()
        self._frame = 0
        self._plane_box.blockSignals(True)
        self._plane_box.clear()
        self._plane_box.blockSignals(False)
        self._readout.setText("no sand field")
        self._canvas.fig.clear()
        self._canvas.draw_idle()

    # ------------------------------------------------------------- following

    def set_frame(self, frame: int) -> None:
        """Show whatever field frame a transport frame maps onto.

        The :class:`~.widgets.FollowsFrame` entry point. The index arriving
        here is the *shot's*, and this view's record is the field's, so it is
        mapped rather than used directly -- and the frame says so.

        Args:
            frame: The transport's sample index.

        Raises:
            ValueError: If a field is loaded and the index is outside the
                transport. Clamping would leave this view showing a different
                moment from the one driving it, which is the whole failure
                linking the views exists to prevent.
        """
        if self._series is None or self._cursor is None:
            return
        self._frame = self._cursor.field_frame(frame)
        self._redraw()

    def _on_plane(self, _index: int) -> None:
        """Cut somewhere else, at the same moment."""
        if self._series is None:
            return
        self._rebuild()

    def _rebuild(self) -> None:
        """Lay the panels out again for a new field or a new cut."""
        series = self._series
        plane = self.plane
        if series is None or plane is None or self._scale is None:
            return
        self._artists = SliceArtists(
            self._canvas.fig, series, plane, self._scale, cursor=self._cursor
        )
        self._redraw()

    def _redraw(self) -> None:
        """Repaint the canvas and restate what this cut is."""
        series = self._series
        if series is None or self._artists is None:
            return
        self._artists.update(self._frame)
        self._canvas.draw_idle()
        provenance = series.provenance
        self._readout.setText(
            f"{series.time_s[self._frame] * 1e3:.3f} ms - "
            f"{provenance.status_label} - "
            f"{provenance.fidelity_tier.value} - {provenance.speed_headline()}"
        )


def _fastest_frame(series: SandFieldSeries) -> int:
    """The frame carrying the fastest reportable sand.

    The moment worth opening on, read off the *masked* speeds: opening on
    the frame with the loudest stencil tail would open on numerics.
    """
    speeds = series.occupied_speed_m_s()
    # Zero-filled before the reduction rather than masked after it: a
    # nanmax over an all-nan row is correct but warns, and frame 0 of
    # every capture is the undisturbed bed.
    peaks = np.nan_to_num(speeds, nan=0.0).max(axis=1)
    return int(np.argmax(peaks))
