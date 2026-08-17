"""The 3-D scene view and the trace panel, as Qt views (issues #8706, #8708).

Neither widget here owns a transport. :class:`~.widgets.SoleLoadFieldWidget`
already has one -- a slider, a timer and a play button, built for the sole
load field -- and #8708 asks for the three views to be scrubbed *in lockstep*,
which is precisely the thing a second slider would break. Both classes below
are therefore :class:`~.widgets.FollowsFrame` implementations: they are handed
a sample index and repaint. The workbench wires them up with
:meth:`~.widgets.SoleLoadFieldWidget.link`.

Neither does physics and neither draws anything of its own. The arithmetic is
:mod:`~.shot3d`'s and :mod:`~.traces`', the drawing is :mod:`~.render3d`'s and
:mod:`~.render_traces`', which is why the same figures can be produced in a
headless test. These classes own a canvas, a camera selector, and the
knowledge of when to repaint.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from .render import viewport_fallback
from .render3d import SceneScale, ShotSceneArtists, scene_scale
from .render_traces import TracePanelArtists
from .shot3d import CameraPreset, ShotScene
from .traces import ShotTraces, ValidityBand
from .widgets import build_canvas_column

__all__ = [
    "ShotViewportWidget",
    "TracePanelWidget",
]

_MIN_SCENE_HEIGHT_PX = 340
_MIN_TRACE_HEIGHT_PX = 420


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
