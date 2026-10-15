"""3-D particle playback inside the shot viewport (issue #9545, slice 9).

The Qt half of the particle viewer. :mod:`.particle_playback` decides what
a frame shows and :mod:`.particle_render_vtk` decides how to draw it; this
panel only lets a person open an artifact and move through it. It does no
physics, reads no HDF5 itself and draws nothing of its own.

What it owns, and why
---------------------

* **Its own transport.** The workbench's one transport
  (:class:`~.widgets.SoleLoadFieldWidget`) steps an F0 shot's *samples*.
  A particle trajectory is a separate artifact on its own simulation clock,
  and nothing in production links the two runs, so following the shot's
  sample index would show a grain frame from a different moment while
  claiming to be in step. The panel therefore drives the existing
  :class:`~.particle_playback.ParticlePlayback` transport -- seek, step,
  scrub by time -- from a :class:`~PyQt6.QtCore.QTimer`, and never builds a
  second playback model.
* **A file the user chose.** No production solver writes the artifact yet:
  the grain-resolving 3-D tier needs ``pychrono``, which the owner has
  deferred. The panel says so and plays only an artifact a person opens.
* **Refusal before rendering.** A file is drawn only after
  :class:`~bunkershot3d.fields.trajectory_reader.ParticleTrajectoryReader`
  has verified its marker, version and digests. A rejected file shows the
  reader's own error, and the previous artifact is dropped too, so nothing
  stale or partial stays on screen beside the error.
* **A reason for every disabled control.** :meth:`~ParticlePlaybackPanel.
  disabled_reasons` maps each control to ``None`` or the sentence the
  panel shows: no trajectory loaded, a one-frame trajectory, or why VTK is
  unavailable (camera and clipping need a real render; play, step, scrub,
  stamps and the time scale work without one).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .particle_playback import BodyTrack, ParticlePlayback
from .particle_render_vtk import (
    CLIP_DESCRIPTION,
    ParticleViewport,
    ParticleViewportUnavailable,
    open_particle_viewport,
)
from .shot3d import CameraPreset

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bunkershot3d.fields.trajectory_reader import ParticleTrajectoryReader

logger = logging.getLogger(__name__)

__all__ = [
    "NO_TRAJECTORY_REASON",
    "SHORTCUTS",
    "TIME_SCALES",
    "ParticlePlaybackPanel",
    "advance_clock",
]

NO_TRAJECTORY_REASON = "no particle trajectory loaded"
_ONE_FRAME_REASON = "the trajectory has only one frame"
_IDLE_STATUS = (
    "No particle trajectory loaded. No production solver writes one yet: the "
    "grain-resolving 3-D tier needs pychrono, which is deferred. Open a "
    "verified artifact (an .h5 file with its .final.json marker) to play it."
)
_NO_BODIES_LINE = "club and ball: not drawn -- this artifact carries no body tracks"
_ALL_ACTIVE = "All particle controls are active."

TIME_SCALES: tuple[tuple[str, float], ...] = (
    ("1/10000x (0.1 ms of simulation per second)", 1.0e-4),
    ("1/1000x (1 ms of simulation per second)", 1.0e-3),
    ("1/100x (10 ms of simulation per second)", 1.0e-2),
)
"""Playback time scales: simulated seconds per wall-clock second."""

SHORTCUTS = {"open": "Ctrl+Shift+O", "play": "P", "step_back": ",", "step_forward": "."}
"""Keyboard shortcuts, active while focus is inside the panel."""

_DEFAULT_SCALE = 1
_TICK_MS = 40
_SCRUB_STEPS = 1000
_WINDOW_SIZE = (640, 400)
_MIN_IMAGE_HEIGHT_PX = 220
_TRANSPORT = ("play", "step_back", "step_forward", "scrub", "speed")
_VTK_ONLY = ("camera", "clipping")
_LABELS = {
    "open": "open",
    "play": "play/pause",
    "step_back": "step back",
    "step_forward": "step forward",
    "scrub": "time scrubber",
    "speed": "time scale",
    "camera": "camera",
    "clipping": "clipping",
}


def advance_clock(clock_s: float, *, wall_s: float, time_scale: float) -> float:
    """Simulation time after ``wall_s`` of playback at ``time_scale``.

    The clock is continuous; snapping it to a retained frame is the
    playback's job, so a tick shorter than half a frame gap still adds up.

    Raises:
        ValueError: If ``wall_s`` is negative or ``time_scale`` is not a
            positive finite number.
    """
    if not math.isfinite(wall_s) or wall_s < 0.0:
        raise ValueError(f"wall_s must be a non-negative time, got {wall_s!r}")
    if not math.isfinite(time_scale) or time_scale <= 0.0:
        raise ValueError(f"time_scale must be positive and finite, got {time_scale!r}")
    return float(clock_s) + float(wall_s) * float(time_scale)


def _open_reader(path: Path) -> ParticleTrajectoryReader:
    """A verified reader for ``path``.

    Raises:
        ValueError: The reader refused the artifact (its own message), or
            h5py is not installed.
        OSError: The file is unreadable or is not HDF5.
    """
    try:
        from bunkershot3d.fields.trajectory_reader import ParticleTrajectoryReader
    except ImportError as error:
        raise ValueError(f"the trajectory reader needs h5py ({error})") from error
    return ParticleTrajectoryReader(path).open()


def _open_viewport(
    playback: ParticlePlayback,
) -> ParticleViewport | ParticleViewportUnavailable:
    """The VTK viewport, or the reason there is none -- never an exception."""
    try:
        return open_particle_viewport(playback, window_size=_WINDOW_SIZE)
    except (
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        OSError,
        ImportError,
    ) as error:
        logger.warning("particle viewport degraded to no-VTK: %s", error)
        return ParticleViewportUnavailable(
            f"the particle layer did not build ({error})."
        )


def _qimage(array: NDArray[np.uint8]) -> QImage:
    """An owned ``QImage`` of an ``(H, W, 3)`` RGB array."""
    pixels = np.ascontiguousarray(array[..., :3], dtype=np.uint8)
    height, width = pixels.shape[:2]
    image = QImage(pixels.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return image.copy()


class ParticlePlaybackPanel(QGroupBox):
    """Open, play, step and scrub a verified particle trajectory."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build an empty panel that states why it is empty."""
        super().__init__("3-D particle playback", parent)
        self._reader: ParticleTrajectoryReader | None = None
        self._playback: ParticlePlayback | None = None
        self._viewport: ParticleViewport | ParticleViewportUnavailable | None = None
        self._has_bodies = False
        self._clock_s = 0.0
        self._pixels: NDArray[np.uint8] | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self.tick)
        self._build_actions()
        self._build_widgets()
        self._build_layout()
        self._status.setText(_IDLE_STATUS)
        self._refresh_controls()

    # ------------------------------------------------------------- building

    def _action(self, text: str, key: str, slot: Callable[[], None]) -> QAction:
        action = QAction(text, self)
        action.setShortcut(QKeySequence(SHORTCUTS[key]))
        action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        action.triggered.connect(lambda _checked=False: slot())
        self.addAction(action)
        return action

    def _build_actions(self) -> None:
        self.open_action = self._action(
            "Open particle trajectory…", "open", self.open_from_dialog
        )
        self.play_action = self._action("Play", "play", self.toggle_play)
        self.step_back_action = self._action("Step back", "step_back", self.step_back)
        self.step_forward_action = self._action(
            "Step forward", "step_forward", self.step_forward
        )
        self._tips = {
            "open": "Open a verified particle-trajectory artifact (Ctrl+Shift+O)",
            "play": "Play or pause the particle trajectory (P)",
            "step_back": "Show the previous retained frame (,)",
            "step_forward": "Show the next retained frame (.)",
            "scrub": "Seek by simulation time; Left/Right and Page keys move it",
            "speed": "Simulated seconds shown per wall-clock second",
            "camera": "Look from one of the shot's named views",
            "clipping": CLIP_DESCRIPTION,
        }

    def _button(self, action: QAction, name: str) -> QToolButton:
        button = QToolButton(self)
        button.setDefaultAction(action)
        button.setAccessibleName(name)
        button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        return button

    def _build_widgets(self) -> None:
        self._open_button = self._button(self.open_action, "Open particle trajectory")
        self._back_button = self._button(self.step_back_action, "Step back one frame")
        self._play_button = self._button(self.play_action, "Play or pause particles")
        self._forward_button = self._button(
            self.step_forward_action, "Step forward one frame"
        )
        self._scrubber = QSlider(Qt.Orientation.Horizontal, self)
        self._scrubber.setRange(0, _SCRUB_STEPS)
        self._scrubber.setPageStep(_SCRUB_STEPS // 10)
        self._scrubber.setAccessibleName("Particle time scrubber (simulation seconds)")
        self._scrubber.valueChanged.connect(self._on_scrub)
        self._speed = QComboBox(self)
        for label, scale in TIME_SCALES:
            self._speed.addItem(label, scale)
        self._speed.setCurrentIndex(_DEFAULT_SCALE)
        self._speed.setAccessibleName("Particle playback time scale")
        self._camera = QComboBox(self)
        for preset in CameraPreset:
            self._camera.addItem(preset.label, preset.value)
        self._camera.setAccessibleName("Particle camera preset")
        self._camera.currentIndexChanged.connect(self._on_view_option)
        self._clip = QCheckBox("Clip at y = 0", self)
        self._clip.setAccessibleName("Particle clipping plane")
        self._clip.toggled.connect(self._on_view_option)
        self._status, self._image, self._stamp, self._time, self._reason = (
            self._text_label(name)
            for name in ("status", "view", "frame stamp", "time", "disabled reasons")
        )
        self._image.setMinimumHeight(_MIN_IMAGE_HEIGHT_PX)
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._time.setWordWrap(False)
        self._controls: dict[str, QAction | QWidget] = {
            "open": self.open_action,
            "play": self.play_action,
            "step_back": self.step_back_action,
            "step_forward": self.step_forward_action,
            "scrub": self._scrubber,
            "speed": self._speed,
            "camera": self._camera,
            "clipping": self._clip,
        }

    def _text_label(self, name: str) -> QLabel:
        label = QLabel("", self)
        label.setWordWrap(True)
        label.setAccessibleName(f"Particle {name}")
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(self._open_button)
        header.addWidget(self._status, stretch=1)
        layout.addLayout(header)
        layout.addWidget(self._image)
        layout.addWidget(self._stamp)
        transport = QHBoxLayout()
        for widget in (self._back_button, self._play_button, self._forward_button):
            transport.addWidget(widget)
        transport.addWidget(self._scrubber, stretch=1)
        transport.addWidget(self._time)
        layout.addLayout(transport)
        options = QHBoxLayout()
        for caption, box in (("Speed:", self._speed), ("View:", self._camera)):
            label = QLabel(caption, self)
            label.setBuddy(box)
            options.addWidget(label)
            options.addWidget(box, stretch=1)
        options.addWidget(self._clip)
        layout.addLayout(options)
        layout.addWidget(self._reason)
        chain = self.focus_chain()
        for earlier, later in zip(chain, chain[1:], strict=False):
            QWidget.setTabOrder(earlier, later)

    # ------------------------------------------------------------ accessors

    def focus_chain(self) -> list[QWidget]:
        """The panel's controls in keyboard (Tab) order."""
        return [
            self._open_button,
            self._back_button,
            self._play_button,
            self._forward_button,
            self._scrubber,
            self._speed,
            self._camera,
            self._clip,
        ]

    @property
    def has_trajectory(self) -> bool:
        """Whether a verified trajectory is loaded."""
        return self._playback is not None

    @property
    def viewport_available(self) -> bool:
        """Whether frames are drawn by VTK (rather than stamped only)."""
        return isinstance(self._viewport, ParticleViewport)

    @property
    def is_playing(self) -> bool:
        """Whether the playback timer is running."""
        return self._timer.isActive()

    @property
    def frame_index(self) -> int:
        """The retained frame shown; ``0`` when nothing is loaded."""
        return 0 if self._playback is None else self._playback.frame_index

    @property
    def scrub_fraction(self) -> float:
        """The scrubber position as a fraction of the run's time span."""
        return self._scrubber.value() / _SCRUB_STEPS

    @property
    def time_scale(self) -> float:
        """Simulated seconds per wall-clock second."""
        return float(self._speed.currentData())

    @property
    def camera(self) -> CameraPreset:
        """The selected named view."""
        return CameraPreset(self._camera.currentData())

    @property
    def clipping(self) -> bool:
        """Whether the clipping toggle is on."""
        return self._clip.isChecked()

    @property
    def status_text(self) -> str:
        """The load status line, including any rejection."""
        return self._status.text()

    @property
    def stamp_text(self) -> str:
        """The frame stamp shown; empty when nothing is loaded."""
        return self._stamp.text()

    @property
    def time_text(self) -> str:
        """The time readout beside the scrubber."""
        return self._time.text()

    @property
    def image_text(self) -> str:
        """What the view area says when it shows no picture."""
        return self._image.text()

    @property
    def reason_text(self) -> str:
        """The visible line naming every disabled control and why."""
        return self._reason.text()

    def rendered_image(self) -> NDArray[np.uint8] | None:
        """The last VTK frame shown, or ``None`` without one."""
        return None if self._pixels is None else self._pixels.copy()

    def control_enabled(self, name: str) -> bool:
        """Whether the named control accepts input."""
        return self._controls[name].isEnabled()

    def disabled_reasons(self) -> dict[str, str | None]:
        """Each control mapped to ``None`` (active) or why it is disabled."""
        playback = self._playback
        transport: str | None = None
        if playback is None:
            transport = NO_TRAJECTORY_REASON
        elif playback.n_frames < 2:
            transport = _ONE_FRAME_REASON
        view: str | None = None
        if playback is None:
            view = NO_TRAJECTORY_REASON
        elif isinstance(self._viewport, ParticleViewportUnavailable):
            view = f"VTK unavailable: {self._viewport.reason}"
        reasons: dict[str, str | None] = {"open": None}
        reasons.update(dict.fromkeys(_TRANSPORT, transport))
        reasons["speed"] = None if playback is not None else NO_TRAJECTORY_REASON
        reasons.update(dict.fromkeys(_VTK_ONLY, view))
        return reasons

    # -------------------------------------------------------------- loading

    def open_from_dialog(self) -> None:
        """Ask for an artifact and load it; a cancelled dialog changes nothing."""
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Open particle trajectory",
            "",
            "Particle trajectory (*.h5 *.hdf5);;All files (*)",
        )
        if chosen:
            self.load_trajectory(chosen)

    def load_trajectory(
        self, path: Path | str, *, bodies: Sequence[BodyTrack] = ()
    ) -> bool:
        """Verify and show an artifact, or show why it was refused.

        Postcondition: on ``False`` nothing is loaded and nothing is drawn
        -- the previous artifact included -- and :attr:`status_text` carries
        the reader's error.

        Args:
            path: The ``.h5`` artifact; its marker sits beside it.
            bodies: Club/ball tracks to draw with it, when a caller has them.
        """
        self.pause()
        target = Path(path)
        try:
            reader = _open_reader(target)
        except (ValueError, OSError, KeyError) as error:
            return self._reject(target, error)
        try:
            playback = ParticlePlayback(reader, bodies=bodies)
        except ValueError as error:
            reader.close()
            return self._reject(target, error)
        self.unload()
        self._reader, self._playback = reader, playback
        self._has_bodies = bool(bodies)
        self._viewport = _open_viewport(playback)
        self._clock_s = playback.time_s
        start, end = playback.time_span_s
        self._status.setText(
            f"Loaded {target.name}: {playback.n_frames} frames, "
            f"t = {start:.6f} to {end:.6f} s."
        )
        self._refresh_controls()
        self._apply_view_options()
        self._render()
        return True

    def _reject(self, path: Path, error: BaseException) -> bool:
        self.unload()
        self._status.setText(
            f"Rejected {path.name}: {error}. Nothing from it is drawn."
        )
        return False

    def unload(self) -> None:
        """Stop, release the render window and the file, and say so."""
        self.pause()
        if isinstance(self._viewport, ParticleViewport):
            self._viewport.close()
        if self._reader is not None:
            self._reader.close()
        self._reader = self._playback = self._viewport = None
        self._has_bodies = False
        self._pixels = None
        self._image.clear()
        self._stamp.clear()
        self._time.clear()
        self._status.setText(_IDLE_STATUS)
        self._sync_scrubber(0.0)
        self._refresh_controls()

    # ------------------------------------------------------------ transport

    def play(self) -> None:
        """Start the timer; from the last frame, restart at the first."""
        playback = self._playback
        if playback is None or self.disabled_reasons()["play"] is not None:
            return
        if playback.frame_index == playback.n_frames - 1:
            playback.seek_frame(0)
            self._render()
        self._clock_s = playback.time_s
        self._timer.start()
        self.play_action.setText("Pause")

    def pause(self) -> None:
        """Stop the timer."""
        self._timer.stop()
        self.play_action.setText("Play")

    def toggle_play(self) -> None:
        """Play if paused, pause if playing."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def step_forward(self) -> None:
        """Show the next retained frame, stopping at the end."""
        self._step(1)

    def step_back(self) -> None:
        """Show the previous retained frame, stopping at the start."""
        self._step(-1)

    def _step(self, delta: int) -> None:
        if self._playback is None:
            return
        self._playback.step(delta)
        self._clock_s = self._playback.time_s
        self._render()

    def tick(self) -> None:
        """Advance the clock by one timer interval at the chosen time scale.

        Postcondition: playback pauses on the last frame rather than wrap,
        so the final frame of a run is always seen.
        """
        playback = self._playback
        if playback is None:
            self.pause()
            return
        self._clock_s = advance_clock(
            self._clock_s, wall_s=_TICK_MS / 1e3, time_scale=self.time_scale
        )
        if self._clock_s >= playback.time_span_s[1]:
            playback.seek_frame(playback.n_frames - 1)
            self.pause()
        else:
            playback.seek_time(self._clock_s)
        self._render()

    def set_scrub_fraction(self, fraction: float) -> None:
        """Move the scrubber to ``fraction`` of the run's time span.

        Raises:
            ValueError: If ``fraction`` is not in ``[0, 1]``.
        """
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError(f"scrub fraction must be in [0, 1], got {fraction!r}")
        self._scrubber.setValue(round(float(fraction) * _SCRUB_STEPS))

    def set_time_scale_index(self, index: int) -> None:
        """Choose one of :data:`TIME_SCALES` by position.

        Raises:
            ValueError: If ``index`` is outside :data:`TIME_SCALES`.
        """
        if not 0 <= int(index) < len(TIME_SCALES):
            raise ValueError(f"time scale {index} is outside 0..{len(TIME_SCALES) - 1}")
        self._speed.setCurrentIndex(int(index))

    def set_camera(self, camera: CameraPreset | str) -> None:
        """Select a named view (shown only when VTK draws)."""
        self._camera.setCurrentIndex(self._camera.findData(CameraPreset(camera).value))

    def set_clipping(self, enabled: bool) -> None:
        """Turn the clipping toggle on or off (shown only when VTK draws)."""
        self._clip.setChecked(bool(enabled))

    def _on_scrub(self, value: int) -> None:
        if self._playback is None:
            return
        self._playback.scrub(value / _SCRUB_STEPS)
        self._clock_s = self._playback.time_s
        self._render(sync_scrubber=False)

    def _on_view_option(self, *_: object) -> None:
        if self._apply_view_options():
            self._render()

    def _apply_view_options(self) -> bool:
        viewport = self._viewport
        if not isinstance(viewport, ParticleViewport):
            return False
        viewport.set_camera(self.camera)
        viewport.set_clipping(self.clipping)
        return True

    # ------------------------------------------------------------ rendering

    def _render(self, *, sync_scrubber: bool = True) -> None:
        playback = self._playback
        if playback is None:
            return
        stamp = self._draw_frame(playback)
        if not self._has_bodies:
            stamp = f"{stamp}\n{_NO_BODIES_LINE}"
        self._stamp.setText(stamp)
        self._time.setText(
            f"t = {playback.time_s:.6f} s ({playback.frame_index + 1}/"
            f"{playback.n_frames})"
        )
        if sync_scrubber:
            self._sync_scrubber(playback.time_fraction())

    def _draw_frame(self, playback: ParticlePlayback) -> str:
        """Draw the current frame and return its stamp; degrade, never raise."""
        viewport = self._viewport
        if isinstance(viewport, ParticleViewport):
            try:
                viewport.show_frame(playback.frame_index)
                self._pixels = viewport.image_array()
            except (
                RuntimeError,
                ValueError,
                TypeError,
                AttributeError,
                OSError,
            ) as error:
                logger.warning("particle render degraded to no-VTK: %s", error)
                viewport.close()
                self._viewport = ParticleViewportUnavailable(
                    f"the render failed ({error})."
                )
                self._refresh_controls()
            else:
                self._image.setPixmap(QPixmap.fromImage(_qimage(self._pixels)))
                return viewport.stamp_text
        self._pixels = None
        if isinstance(self._viewport, ParticleViewportUnavailable):
            self._image.setText(self._viewport.describe())
        return "\n".join(playback.view().stamp_lines)

    def _sync_scrubber(self, fraction: float) -> None:
        self._scrubber.blockSignals(True)
        self._scrubber.setValue(round(fraction * _SCRUB_STEPS))
        self._scrubber.blockSignals(False)

    def _refresh_controls(self) -> None:
        """Enable each control or disable it with its visible reason."""
        reasons = self.disabled_reasons()
        grouped: dict[str, list[str]] = {}
        for name, reason in reasons.items():
            control = self._controls[name]
            control.setEnabled(reason is None)
            tip = self._tips[name] if reason is None else f"Disabled: {reason}"
            control.setToolTip(tip)
            if reason is not None:
                grouped.setdefault(reason, []).append(_LABELS[name])
        lines = [
            f"Disabled ({', '.join(names)}): {why}" for why, names in grouped.items()
        ]
        self._reason.setText("\n".join(lines) if lines else _ALL_ACTIVE)
