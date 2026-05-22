"""Animated full-trajectory preview helpers for the starting-pose matcher.

This module is the home of the new :class:`PlaybackController` (issue #4482)
that drives a :class:`matplotlib.animation.FuncAnimation` over a stack of
per-layer artists. The matcher's main ``gui.py`` module embeds an instance
of this controller; the controller itself is GUI-framework-agnostic so the
headless tests in ``tests/unit/tools/starting_pose_matcher/test_animation.py``
can exercise it under the matplotlib ``Agg`` backend without a Qt event
loop.

Each rendering layer (mocap markers, body skeleton segments, club mid-hands
trace, clubface trace, clubface triad, ball impact, model skeleton) is a
small object that owns one or more matplotlib ``Artist`` instances and
exposes:

* ``visible``   — toggling does not recreate the artist; it only toggles
  ``Artist.set_visible``. This is what keeps layer visibility changes from
  tearing playback (acceptance criterion).
* ``update(frame)`` — updates the artist data via ``set_data_3d`` /
  ``set_segments`` rather than recreating the artist each frame.
* ``artists()`` — returns the underlying matplotlib artists so the
  ``FuncAnimation`` ``blit=False`` path can re-draw them in one pass.

The controller is intentionally generic and accepts duck-typed inputs:
either a fully-typed ``BodyTarget`` / ``ClubTarget`` / ``BallImpactState``
(once those land — see issue body), or any object exposing the equivalent
attributes / arrays. This keeps the controller usable in tests and ahead
of the dependent PRs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.axes3d import Axes3D

from .session_schema import ALLOWED_SPEEDS, PlaybackState

logger = logging.getLogger(__name__)

# Generic UI labels used by the layer visibility checkboxes. Per the issue
# body, no source / vendor / lab / person names appear here.
LAYER_LABELS: dict[str, str] = {
    "body_markers": "Mocap markers",
    "body_skeleton": "Body skeleton",
    "club_midhands_trace": "Club mid-hands trace",
    "clubface_trace": "Clubface trace",
    "clubface_triad": "Clubface triad",
    "ball_impact": "Ball impact",
    "model_skeleton": "Model skeleton",
}


# --------------------------------------------------------------------------- #
# Layer base + concrete layers                                                #
# --------------------------------------------------------------------------- #


@dataclass
class _LayerBase:
    """Shared behaviour for animation layers.

    Subclasses must populate ``_artists`` in :meth:`build` and implement
    :meth:`update` to mutate artist data for a given frame index.
    """

    key: str
    _visible: bool = True
    _artists: list[Any] = field(default_factory=list, repr=False)

    @property
    def label(self) -> str:
        return LAYER_LABELS.get(self.key, self.key)

    @property
    def visible(self) -> bool:
        return self._visible

    def set_visible(self, visible: bool) -> None:
        """Toggle visibility without recreating artists.

        This is the path that keeps the artist count constant across
        toggles — the test ``test_layer_toggle_does_not_tear`` asserts
        that.
        """
        self._visible = bool(visible)
        for art in self._artists:
            art.set_visible(self._visible)

    def artists(self) -> list[Any]:
        return list(self._artists)

    def build(self, ax: Axes3D) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def update(self, frame: int) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class BodyMarkerLayer(_LayerBase):
    """Body mocap markers as a 3-D scatter line.

    Source: ``marker_xyz`` array of shape ``(T, M, 3)`` (metres). When
    ``BodyTarget`` lands (#4476) callers can pass ``body_target.marker_xyz``
    directly.
    """

    marker_xyz: np.ndarray | None = None  # (T, M, 3)

    def build(self, ax: Axes3D) -> None:
        if self.marker_xyz is None or self.marker_xyz.size == 0:
            return
        first = self.marker_xyz[0]
        (line,) = ax.plot(
            first[:, 0],
            first[:, 1],
            first[:, 2],
            linestyle="none",
            marker="o",
            markersize=3,
            color="tab:blue",
            label=self.label,
        )
        line.set_visible(self._visible)
        self._artists = [line]

    def update(self, frame: int) -> None:
        if self.marker_xyz is None or not self._artists:
            return
        t = int(np.clip(frame, 0, self.marker_xyz.shape[0] - 1))
        pts = self.marker_xyz[t]
        # Line3D.set_data_3d expects three 1-D arrays.
        self._artists[0].set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])


@dataclass
class BodySkeletonLayer(_LayerBase):
    """Body skeleton segments as a Line3DCollection.

    ``segments`` is the precomputed array of shape ``(T, S, 2, 3)`` — for
    each frame, ``S`` segments each with two ``(x, y, z)`` endpoints.
    Pre-computation happens once after loading (issue body, performance
    section) so per-frame updates are a cheap ``set_segments`` call.
    """

    segments: np.ndarray | None = None  # (T, S, 2, 3)

    def build(self, ax: Axes3D) -> None:
        if self.segments is None or self.segments.size == 0:
            return
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        first = self.segments[0]
        coll = Line3DCollection(
            [seg.tolist() for seg in first],
            colors="tab:cyan",
            linewidths=1.5,
        )
        ax.add_collection3d(coll)
        coll.set_visible(self._visible)
        self._artists = [coll]

    def update(self, frame: int) -> None:
        if self.segments is None or not self._artists:
            return
        t = int(np.clip(frame, 0, self.segments.shape[0] - 1))
        self._artists[0].set_segments(self.segments[t].tolist())


@dataclass
class ClubTraceLayer(_LayerBase):
    """A growing 3-D polyline for club mid-hands or clubface trajectory.

    ``positions`` has shape ``(T, 3)``. The trace draws ``[0:t+1]``
    each frame (full path up to the current frame, per the issue body).
    """

    positions: np.ndarray | None = None  # (T, 3)
    color: str = "tab:orange"

    def build(self, ax: Axes3D) -> None:
        if self.positions is None or self.positions.size == 0:
            return
        (line,) = ax.plot(
            self.positions[:1, 0],
            self.positions[:1, 1],
            self.positions[:1, 2],
            color=self.color,
            linewidth=1.4,
            label=self.label,
        )
        line.set_visible(self._visible)
        self._artists = [line]

    def update(self, frame: int) -> None:
        if self.positions is None or not self._artists:
            return
        t = int(np.clip(frame, 0, self.positions.shape[0] - 1))
        seg = self.positions[: t + 1]
        self._artists[0].set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])


@dataclass
class ClubfaceTriadLayer(_LayerBase):
    """Three unit-vector axes drawn at the clubhead each frame.

    ``origin`` is the clubhead position (T, 3). ``triad`` is the (T, 3, 3)
    array of body-frame axes — column j of ``triad[t]`` is axis j.
    """

    origin: np.ndarray | None = None  # (T, 3)
    triad: np.ndarray | None = None  # (T, 3, 3)
    length: float = 0.1

    def build(self, ax: Axes3D) -> None:
        if self.origin is None or self.triad is None:
            return
        if self.origin.size == 0:
            return
        colors = ("tab:red", "tab:green", "tab:blue")
        self._artists = []
        for j in range(3):
            o = self.origin[0]
            d = self.triad[0, :, j] * self.length
            (line,) = ax.plot(
                [o[0], o[0] + d[0]],
                [o[1], o[1] + d[1]],
                [o[2], o[2] + d[2]],
                color=colors[j],
                linewidth=1.5,
            )
            line.set_visible(self._visible)
            self._artists.append(line)

    def update(self, frame: int) -> None:
        if self.origin is None or self.triad is None or not self._artists:
            return
        t = int(np.clip(frame, 0, self.origin.shape[0] - 1))
        o = self.origin[t]
        for j, art in enumerate(self._artists):
            d = self.triad[t, :, j] * self.length
            art.set_data_3d(
                np.array([o[0], o[0] + d[0]]),
                np.array([o[1], o[1] + d[1]]),
                np.array([o[2], o[2] + d[2]]),
            )


@dataclass
class BallImpactLayer(_LayerBase):
    """Static marker drawn at the ball impact point.

    ``position`` is the (3,) ball impact point in metres.
    """

    position: np.ndarray | None = None

    def build(self, ax: Axes3D) -> None:
        if self.position is None:
            return
        (line,) = ax.plot(
            [self.position[0]],
            [self.position[1]],
            [self.position[2]],
            linestyle="none",
            marker="*",
            markersize=10,
            color="gold",
            markeredgecolor="black",
            label=self.label,
        )
        line.set_visible(self._visible)
        self._artists = [line]

    def update(self, frame: int) -> None:  # noqa: ARG002 - position is static
        return


@dataclass
class TrailLayer(_LayerBase):
    """Fading polyline trails for the last ``trail_frames`` of each marker.

    Implemented as a single Line3DCollection segments array of shape
    ``(M*(W-1), 2, 3)`` rebuilt per frame; cheap because W is small (default
    30) and segment data is updated via ``set_segments``.
    """

    marker_xyz: np.ndarray | None = None  # (T, M, 3)
    trail_frames: int = 30
    color: str = "tab:blue"

    def build(self, ax: Axes3D) -> None:
        if self.marker_xyz is None or self.marker_xyz.size == 0:
            return
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        # Seed with a single zero-length segment per marker so matplotlib's
        # autoscale path has something to work with on add_collection3d.
        m0 = self.marker_xyz[0]
        seed = [[m0[i].tolist(), m0[i].tolist()] for i in range(m0.shape[0])]
        coll = Line3DCollection(seed, colors=self.color, linewidths=0.8, alpha=0.4)
        ax.add_collection3d(coll)
        coll.set_visible(self._visible)
        self._artists = [coll]

    def update(self, frame: int) -> None:
        if self.marker_xyz is None or not self._artists:
            return
        t = int(np.clip(frame, 0, self.marker_xyz.shape[0] - 1))
        start = max(0, t - max(1, int(self.trail_frames)))
        window = self.marker_xyz[start : t + 1]  # (W, M, 3)
        if window.shape[0] < 2:
            self._artists[0].set_segments([])
            return
        # Build segment list: for each marker, consecutive (W-1) segments.
        # Shape (M, W-1, 2, 3) → flatten to (M*(W-1), 2, 3).
        segs = np.stack([window[:-1], window[1:]], axis=2)  # (W-1, M, 2, 3)
        segs = np.transpose(segs, (1, 0, 2, 3)).reshape(-1, 2, 3)
        self._artists[0].set_segments(segs.tolist())


# --------------------------------------------------------------------------- #
# Controller                                                                  #
# --------------------------------------------------------------------------- #


_BASE_FPS = 30.0


@dataclass
class PlaybackController:
    """Drives ``FuncAnimation`` over a stack of layers.

    The controller is created with a matplotlib ``Axes3D``, a number of
    frames ``n_frames``, an iterable of layers, and an initial
    :class:`PlaybackState`. It does **not** start the animation in
    ``__init__`` — call :meth:`start` for that. Tests that just need to
    drive frames synchronously can call :meth:`step` directly.

    Frame counter callback (``on_frame``) is invoked after each update so
    the host GUI can refresh its "12 / 301" label and the slider.
    """

    ax: Axes3D
    n_frames: int
    layers: list[_LayerBase]
    state: PlaybackState = field(default_factory=PlaybackState)
    on_frame: Callable[[int], None] | None = None

    _anim: FuncAnimation | None = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_frames < 0:
            raise ValueError(f"n_frames must be >= 0, got {self.n_frames}")
        for layer in self.layers:
            layer.build(self.ax)
        # Render the initial frame so the static view matches the state.
        if self.n_frames > 0:
            self._render_frame(self.state.current_frame)

    # -- layer access -------------------------------------------------------- #

    def layer(self, key: str) -> _LayerBase:
        for layer in self.layers:
            if layer.key == key:
                return layer
        raise KeyError(f"unknown layer: {key!r}")

    def set_layer_visible(self, key: str, visible: bool) -> None:
        """Toggle a layer mid-playback without tearing.

        The artist count is invariant across this call: existing artists
        are kept and only their ``visible`` flag flips. The animation
        continues to run from the same frame.
        """
        self.layer(key).set_visible(visible)

    def all_artists(self) -> list[Any]:
        out: list[Any] = []
        for layer in self.layers:
            out.extend(layer.artists())
        return out

    # -- playback ----------------------------------------------------------- #

    def step(self, count: int = 1) -> None:
        """Advance ``count`` frames synchronously.

        Used by headless tests to play through the whole timeline without
        a real-time clock.
        """
        for _ in range(count):
            self._advance_one_frame()

    def _advance_one_frame(self) -> None:
        nxt = self.state.current_frame + 1
        if nxt >= self.n_frames:
            if self.state.loop:
                nxt = 0
            else:
                self._running = False
                return
        self.state.current_frame = nxt
        self._render_frame(nxt)

    def _render_frame(self, frame: int) -> None:
        for layer in self.layers:
            try:
                layer.update(frame)
            except Exception:  # pragma: no cover - defensive
                logger.exception("layer %s update failed", layer.key)
        if self.on_frame is not None:
            self.on_frame(frame)

    def seek(self, frame: int) -> None:
        """Jump directly to ``frame`` (clamped to the valid range)."""
        if self.n_frames == 0:
            return
        f = int(np.clip(frame, 0, self.n_frames - 1))
        self.state.current_frame = f
        self._render_frame(f)

    def set_speed(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError(f"speed must be > 0, got {speed}")
        self.state.speed = float(speed)
        if self._anim is not None and hasattr(self._anim, "event_source"):
            es = self._anim.event_source
            if es is not None:
                es.interval = self._interval_ms()

    def set_loop(self, loop: bool) -> None:
        self.state.loop = bool(loop)

    def set_trail_frames(self, n: int) -> None:
        if n < 0:
            raise ValueError(f"trail_frames must be >= 0, got {n}")
        self.state.trail_frames = int(n)
        for layer in self.layers:
            if isinstance(layer, TrailLayer):
                layer.trail_frames = int(n)
                layer.update(self.state.current_frame)

    def _interval_ms(self) -> int:
        return max(1, int(round(1000.0 / (_BASE_FPS * self.state.speed))))

    def start(self) -> FuncAnimation | None:
        """Start a real ``FuncAnimation`` bound to ``self.ax`` figure.

        In headless tests this is generally avoided in favour of
        :meth:`step`; in the live GUI it returns the running animation
        (``gui.py`` keeps a reference to prevent GC).
        """
        if self.n_frames == 0:
            return None
        from matplotlib.animation import FuncAnimation

        def _frame_callable(_i: int) -> Sequence[Any]:
            self._advance_one_frame()
            return self.all_artists()

        fig = self.ax.figure
        self._anim = FuncAnimation(
            fig,
            _frame_callable,
            interval=self._interval_ms(),
            blit=False,
            cache_frame_data=False,
        )
        self._running = True
        return self._anim

    def pause(self) -> None:
        """Pause the animation if running. Safe to call multiple times.

        Wired to ``QHideEvent`` in the host GUI so playback halts when
        the window is hidden.
        """
        if self._anim is not None:
            try:
                self._anim.pause()
            except AttributeError:  # pragma: no cover - older matplotlib
                if self._anim.event_source is not None:
                    self._anim.event_source.stop()
        self._running = False

    def resume(self) -> None:
        """Resume the animation if it was paused."""
        if self._anim is not None:
            try:
                self._anim.resume()
            except AttributeError:  # pragma: no cover - older matplotlib
                if self._anim.event_source is not None:
                    self._anim.event_source.start()
            self._running = True

    @property
    def is_running(self) -> bool:
        return self._running


def precompute_segments_from_pairs(
    marker_xyz: np.ndarray,
    pairs: Iterable[tuple[int, int]],
) -> np.ndarray:
    """Pre-compute body-skeleton segment endpoints once per dataset.

    Parameters
    ----------
    marker_xyz : np.ndarray
        Shape ``(T, M, 3)``.
    pairs : iterable of (int, int)
        Marker-index pairs that define each skeletal segment. This will
        be supplied by ``default_body_segments`` (issue #4483) once that
        lands; until then callers may pass any pair list.

    Returns
    -------
    np.ndarray
        Shape ``(T, S, 2, 3)``, suitable for :class:`BodySkeletonLayer`.
    """
    pair_arr = np.asarray(list(pairs), dtype=int)
    if pair_arr.size == 0:
        return np.zeros((marker_xyz.shape[0], 0, 2, 3), dtype=marker_xyz.dtype)
    a = marker_xyz[:, pair_arr[:, 0], :]  # (T, S, 3)
    b = marker_xyz[:, pair_arr[:, 1], :]  # (T, S, 3)
    return np.stack([a, b], axis=2)  # (T, S, 2, 3)


__all__ = [
    "ALLOWED_SPEEDS",
    "BallImpactLayer",
    "BodyMarkerLayer",
    "BodySkeletonLayer",
    "ClubTraceLayer",
    "ClubfaceTriadLayer",
    "LAYER_LABELS",
    "PlaybackController",
    "PlaybackState",
    "TrailLayer",
    "precompute_segments_from_pairs",
]
