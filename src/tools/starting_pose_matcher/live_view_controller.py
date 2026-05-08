"""Live multi-source rendering controller for the starting-pose matcher.

Wires the loaded :class:`~motion_matching.body_target.BodyTarget` (and any
companion club / ball targets) to the matcher's existing 3-D matplotlib
axes so the user can scrub the timeline and visibly see the markers move.

This module is the orchestration glue called out in issue #4512. It owns
*instances* of the layer classes already living in
:mod:`gui_playback` — :class:`BodyMarkerLayer`, :class:`BodySkeletonLayer`,
:class:`ClubTraceLayer`, :class:`ClubfaceTriadLayer`, :class:`BallImpactLayer`
and :class:`TrailLayer` — and re-binds them whenever a new target is
selected. It does *not* duplicate any rendering logic; it is a thin
controller.

The controller is duck-typed on its inputs so it works ahead of (or
without) :class:`MultiSourceTarget` from PR #4505: callers pass body /
club / ball arguments directly to :meth:`set_target`. Every layer must
accept missing slots gracefully — selecting only a body file produces a
working live view with no club or ball artists.

Headless tests under ``QT_QPA_PLATFORM=offscreen`` and the matplotlib
``Agg`` backend exercise :meth:`set_target` and :meth:`set_frame` without
a real Qt event loop; :meth:`set_layer_visible` is also covered.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .gui_playback import (
    BallImpactLayer,
    BodyMarkerLayer,
    BodySkeletonLayer,
    ClubfaceTriadLayer,
    ClubTraceLayer,
    TrailLayer,
    _LayerBase,
    precompute_segments_from_pairs,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.backend_bases import FigureCanvasBase
    from mpl_toolkits.mplot3d.axes3d import Axes3D

logger = logging.getLogger(__name__)


def _default_body_segments_safe(marker_names: tuple[str, ...]) -> list[tuple[int, int]]:
    """Return body-skeleton segment index pairs.

    Tries to import ``default_body_segments`` from the body-skeleton
    module first (PR #4496). Falls back to a minimal heuristic — pairs
    of consecutive markers — if the module is not yet on this branch.
    The fallback keeps the live view working even if the upstream PR
    has not merged yet; the test does not depend on the exact pairing.
    """
    try:
        from motion_matching.body_skeleton import (  # type: ignore[import-not-found]
            default_body_segments,
        )

        return list(default_body_segments(marker_names))
    except Exception:  # pragma: no cover - branch-dependent import
        m = len(marker_names)
        return [(i, i + 1) for i in range(m - 1)]


class LiveViewController:
    """Orchestrates rendering of a multi-source target on a 3-D axis.

    The controller owns a stack of :class:`_LayerBase` instances. On
    :meth:`set_target` it tears the old stack down, builds a fresh one
    for the new target, and seeds the axes with frame 0. On
    :meth:`set_frame` it updates each layer in place and asks the
    canvas for a single redraw via ``draw_idle``.

    The controller is GUI-framework-agnostic — it takes a matplotlib
    ``Axes3D`` and a ``FigureCanvasBase`` (the existing matcher already
    uses ``FigureCanvasQTAgg``). Headless tests pass an ``Agg`` canvas.
    """

    def __init__(
        self,
        ax: Axes3D,
        canvas: FigureCanvasBase,
    ) -> None:
        self._ax = ax
        self._canvas = canvas
        self._layers: dict[str, _LayerBase] = {}
        self._n_frames: int = 0
        self._current_frame: int = 0
        self._auto_fit_done: bool = False
        # Cached marker positions so :meth:`equalize_3d_axes` can be
        # called once per target without re-loading.
        self._all_xyz_for_fit: np.ndarray | None = None

    # -- accessors --------------------------------------------------------- #

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def current_frame(self) -> int:
        return self._current_frame

    def layers(self) -> dict[str, _LayerBase]:
        """Return the live layer dict (key -> layer)."""
        return dict(self._layers)

    def has_layer(self, key: str) -> bool:
        return key in self._layers

    # -- target wiring ----------------------------------------------------- #

    def set_target(
        self,
        body: Any | None = None,
        club: Any | None = None,
        ball: Any | None = None,
    ) -> None:
        """Rebuild the layer stack for a new target.

        Parameters
        ----------
        body
            Optional :class:`BodyTarget` (or duck-typed object exposing
            ``marker_xyz`` shape ``(T, M, 3)`` and a ``marker_names``
            tuple).
        club
            Optional :class:`ClubTarget` (or duck-typed: ``mid_hands``,
            ``clubhead``, ``clubface_triad``).
        ball
            Optional ball-impact-state object exposing ``position``
            (shape ``(3,)``) — usually attached to a ``ClubBallTarget``.

        Missing slots are silently skipped: a body-only target produces a
        live view with body markers and a body skeleton only.
        """
        self._teardown_layers()

        layers: dict[str, _LayerBase] = {}
        n_frames = 0
        fit_points: list[np.ndarray] = []

        if body is not None:
            marker_xyz = np.asarray(body.marker_xyz)
            if marker_xyz.ndim != 3 or marker_xyz.shape[2] != 3:
                raise ValueError(
                    f"body.marker_xyz must have shape (T, M, 3), got {marker_xyz.shape}"
                )
            n_frames = max(n_frames, int(marker_xyz.shape[0]))
            layers["body_markers"] = BodyMarkerLayer(
                key="body_markers", marker_xyz=marker_xyz
            )
            pairs = _default_body_segments_safe(tuple(body.marker_names))
            segments = precompute_segments_from_pairs(marker_xyz, pairs)
            layers["body_skeleton"] = BodySkeletonLayer(
                key="body_skeleton", segments=segments
            )
            layers["body_trail"] = TrailLayer(
                key="body_markers",  # share visibility key with markers
                marker_xyz=marker_xyz,
            )
            layers["body_trail"].key = "body_trail"
            finite = np.isfinite(marker_xyz).all(axis=-1)
            if finite.any():
                fit_points.append(marker_xyz[finite].reshape(-1, 3))

        if club is not None:
            mid_hands = _maybe_xyz(club, ("mid_hands", "midhands", "mid_hands_xyz"))
            clubhead = _maybe_xyz(club, ("clubhead", "clubhead_xyz", "head"))
            triad = _maybe_xyz(club, ("clubface_triad", "triad"))
            if mid_hands is not None:
                layers["club_midhands_trace"] = ClubTraceLayer(
                    key="club_midhands_trace",
                    positions=mid_hands,
                    color="tab:orange",
                )
                n_frames = max(n_frames, int(mid_hands.shape[0]))
                fit_points.append(_finite_rows(mid_hands))
            if clubhead is not None:
                layers["clubface_trace"] = ClubTraceLayer(
                    key="clubface_trace",
                    positions=clubhead,
                    color="tab:purple",
                )
                n_frames = max(n_frames, int(clubhead.shape[0]))
                fit_points.append(_finite_rows(clubhead))
            if clubhead is not None and triad is not None and triad.ndim == 3:
                layers["clubface_triad"] = ClubfaceTriadLayer(
                    key="clubface_triad",
                    origin=clubhead,
                    triad=triad,
                )

        if ball is not None:
            pos = getattr(ball, "position", None)
            if pos is not None:
                pos_arr = np.asarray(pos, dtype=float).reshape(-1)
                if pos_arr.shape == (3,):
                    layers["ball_impact"] = BallImpactLayer(
                        key="ball_impact", position=pos_arr
                    )

        # Build artists on the axes.
        for layer in layers.values():
            try:
                layer.build(self._ax)
            except Exception:  # pragma: no cover - defensive
                logger.exception("layer %s build failed", layer.key)

        self._layers = layers
        self._n_frames = n_frames
        self._current_frame = 0

        # Auto-fit the first time we see any data, so the user's first
        # load lands a properly framed view (acceptance criterion).
        if fit_points and not self._auto_fit_done:
            try:
                from motion_matching.diagnostics._skeleton_render import (
                    equalize_3d_axes,
                )

                pts = np.concatenate(fit_points, axis=0)
                if pts.size:
                    equalize_3d_axes(self._ax, pts)
                    self._auto_fit_done = True
            except Exception:  # pragma: no cover - defensive
                logger.exception("equalize_3d_axes failed")

        # Seed frame 0 and request a single redraw.
        self._render_frame(0)

    def clear(self) -> None:
        """Drop all layers and forget any cached target."""
        self._teardown_layers()
        self._n_frames = 0
        self._current_frame = 0
        self._request_redraw()

    # -- frame / visibility ------------------------------------------------ #

    def set_frame(self, frame: int) -> None:
        """Update every layer to ``frame`` and request a redraw."""
        if self._n_frames == 0:
            return
        f = int(np.clip(frame, 0, self._n_frames - 1))
        self._current_frame = f
        self._render_frame(f)

    def set_layer_visible(self, key: str, visible: bool) -> None:
        """Toggle visibility on a layer; immediately redraw the canvas.

        ``key`` is the logical layer name (``"body_markers"``,
        ``"body_skeleton"``, etc.). When toggling ``"body_markers"`` the
        trail layer hides too, since the trail is meaningless without
        the markers.
        """
        if key == "body_markers":
            for k in ("body_markers", "body_trail"):
                layer = self._layers.get(k)
                if layer is not None:
                    layer.set_visible(visible)
        else:
            layer = self._layers.get(key)
            if layer is None:
                return
            layer.set_visible(visible)
        self._request_redraw()

    # -- internals --------------------------------------------------------- #

    def _render_frame(self, frame: int) -> None:
        for layer in self._layers.values():
            try:
                layer.update(frame)
            except Exception:  # pragma: no cover - defensive
                logger.exception("layer %s update failed", layer.key)
        self._request_redraw()

    def _request_redraw(self) -> None:
        try:
            self._canvas.draw_idle()
        except Exception:  # pragma: no cover - defensive
            logger.exception("canvas draw_idle failed")

    def _teardown_layers(self) -> None:
        for layer in self._layers.values():
            for art in layer.artists():
                # Artist may already be gone if the host cleared the
                # axes (e.g. via _setup_axes); not fatal.
                with contextlib.suppress(ValueError, AttributeError):
                    art.remove()
        self._layers = {}


def _maybe_xyz(obj: Any, names: tuple[str, ...]) -> np.ndarray | None:
    """Return the first ``names``-attribute that is a non-empty ndarray."""
    for name in names:
        val = getattr(obj, name, None)
        if val is None:
            continue
        arr = np.asarray(val)
        if arr.size == 0:
            continue
        return arr
    return None


def _finite_rows(positions: np.ndarray) -> np.ndarray:
    """Return only rows of ``positions`` (shape ``(T, 3)``) that are finite."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        return positions.reshape(-1, 3)
    mask = np.isfinite(positions).all(axis=-1)
    if not mask.any():
        return positions[:0]
    return positions[mask]


__all__ = ["LiveViewController"]
