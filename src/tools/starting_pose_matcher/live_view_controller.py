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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

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

# Body skeleton renderer style. Mirrors
# :data:`session_schema.BODY_SKELETON_STYLES`. Re-declared here as a
# module constant so the controller does not import the schema module.
BodySkeletonStyle = Literal["lines", "library_shapes"]
BODY_SKELETON_STYLES: tuple[BodySkeletonStyle, ...] = ("lines", "library_shapes")
DEFAULT_BODY_SKELETON_STYLE: BodySkeletonStyle = "lines"


@dataclass
class BodyLibraryShapesLayer(_LayerBase):
    """Body skeleton rendered with :mod:`body_part_viz` library shapes.

    Wraps a :class:`body_part_viz.renderers.MatplotlibRenderer` and a list
    of fitted shapes resolved from
    :class:`body_part_viz.asset_library.ShapeLibrary.default`. Per-frame
    updates delegate to ``update_frame(handle)`` on the renderer; the
    axes are never cleared.

    Bindings whose required markers are absent from ``markers_xyz`` are
    silently skipped — the layer renders whichever subset of canonical
    body parts is present in the loaded data.
    """

    marker_xyz: np.ndarray | None = None  # (T, M, 3) — kept for label parity
    marker_names: tuple[str, ...] = ()
    _shape_names: tuple[str, ...] = ()
    _handles: list[str] = field(default_factory=list, repr=False)
    _renderer: Any | None = None  # body_part_viz MatplotlibRenderer
    _ax: Any | None = field(default=None, repr=False)

    def build(self, ax: Axes3D) -> None:
        """Resolve library shapes for the loaded markers and add them.

        Imports body_part_viz lazily so importing this controller does
        not pull matplotlib internals on test paths that do not need it.
        """
        if self.marker_xyz is None or self.marker_xyz.size == 0:
            return
        if not self.marker_names:
            return

        # Lazy imports: keep the rest of the controller usable when
        # body_part_viz is not on the path (legacy tests, pruned envs).
        try:
            from src.shared.python.body_part_viz.asset_library import ShapeLibrary
            from src.shared.python.body_part_viz.fitters.between_two import (
                BetweenTwoMarkersFitter,
            )
            from src.shared.python.body_part_viz.renderers.matplotlib_renderer import (
                MatplotlibRenderer,
            )
            from src.shared.python.body_part_viz.theme import ShapeTheme
        except Exception:  # pragma: no cover - optional dependency
            logger.exception("body_part_viz import failed; library shapes unavailable")
            return

        try:
            library = ShapeLibrary.default()
        except Exception:  # pragma: no cover - missing asset bundle
            logger.exception("ShapeLibrary.default() failed")
            return

        # Build a marker-name -> (T, 3) trajectory dict from the (T, M, 3)
        # tensor. The fitter only reads the names it needs.
        markers_xyz: dict[str, np.ndarray] = {
            name: np.asarray(self.marker_xyz[:, idx, :], dtype=float)
            for idx, name in enumerate(self.marker_names)
        }

        renderer = MatplotlibRenderer(ax)
        fitter = BetweenTwoMarkersFitter()
        # Iterate library names; default theme palette per group keeps
        # the figure visually consistent across body parts.
        theme = ShapeTheme(color="#7dd3fc", opacity=0.55, edge_color="#0c4a6e")

        handles: list[str] = []
        used: list[str] = []
        for name in library.names():
            try:
                binding = library.binding_template(name)
            except Exception:
                logger.debug("library binding for %r unavailable", name, exc_info=True)
                continue
            # Only between-two bindings work without a cluster fitter
            # implementation here. cluster/on-marker shapes are skipped
            # for this integration; #4767 explicitly limits the canonical
            # body parts to between-two segments.
            if binding.kind.value != "between_two":
                continue
            if any(m not in markers_xyz for m in binding.marker_names):
                continue
            try:
                shape = library.get(name)
                fitted = fitter.fit(shape, binding, markers_xyz)
                handle = renderer.add_shape(shape, fitted, theme)
            except Exception:
                logger.debug(
                    "library shape %r could not be fitted/added", name, exc_info=True
                )
                continue
            handles.append(handle)
            used.append(name)

        self._renderer = renderer
        self._handles = handles
        self._shape_names = tuple(used)
        self._ax = ax
        # Expose underlying matplotlib artists so the controller's
        # tear-down + ``set_visible`` paths see them like any other
        # layer's artists.
        self._artists = [
            entry.artist
            for entry in renderer._entries.values()  # noqa: SLF001
        ]
        for art in self._artists:
            art.set_visible(self._visible)

    def update(self, frame: int) -> None:
        if self._renderer is None or not self._handles:
            return
        if self.marker_xyz is None:
            return
        t = int(np.clip(frame, 0, self.marker_xyz.shape[0] - 1))
        for handle in self._handles:
            try:
                self._renderer.update_frame(handle, t)
            except Exception:  # pragma: no cover - defensive
                logger.exception("library-shape update_frame failed")

    @property
    def shape_count(self) -> int:
        """Return the number of library shapes currently rendered."""
        return len(self._handles)

    @property
    def shape_names(self) -> tuple[str, ...]:
        """Return the resolved library shape names in render order."""
        return self._shape_names


def _default_body_segments_safe(marker_names: tuple[str, ...]) -> list[tuple[int, int]]:
    """Return body-skeleton segment index pairs.

    Tries to import ``default_body_segments`` from the body-skeleton
    module first. That helper returns ``BodySegment`` dataclasses keyed
    on marker NAMES — we map back to integer indices into
    ``marker_names`` here so :func:`precompute_segments_from_pairs`
    receives the ``(int, int)`` pairs it expects. Falls back to a minimal
    heuristic (pairs of consecutive markers) when the helper is absent
    or any segment endpoint is missing from ``marker_names``.
    """
    try:
        from src.shared.python.motion_matching.body_skeleton import (
            default_body_segments,
        )

        name_to_idx = {n: i for i, n in enumerate(marker_names)}
        pairs: list[tuple[int, int]] = []
        for seg in default_body_segments(marker_names):
            ia = name_to_idx.get(getattr(seg, "a", None))
            ib = name_to_idx.get(getattr(seg, "b", None))
            if ia is not None and ib is not None:
                pairs.append((ia, ib))
        if pairs:
            return pairs
        # Empty result -> fall through to consecutive-pairs fallback
    except Exception:  # pragma: no cover - branch-dependent import
        pass
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
        body_skeleton_style: BodySkeletonStyle = DEFAULT_BODY_SKELETON_STYLE,
    ) -> None:
        if body_skeleton_style not in BODY_SKELETON_STYLES:
            raise ValueError(
                "body_skeleton_style must be one of "
                f"{BODY_SKELETON_STYLES!r}; got {body_skeleton_style!r}"
            )
        self._ax = ax
        self._canvas = canvas
        self._layers: dict[str, _LayerBase] = {}
        self._n_frames: int = 0
        self._current_frame: int = 0
        self._auto_fit_done: bool = False
        # Cached marker positions so :meth:`equalize_3d_axes` can be
        # called once per target without re-loading.
        self._all_xyz_for_fit: np.ndarray | None = None
        # Currently active body-skeleton renderer style. Switching styles
        # via :meth:`set_body_skeleton_style` is non-destructive — the
        # cached body data is preserved so only the rendering swaps.
        self._body_skeleton_style: BodySkeletonStyle = body_skeleton_style
        self._cached_body: Any | None = None
        self._cached_club: Any | None = None
        self._cached_ball: Any | None = None

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
            marker_names = tuple(body.marker_names)
            if self._body_skeleton_style == "library_shapes":
                layers["body_skeleton"] = BodyLibraryShapesLayer(
                    key="body_skeleton",
                    marker_xyz=marker_xyz,
                    marker_names=marker_names,
                )
            else:
                pairs = _default_body_segments_safe(marker_names)
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
        # Cache the inputs so a non-destructive style swap can rebuild
        # without the caller reloading the C3D / CSV.
        self._cached_body = body
        self._cached_club = club
        self._cached_ball = ball

        # Auto-fit the first time we see any data, so the user's first
        # load lands a properly framed view (acceptance criterion).
        if fit_points and not self._auto_fit_done:
            try:
                from src.shared.python.motion_matching.diagnostics._skeleton_render import (  # noqa: E501
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
        self._cached_body = None
        self._cached_club = None
        self._cached_ball = None
        self._request_redraw()

    @property
    def body_skeleton_style(self) -> BodySkeletonStyle:
        """Return the active body-skeleton renderer style."""
        return self._body_skeleton_style

    def set_body_skeleton_style(self, style: BodySkeletonStyle) -> None:
        """Swap the body-skeleton renderer non-destructively.

        Rebuilds only the body-skeleton layer (keeps the loaded body
        target, the body-marker layer, and the trail layer untouched
        from the caller's perspective). The current frame is preserved.

        ``style`` must be one of :data:`BODY_SKELETON_STYLES`. A no-op
        is performed when ``style`` already matches the active mode.
        """
        if style not in BODY_SKELETON_STYLES:
            raise ValueError(
                f"style must be one of {BODY_SKELETON_STYLES!r}; got {style!r}"
            )
        if style == self._body_skeleton_style:
            return
        self._body_skeleton_style = style
        # If no body has been loaded yet just record the choice; the
        # next ``set_target`` call will pick it up.
        if self._cached_body is None:
            return

        # Non-destructive swap: keep the cached body / club / ball, the
        # current frame, and re-bind a fresh layer stack so the new
        # body_skeleton picks up the new style. Reuses ``set_target``
        # to avoid duplicating the layer-build path.
        prev_frame = self._current_frame
        # ``set_target`` resets ``_current_frame`` to 0; restore it.
        self.set_target(
            body=self._cached_body,
            club=self._cached_club,
            ball=self._cached_ball,
        )
        if self._n_frames:
            self.set_frame(prev_frame)

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


__all__ = [
    "BODY_SKELETON_STYLES",
    "DEFAULT_BODY_SKELETON_STYLE",
    "BodyLibraryShapesLayer",
    "BodySkeletonStyle",
    "LiveViewController",
]
