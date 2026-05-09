"""Matplotlib backend for the :class:`MarkerRenderer` Protocol.

Implements :class:`MatplotlibMarkerRenderer` with backends for both 2D
``Axes`` and 3D ``Axes3D``. The class satisfies the stateful
:class:`~src.shared.python.plot_style.contracts.MarkerRenderer` Protocol
(``add_markers`` / ``update_frame`` / ``update_style`` / ``set_visible`` /
``remove``) and additionally exposes a stateless :meth:`draw` helper for
callers that already have a resolved RGBA array in hand.

Built-in shapes are mapped to matplotlib's ``marker=`` parameter wherever
possible (sphere -> 'o', cube -> 's', cross -> 'x', star -> '*',
diamond -> 'D', plus -> '+', point -> '.'). For
:class:`CustomMeshSpec` markers, the 3D backend draws each marker via
``ax.plot_trisurf`` (the slow path), and the 2D backend falls back to a
plain 'o' with a warning.
"""

from __future__ import annotations

import contextlib
import logging
import uuid
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Path3DCollection, Poly3DCollection

from ..colors import DataDrivenColor, PaletteColor, StaticColor
from ..markers import CustomMeshSpec, MarkerShape, MarkerStyle

if TYPE_CHECKING:
    pass

__all__ = ["MatplotlibMarkerRenderer"]

logger = logging.getLogger(__name__)


# Mapping from MarkerShape enum -> matplotlib `marker=` glyph.
_MPL_MARKER_GLYPHS: dict[MarkerShape, str] = {
    MarkerShape.SPHERE: "o",
    MarkerShape.CUBE: "s",
    MarkerShape.CROSS: "x",
    MarkerShape.STAR: "*",
    MarkerShape.DIAMOND: "D",
    MarkerShape.PLUS: "+",
    MarkerShape.POINT: ".",
}


def _shape_to_glyph(shape: MarkerShape) -> str:
    """Return the matplotlib marker glyph for a built-in shape."""
    return _MPL_MARKER_GLYPHS.get(shape, "o")


def _resolve_rgba_array(
    style: MarkerStyle,
    n_markers: int,
    n_frames: int = 1,
) -> np.ndarray:
    """Resolve ``style.fill_color`` into a per-marker ``(N, 4)`` RGBA array.

    For static / palette colors the output is broadcast to all markers.
    For data-driven colors the channel is sampled at frame 0 marker-by-marker
    (used as a default when no pre-resolved array is supplied).
    """
    fill = style.fill_color
    rgba = np.zeros((n_markers, 4), dtype=np.float64)
    if isinstance(fill, (StaticColor, PaletteColor)):
        single = fill.resolve(0, 0)
        rgba[:, :] = np.asarray(single, dtype=np.float64)
    elif isinstance(fill, DataDrivenColor):
        for m in range(n_markers):
            rgba[m, :] = np.asarray(fill.resolve(0, m), dtype=np.float64)
    else:  # pragma: no cover - guarded by MarkerStyle.__post_init__
        raise TypeError(f"unsupported ColorScale variant: {type(fill).__name__}")
    rgba[:, 3] *= float(style.opacity)
    np.clip(rgba, 0.0, 1.0, out=rgba)
    del n_frames
    return rgba


def _is_3d_axes(ax: Axes) -> bool:
    """Return ``True`` if ``ax`` is a 3D ``Axes3D`` instance."""
    return isinstance(ax, Axes3D)


def _validate_positions(
    positions: np.ndarray, *, expected_dim: int | None = None
) -> np.ndarray:
    """Validate and reshape position array to ``(N, D)`` with D in {2, 3}."""
    if not isinstance(positions, np.ndarray):
        raise TypeError(
            f"positions must be numpy.ndarray; got {type(positions).__name__}"
        )
    if positions.ndim == 1:
        # Single point. Auto-promote.
        positions = positions.reshape(1, -1)
    if positions.ndim != 2:
        raise ValueError(f"positions must be 1-D or 2-D; got shape={positions.shape}")
    dim = positions.shape[1]
    if dim not in (2, 3):
        raise ValueError(f"positions must have 2 or 3 columns; got {dim}")
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(
            f"axis dimensionality mismatch: ax expects {expected_dim}D "
            f"positions, got {dim}D"
        )
    return positions


def _validate_colors(colors: np.ndarray, n_markers: int) -> np.ndarray:
    """Validate ``colors`` against ``n_markers`` and return as ``(N, 4)``."""
    if not isinstance(colors, np.ndarray):
        raise TypeError(f"colors must be numpy.ndarray; got {type(colors).__name__}")
    if colors.ndim != 2 or colors.shape[1] != 4:
        raise ValueError(f"colors must have shape (N, 4); got {colors.shape}")
    if colors.shape[0] != n_markers:
        raise ValueError(
            "colors and positions length mismatch: "
            f"colors={colors.shape[0]}, positions={n_markers}"
        )
    return colors.astype(np.float64, copy=False)


def _make_unit_cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices, faces)`` for a unit cube of half-extent 1."""
    v = np.array(
        [
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1],
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom
            [4, 6, 5],
            [4, 7, 6],  # top
            [0, 4, 5],
            [0, 5, 1],  # front
            [1, 5, 6],
            [1, 6, 2],  # right
            [2, 6, 7],
            [2, 7, 3],  # back
            [3, 7, 4],
            [3, 4, 0],  # left
        ],
        dtype=np.int64,
    )
    return v, f


@dataclass
class _Handle:
    """Internal record for a registered marker primitive."""

    handle_id: str
    style: MarkerStyle
    positions: np.ndarray  # (T, M, D) or (T, D)
    label: str
    artists: list[Artist] = field(default_factory=list)
    visible: bool = True
    ax: Axes | None = None


class MatplotlibMarkerRenderer:
    """Matplotlib-backed :class:`MarkerRenderer`.

    Constructable with a single ``ax`` (the default scene axes) or
    used purely as a stateless renderer via :meth:`draw`.

    Parameters
    ----------
    ax:
        Default axes used by :meth:`add_markers`. Optional — a per-call
        ``ax`` may be supplied to :meth:`draw` instead.
    """

    def __init__(self, ax: Axes | None = None) -> None:
        if ax is not None and not isinstance(ax, Axes):
            raise TypeError(
                f"ax must be matplotlib Axes or None; got {type(ax).__name__}"
            )
        self._default_ax = ax
        self._handles: dict[str, _Handle] = {}

    # ------------------------------------------------------------------
    # Stateless draw helper
    # ------------------------------------------------------------------

    def draw(
        self,
        ax: Axes,
        positions: np.ndarray,
        style: MarkerStyle,
        colors: np.ndarray,
    ) -> Artist | list[Artist]:
        """Draw markers at ``positions`` with given ``style`` and ``colors``.

        Parameters
        ----------
        ax:
            Matplotlib ``Axes`` (2D) or ``Axes3D`` (3D) to draw on.
        positions:
            ``(N, 2)`` (2D) or ``(N, 3)`` (3D) ndarray of marker world
            positions.
        style:
            :class:`MarkerStyle` driving glyph / size / edges.
        colors:
            ``(N, 4)`` RGBA array, components in ``[0, 1]``. Typically
            produced by a :class:`ColorResolver`.

        Returns
        -------
        artist:
            For built-in shapes a single ``PathCollection`` /
            ``Path3DCollection``. For ``CUSTOM_MESH`` in 3D, a list of
            per-marker ``Poly3DCollection`` artists.
        """
        if not isinstance(ax, Axes):
            raise TypeError(f"ax must be matplotlib Axes; got {type(ax).__name__}")
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")
        is_3d = _is_3d_axes(ax)
        positions = _validate_positions(positions, expected_dim=3 if is_3d else 2)
        n = positions.shape[0]
        colors = _validate_colors(colors, n)

        if style.shape is MarkerShape.CUSTOM_MESH:
            mesh = style.custom_mesh
            assert mesh is not None  # enforced by MarkerStyle.__post_init__
            if is_3d:
                return self._draw_custom_mesh_3d(
                    cast(Axes3D, ax), positions, style, colors, mesh
                )
            warnings.warn(
                "MatplotlibMarkerRenderer: CUSTOM_MESH not supported in 2D; "
                "falling back to 'o' marker.",
                stacklevel=2,
            )
            return self._draw_scatter(ax, positions, style, colors, glyph="o")

        glyph = _shape_to_glyph(style.shape)
        return self._draw_scatter(ax, positions, style, colors, glyph=glyph)

    # ------------------------------------------------------------------
    # MarkerRenderer Protocol
    # ------------------------------------------------------------------

    def add_markers(
        self,
        positions: np.ndarray,
        style: MarkerStyle,
        label: str = "",
    ) -> str:
        """Add a marker primitive to the default axes and return a handle.

        ``positions`` may be ``(T, M, D)`` (per-frame), ``(T, D)`` (single
        marker per frame), or ``(M, D)`` / ``(D,)`` (single static frame).
        """
        if self._default_ax is None:
            raise RuntimeError(
                "MatplotlibMarkerRenderer.add_markers requires a default "
                "ax — pass one to the constructor."
            )
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")
        if not isinstance(positions, np.ndarray):
            raise TypeError(
                f"positions must be numpy.ndarray; got {type(positions).__name__}"
            )
        norm = self._normalise_positions(positions)
        ax = self._default_ax
        is_3d = _is_3d_axes(ax)
        expected_d = 3 if is_3d else 2
        if norm.shape[-1] != expected_d:
            raise ValueError(
                f"positions last-dim must be {expected_d} for this axes; "
                f"got {norm.shape[-1]}"
            )

        frame0 = norm[0]  # (M, D)
        if frame0.ndim == 1:
            frame0 = frame0.reshape(1, -1)
        n = frame0.shape[0]
        rgba = _resolve_rgba_array(style, n)
        artists_raw = self.draw(ax, frame0, style, rgba)
        artists: list[Artist] = (
            artists_raw if isinstance(artists_raw, list) else [artists_raw]
        )

        handle_id = uuid.uuid4().hex
        record = _Handle(
            handle_id=handle_id,
            style=style,
            positions=norm,
            label=label,
            artists=artists,
            ax=ax,
        )
        self._handles[handle_id] = record
        return handle_id

    def update_frame(self, handle: str, frame_idx: int) -> None:
        """Move the markers identified by ``handle`` to ``frame_idx``."""
        record = self._require(handle)
        positions = record.positions
        if positions.ndim < 2:
            return
        n_frames = positions.shape[0]
        if frame_idx < 0 or frame_idx >= n_frames:
            raise IndexError(f"frame_idx {frame_idx} out of bounds [0, {n_frames})")
        frame = positions[frame_idx]
        if frame.ndim == 1:
            frame = frame.reshape(1, -1)
        # For built-in shapes there is exactly one PathCollection.
        if record.style.shape is MarkerShape.CUSTOM_MESH:
            # Slow path: rebuild custom mesh artists from scratch.
            for art in record.artists:
                art.remove()
            ax = record.ax
            assert ax is not None
            n = frame.shape[0]
            rgba = _resolve_rgba_array(record.style, n)
            new_artists_raw = self.draw(ax, frame, record.style, rgba)
            record.artists = (
                new_artists_raw
                if isinstance(new_artists_raw, list)
                else [new_artists_raw]
            )
            return
        artist = record.artists[0]
        if isinstance(artist, Path3DCollection):
            # 3D scatter: use private API exposed by mpl.
            artist._offsets3d = (  # type: ignore[attr-defined]
                frame[:, 0],
                frame[:, 1],
                frame[:, 2],
            )
        elif isinstance(artist, PathCollection):
            artist.set_offsets(frame[:, :2])

    def update_style(self, handle: str, style: MarkerStyle) -> None:
        """Replace the style applied to ``handle``."""
        record = self._require(handle)
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")
        for art in record.artists:
            art.remove()
        record.style = style
        ax = record.ax
        assert ax is not None
        frame0 = record.positions[0]
        if frame0.ndim == 1:
            frame0 = frame0.reshape(1, -1)
        n = frame0.shape[0]
        rgba = _resolve_rgba_array(style, n)
        new_artists_raw = self.draw(ax, frame0, style, rgba)
        record.artists = (
            new_artists_raw if isinstance(new_artists_raw, list) else [new_artists_raw]
        )
        for art in record.artists:
            art.set_visible(record.visible)

    def set_visible(self, handle: str, visible: bool) -> None:
        """Show or hide the markers identified by ``handle``."""
        record = self._require(handle)
        record.visible = bool(visible)
        for art in record.artists:
            art.set_visible(record.visible)

    def remove(self, handle: str) -> None:
        """Remove the markers identified by ``handle`` from the scene."""
        record = self._require(handle)
        for art in record.artists:
            with contextlib.suppress(ValueError, NotImplementedError):
                art.remove()
        del self._handles[handle]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require(self, handle: str) -> _Handle:
        if handle not in self._handles:
            raise KeyError(f"unknown handle: {handle!r}")
        return self._handles[handle]

    @staticmethod
    def _normalise_positions(positions: np.ndarray) -> np.ndarray:
        """Reshape positions to ``(T, M, D)``."""
        if positions.ndim == 1:
            # Single static marker, single frame.
            return positions.reshape(1, 1, -1)
        if positions.ndim == 2:
            # (M, D) interpreted as a single frame.
            return positions.reshape(1, *positions.shape)
        if positions.ndim == 3:
            return positions
        raise ValueError(
            f"positions must have ndim in {{1, 2, 3}}; got shape={positions.shape}"
        )

    @staticmethod
    def _draw_scatter(
        ax: Axes,
        positions: np.ndarray,
        style: MarkerStyle,
        colors: np.ndarray,
        *,
        glyph: str,
    ) -> Artist:
        """Draw a scatter on ``ax`` using a single matplotlib glyph."""
        size_pts2 = float(style.size_px) ** 2
        kwargs: dict[str, Any] = {
            "s": size_pts2,
            "c": colors,
            "marker": glyph,
            "edgecolors": style.edge_color,
            "linewidths": float(style.edge_width),
        }
        if _is_3d_axes(ax):
            ax3 = cast(Axes3D, ax)
            artist = ax3.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                **kwargs,
            )
        else:
            artist = ax.scatter(positions[:, 0], positions[:, 1], **kwargs)
        return cast(Artist, artist)

    @staticmethod
    def _draw_custom_mesh_3d(
        ax: Axes3D,
        positions: np.ndarray,
        style: MarkerStyle,
        colors: np.ndarray,
        mesh: CustomMeshSpec,
    ) -> list[Artist]:
        """Draw one ``plot_trisurf`` per marker for a custom mesh."""
        radius = float(style.size_px) / 2.0
        verts = mesh.vertices.astype(np.float64, copy=False)
        # Normalise so bounding sphere = 1, then scale by radius.
        centroid = (verts.min(axis=0) + verts.max(axis=0)) * 0.5
        centred = verts - centroid
        radii = np.linalg.norm(centred, axis=1)
        max_r = float(radii.max()) if radii.size else 1.0
        if max_r <= 0.0:
            max_r = 1.0
        unit = centred / max_r
        scaled = unit * radius

        faces = mesh.faces.astype(np.int64, copy=False)
        artists: list[Artist] = []
        for i in range(positions.shape[0]):
            cx, cy, cz = positions[i]
            v = scaled + np.array([cx, cy, cz], dtype=np.float64)
            color = tuple(float(x) for x in colors[i])
            tri = ax.plot_trisurf(
                v[:, 0],
                v[:, 1],
                v[:, 2],
                triangles=faces,
                color=color,
                edgecolor=style.edge_color,
                linewidth=float(style.edge_width),
                shade=False,
            )
            artists.append(cast(Artist, tri))
        # Reference for type checkers / future use.
        _ = _make_unit_cube_mesh
        return artists


# Re-export of an internal helper so static analyzers do not flag
# `_make_unit_cube_mesh` as unused (kept as a quick fallback for future
# CUSTOM_MESH default geometry).
_ = Poly3DCollection
