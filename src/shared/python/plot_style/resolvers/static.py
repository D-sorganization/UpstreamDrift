"""Static color resolver — passes a single hex / RGBA through unchanged.

Implements :class:`~src.shared.python.plot_style.contracts.ColorResolver`
for :class:`~src.shared.python.plot_style.colors.StaticColor` scales.
The resolver is stateless beyond a cached RGBA tuple; both per-point and
bulk paths broadcast the same color.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from matplotlib.colors import is_color_like, to_rgba

from .._types import RGBATuple
from ..colors import StaticColor as StaticColorScale

__all__ = ["StaticColor"]


def _hex_to_rgba(hex_value: str) -> RGBATuple:
    """Convert a matplotlib-recognised color string to an ``RGBATuple``.

    Mirrors :func:`src.shared.python.plot_style.colors._hex_to_rgba` but
    is duplicated here to keep the resolvers package free of private
    cross-module imports.
    """
    if not isinstance(hex_value, str) or not hex_value:
        raise ValueError(f"hex_value must be a non-empty string; got {hex_value!r}")
    if not is_color_like(hex_value):
        raise ValueError(f"hex_value {hex_value!r} is not a parseable matplotlib color")
    rgba = to_rgba(cast(object, hex_value))  # type: ignore[arg-type]
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))


class StaticColor:
    """Resolver that returns a single constant RGBA for every query.

    The resolver may be constructed directly from a hex / matplotlib
    color string (the common case) or from a pre-existing
    :class:`~src.shared.python.plot_style.colors.StaticColor` scale via
    :meth:`from_scale`.

    Parameters
    ----------
    hex_value:
        Any matplotlib-recognised color string. Validated eagerly so a
        misconfigured style fails loudly at construction time rather
        than at the first frame.
    """

    __slots__ = ("_hex_value", "_rgba")

    def __init__(self, hex_value: str) -> None:
        self._rgba = _hex_to_rgba(hex_value)
        self._hex_value = hex_value

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_scale(cls, scale: StaticColorScale) -> StaticColor:
        """Construct from a :class:`StaticColor` scale instance."""
        if not isinstance(scale, StaticColorScale):
            raise TypeError(
                f"scale must be a colors.StaticColor; got {type(scale).__name__}"
            )
        return cls(scale.hex_value)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hex_value(self) -> str:
        """The configured matplotlib color string."""
        return self._hex_value

    @property
    def rgba(self) -> RGBATuple:
        """The cached RGBA tuple."""
        return self._rgba

    # ------------------------------------------------------------------
    # ColorResolver Protocol
    # ------------------------------------------------------------------

    def resolve_one(
        self,
        scale: StaticColorScale,
        frame_idx: int,
        marker_idx: int | None = None,
    ) -> RGBATuple:
        """Return the cached RGBA, ignoring all positional indices."""
        del scale, frame_idx, marker_idx  # interface compatibility
        return self._rgba

    def resolve_array(
        self,
        scale: StaticColorScale,
        n_frames: int,
        n_markers: int | None = None,
    ) -> np.ndarray:
        """Broadcast the constant RGBA into ``(n_frames, [n_markers,] 4)``."""
        del scale  # interface compatibility — colour comes from self
        if not isinstance(n_frames, int) or n_frames < 0:
            raise ValueError(f"n_frames must be a non-negative int; got {n_frames!r}")
        if n_markers is not None and (not isinstance(n_markers, int) or n_markers < 0):
            raise ValueError(
                f"n_markers must be a non-negative int or None; got {n_markers!r}"
            )
        rgba = np.asarray(self._rgba, dtype=np.float64)
        if n_markers is None:
            out = np.broadcast_to(rgba, (n_frames, 4)).copy()
        else:
            out = np.broadcast_to(rgba, (n_frames, n_markers, 4)).copy()
        return out
