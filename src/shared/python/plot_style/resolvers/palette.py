"""Palette color resolver — categorical lookup into a named palette.

Implements :class:`~src.shared.python.plot_style.contracts.ColorResolver`
for :class:`~src.shared.python.plot_style.colors.PaletteColor` scales.
The resolver pulls from matplotlib's built-in qualitative palettes
(``tab10``, ``Set2``, ...) plus any custom colormap registered through
:func:`~src.shared.python.plot_style.registry.register_custom_colormap`.

Out-of-bounds indices raise :class:`IndexError` with a descriptive
message — palette lookups are categorical and silent wrap-around hides
configuration mistakes.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Colormap

from .._types import RGBATuple
from ..colors import PaletteColor as PaletteColorScale
from ..registry import _CUSTOM_COLORMAPS  # noqa: PLC2701  — internal lookup OK

__all__ = ["PaletteColor"]


def _resolve_palette(palette_name: str) -> Colormap:
    """Return a matplotlib :class:`Colormap` for ``palette_name``.

    Searches matplotlib's built-in registry first, then the custom
    colormap registry. Raises :class:`KeyError` if neither contains the
    requested name.
    """
    try:
        return cast(Colormap, colormaps[palette_name])
    except (KeyError, ValueError):
        pass
    if palette_name in _CUSTOM_COLORMAPS:
        return _CUSTOM_COLORMAPS[palette_name][1]
    raise KeyError(
        f"palette {palette_name!r} is not a matplotlib colormap nor a "
        "registered custom colormap"
    )


def _palette_size(cmap: Colormap) -> int:
    """Return the number of categorical entries in ``cmap``.

    Falls back to ``256`` for continuous colormaps (matplotlib's default
    ``N`` for non-qualitative maps).
    """
    n = getattr(cmap, "N", None)
    if n is None or n <= 0:
        return 256
    return int(n)


class PaletteColor:
    """Resolver that picks RGBA values from a named palette by index.

    Parameters
    ----------
    palette_name:
        Name of a matplotlib colormap (qualitative or continuous) or a
        custom colormap registered via
        :func:`register_custom_colormap`.
    palette_index:
        Non-negative integer index into the palette. Indices ``>=`` the
        palette size raise :class:`IndexError` rather than wrapping.
    """

    __slots__ = ("_palette_index", "_palette_name", "_rgba", "_size")

    def __init__(self, palette_name: str, palette_index: int) -> None:
        if not isinstance(palette_name, str) or not palette_name:
            raise ValueError(
                f"palette_name must be a non-empty string; got {palette_name!r}"
            )
        if not isinstance(palette_index, int) or isinstance(palette_index, bool):
            raise TypeError(
                f"palette_index must be int; got {type(palette_index).__name__}"
            )
        if palette_index < 0:
            raise ValueError(f"palette_index must be non-negative; got {palette_index}")

        cmap = _resolve_palette(palette_name)
        size = _palette_size(cmap)
        if palette_index >= size:
            raise IndexError(
                f"palette_index {palette_index} is out of range for palette "
                f"{palette_name!r} of size {size}"
            )
        rgba = cmap(palette_index)
        self._palette_name = palette_name
        self._palette_index = palette_index
        self._size = size
        self._rgba = (
            float(rgba[0]),
            float(rgba[1]),
            float(rgba[2]),
            float(rgba[3]),
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_scale(cls, scale: PaletteColorScale) -> PaletteColor:
        """Construct from a :class:`PaletteColor` scale instance.

        The scale dataclass permits modulo wrap; the resolver does not.
        Wrap-eligible indices that exceed the palette size are
        normalised here to preserve scale-level semantics.
        """
        if not isinstance(scale, PaletteColorScale):
            raise TypeError(
                f"scale must be a colors.PaletteColor; got {type(scale).__name__}"
            )
        cmap = _resolve_palette(scale.palette_name)
        size = _palette_size(cmap)
        normalised_index = scale.palette_index % size
        return cls(scale.palette_name, normalised_index)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def palette_name(self) -> str:
        """The configured palette name."""
        return self._palette_name

    @property
    def palette_index(self) -> int:
        """The configured palette index."""
        return self._palette_index

    @property
    def palette_size(self) -> int:
        """The number of categorical entries in the resolved palette."""
        return self._size

    @property
    def rgba(self) -> RGBATuple:
        """The cached RGBA tuple at ``palette_index``."""
        return self._rgba

    # ------------------------------------------------------------------
    # ColorResolver Protocol
    # ------------------------------------------------------------------

    def resolve_one(
        self,
        scale: PaletteColorScale,
        frame_idx: int,
        marker_idx: int | None = None,
    ) -> RGBATuple:
        """Return the cached RGBA at ``palette_index``."""
        del scale, frame_idx, marker_idx  # interface compatibility
        return self._rgba

    def resolve_array(
        self,
        scale: PaletteColorScale,
        n_frames: int,
        n_markers: int | None = None,
    ) -> np.ndarray:
        """Broadcast the palette colour into ``(n_frames, [n_markers,] 4)``."""
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
