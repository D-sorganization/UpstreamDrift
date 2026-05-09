"""Color-scale dataclasses for marker fill colors.

A *color scale* is anything that, given a ``(frame_idx, marker_idx)``
pair, produces an ``(r, g, b, a)`` tuple in ``[0, 1]``. Three concrete
variants are supported:

* :class:`StaticColor` — single constant color.
* :class:`PaletteColor` — categorical pick from a named palette.
* :class:`DataDrivenColor` — channel value normalised through a
  colormap.

The :data:`ColorScale` alias unifies the three variants.

Design-by-Contract
------------------
Each dataclass is frozen, validates its arguments in ``__post_init__``,
and provides a ``resolve(frame_idx, marker_idx)`` method that returns a
4-tuple of floats in ``[0, 1]``. ``resolve`` is total — non-finite
inputs map to the ``nan_color`` (for :class:`DataDrivenColor`) or to the
configured static color.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from matplotlib import cm, colormaps
from matplotlib.colors import Colormap, is_color_like, to_rgba

from ._types import RGBATuple
from .channels import DataChannel
from .colormaps import ColormapId, resolve_colormap_alias

__all__ = [
    "ColorScale",
    "DataDrivenColor",
    "PaletteColor",
    "StaticColor",
]


def _hex_to_rgba(hex_value: str) -> RGBATuple:
    """Convert a matplotlib-recognised color string to an ``RGBATuple``."""
    # matplotlib stubs are over-narrow on the input type — to_rgba accepts
    # any color-like (str / tuple / ndarray) at runtime.
    rgba = to_rgba(cast(Any, hex_value))
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))


def _get_matplotlib_colormap(cmap_id: ColormapId) -> Colormap:
    """Resolve a :class:`ColormapId` (incl. semantic aliases) to a Colormap."""
    resolved = resolve_colormap_alias(cmap_id)
    try:
        return cast(Colormap, colormaps[resolved.value])
    except (KeyError, ValueError):
        # Fallback: matplotlib older API
        return cast(Colormap, cm.get_cmap(resolved.value))


@dataclass(frozen=True)
class StaticColor:
    """A constant color shared by every marker on every frame.

    Attributes
    ----------
    hex_value:
        Any matplotlib-recognised color string (hex, name, rgb tuple as
        string, etc.).
    """

    hex_value: str

    def __post_init__(self) -> None:
        if not isinstance(self.hex_value, str) or not self.hex_value:
            raise ValueError(
                f"hex_value must be a non-empty string; got {self.hex_value!r}"
            )
        if not is_color_like(self.hex_value):
            raise ValueError(
                f"hex_value {self.hex_value!r} is not a parseable matplotlib color"
            )

    def resolve(
        self,
        frame_idx: int,
        marker_idx: int | None = None,
    ) -> RGBATuple:
        """Return the constant ``(r, g, b, a)`` tuple in ``[0, 1]``.

        Parameters are accepted for interface compatibility and ignored.
        """
        del frame_idx, marker_idx  # interface compatibility
        return _hex_to_rgba(self.hex_value)


@dataclass(frozen=True)
class PaletteColor:
    """A color picked from a named matplotlib qualitative palette.

    Attributes
    ----------
    palette_name:
        Name of a matplotlib colormap (e.g. ``"tab10"``, ``"Set2"``).
    palette_index:
        Non-negative integer index into the palette. Matplotlib palettes
        wrap modulo the palette size; we record the raw index and let
        the colormap handle wrap so palette-index 12 in ``"tab10"`` is
        deterministic.
    """

    palette_name: str
    palette_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.palette_name, str) or not self.palette_name:
            raise ValueError(
                f"palette_name must be a non-empty string; got {self.palette_name!r}"
            )
        if not isinstance(self.palette_index, int) or isinstance(
            self.palette_index, bool
        ):
            raise TypeError(
                f"palette_index must be int; got {type(self.palette_index).__name__}"
            )
        if self.palette_index < 0:
            raise ValueError(
                f"palette_index must be non-negative; got {self.palette_index}"
            )
        # Verify the palette actually exists.
        try:
            _ = colormaps[self.palette_name]
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"palette_name {self.palette_name!r} is not a registered "
                "matplotlib colormap"
            ) from exc

    def resolve(
        self,
        frame_idx: int,
        marker_idx: int | None = None,
    ) -> RGBATuple:
        """Return the palette color at ``palette_index`` (modulo palette size)."""
        del frame_idx, marker_idx  # interface compatibility
        cmap = cast(Colormap, colormaps[self.palette_name])
        size = getattr(cmap, "N", 10) or 10
        index = self.palette_index % size
        rgba = cmap(index)
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))


@dataclass(frozen=True)
class DataDrivenColor:
    """Color sampled from a colormap by normalised channel value.

    For each ``(frame_idx, marker_idx)`` query, the channel scalar is
    looked up, normalised against ``vmin`` / ``vmax`` (auto-detected
    from the channel if either is ``None``), clamped to ``[0, 1]``, and
    used to sample :attr:`colormap`. Non-finite values map to
    :attr:`nan_color`.

    Attributes
    ----------
    channel:
        Source :class:`DataChannel`.
    colormap:
        Colormap identifier (built-in or semantic alias).
    vmin, vmax:
        Optional explicit normalisation bounds. ``None`` means
        auto-detect via :meth:`DataChannel.auto_range`.
    nan_color:
        Color (matplotlib-recognised string) used for non-finite values
        and the case when ``vmin == vmax`` (degenerate range).
    """

    channel: DataChannel
    colormap: ColormapId
    vmin: float | None = None
    vmax: float | None = None
    nan_color: str = "#888888"

    def __post_init__(self) -> None:
        if not isinstance(self.channel, DataChannel):
            raise TypeError(
                f"channel must be DataChannel; got {type(self.channel).__name__}"
            )
        if not isinstance(self.colormap, ColormapId):
            raise TypeError(
                f"colormap must be ColormapId; got {type(self.colormap).__name__}"
            )
        for attr_name, attr_val in (("vmin", self.vmin), ("vmax", self.vmax)):
            if attr_val is None:
                continue
            if not isinstance(attr_val, (int, float)) or isinstance(attr_val, bool):
                raise TypeError(
                    f"{attr_name} must be numeric or None; got {attr_val!r}"
                )
            if not math.isfinite(float(attr_val)):
                raise ValueError(
                    f"{attr_name} must be finite when supplied; got {attr_val!r}"
                )
        if (
            self.vmin is not None
            and self.vmax is not None
            and float(self.vmax) <= float(self.vmin)
        ):
            raise ValueError(
                "vmax must be strictly greater than vmin when both are set; "
                f"got vmin={self.vmin}, vmax={self.vmax}"
            )
        if not isinstance(self.nan_color, str) or not self.nan_color:
            raise ValueError(
                f"nan_color must be a non-empty string; got {self.nan_color!r}"
            )
        if not is_color_like(self.nan_color):
            raise ValueError(
                f"nan_color {self.nan_color!r} is not a parseable matplotlib color"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolved_bounds(self) -> tuple[float, float]:
        """Return ``(vmin, vmax)`` with auto-detect filled in.

        Both values may be NaN if the channel has no finite samples.
        """
        if self.vmin is not None and self.vmax is not None:
            return (float(self.vmin), float(self.vmax))
        auto_lo, auto_hi = self.channel.auto_range()
        lo = float(self.vmin) if self.vmin is not None else auto_lo
        hi = float(self.vmax) if self.vmax is not None else auto_hi
        return (lo, hi)

    # ------------------------------------------------------------------
    # Resolve
    # ------------------------------------------------------------------

    def resolve(
        self,
        frame_idx: int,
        marker_idx: int | None = None,
    ) -> RGBATuple:
        """Return the ``(r, g, b, a)`` color for the given index pair."""
        scalar = self.channel.value_at(frame_idx, marker_idx)
        if not math.isfinite(scalar):
            return _hex_to_rgba(self.nan_color)
        lo, hi = self._resolved_bounds()
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            return _hex_to_rgba(self.nan_color)
        normalised = (scalar - lo) / (hi - lo)
        normalised = float(np.clip(normalised, 0.0, 1.0))
        cmap = _get_matplotlib_colormap(self.colormap)
        rgba = cmap(normalised)
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))


# Public union over the three variants.
ColorScale = StaticColor | PaletteColor | DataDrivenColor
