"""Data-driven color resolver — channel value → colormap RGBA.

Implements :class:`~src.shared.python.plot_style.contracts.ColorResolver`
for :class:`~src.shared.python.plot_style.colors.DataDrivenColor`
scales.

Two paths are exposed:

* :meth:`resolve_one` — single ``(frame_idx, marker_idx)`` lookup. Used
  by interactive callers that want a one-shot RGBA without allocating
  a frame-shaped array.
* :meth:`resolve_array` — bulk per-frame (or per-(frame, marker)) RGBA
  array. Pre-computes a 256-entry colormap LUT once, normalises the
  channel values into LUT indices, and gathers in a single vectorised
  step. Designed to clear ≥ 60 fps for 1000 frames × 32 markers.
"""

from __future__ import annotations

import math
from typing import Final, cast

import numpy as np
from matplotlib.colors import is_color_like, to_rgba

from .._types import RGBATuple
from ..colors import DataDrivenColor as DataDrivenColorScale
from ..registry import get_colormap

__all__ = ["DataDrivenColor"]


# Pre-computed LUT resolution. 256 entries matches matplotlib's default
# colormap size and is the inflection point for 8-bit display banding.
_LUT_SIZE: Final[int] = 256


def _hex_to_rgba(hex_value: str) -> RGBATuple:
    """Convert a matplotlib-recognised color string to an ``RGBATuple``."""
    if not is_color_like(hex_value):
        raise ValueError(f"hex_value {hex_value!r} is not a parseable matplotlib color")
    rgba = to_rgba(cast(object, hex_value))  # type: ignore[arg-type]
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))


class DataDrivenColor:
    """Resolver that samples a colormap by normalised channel value.

    Parameters
    ----------
    scale:
        The :class:`DataDrivenColor` scale to bind. The resolver caches
        a colormap LUT and the resolved ``(vmin, vmax)`` bounds at
        construction time so per-frame resolution is allocation-free
        beyond the output array.
    """

    __slots__ = (
        "_lut",
        "_nan_rgba",
        "_scale",
        "_vmax",
        "_vmin",
    )

    def __init__(self, scale: DataDrivenColorScale) -> None:
        if not isinstance(scale, DataDrivenColorScale):
            raise TypeError(
                f"scale must be a colors.DataDrivenColor; got {type(scale).__name__}"
            )
        self._scale = scale
        self._nan_rgba = _hex_to_rgba(scale.nan_color)

        # Resolve bounds eagerly so we can fail fast and so that
        # ``resolve_one`` doesn't pay the cost on every call.
        if scale.vmin is not None and scale.vmax is not None:
            lo, hi = float(scale.vmin), float(scale.vmax)
        else:
            auto_lo, auto_hi = scale.channel.auto_range()
            lo = float(scale.vmin) if scale.vmin is not None else auto_lo
            hi = float(scale.vmax) if scale.vmax is not None else auto_hi
        self._vmin = lo
        self._vmax = hi

        # Pre-compute the LUT once. ``get_colormap`` accepts the enum
        # directly and resolves semantic aliases (VELOCITY -> PLASMA).
        cmap = get_colormap(scale.colormap)
        sample_points = np.linspace(0.0, 1.0, _LUT_SIZE, dtype=np.float64)
        # ``cmap`` returns an ndarray of shape ``(N, 4)`` for an
        # ndarray input — broadcast-friendly for downstream gather.
        self._lut = np.asarray(cmap(sample_points), dtype=np.float64)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_scale(cls, scale: DataDrivenColorScale) -> DataDrivenColor:
        """Construct from a :class:`DataDrivenColor` scale instance."""
        return cls(scale)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vmin(self) -> float:
        """Resolved lower normalisation bound (may be NaN)."""
        return self._vmin

    @property
    def vmax(self) -> float:
        """Resolved upper normalisation bound (may be NaN)."""
        return self._vmax

    @property
    def nan_rgba(self) -> RGBATuple:
        """Cached RGBA used for non-finite values."""
        return self._nan_rgba

    @property
    def lut(self) -> np.ndarray:
        """Read-only view of the pre-computed 256-entry colormap LUT."""
        view = self._lut.view()
        view.flags.writeable = False
        return view

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bounds_degenerate(self) -> bool:
        """Return True iff bounds are non-finite or collapsed."""
        return (
            not math.isfinite(self._vmin)
            or not math.isfinite(self._vmax)
            or self._vmax <= self._vmin
        )

    def _sample_lut(self, normalised: float) -> RGBATuple:
        """Index into the LUT by normalised value in ``[0, 1]``."""
        clipped = float(np.clip(normalised, 0.0, 1.0))
        idx = int(round(clipped * (_LUT_SIZE - 1)))
        rgba = self._lut[idx]
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))

    # ------------------------------------------------------------------
    # ColorResolver Protocol
    # ------------------------------------------------------------------

    def resolve(
        self,
        scale: DataDrivenColorScale,
        frame_idx: int,
        marker_idx: int | None = None,
    ) -> RGBATuple:
        """Per-frame (or per-(frame, marker)) RGBA via LUT lookup.

        Alias for :meth:`resolve_one`; kept under the issue-spec name
        for symmetry with :meth:`resolve_array`.
        """
        return self.resolve_one(scale, frame_idx, marker_idx)

    def resolve_one(
        self,
        scale: DataDrivenColorScale,
        frame_idx: int,
        marker_idx: int | None = None,
    ) -> RGBATuple:
        """Return RGBA at ``(frame_idx, marker_idx)`` via LUT lookup."""
        del scale  # interface compatibility — bound at construction
        if self._bounds_degenerate():
            return self._nan_rgba
        scalar = self._scale.channel.value_at(frame_idx, marker_idx)
        if not math.isfinite(scalar):
            return self._nan_rgba
        normalised = (scalar - self._vmin) / (self._vmax - self._vmin)
        return self._sample_lut(normalised)

    def resolve_array(
        self,
        scale: DataDrivenColorScale,
        n_frames: int,
        n_markers: int | None = None,
    ) -> np.ndarray:
        """Bulk-resolve into ``(n_frames, [n_markers,] 4)`` via LUT gather.

        Slices the channel values once, normalises, clips, rounds to
        integer LUT indices, and gathers. NaN values yield
        :attr:`nan_rgba`.
        """
        del scale  # interface compatibility — bound at construction
        if not isinstance(n_frames, int) or n_frames < 0:
            raise ValueError(f"n_frames must be a non-negative int; got {n_frames!r}")
        if n_markers is not None and (not isinstance(n_markers, int) or n_markers < 0):
            raise ValueError(
                f"n_markers must be a non-negative int or None; got {n_markers!r}"
            )

        nan_rgba = np.asarray(self._nan_rgba, dtype=np.float64)
        out_shape: tuple[int, ...] = (
            (n_frames, 4) if n_markers is None else (n_frames, n_markers, 4)
        )

        if self._bounds_degenerate() or n_frames == 0 or n_markers == 0:
            return np.broadcast_to(nan_rgba, out_shape).copy()

        channel = self._scale.channel
        values = channel.values

        # Build a (n_frames,) or (n_frames, n_markers) slab matching
        # the requested output. Cropping / padding tolerates callers
        # that ask for more frames than the channel actually has.
        if n_markers is None:
            slab = self._slab_per_frame(values, n_frames)
        else:
            slab = self._slab_per_marker(values, n_frames, n_markers)

        finite_mask = np.isfinite(slab)
        normalised = (slab - self._vmin) / (self._vmax - self._vmin)
        normalised = np.clip(normalised, 0.0, 1.0)
        # ``np.where`` keeps NaN-derived entries finite (0.0) so the
        # subsequent integer cast does not raise on platforms that
        # error on NaN -> int conversions.
        normalised = np.where(finite_mask, normalised, 0.0)
        indices = np.rint(normalised * (_LUT_SIZE - 1)).astype(np.int64)
        gathered = self._lut[indices]
        gathered = np.where(finite_mask[..., None], gathered, nan_rgba)
        return np.ascontiguousarray(gathered, dtype=np.float64)

    # ------------------------------------------------------------------
    # Slab helpers
    # ------------------------------------------------------------------

    def _slab_per_frame(self, values: np.ndarray, n_frames: int) -> np.ndarray:
        """Return a 1-D ``(n_frames,)`` slab (NaN-padded if needed)."""
        if values.ndim == 1:
            source = values
        else:
            # Per-frame mean over markers, NaN-respecting via nanmean.
            with np.errstate(invalid="ignore"):
                source = np.nanmean(values, axis=1)
        return self._fit_axis0(source, n_frames)

    def _slab_per_marker(
        self,
        values: np.ndarray,
        n_frames: int,
        n_markers: int,
    ) -> np.ndarray:
        """Return a 2-D ``(n_frames, n_markers)`` slab (NaN-padded)."""
        if values.ndim == 1:
            broadcast = np.broadcast_to(values[:, None], (values.shape[0], n_markers))
            return self._fit_axis0(np.asarray(broadcast), n_frames)
        cropped = values[:n_frames, :n_markers]
        out = np.full((n_frames, n_markers), np.nan, dtype=np.float64)
        out[: cropped.shape[0], : cropped.shape[1]] = cropped
        return out

    @staticmethod
    def _fit_axis0(source: np.ndarray, n_frames: int) -> np.ndarray:
        """Crop or NaN-pad ``source`` along axis 0 to ``n_frames``."""
        actual = source.shape[0]
        if actual == n_frames:
            return np.asarray(source, dtype=np.float64)
        out_shape = (n_frames,) + tuple(source.shape[1:])
        out = np.full(out_shape, np.nan, dtype=np.float64)
        keep = min(actual, n_frames)
        out[:keep] = source[:keep]
        return out
