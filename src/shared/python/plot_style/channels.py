"""Per-frame (and per-(frame, marker)) scalar data channels.

A :class:`DataChannel` wraps a numeric ``numpy`` array of shape ``(T,)``
or ``(T, M)`` and exposes safe ``value_at`` / ``auto_range`` helpers used
by :class:`~src.shared.python.plot_style.colors.DataDrivenColor`.

Design-by-Contract
------------------
``DataChannel`` is frozen and validates every field in ``__post_init__``.
``value_at`` returns ``NaN`` for out-of-bounds indices instead of
raising — callers may safely vectorise without bounds-guarding.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DataChannel",
    "derivative_channel",
    "magnitude_channel",
    "slice_channel",
]


def _is_numeric_dtype(dtype: np.dtype[Any]) -> bool:
    """Return True if ``dtype`` is a real numeric kind (i, u, f)."""
    return dtype.kind in ("i", "u", "f")


@dataclass(frozen=True)
class DataChannel:
    """A per-frame (or per-(frame, marker)) scalar source.

    Attributes
    ----------
    name:
        Non-empty unique identifier (e.g. ``"clubhead_speed"``).
    values:
        ``numpy.ndarray`` with ``ndim`` in ``{1, 2}``. A 1-D array of
        shape ``(T,)`` denotes one scalar per frame. A 2-D array of
        shape ``(T, M)`` denotes one scalar per (frame, marker).
    unit:
        Display unit string. May be empty.
    """

    name: str
    values: np.ndarray = field(repr=False)
    unit: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty string; got {self.name!r}")
        if not isinstance(self.unit, str):
            raise TypeError(f"unit must be a string; got {type(self.unit).__name__}")
        if not isinstance(self.values, np.ndarray):
            raise TypeError(
                f"values must be a numpy.ndarray; got {type(self.values).__name__}"
            )
        if self.values.ndim not in (1, 2):
            raise ValueError(
                "values must have ndim in {1, 2}; "
                f"got ndim={self.values.ndim} (shape={self.values.shape})"
            )
        if not _is_numeric_dtype(self.values.dtype):
            raise TypeError(
                "values dtype must be numeric (int / uint / float); "
                f"got {self.values.dtype}"
            )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_array(
        cls,
        name: str,
        values: np.ndarray,
        unit: str = "",
    ) -> DataChannel:
        """Construct a :class:`DataChannel` from an ``ndarray``.

        Equivalent to the regular constructor; provided as an explicit
        factory for readability at call sites.
        """
        return cls(name=name, values=values, unit=unit)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Number of frames (length of axis 0)."""
        return int(self.values.shape[0])

    @property
    def n_markers(self) -> int | None:
        """Number of markers (length of axis 1) or ``None`` for 1-D channels."""
        if self.values.ndim == 1:
            return None
        return int(self.values.shape[1])

    @property
    def is_per_marker(self) -> bool:
        """True iff the channel has a marker axis (2-D)."""
        return self.values.ndim == 2

    def value_at(self, frame_idx: int, marker_idx: int | None = None) -> float:
        """Return the scalar value at ``(frame_idx, marker_idx)``.

        Returns ``NaN`` for out-of-bounds indices. For a 1-D channel,
        ``marker_idx`` is ignored. For a 2-D channel, ``marker_idx`` of
        ``None`` returns the *mean* over the marker axis (excluding NaN).
        """
        if not isinstance(frame_idx, int):
            raise TypeError(f"frame_idx must be int; got {type(frame_idx).__name__}")
        if marker_idx is not None and not isinstance(marker_idx, int):
            raise TypeError(
                f"marker_idx must be int or None; got {type(marker_idx).__name__}"
            )

        if frame_idx < 0 or frame_idx >= self.n_frames:
            return float("nan")

        if self.values.ndim == 1:
            return float(self.values[frame_idx])

        # 2-D path
        n_markers = self.n_markers
        assert n_markers is not None  # noqa: S101 — invariant guarded above
        if marker_idx is None:
            row = self.values[frame_idx]
            finite_mask = np.isfinite(row)
            if not bool(finite_mask.any()):
                return float("nan")
            return float(np.mean(row[finite_mask]))
        if marker_idx < 0 or marker_idx >= n_markers:
            return float("nan")
        return float(self.values[frame_idx, marker_idx])

    def auto_range(self) -> tuple[float, float]:
        """Return ``(min, max)`` over all finite values.

        Returns ``(NaN, NaN)`` if the channel contains no finite values.
        """
        finite = self.values[np.isfinite(self.values)]
        if finite.size == 0:
            nan = float("nan")
            return (nan, nan)
        return (float(np.min(finite)), float(np.max(finite)))

    def has_finite_range(self) -> bool:
        """True iff :meth:`auto_range` returns finite values."""
        lo, hi = self.auto_range()
        return math.isfinite(lo) and math.isfinite(hi)


# ---------------------------------------------------------------------------
# Module-level derived-channel helpers (issue #4809)
# ---------------------------------------------------------------------------


def magnitude_channel(
    name: str,
    vector_per_frame: np.ndarray,
    unit: str = "",
) -> DataChannel:
    """Build a :class:`DataChannel` of magnitudes from a vector field.

    Parameters
    ----------
    name:
        Non-empty channel identifier.
    vector_per_frame:
        Either a 2-D array of shape ``(T, 3)`` (one 3-vector per frame)
        or a 3-D array of shape ``(T, M, 3)`` (one 3-vector per
        ``(frame, marker)``). The last axis must be of length 3.
    unit:
        Display unit string of the resulting magnitudes.

    Returns
    -------
    DataChannel
        For ``(T, 3)`` input the result is 1-D of shape ``(T,)``. For
        ``(T, M, 3)`` input the result is 2-D of shape ``(T, M)``.

    Raises
    ------
    TypeError
        If ``vector_per_frame`` is not a ``numpy.ndarray``.
    ValueError
        If ``vector_per_frame`` has unsupported ``ndim`` or its last
        axis is not of length 3.
    """
    if not isinstance(vector_per_frame, np.ndarray):
        raise TypeError(
            "vector_per_frame must be a numpy.ndarray; "
            f"got {type(vector_per_frame).__name__}"
        )
    if vector_per_frame.ndim not in (2, 3):
        raise ValueError(
            "vector_per_frame must have ndim in {2, 3}; "
            f"got ndim={vector_per_frame.ndim} (shape={vector_per_frame.shape})"
        )
    if vector_per_frame.shape[-1] != 3:
        raise ValueError(
            "vector_per_frame last axis must be of length 3; "
            f"got shape={vector_per_frame.shape}"
        )

    # ``np.linalg.norm`` propagates NaN — preserves the contract used by
    # ``auto_range`` and ``value_at``.
    magnitudes = np.linalg.norm(vector_per_frame, axis=-1)
    return DataChannel(name=name, values=magnitudes, unit=unit)


def derivative_channel(
    name: str,
    base_channel: DataChannel,
    *,
    timestep_s: float,
    unit_suffix: str = "/s",
) -> DataChannel:
    """Numerical time derivative of a scalar :class:`DataChannel`.

    Uses :func:`numpy.gradient` along the frame axis (axis 0) with a
    uniform spacing of ``timestep_s``. NaNs in the source propagate
    through the derivative.

    Parameters
    ----------
    name:
        Non-empty identifier of the new channel.
    base_channel:
        Source channel whose values will be differentiated.
    timestep_s:
        Uniform sample period in seconds. Must be strictly positive.
    unit_suffix:
        Suffix appended to ``base_channel.unit`` to form the derived
        unit (default ``"/s"``). When the base channel has no unit, the
        suffix is used as-is for transparency.

    Raises
    ------
    TypeError
        If ``base_channel`` is not a :class:`DataChannel`.
    ValueError
        If ``timestep_s`` is not finite or not strictly positive, or if
        the base channel has fewer than 2 frames (gradient undefined).
    """
    if not isinstance(base_channel, DataChannel):
        raise TypeError(
            f"base_channel must be a DataChannel; got {type(base_channel).__name__}"
        )
    if not isinstance(timestep_s, (int, float)) or isinstance(timestep_s, bool):
        raise TypeError(
            f"timestep_s must be a real number; got {type(timestep_s).__name__}"
        )
    if not math.isfinite(timestep_s) or timestep_s <= 0.0:
        raise ValueError(f"timestep_s must be finite and > 0; got {timestep_s!r}")
    if base_channel.n_frames < 2:
        raise ValueError(
            "base_channel must have at least 2 frames to compute a derivative; "
            f"got n_frames={base_channel.n_frames}"
        )

    derivative = np.gradient(base_channel.values, float(timestep_s), axis=0)
    derived_unit = (
        f"{base_channel.unit}{unit_suffix}" if base_channel.unit else unit_suffix
    )
    return DataChannel(name=name, values=derivative, unit=derived_unit)


def slice_channel(
    base_channel: DataChannel,
    frame_range: slice,
    marker_subset: Sequence[int] | None = None,
) -> DataChannel:
    """Build a derived :class:`DataChannel` by slicing the source.

    The resulting channel re-uses ``base_channel.name`` and ``unit`` —
    the slice is a *view-like* derived channel, not a rename. NaN
    handling in :meth:`DataChannel.auto_range` and
    :meth:`DataChannel.value_at` is preserved because the underlying
    ``ndarray`` slice keeps the same dtype and finite-mask semantics.

    Parameters
    ----------
    base_channel:
        Source channel.
    frame_range:
        Python ``slice`` object describing the frame window. ``start``
        and ``stop`` (when given) must lie within
        ``[0, base_channel.n_frames]``. Negative bounds are rejected to
        keep DbC explicit.
    marker_subset:
        Optional sequence of marker indices for 2-D channels. Each
        index must lie in ``[0, n_markers)``. Ignored for 1-D channels.

    Raises
    ------
    TypeError
        If types are wrong (e.g. ``frame_range`` not a ``slice``).
    ValueError
        If ``frame_range`` bounds are out of range, if
        ``marker_subset`` is provided for a 1-D channel, or if any
        marker index is OOB.
    """
    if not isinstance(base_channel, DataChannel):
        raise TypeError(
            f"base_channel must be a DataChannel; got {type(base_channel).__name__}"
        )
    if not isinstance(frame_range, slice):
        raise TypeError(
            f"frame_range must be a slice; got {type(frame_range).__name__}"
        )

    n_frames = base_channel.n_frames
    start = frame_range.start
    stop = frame_range.stop
    if start is not None and (
        not isinstance(start, int) or start < 0 or start > n_frames
    ):
        raise ValueError(f"frame_range.start must be in [0, {n_frames}]; got {start!r}")
    if stop is not None and (not isinstance(stop, int) or stop < 0 or stop > n_frames):
        raise ValueError(f"frame_range.stop must be in [0, {n_frames}]; got {stop!r}")

    if marker_subset is not None:
        if not base_channel.is_per_marker:
            raise ValueError(
                "marker_subset is only valid for per-marker (2-D) channels"
            )
        if not isinstance(marker_subset, Sequence) or isinstance(marker_subset, str):
            raise TypeError(
                "marker_subset must be a sequence of ints; "
                f"got {type(marker_subset).__name__}"
            )
        n_markers = base_channel.n_markers
        assert n_markers is not None  # noqa: S101 — guarded above
        for idx in marker_subset:
            if not isinstance(idx, (int, np.integer)) or isinstance(idx, bool):
                raise TypeError(
                    f"marker_subset entries must be int; got {type(idx).__name__}"
                )
            if idx < 0 or idx >= n_markers:
                raise ValueError(
                    f"marker_subset index {idx} out of range [0, {n_markers})"
                )

    sliced = base_channel.values[frame_range]
    if marker_subset is not None and base_channel.is_per_marker:
        sliced = sliced[:, list(marker_subset)]

    # ``np.ndarray`` slicing returns views; ``DataChannel`` is frozen so
    # rewrapping is sufficient and avoids unnecessary copies.
    return DataChannel(
        name=base_channel.name,
        values=np.asarray(sliced),
        unit=base_channel.unit,
    )
