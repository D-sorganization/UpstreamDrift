"""Pick the markers a simulation result can actually plot (#12945).

The golf dashboard used to index ``marker_positions["Hand"]`` and
``["ClubHead"]``, names neither shipped default model defines. This is the one
place that decides which recorded markers get drawn.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = ["PREFERRED_MARKERS", "resolve_trajectory_markers"]

#: Preferred (hand, club) marker names when a model defines them.
PREFERRED_MARKERS: tuple[str, ...] = ("Hand", "ClubHead")


def _plottable(track: object) -> bool:
    """A track is plottable when it is ``[N, >=2]`` with at least one sample."""
    return (
        isinstance(track, np.ndarray)
        and track.ndim == 2
        and track.shape[0] > 0
        and track.shape[1] >= 2
    )


def resolve_trajectory_markers(
    marker_positions: Mapping[str, np.ndarray],
    preferred: tuple[str, ...] = PREFERRED_MARKERS,
) -> tuple[str, ...]:
    """Return up to ``len(preferred)`` marker names to plot, possibly none.

    Preferred names present in the result come first (in order); remaining
    slots are filled with the model's other markers in alphabetical order.
    An empty tuple means the model recorded nothing plottable.

    Raises:
        TypeError: ``marker_positions`` is not a mapping.
    """
    if not isinstance(marker_positions, Mapping):
        raise TypeError(
            f"marker_positions must be a mapping, got {type(marker_positions).__name__}"
        )
    usable = {n for n, track in marker_positions.items() if _plottable(track)}
    chosen = [name for name in preferred if name in usable]
    chosen += sorted(usable - set(chosen))
    return tuple(chosen[: len(preferred)])
