"""Ball phase marks for playback jumps (#14473, epic #14452).

A putt's trajectory is a run of solver modes (:class:`~.types.Mode`): the
brief post-impact hop (airborne), the skid (slide), pure roll and rest. A
:class:`PhaseMark` is where one run starts, so a player can jump the
playback to the start of the skid, the roll or the rest. The marks are
read off the solver's own per-sample modes; nothing is re-derived.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .types import Mode, TrajectorySample

__all__ = ["PhaseMark", "phase_marks"]


@dataclass(frozen=True)
class PhaseMark:
    """The start of one run of a mode: its mode, time [s] and sample index."""

    mode: Mode
    t_s: float
    index: int


def phase_marks(samples: Sequence[TrajectorySample]) -> tuple[PhaseMark, ...]:
    """Where each run of a mode starts in ``samples``.

    Postconditions: the first mark is sample 0, every mark's sample has its
    mode, and the sample before a mark has a different mode.

    Raises:
        ValueError: On no samples, or samples out of time order.
    """
    if not samples:
        raise ValueError("phase_marks needs at least one sample")
    for before, after in zip(samples, samples[1:], strict=False):
        if after.t_s < before.t_s:
            raise ValueError("samples must be in time order")
    return tuple(
        PhaseMark(sample.mode, sample.t_s, i)
        for i, sample in enumerate(samples)
        if i == 0 or samples[i - 1].mode is not sample.mode
    )
