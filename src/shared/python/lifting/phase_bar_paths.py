"""Phase-correct planar bar paths of the multi-phase lifts (#11745).

The Tools Movement Optimizer's ``bar_position(q, convention)`` places the bar
from the configuration alone, so one convention holds one bar placement for
the whole lift.  The snatch and the full clean and jerk move the bar between
placements as the lift goes through its phases, so their bar path also needs
the time of each sample.  The vendored Tools exercises provide those paths:

* ``snatch`` (:func:`movement_optimizer.exercises.snatch.snatch_bar_path`):
  hanging through the pulls, overhead from the catch on;
* ``clean_and_jerk``
  (:func:`movement_optimizer.exercises.clean_and_jerk.clean_and_jerk_bar_path`):
  hanging through the pulls, on the shoulders (front rack) from the catch to
  the jerk drive, overhead from the overhead catch on.

Split from :mod:`.bundles` to keep that module inside its size budget.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

#: Bar convention -> (vendored Tools module, phase-correct bar path function).
PHASE_BAR_CONVENTIONS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "snatch": ("movement_optimizer.exercises.snatch", "snatch_bar_path"),
        "clean_and_jerk": (
            "movement_optimizer.exercises.clean_and_jerk",
            "clean_and_jerk_bar_path",
        ),
    }
)


def phase_bar_path(dyn: Any, angles: Array, time_s: Array, convention: str) -> Array:
    """Planar bar positions (N, 2) of the optimizer's *convention* bar model.

    A :data:`PHASE_BAR_CONVENTIONS` convention is evaluated at each sample's
    fraction of the lift, ``(time_s - time_s[0]) / (time_s[-1] - time_s[0])``;
    any other convention is the configuration-only ``dyn.bar_position``.

    Raises:
        ValueError: For a phase convention whose *time_s* has fewer than two
            samples or no positive span.
    """
    if convention not in PHASE_BAR_CONVENTIONS:
        return np.array([dyn.bar_position(q, convention) for q in angles])
    time_s = np.asarray(time_s, dtype=float)
    if time_s.ndim != 1 or time_s.size < 2 or not time_s[-1] > time_s[0]:
        raise ValueError("time_s must be 1-D with a positive span")
    module, function = PHASE_BAR_CONVENTIONS[convention]
    path_fn = getattr(importlib.import_module(module), function)
    frac = (time_s - time_s[0]) / float(time_s[-1] - time_s[0])
    return np.asarray(path_fn(dyn, angles, frac), dtype=float)
