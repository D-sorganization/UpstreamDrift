"""Force-event namespace and sourced impact anchor (OF-04).

Plate events (``max_trail``, ``impact`` ...) are force events, not swing
positions: ``max_trail`` is not ``top``.  They are published under the
``force.`` prefix so they can sit beside model events (address, top, impact,
finish) on one plot without ever being mistaken for them or silently
overwriting them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from src.shared.python.force_plates.alignment import PlateTimeAlignment
from src.shared.python.force_plates.contracts import VerticalPlateTrial

__all__ = [
    "FORCE_EVENT_PREFIX",
    "force_event_markers",
    "impact_anchor",
    "merge_event_markers",
]

FORCE_EVENT_PREFIX = "force."
IMPACT_EVENT = "impact"


def force_event_markers(
    trial: VerticalPlateTrial, *, alignment: PlateTimeAlignment | None = None
) -> dict[str, float]:
    """Return ``{"force.<name>": t}`` for every event of ``trial`` with a time.

    Events whose time is ``None`` are omitted, never reported as zero.  Times
    are on the trial clock, or on the model clock when ``alignment`` is given.

    Raises:
        TypeError: if ``trial`` or ``alignment`` has the wrong type.
    """
    if not isinstance(trial, VerticalPlateTrial):
        raise TypeError("trial must be a VerticalPlateTrial")
    if alignment is not None and not isinstance(alignment, PlateTimeAlignment):
        raise TypeError("alignment must be a PlateTimeAlignment or None")
    out: dict[str, float] = {}
    for name, when in trial.events_s.items():
        if when is None:
            continue
        out[FORCE_EVENT_PREFIX + name] = (
            float(when) if alignment is None else alignment.model_time(when)
        )
    return out


def _check_markers(markers: Mapping[str, float], label: str) -> None:
    for key, value in markers.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings, got {key!r}")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{label}[{key!r}] must be a real number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{label}[{key!r}] must be finite, got {value!r}")


def merge_event_markers(
    model_events: Mapping[str, float], force_events: Mapping[str, float]
) -> dict[str, float]:
    """Merge model and force event times into a new dict.

    Raises:
        ValueError: naming the key, for a force key without the ``force.``
            prefix, a model key carrying it, a key present in both, or a
            non-finite or non-numeric time.
    """
    _check_markers(model_events, "model_events")
    _check_markers(force_events, "force_events")
    for key in force_events:
        if not key.startswith(FORCE_EVENT_PREFIX):
            raise ValueError(
                f"force event {key!r} lacks the {FORCE_EVENT_PREFIX!r} prefix"
            )
    for key in model_events:
        if key.startswith(FORCE_EVENT_PREFIX):
            raise ValueError(f"model event {key!r} must not use the force namespace")
    for key in force_events:
        if key in model_events:
            raise ValueError(f"event {key!r} collides between model and force events")
    return {**model_events, **force_events}


def impact_anchor(trial: VerticalPlateTrial) -> float | None:
    """Trial-clock time [s] of the trial's ``impact`` event, or ``None``.

    OF-02 only accepts an ``impact`` event together with an ``impact_source``,
    so a returned time is sourced.  Whether that source was a shot or a manual
    mark is not retained: callers declare it as the alignment kind.
    """
    if not isinstance(trial, VerticalPlateTrial):
        raise TypeError("trial must be a VerticalPlateTrial")
    when = trial.events_s.get(IMPACT_EVENT)
    return None if when is None else float(when)
