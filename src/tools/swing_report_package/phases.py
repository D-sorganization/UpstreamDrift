"""Key swing phases from segment directions (#15721, SRP-2).

A phase is an anchor of the swing (``start``, ``top``, ``impact``) or the
first instant, between two earlier phases or anchors, at which a segment
(the shaft, the lead arm) is level: its unit vector's vertical part changes
sign. The phases, their labels, segments and windows are settings
(``phase_settings.json``). The detector is pure: it takes world points per
state, so any engine's forward kinematics can feed it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ANCHORS",
    "PHASE_SETTINGS_PATH",
    "RULES",
    "KeyPhase",
    "PhaseSettings",
    "detect_phases",
    "load_phase_settings",
]

PHASE_SETTINGS_PATH = Path(__file__).with_name("phase_settings.json")
#: window bounds every swing has
ANCHORS = ("start", "top", "impact", "end")
RULES = ("start", "top", "impact", "level")


@dataclass(frozen=True)
class KeyPhase:
    """One key phase: its id, display label, nearest state and time (s)."""

    id: str
    label: str
    index: int
    time_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseSettings:
    """Segments (two point names each) and the ordered phase rules."""

    segments: Mapping[str, tuple[str, str]]
    phases: tuple[Mapping[str, str], ...]


def load_phase_settings(path: Path | None = None) -> PhaseSettings:
    """The phase settings, checked.

    Raises:
        ValueError: an unknown rule, a level phase without a known segment, or
            a window bound that is neither an anchor (:data:`ANCHORS`) nor an
            earlier phase.
    """
    doc = json.loads((path or PHASE_SETTINGS_PATH).read_text(encoding="utf-8"))
    segments = {k: tuple(v) for k, v in doc.get("segments", {}).items()}
    if any(len(v) != 2 for v in segments.values()):
        raise ValueError("each phase segment names exactly two points")
    known = set(ANCHORS)
    for phase in doc.get("phases", ()):
        pid, rule = phase.get("id"), phase.get("rule")
        if not pid or not phase.get("label"):
            raise ValueError(f"phase {phase!r} needs an id and a label")
        if rule not in RULES:
            raise ValueError(f"phase {pid!r} rule must be one of {RULES}, got {rule!r}")
        if rule == "level":
            if phase.get("segment") not in segments:
                raise ValueError(f"phase {pid!r} levels an unknown segment")
            for bound in ("after", "before"):
                ref = phase.get(bound)
                if ref not in known:
                    raise ValueError(f"phase {pid!r} {bound} {ref!r} is not known")
        known.add(pid)
    return PhaseSettings(segments, tuple(doc["phases"]))


def _nearest(t: NDArray[np.float64], time_s: float) -> int:
    return int(np.argmin(np.abs(t - time_s)))


def _first_level(
    t: NDArray[np.float64], z: NDArray[np.float64], lo: int, hi: int
) -> tuple[int, float] | None:
    """``(index, time)`` of the first sign change of ``z`` in ``(lo, hi]``."""
    for k in range(lo, hi):
        a, b = z[k], z[k + 1]
        if a == 0.0 or a * b < 0.0:
            time_s = float(t[k] + (t[k + 1] - t[k]) * a / (a - b)) if a != b else t[k]
            return _nearest(t, time_s), float(time_s)
    return None


def _unit_z(points: Mapping[str, Any], pair: Sequence[str], n: int) -> NDArray:
    for name in pair:
        if name not in points:
            raise ValueError(f"phase points need {name!r}")
    seg = np.asarray(points[pair[1]], float) - np.asarray(points[pair[0]], float)
    if seg.shape != (n, 3) or not np.all(np.isfinite(seg)):
        raise ValueError(f"points {pair} must be finite, shape ({n}, 3)")
    return seg[:, 2] / np.linalg.norm(seg, axis=1)


def detect_phases(
    time_s: Any,
    points: Mapping[str, Any],
    impact_time_s: float,
    top_time_s: float,
    settings: PhaseSettings | None = None,
) -> tuple[KeyPhase, ...]:
    """The key phases of a swing, in swing order.

    Args:
        time_s: state times ``(n,)``, increasing.
        points: world points ``(n, 3)`` per name used by the segments.
        impact_time_s, top_time_s: the swing's impact and top, inside
            ``time_s``.
        settings: phase settings (default: :data:`PHASE_SETTINGS_PATH`).

    Postconditions: phase times are non-decreasing in settings order.

    Raises:
        ValueError: bad times or points, or a segment that is never level in
            its window (named, never guessed).
    """
    look = settings or load_phase_settings()
    t = np.asarray(time_s, dtype=float)
    n = t.shape[0]
    if t.ndim != 1 or n < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be increasing, at least two states")
    for name, value in (("impact", impact_time_s), ("top", top_time_s)):
        if not t[0] <= value <= t[-1]:
            raise ValueError(f"{name} time {value} lies outside the swing")
    found: dict[str, tuple[int, float]] = {
        "start": (0, float(t[0])),
        "top": (_nearest(t, top_time_s), float(top_time_s)),
        "impact": (_nearest(t, impact_time_s), float(impact_time_s)),
        "end": (n - 1, float(t[-1])),
    }
    vertical = {name: _unit_z(points, pair, n) for name, pair in look.segments.items()}
    out = []
    for phase in look.phases:
        pid, rule = phase["id"], phase["rule"]
        if rule == "level":
            lo, hi = found[phase["after"]][0], found[phase["before"]][0]
            hit = _first_level(t, vertical[phase["segment"]], lo, hi)
            if hit is None:
                raise ValueError(
                    f"{pid}: the {phase['segment']} is never level between "
                    f"{phase['after']} and {phase['before']}"
                )
            found[pid] = hit
        else:
            found[pid] = found[rule]
        index, at = found[pid]
        out.append(KeyPhase(pid, phase["label"], index, at))
    times = [p.time_s for p in out]
    if times != sorted(times):
        raise ValueError(f"phases are out of swing order: {times}")
    return tuple(out)
