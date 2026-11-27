"""Open-loop replay horizon backward from impact (#15150 D2).

Full-horizon open-loop replay of saved efforts diverges (the standing body is
unstable without feedback; see ``same_input/scoring.py``), so open-loop
results are only meaningful over a window just before impact.  This module
measures that window: given a replay callable that starts an open-loop replay
of the recorded efforts from the tracked state at ``start_s`` and returns the
simulated club delivery at impact, it walks candidate starts backward from
impact and reports the earliest start whose face angle, path and head speed
all stay within tolerance, with the per-candidate errors for a report.

The search assumes agreement worsens monotonically with lead time.  It checks
that assumption instead of trusting it: if a start passes after a later start
failed, the window is reported ``non_contiguous`` and no earliest start is
picked.  A replay that diverges (non-finite state, integrator error) raises
:class:`ReplayDiverged`; that candidate is recorded as a failure with the
reason instead of aborting the search.  Tolerances, the candidate step and the maximum horizon come from
``src/config/open_loop_horizon.json``.

The search is engine-free; building the replay callable from a MuJoCo efforts
bundle is the caller's job.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

SETTINGS_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "open_loop_horizon.json"
)
STATUSES = ("accepted", "none_accepted", "non_contiguous")
# Relative slack when counting whole steps inside the horizon (round-off only).
_STEP_SLACK = 1e-9

__all__ = [
    "STATUSES",
    "CandidateResult",
    "HorizonResult",
    "HorizonSettings",
    "ImpactDelivery",
    "ReplayDiverged",
    "load_settings",
    "search_horizon",
]


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _positive(value: float, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


class ReplayDiverged(RuntimeError):
    """A replay that could not reach impact (non-finite state, integrator error).

    Raised by a replay callable; :func:`search_horizon` records the candidate
    as failed with ``str(exc)`` as the reason.  Precondition errors (bad start
    times) stay ``ValueError`` and still abort the search.
    """


@dataclass(frozen=True)
class ImpactDelivery:
    """Club delivery at impact: face angle and path [deg], head speed [m/s]."""

    face_angle_deg: float
    path_deg: float
    head_speed_mps: float

    def __post_init__(self) -> None:
        for item in fields(self):
            _finite(getattr(self, item.name), item.name)


@dataclass(frozen=True)
class HorizonSettings:
    """Validated search settings (see ``open_loop_horizon.json``)."""

    face_tolerance_deg: float
    path_tolerance_deg: float
    head_speed_tolerance_mps: float
    step_s: float
    max_horizon_s: float

    def __post_init__(self) -> None:
        for item in fields(self):
            _positive(getattr(self, item.name), item.name)
        if self.max_horizon_s < self.step_s:
            raise ValueError("max_horizon_s must be at least one step_s")

    def candidate_leads(self, available_s: float) -> list[float]:
        """Lead times ``k * step_s`` up to the smaller of the horizon and ``available_s``."""
        limit = min(self.max_horizon_s, available_s)
        count = math.floor(limit / self.step_s * (1.0 + _STEP_SLACK))
        return [k * self.step_s for k in range(1, count + 1)]


def load_settings(path: Path | str = SETTINGS_PATH) -> HorizonSettings:
    """Read the settings file; ``ValueError`` on a missing or invalid field."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    names = [item.name for item in fields(HorizonSettings)]
    missing = [name for name in names if name not in document]
    if missing:
        raise ValueError(f"open-loop horizon settings miss {missing}")
    return HorizonSettings(**{name: document[name] for name in names})


def _angle_error_deg(a: float, b: float) -> float:
    """Absolute angular difference wrapped to [0, 180] deg."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class CandidateResult:
    """One open-loop replay start and its delivery errors at impact.

    A diverged replay has ``failure`` set, every error ``None`` and
    ``passed`` false.
    """

    lead_s: float
    start_s: float
    face_error_deg: float | None
    path_error_deg: float | None
    head_speed_error_mps: float | None
    passed: bool
    failure: str | None = None

    def as_dict(self) -> dict[str, float | bool | str | None]:
        """Strict-JSON record for a report."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


def _evaluate(
    replay: Callable[[float], ImpactDelivery],
    reference: ImpactDelivery,
    start_s: float,
    lead_s: float,
    settings: HorizonSettings,
) -> CandidateResult:
    try:
        simulated = replay(start_s)
    except ReplayDiverged as exc:
        return CandidateResult(lead_s, start_s, None, None, None, False, str(exc))
    if not isinstance(simulated, ImpactDelivery):
        raise TypeError(
            f"replay must return an ImpactDelivery, got {type(simulated).__name__}"
        )
    face = _angle_error_deg(simulated.face_angle_deg, reference.face_angle_deg)
    path = _angle_error_deg(simulated.path_deg, reference.path_deg)
    speed = abs(simulated.head_speed_mps - reference.head_speed_mps)
    passed = (
        face <= settings.face_tolerance_deg
        and path <= settings.path_tolerance_deg
        and speed <= settings.head_speed_tolerance_mps
    )
    return CandidateResult(lead_s, start_s, face, path, speed, passed)


@dataclass(frozen=True)
class HorizonResult:
    """Outcome of the backward search.

    ``horizon_s`` is the lead of the last start in the passing run that begins
    at the latest candidate (0.0 when the latest candidate already fails).
    ``earliest_start_s`` is set only when ``status == "accepted"``; for a
    ``non_contiguous`` window, ``passes_after_failure_s`` lists the leads that
    passed after a failure, and nothing is picked.
    """

    impact_s: float
    status: str
    horizon_s: float
    earliest_start_s: float | None
    candidates: tuple[CandidateResult, ...]
    passes_after_failure_s: tuple[float, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if (self.earliest_start_s is None) == (self.status == "accepted"):
            raise ValueError("earliest_start_s is set exactly when accepted")

    @property
    def contiguous(self) -> bool:
        """False when an earlier start passed after a later one failed."""
        return self.status != "non_contiguous"

    def summary(self) -> dict[str, Any]:
        """Strict-JSON report of the search and every candidate."""
        return {
            "impact_s": self.impact_s,
            "status": self.status,
            "contiguous": self.contiguous,
            "horizon_s": self.horizon_s,
            "earliest_start_s": self.earliest_start_s,
            "passes_after_failure_s": list(self.passes_after_failure_s),
            "diverged_leads_s": [c.lead_s for c in self.candidates if c.failure],
            "candidates": [c.as_dict() for c in self.candidates],
        }


def _classify(impact_s: float, results: list[CandidateResult]) -> HorizonResult:
    passed = [c.passed for c in results]
    run = passed.index(False) if False in passed else len(passed)
    horizon = results[run - 1].lead_s if run else 0.0
    late = tuple(c.lead_s for c in results[run:] if c.passed)
    if late:
        status, earliest = "non_contiguous", None
    elif run:
        status, earliest = "accepted", results[run - 1].start_s
    else:
        status, earliest = "none_accepted", None
    return HorizonResult(impact_s, status, horizon, earliest, tuple(results), late)


def search_horizon(
    replay: Callable[[float], ImpactDelivery],
    reference: ImpactDelivery,
    *,
    impact_s: float,
    earliest_s: float,
    settings: HorizonSettings,
) -> HorizonResult:
    """Search backward from impact for the open-loop agreement horizon.

    Preconditions: ``replay`` is callable and maps a start time [s] to the
    simulated :class:`ImpactDelivery`; ``reference`` is the recorded delivery;
    ``earliest_s < impact_s`` (the earliest start the recording supports) with
    at least one candidate step between them.

    Every candidate up to the horizon is evaluated, so the report carries the
    full error curve and the monotonic assumption can be checked.

    Postcondition: candidates are ordered by increasing lead time, and the
    result's ``status`` is one of :data:`STATUSES`.
    """
    if not callable(replay):
        raise TypeError("replay must be callable (start_s -> ImpactDelivery)")
    if not isinstance(reference, ImpactDelivery):
        raise TypeError("reference must be an ImpactDelivery")
    if not isinstance(settings, HorizonSettings):
        raise TypeError("settings must be HorizonSettings")
    impact = _finite(impact_s, "impact_s")
    earliest = _finite(earliest_s, "earliest_s")
    if not earliest < impact:
        raise ValueError("earliest_s must be before impact_s")
    leads = settings.candidate_leads(impact - earliest)
    if not leads:
        raise ValueError("no candidate start fits one step_s before impact")
    results = [
        _evaluate(replay, reference, impact - lead, lead, settings) for lead in leads
    ]
    return _classify(impact, results)
