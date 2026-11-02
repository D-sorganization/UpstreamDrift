"""JSON-in, JSON-out putting studies for tool adapters such as MCP (#14177).

Every request is a plain mapping so an adapter can pass tool arguments
straight through, and every answer is a plain dict of floats so it can be
serialised as is. The green is a uniform plane (grade, the direction it falls
toward, stimp); the stroke has Gaussian start-line error and lognormal speed
error. The miss-pricing curve (``baseline``) is always supplied by the caller:
this module ships no reference make-rate data (see #14176 for why).

The work is bounded: ``candidates x n`` rollouts may not exceed
:data:`MAX_ROLLOUTS`, so a tool call cannot pin the server for minutes.

Input checks here use :func:`_check`, which always raises ``ValueError``, not
the project ``require``: contract checks can be switched off globally, and a
tool boundary (in particular the work cap) must hold regardless.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.putting_analytics.aggregate import (
    DEFAULT_GRID_FT,
    AggregateConfig,
    aggregate,
)
from src.shared.python.putting_analytics.expected_putts import (
    ExpectedPuttsCurve,
    PuttAction,
)
from src.shared.python.putting_analytics.library import (
    DISTANCE_RANGE_FT,
    FLAT,
    ORIENTATION_ASPECT_DEG,
    SLOPE_RANGE_PCT,
    ClassLibrary,
    PuttClass,
)
from src.shared.python.putting_analytics.optimize import (
    OBJECTIVES,
    ActionEvaluation,
    best_action,
    evaluate_outcomes,
)
from src.shared.python.putting_analytics.perceived_green import (
    GreenRead,
    PerceivedGreenPlanner,
    read_error_cost,
)
from src.shared.python.putting_analytics.roll_model import BatchRollModel
from src.shared.python.putting_dynamics.surfaces import HeightField, SurfaceSpec

MAX_ROLLOUTS = 10_000  # ~7 s of batch rolling on one core
MAX_MISREAD_ROLLOUTS = 4 * MAX_ROLLOUTS  # one plan per distinct read, plus play
MAX_GRADE_PCT = 10.0
MAX_DISTANCE_FT = 100.0
STIMP_RANGE = (5.0, 16.0)
MAX_ABS_AIM_DEG = 45.0
PACE_RANGE_FT = (0.0, 10.0)
DEFAULT_AIMS_DEG = tuple(float(a) for a in np.arange(-12.0, 12.01, 1.0))
DEFAULT_PACES_FT = (0.5, 1.0, 2.0, 3.0)
TOP_K = 5
MAX_LIBRARY_CLASSES = 24
MAX_LIBRARY_ROLLOUTS = 4 * MAX_ROLLOUTS
LIBRARY_GRID_FT = (*DEFAULT_GRID_FT, 60.0)  # reaches the library's longest class


def _check(condition: bool, message: str) -> None:
    """Unconditional input check for untrusted tool arguments."""
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class StrokeNoise:
    """Start-line error ~ N(0, line_sd_deg); speed ratio ~ LogNormal(0, speed_sd)."""

    line_sd_deg: float = 0.0
    speed_sd: float = 0.0

    def __post_init__(self) -> None:
        for name in ("line_sd_deg", "speed_sd"):
            value = getattr(self, name)
            _check(math.isfinite(value) and value >= 0.0, f"{name} must be >= 0")

    def __call__(
        self, rng: np.random.Generator, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-putt (line error [deg], speed ratio)."""
        line_err = rng.normal(0.0, self.line_sd_deg, n)
        return line_err, np.exp(rng.normal(0.0, self.speed_sd, n))


@dataclass(frozen=True)
class PuttStudy:
    """One validated putt question: where, how well struck, priced how."""

    distance_ft: float
    surface: SurfaceSpec
    noise: StrokeNoise
    baseline: ExpectedPuttsCurve
    candidates: tuple[PuttAction, ...]
    objective: str
    n: int
    seed: int


def _number(args: Mapping[str, Any], key: str, default: float | None = None) -> float:
    value = args.get(key, default)
    _check(value is not None, f"'{key}' is required")
    _check(
        isinstance(value, int | float) and not isinstance(value, bool),
        f"'{key}' must be a number",
    )
    _check(math.isfinite(float(value)), f"'{key}' must be finite")
    return float(value)


def _floats(args: Mapping[str, Any], key: str, default: Sequence[float]) -> list[float]:
    values = args.get(key, default)
    _check(isinstance(values, list | tuple) and len(values) > 0, f"'{key}' is empty")
    return [_number({key: v}, key) for v in values]


def surface_from_args(args: Mapping[str, Any]) -> SurfaceSpec:
    """Uniform planar green from ``grade_pct``, ``aspect_deg`` and ``stimp``."""
    grade = _number(args, "grade_pct", 0.0)
    _check(0.0 <= grade <= MAX_GRADE_PCT, f"grade_pct must be in [0, {MAX_GRADE_PCT}]")
    stimp = _number(args, "stimp", 10.0)
    _check(STIMP_RANGE[0] <= stimp <= STIMP_RANGE[1], f"stimp must be in {STIMP_RANGE}")
    flat = SurfaceSpec.flat_uniform(stimp)
    if grade == 0.0:
        return flat
    plane = HeightField.planar(grade, _number(args, "aspect_deg", 0.0))
    return dataclasses.replace(flat, height=plane)


def baseline_from_args(args: Mapping[str, Any]) -> ExpectedPuttsCurve:
    """Miss-pricing curve from ``baseline = {distances_ft, expected_putts}``."""
    raw = args.get("baseline")
    if not isinstance(raw, Mapping):
        raise ValueError("'baseline' must give distances_ft and E(d)")
    distances = np.asarray(_floats(raw, "distances_ft", ()), dtype=float)
    putts = np.asarray(_floats(raw, "expected_putts", ()), dtype=float)
    return ExpectedPuttsCurve(distances, putts)


def candidates_from_args(args: Mapping[str, Any]) -> tuple[PuttAction, ...]:
    """Every (aim, pace) pair from ``aims_deg`` x ``paces_ft``."""
    aims = _floats(args, "aims_deg", DEFAULT_AIMS_DEG)
    paces = _floats(args, "paces_ft", DEFAULT_PACES_FT)
    _check(all(abs(a) <= MAX_ABS_AIM_DEG for a in aims), "aims_deg out of range")
    lo, hi = PACE_RANGE_FT
    _check(all(lo <= p <= hi for p in paces), f"paces_ft must be in {PACE_RANGE_FT}")
    return tuple(PuttAction(aim, pace) for aim in aims for pace in paces)


def _samples(args: Mapping[str, Any]) -> int:
    n = int(_number(args, "n", 100))
    _check(n > 1, "'n' must be at least 2")
    return n


def _objective(args: Mapping[str, Any]) -> str:
    objective = str(args.get("objective", "sg"))
    _check(objective in OBJECTIVES, f"objective must be one of {OBJECTIVES}")
    return objective


def noise_from_args(args: Mapping[str, Any]) -> StrokeNoise:
    """Stroke noise from ``line_sd_deg`` and ``speed_sd`` (both default 0)."""
    return StrokeNoise(
        _number(args, "line_sd_deg", 0.0), _number(args, "speed_sd", 0.0)
    )


def study_from_args(args: Mapping[str, Any]) -> PuttStudy:
    """Validate tool arguments into a :class:`PuttStudy` (raises ``ValueError``)."""
    candidates = candidates_from_args(args)
    n = _samples(args)
    _check(
        n * len(candidates) <= MAX_ROLLOUTS,
        f"candidates x n must be <= {MAX_ROLLOUTS}",
    )
    objective = _objective(args)
    distance = _number(args, "distance_ft")
    _check(
        0.0 < distance <= MAX_DISTANCE_FT,
        f"'distance_ft' must be in (0, {MAX_DISTANCE_FT}]",
    )
    return PuttStudy(
        distance,
        surface_from_args(args),
        noise_from_args(args),
        baseline_from_args(args),
        candidates,
        objective,
        n,
        int(_number(args, "seed", 0)),
    )


def _evaluation_dict(evaluation: ActionEvaluation) -> dict[str, Any]:
    """``{"action": {aim_offset_deg, pace_ft_past}, make_rate, ...}``."""
    return dataclasses.asdict(evaluation)


def _ranked(evaluations: list[ActionEvaluation], objective: str) -> list[Any]:
    if objective == "sg":
        return sorted(evaluations, key=lambda e: (e.expected_strokes, -e.make_rate))
    return sorted(evaluations, key=lambda e: (-e.make_rate, e.expected_strokes))


def optimize_putt(args: Mapping[str, Any]) -> dict[str, Any]:
    """Best aim and pace for one putt under the player's stroke noise.

    Aim is degrees left (CCW) of the hole line; pace is feet past the hole.
    All candidates share one noise draw, so their differences are paired.
    """
    study = study_from_args(args)
    model = BatchRollModel(study.surface, study.noise)
    outcomes = model.simulate_many(
        study.distance_ft, study.candidates, study.n, study.seed
    )
    evaluations = [
        evaluate_outcomes(action, sample, study.baseline)
        for action, sample in zip(study.candidates, outcomes, strict=True)
    ]
    best = best_action(evaluations, study.objective)
    return {
        "objective": study.objective,
        "distance_ft": study.distance_ft,
        "best": _evaluation_dict(best),
        "top": [
            _evaluation_dict(e) for e in _ranked(evaluations, study.objective)[:TOP_K]
        ],
        "candidates": len(study.candidates),
        "n": study.n,
        "seed": study.seed,
    }


def _reads_from_args(args: Mapping[str, Any]) -> list[GreenRead]:
    raw = args.get("reads")
    if not isinstance(raw, list) or not raw:
        raise ValueError("'reads' must be a non-empty list")
    reads = []
    for read in raw:
        _check(isinstance(read, Mapping), "each read must be an object")
        scale = _number(read, "slope_scale", 1.0)
        ratio = _number(read, "stimp_ratio", 1.0)
        _check(scale > 0.0 and ratio > 0.0, "read scales must be positive")
        reads.append(GreenRead(scale, ratio))
    return reads


def misread_cost(args: Mapping[str, Any]) -> dict[str, Any]:
    """Expected-strokes cost of the sampled ``reads`` against a perfect read.

    Each read is planned on the green the player perceives and played on the
    true green, so a misread changes both aim and pace.
    """
    study = study_from_args(args)
    reads = _reads_from_args(args)
    work = (len(reads) + 1) * study.n * len(study.candidates)
    _check(work <= MAX_MISREAD_ROLLOUTS, f"(reads+1) x candidates x n is {work}")
    planner = PerceivedGreenPlanner(
        study.surface,
        study.candidates,
        study.baseline,
        study.n,
        study.noise,
        study.objective,
        plan_seed=study.seed,
    )
    cost = read_error_cost(planner, study.distance_ft, reads, study.seed)
    return {
        "objective": study.objective,
        "distance_ft": study.distance_ft,
        "reads": len(reads),
        **dataclasses.asdict(cost),
    }


def _putt_class(raw: Any) -> tuple[PuttClass, float]:
    """One ``{distance_ft, slope_pct, orientation, stimp, count}`` entry."""
    _check(isinstance(raw, Mapping), "each class must be an object")
    distance = _number(raw, "distance_ft")
    lo, hi = DISTANCE_RANGE_FT
    _check(lo <= distance <= hi, f"class distance_ft must be in [{lo}, {hi}]")
    slope = _number(raw, "slope_pct", 0.0)
    lo, hi = SLOPE_RANGE_PCT
    _check(lo <= slope <= hi, f"class slope_pct must be in [{lo}, {hi}]")
    orientation = raw.get("orientation", FLAT)
    _check(orientation in ORIENTATION_ASPECT_DEG, "unknown class orientation")
    _check((slope == 0.0) == (orientation == FLAT), "use 'flat' exactly at 0 slope")
    stimp = _number(raw, "stimp", 10.0)
    _check(STIMP_RANGE[0] <= stimp <= STIMP_RANGE[1], f"stimp must be in {STIMP_RANGE}")
    count = _number(raw, "count", 1.0)
    _check(count > 0.0, "class count must be positive")
    return PuttClass(distance, slope, str(orientation), stimp), count


def library_from_args(args: Mapping[str, Any]) -> ClassLibrary:
    """Putt classes weighted by their ``count`` (default 1 each)."""
    raw = args.get("classes")
    if not isinstance(raw, list) or not 0 < len(raw) <= MAX_LIBRARY_CLASSES:
        raise ValueError(f"'classes' must list 1 to {MAX_LIBRARY_CLASSES} putt classes")
    pairs = [_putt_class(item) for item in raw]
    classes = [putt_class for putt_class, _ in pairs]
    _check(len(set(classes)) == len(classes), "putt classes must be distinct")
    return ClassLibrary.from_counts(classes, [count for _, count in pairs])


def class_surface(putt_class: PuttClass) -> SurfaceSpec:
    """Uniform plane for one class: its slope, fall line and stimp."""
    aspect = ORIENTATION_ASPECT_DEG[putt_class.orientation]
    plane = {"grade_pct": putt_class.slope_pct, "aspect_deg": aspect}
    return surface_from_args({**plane, "stimp": putt_class.stimp})


def _grid_covering(distance_ft: float) -> tuple[float, ...]:
    """The library grid up to its first point at or beyond ``distance_ft``."""
    reach = next(d for d in LIBRARY_GRID_FT if d >= distance_ft)
    return tuple(d for d in LIBRARY_GRID_FT if d <= reach)


def library_study(args: Mapping[str, Any]) -> dict[str, Any]:
    """Weighted putting report over a library of putt classes (#14174).

    Each class is rolled on its own plane with the caller's stroke noise; the
    report gives expected putts per round, make rate by distance band, optimal
    pace and strokes gained against ``baseline`` (the reference curve, for
    example tour), each with a Monte Carlo standard error.
    """
    library = library_from_args(args)
    candidates = candidates_from_args(args)
    n = _samples(args)
    grid = _grid_covering(max(c.distance_ft for c in library.classes))
    work = len(library.classes) * (len(grid) + 2) * len(candidates) * n
    _check(
        work <= MAX_LIBRARY_ROLLOUTS, f"classes x (grid+2) x candidates x n is {work}"
    )
    cfg = AggregateConfig(
        candidates,
        baseline_from_args(args),
        n,
        int(_number(args, "seed", 0)),
        grid_ft=grid,
        objective=_objective(args),
    )
    noise = noise_from_args(args)
    report = aggregate(library, lambda c: BatchRollModel(class_surface(c), noise), cfg)
    result: dict[str, Any] = json.loads(report.to_json())
    return result
