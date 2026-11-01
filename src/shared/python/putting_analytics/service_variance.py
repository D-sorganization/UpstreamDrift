"""JSON service for the putting variance-attribution study (#14177).

Answers "which error source costs this player the most strokes?" for a library
of putt classes. Each class is played by :class:`.SkillRollModel` (lines are
planned on the green as read and rolled on the true green); the study reports,
per class, Sobol indices of the stroke outcome and the strokes gained by
removing or cutting each error source, then weights the gains into a per-round
figure with the library's encounter weights.

Arguments are untrusted tool input, so every number is bounded here with
:func:`.service._check` before any contract-checked constructor sees it
(``require`` can be switched off globally).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from src.shared.python.putting_analytics.library import (
    ORIENTATION_ASPECT_DEG,
    ClassLibrary,
    PuttClass,
)
from src.shared.python.putting_analytics.service import (
    MAX_LIBRARY_ROLLOUTS,
    PACE_RANGE_FT,
    _check,
    _number,
    baseline_from_args,
    library_from_args,
)
from src.shared.python.putting_analytics.skill import (
    CHANNELS,
    PuttingSkillProfile,
    ReadDispersion,
    StrokeDispersion,
)
from src.shared.python.putting_analytics.skill_roll_model import (
    PlaneGreen,
    ReadLattice,
    SkillRollModel,
)
from src.shared.python.putting_analytics.variance import (
    AttributionConfig,
    ClassAttribution,
    attribute_variance,
    strokes_gained_per_round,
)

# The skill roll model ignores strike location, so its factor would always
# report zero; it is not offered.
MODELLED_FACTORS: tuple[str, ...] = tuple(c for c in CHANNELS if c != "strike")
MAX_VARIANCE_ROLLOUTS = MAX_LIBRARY_ROLLOUTS
DEFAULT_FIRST_PUTTS = 18.0  # one first putt per hole

# key: (default, low, high), inclusive. Bounds are physical sanity limits.
_STROKE_BOUNDS = {
    "face_sd_deg": (1.5, 0.0, 10.0),
    "path_sd_deg": (2.0, 0.0, 15.0),
    "speed_cv": (0.08, 0.0, 0.5),
    "face_bias_deg": (0.0, -5.0, 5.0),
    "speed_bias": (0.0, -0.5, 0.5),
}
_READ_BOUNDS = {
    "grade_rel_sd": (0.0, 0.0, 1.0),
    "aspect_sd_deg": (0.0, 0.0, 45.0),
    "stimp_rel_sd": (0.0, 0.0, 0.5),
    "grade_rel_bias": (0.0, -0.9, 1.0),
}
_CONFIG_BOUNDS = {
    "base_samples": (128, 2, 4096),
    "n_samples": (500, 2, 20_000),
    "n_bootstrap": (100, 10, 1000),
}


def _bounded(args: Mapping[str, Any], key: str, spec: tuple[float, ...]) -> float:
    default, low, high = spec
    value = _number(args, key, default)
    _check(low <= value <= high, f"'{key}' must be in [{low}, {high}]")
    return value


def _count(args: Mapping[str, Any], key: str) -> int:
    value = _bounded(args, key, _CONFIG_BOUNDS[key])
    _check(value == int(value), f"'{key}' must be a whole number")
    return int(value)


def profile_from_args(args: Mapping[str, Any]) -> PuttingSkillProfile:
    """Stroke and read dispersion of the player under study."""
    stroke = {k: _bounded(args, k, spec) for k, spec in _STROKE_BOUNDS.items()}
    read = {k: _bounded(args, k, spec) for k, spec in _READ_BOUNDS.items()}
    face_weight = _bounded(args, "face_weight", (0.83, 0.0, 1.0))
    return PuttingSkillProfile(
        "tool", StrokeDispersion(**stroke), ReadDispersion(**read), face_weight
    )


def factors_from_args(args: Mapping[str, Any]) -> tuple[str, ...]:
    """Error sources to attribute; defaults to every modelled channel."""
    raw = args.get("factors", list(MODELLED_FACTORS))
    _check(isinstance(raw, list | tuple) and len(raw) > 0, "'factors' is empty")
    for name in raw:
        _check(name in MODELLED_FACTORS, f"factors must be in {MODELLED_FACTORS}")
    _check(len(set(raw)) == len(raw), "factors must be unique")
    return tuple(str(name) for name in raw)


def config_from_args(args: Mapping[str, Any]) -> AttributionConfig:
    """Sample sizes, bootstrap, seed and the partial-cut fraction."""
    seed = _number(args, "seed", 0)
    _check(seed >= 0 and seed == int(seed), "'seed' must be a non-negative integer")
    reduction = _number(args, "reduction", 0.1)
    _check(0.0 < reduction <= 1.0, "'reduction' must be in (0, 1]")
    return AttributionConfig(
        factors=factors_from_args(args),
        base_samples=_count(args, "base_samples"),
        n_samples=_count(args, "n_samples"),
        seed=int(seed),
        n_bootstrap=_count(args, "n_bootstrap"),
        reduction=reduction,
    )


def _work(config: AttributionConfig, n_classes: int) -> int:
    """Putts rolled: the Saltelli design plus the paired elimination runs."""
    k = len(config.factors)
    per_class = config.base_samples * (k + 2) + config.n_samples * (1 + 2 * k)
    return n_classes * per_class


def _models(
    library: ClassLibrary, args: Mapping[str, Any], profile: PuttingSkillProfile
) -> dict[str, SkillRollModel]:
    """One model per class, keyed ``c0``, ``c1``... in library order."""
    baseline = baseline_from_args(args)
    lattice = ReadLattice.from_dispersion(profile.read)
    pace = _bounded(args, "pace_ft_past", (1.5, *PACE_RANGE_FT))
    models = {}
    for i, putt_class in enumerate(library.classes):
        aspect = ORIENTATION_ASPECT_DEG[putt_class.orientation]
        green = PlaneGreen(putt_class.slope_pct, aspect, putt_class.stimp)
        models[f"c{i}"] = SkillRollModel(
            green,
            putt_class.distance_ft,
            baseline,
            lattice,
            pace_ft_past=pace,
            face_weight=profile.face_weight,
        )
    return models


def finite_json(value: Any) -> Any:
    """Replace NaN and infinities (e.g. Sobol indices of a constant output) by None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [finite_json(item) for item in value]
    return value


def _class_entry(
    putt_class: PuttClass, weight: float, result: ClassAttribution
) -> dict[str, Any]:
    eliminations = [
        {
            "factor": row.factor,
            "gain": row.gain,
            "gain_se": row.gain_se,
            "gain_reduced": row.gain_reduced,
        }
        for row in result.eliminations
    ]
    return {
        "distance_ft": putt_class.distance_ft,
        "slope_pct": putt_class.slope_pct,
        "orientation": putt_class.orientation,
        "stimp": putt_class.stimp,
        "weight": weight,
        "baseline_strokes": result.baseline_strokes,
        "eliminations": eliminations,
        "sobol": result.sobol.to_json_dict(),
    }


def _per_round(
    library: ClassLibrary, result: Mapping[str, ClassAttribution], first_putts: float
) -> dict[str, Any]:
    weights = {f"c{i}": first_putts * w for i, w in enumerate(library.weights)}
    totals = strokes_gained_per_round(result, weights)
    ranked = sorted(totals.items(), key=lambda item: item[1][0], reverse=True)
    rows = [
        {"factor": factor, "gain_removed": removed, "gain_cut": cut}
        for factor, (removed, cut) in ranked
    ]
    return {"first_putts": first_putts, "factors": rows}


def variance_study(args: Mapping[str, Any]) -> dict[str, Any]:
    """Attribute stroke variance and strokes gained to each error source.

    Per class: ``baseline_strokes`` (expected strokes with every error source),
    ``eliminations`` (strokes gained by removing each source, its standard
    error, and the gain from cutting it to ``reduction`` of its spread) and the
    ``sobol`` report. ``per_round`` weights the gains by encounter weight times
    ``first_putts_per_round`` and ranks the sources.
    """
    library = library_from_args(args)
    profile = profile_from_args(args)
    config = config_from_args(args)
    work = _work(config, len(library.classes))
    _check(
        work <= MAX_VARIANCE_ROLLOUTS,
        f"work is {work} putts; cap is {MAX_VARIANCE_ROLLOUTS}",
    )
    first_putts = _number(args, "first_putts_per_round", DEFAULT_FIRST_PUTTS)
    _check(0.0 < first_putts <= 100.0, "'first_putts_per_round' must be in (0, 100]")
    result = attribute_variance(_models(library, args, profile), profile, config)
    classes = [
        _class_entry(c, w, result[f"c{i}"])
        for i, (c, w) in enumerate(zip(library.classes, library.weights, strict=True))
    ]
    report = {
        "schema_version": 1,
        "factors": list(config.factors),
        "classes": classes,
        "per_round": _per_round(library, result, first_putts),
    }
    cleaned: dict[str, Any] = finite_json(report)
    return cleaned
