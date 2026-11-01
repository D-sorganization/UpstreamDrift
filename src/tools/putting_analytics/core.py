"""Pure-data layer of the putting analytics tool (#14177).

Turns the form's text fields into the JSON arguments the shared putting
service takes (:mod:`src.shared.python.putting_analytics.service`, the same
functions the MCP tools and the ``/tools/putting-analytics`` API call) and
turns each study's answer into table rows. Nothing here imports Qt, so the
layer is tested headless.

The miss-pricing curve (``baseline``) always comes from the user: the tool,
like the service, ships no reference make-rate data (see #14176).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

MODES: tuple[str, ...] = ("optimize", "misread_cost", "library_study")
SERVICE_FUNCTIONS: dict[str, str] = {
    "optimize": "optimize_putt",
    "misread_cost": "misread_cost",
    "library_study": "library_study",
}
CLASS_FIELDS: tuple[str, ...] = ("distance_ft", "slope_pct", "orientation", "stimp")

Rows = list[tuple[str, str]]


def parse_floats(text: str, name: str) -> list[float]:
    """Comma- or space-separated numbers; raises ``ValueError`` naming the field."""
    tokens = text.replace(",", " ").split()
    if not tokens:
        raise ValueError(f"{name} is empty")
    try:
        return [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc


def _lines(text: str) -> list[list[str]]:
    rows = (line.replace(",", " ").split() for line in text.splitlines())
    return [row for row in rows if row and not row[0].startswith("#")]


def parse_reads(text: str) -> list[dict[str, float]]:
    """One read per line: ``slope_scale stimp_ratio`` (1 1 is a perfect read)."""
    reads = []
    for row in _lines(text):
        if len(row) != 2:
            raise ValueError("each read is 'slope_scale stimp_ratio'")
        scale, ratio = parse_floats(" ".join(row), "read")
        reads.append({"slope_scale": scale, "stimp_ratio": ratio})
    if not reads:
        raise ValueError("enter at least one read")
    return reads


def parse_classes(text: str) -> list[dict[str, Any]]:
    """One class per line: ``distance_ft slope_pct orientation stimp count``."""
    classes = []
    for row in _lines(text):
        if len(row) != 5:
            raise ValueError(
                "each class is 'distance_ft slope_pct orientation stimp count'"
            )
        distance, slope = parse_floats(" ".join(row[:2]), "class")
        stimp, count = parse_floats(" ".join(row[3:]), "class")
        values = (distance, slope, row[2], stimp)
        classes.append({**dict(zip(CLASS_FIELDS, values, strict=True)), "count": count})
    if not classes:
        raise ValueError("enter at least one putt class")
    return classes


@dataclass(frozen=True)
class StudyForm:
    """The form as entered: numbers already parsed, lists still as text."""

    distance_ft: float = 10.0
    grade_pct: float = 0.0
    aspect_deg: float = 0.0
    stimp: float = 10.0
    line_sd_deg: float = 1.0
    speed_sd: float = 0.08
    objective: str = "sg"
    n: int = 200
    seed: int = 0
    baseline_distances: str = ""
    baseline_putts: str = ""
    aims: str = "-4 -2 0 2 4"
    paces: str = "0.5 1 2 3"
    extra: dict[str, str] = field(default_factory=dict)


def study_args(form: StudyForm, mode: str) -> dict[str, Any]:
    """Service arguments for ``mode`` (raises ``ValueError`` on bad input)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    args: dict[str, Any] = {
        "distance_ft": form.distance_ft,
        "grade_pct": form.grade_pct,
        "aspect_deg": form.aspect_deg,
        "stimp": form.stimp,
        "line_sd_deg": form.line_sd_deg,
        "speed_sd": form.speed_sd,
        "objective": form.objective,
        "n": form.n,
        "seed": form.seed,
        "baseline": {
            "distances_ft": parse_floats(form.baseline_distances, "baseline distances"),
            "expected_putts": parse_floats(form.baseline_putts, "baseline putts"),
        },
        "aims_deg": parse_floats(form.aims, "aims"),
        "paces_ft": parse_floats(form.paces, "paces"),
    }
    if mode == "misread_cost":
        args["reads"] = parse_reads(form.extra.get("reads", ""))
    if mode == "library_study":
        args["classes"] = parse_classes(form.extra.get("classes", ""))
    return args


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _action(entry: Mapping[str, Any]) -> str:
    action = entry["action"]
    return (
        f"aim {action['aim_offset_deg']:+.1f} deg, pace {action['pace_ft_past']:.1f} ft"
    )


def optimize_rows(result: Mapping[str, Any]) -> Rows:
    """Best action first, then the ranked top candidates."""
    best = result["best"]
    rows: Rows = [("Best", _action(best))]
    rows += [(key, _fmt(value)) for key, value in best.items() if key != "action"]
    rows += [
        (f"#{rank}", f"{_action(entry)}: {entry['expected_strokes']:.3f} strokes")
        for rank, entry in enumerate(result["top"], start=1)
    ]
    return rows


def flat_rows(result: Mapping[str, Any], prefix: str = "") -> Rows:
    """Every scalar in a nested answer, keyed by its dotted (and indexed) path."""
    rows: Rows = []
    for key, value in result.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            rows += flat_rows(value, f"{name}.")
        elif isinstance(value, list) and all(isinstance(v, Mapping) for v in value):
            for index, item in enumerate(value):
                rows += flat_rows(item, f"{name}[{index}].")
        elif isinstance(value, list):
            rows.append((name, ", ".join(_fmt(item) for item in value)))
        else:
            rows.append((name, _fmt(value)))
    return rows


def result_rows(mode: str, result: Mapping[str, Any]) -> Rows:
    """Table rows for a study's answer."""
    if mode == "optimize":
        return optimize_rows(result)
    return flat_rows(result)
