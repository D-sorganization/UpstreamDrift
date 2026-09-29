"""Conservative semantic matching helpers for numeric-contract scaffolding."""

from __future__ import annotations

import math
from pathlib import Path
import re


TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9]+")
SCALES = (1.0, 100.0, 0.01, 1000.0, 0.001, 180.0 / math.pi, math.pi / 180.0)


def _rounding_tolerance(text: str) -> float:
    plain = text.replace(",", "").lower().lstrip("+")
    mantissa, _, exponent_text = plain.partition("e")
    exponent = int(exponent_text) if exponent_text else 0
    decimals = len(mantissa.partition(".")[2]) if "." in mantissa else 0
    return 0.5000001 * 10.0 ** (exponent - decimals)


def _tokens(text: str) -> set[str]:
    result = set(TOKEN_PATTERN.findall(text.lower().replace("squared", "square")))
    aliases = {
        "event": {"time"},
        "timing": {"time"},
        "duration": {"time", "horizon"},
        "velocity": {"speed"},
        "speed": {"velocity"},
        "sample": {"count"},
        "samples": {"count"},
        "cases": {"count"},
        "cells": {"count"},
        "trajectories": {"count"},
        "error": {"residual", "discrepancy"},
        "errors": {"residual", "discrepancy"},
        "ratio": {"fraction"},
        "percent": {"fraction"},
    }
    for token in tuple(result):
        result.update(aliases.get(token, set()))
    return result


def _rank_candidate(
    candidate: tuple[str, str, float, float],
    *,
    statement: str,
    context: str,
    artifact_order: dict[str, int],
    expected: float,
) -> tuple[float, int, int, int, str, str]:
    artifact, pointer, value, scale = candidate
    path_tokens = _tokens(Path(artifact).stem + " " + pointer.replace("/", " "))
    context_overlap = len(path_tokens & _tokens(context))
    statement_overlap = len(path_tokens & _tokens(statement))
    indexed_tokens = sum(token.isdigit() for token in pointer.split("/"))
    exactness = abs(value * scale - expected)
    return (
        exactness,
        0 if scale == 1.0 else 1,
        -context_overlap,
        -statement_overlap + indexed_tokens,
        f"{artifact_order[artifact]:04d}",
        pointer,
    )


def _has_semantic_pointer_match(pointer: str, context: str) -> bool:
    """Require at least one contextual term in the selected JSON path."""
    ignored = {
        "data",
        "result",
        "results",
        "summary",
        "value",
        "values",
        "model",
        "record",
        "registered",
        "grid",
        "row",
        "rows",
        "case",
        "cases",
        "configuration",
        "design",
        "baseline",
        "reference",
        "program",
        "programs",
        "study",
        "audit",
        "selected",
        "variant",
        "variants",
        "index",
        "pair",
        "pairs",
        "rate",
        "matching",
        "matched",
    }
    return bool((_tokens(pointer) - ignored) & (_tokens(context) - ignored))


def _scale_is_semantically_valid(scale: float, pointer: str, context: str) -> bool:
    """Allow unit transforms only when path and prose declare compatible units."""
    if scale == 1.0:
        return True
    prose = context.lower()
    path = pointer.lower()
    if scale == 100.0:
        return ("percent" in prose or "%" in prose) and "fraction" in path
    if scale == 0.01:
        return ("fraction" in prose or "ratio" in prose) and "percent" in path
    if scale == 1000.0:
        return (
            " ms" in prose
            and any(term in path for term in ("time", "duration", "horizon", "step"))
            and path.endswith("_s")
        ) or (" mm" in prose and path.endswith("_m"))
    if scale == 0.001:
        return (" kw" in prose and path.endswith("_w")) or (
            " m" in prose and path.endswith("_mm")
        )
    if math.isclose(scale, 180.0 / math.pi):
        return ("deg" in prose or "degree" in prose) and path.endswith("_rad")
    if math.isclose(scale, math.pi / 180.0):
        return "rad" in prose and path.endswith("_deg")
    return False


def _pointer_matches_declared_quantity(
    pointer: str, *, before: str, after: str
) -> bool:
    """Reject coincidental equal values for a different declared quantity."""
    path = pointer.lower()
    path_tokens = _tokens(path)
    prefix = after.lower().lstrip(" -–—/")
    if before.rstrip().endswith("="):
        return False
    if re.match(r"ms\b", prefix):
        return bool(path_tokens & {"time", "duration", "horizon", "step"})
    if re.match(r"s\b", prefix):
        return bool(path_tokens & {"time", "duration", "horizon", "step"})
    if re.match(r"(?:%|percent\b)", prefix):
        return bool(path_tokens & {"percent", "fraction", "ratio", "tolerance"})
    if re.match(r"n\s*m\b", prefix):
        return path.endswith("_nm") or bool(
            path_tokens & {"moment", "couple", "torque"}
        )
    if re.match(r"n\b", prefix):
        return path.endswith("_n") or bool(path_tokens & {"force", "load"})
    if re.match(r"j\b", prefix):
        return path.endswith("_j") or bool(path_tokens & {"work", "energy"})
    if re.match(r"w\b", prefix):
        return path.endswith("_w") or "power" in path_tokens
    if re.match(r"(?:m/s|m\s+s-?1)\b", prefix):
        return path.endswith("_m_s") or bool(path_tokens & {"speed", "velocity"})
    if re.match(r"(?:rad/s|rad\s+s-?1)\b", prefix):
        return path.endswith("_rad_s") or bool(path_tokens & {"angular", "velocity"})
    if re.match(r"rad\b", prefix):
        return path.endswith("_rad") or bool(path_tokens & {"angle", "rotation"})
    count_noun = re.search(
        r"\b(golfer|participant|case|cell|program|input|sample|state|trajectory|"
        r"comparison|coordinate|profile|configuration|branch|pair)s?\b",
        prefix[:48],
    )
    if count_noun:
        return bool(path_tokens & {"count", "total", "size"})
    if re.search(r"(?:schema|version)\s+[a-z-]*v?\s*$", before.lower()):
        return any(term in path for term in ("schema", "version"))
    return True
