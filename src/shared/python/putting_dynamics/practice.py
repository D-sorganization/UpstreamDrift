"""Practice feedback and putt-result documents (#14141 step 10c).

Ported from the legacy engine's ``_practice_mode`` (``simulate_with_feedback``,
``export_result`` and ``load_result``) so the features survive the removal
of its duplicate physics (ADR-0058: no unique feature is deleted). Scatter,
aim and green reading already live in the shared core (``read_line``,
``solve_aim``) and the ``/scatter`` route.

A result document always names its roll model, and the reader is fail-closed
(ADR-0045 F1): it refuses a document with no roll model, or one produced by a
model this core does not run, rather than replaying it under the wrong law.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from src.shared.python.contracts import require_finite

from .friction import USGA_STIMP_ROLL_MODEL
from .types import CollisionReport, Mode, PuttResult, TrajectorySample

__all__ = [
    "PUTT_RESULT_SCHEMA",
    "PracticeFeedback",
    "load_putt_result",
    "practice_feedback",
    "putt_result_from_dict",
    "putt_result_to_dict",
    "save_putt_result",
]

PUTT_RESULT_SCHEMA = "putting_dynamics.putt_result/1"

#: Rest within this distance of the hole counts as a pace miss [m]; the
#: legacy practice mode's threshold, kept unchanged.
NEAR_MISS_M = 0.5

_SCALARS = (
    "holed",
    "rest_x_m",
    "rest_y_m",
    "total_distance_m",
    "time_s",
    "skid_distance_m",
    "speed_at_hole_mps",
)


@dataclasses.dataclass(frozen=True)
class PracticeFeedback:
    """Feedback on one practice putt.

    Attributes:
        distance_from_hole_m: Rest distance from the hole centre [m].
        holed: Whether the ball was captured.
        total_distance_m: Ground-path length [m].
        suggestion: One of "Great putt", "Hit slightly firmer",
            "Hit slightly softer" or "Check your aim line".
        roll_model: Roll model that produced the putt.
    """

    distance_from_hole_m: float
    holed: bool
    total_distance_m: float
    suggestion: str
    roll_model: str = USGA_STIMP_ROLL_MODEL


def practice_feedback(
    result: PuttResult, hole_xy: tuple[float, float]
) -> PracticeFeedback:
    """Grade a putt against the hole at ``hole_xy``.

    A near miss (rest within :data:`NEAR_MISS_M`) is short or long by the
    sign of the miss along the start-to-hole line; a wider miss points at
    the aim.

    Raises:
        PreconditionError: If ``hole_xy`` is not finite.
    """
    hole_x, hole_y = (float(hole_xy[0]), float(hole_xy[1]))
    require_finite(hole_x, "hole_x")
    require_finite(hole_y, "hole_y")
    start = result.samples[0]
    miss_x, miss_y = result.rest_x_m - hole_x, result.rest_y_m - hole_y
    distance = math.hypot(miss_x, miss_y)
    if result.holed:
        suggestion = "Great putt"
    elif distance < NEAR_MISS_M:
        along = miss_x * (hole_x - start.x_m) + miss_y * (hole_y - start.y_m)
        suggestion = "Hit slightly firmer" if along < 0.0 else "Hit slightly softer"
    else:
        suggestion = "Check your aim line"
    return PracticeFeedback(
        distance_from_hole_m=distance,
        holed=result.holed,
        total_distance_m=result.total_distance_m,
        suggestion=suggestion,
    )


def putt_result_to_dict(result: PuttResult) -> dict[str, Any]:
    """JSON-ready document for ``result``, naming its roll model."""
    collision = result.collision
    return {
        "schema": PUTT_RESULT_SCHEMA,
        "roll_model": USGA_STIMP_ROLL_MODEL,
        **{name: getattr(result, name) for name in _SCALARS},
        "collision": None if collision is None else dataclasses.asdict(collision),
        "samples": [
            {**dataclasses.asdict(sample), "mode": sample.mode.value}
            for sample in result.samples
        ],
    }


def putt_result_from_dict(document: dict[str, Any]) -> PuttResult:
    """Rebuild a :class:`PuttResult`, refusing documents this core cannot reproduce.

    Raises:
        ValueError: If the schema is unknown, the roll model is missing,
            or the putt came from a different roll model.
    """
    if document.get("schema") != PUTT_RESULT_SCHEMA:
        raise ValueError(f"unknown putt result schema: {document.get('schema')!r}")
    model = document.get("roll_model")
    if model is None:
        raise ValueError("putt result names no roll model (ADR-0045 F1)")
    if model != USGA_STIMP_ROLL_MODEL:
        raise ValueError(
            f"putt result was produced by {model}; this core runs "
            f"{USGA_STIMP_ROLL_MODEL} only (ADR-0058)"
        )
    collision = document["collision"]
    return PuttResult(
        **{name: document[name] for name in _SCALARS},
        collision=None if collision is None else _collision(collision),
        samples=tuple(
            TrajectorySample(**{**sample, "mode": Mode(sample["mode"])})
            for sample in document["samples"]
        ),
    )


def save_putt_result(result: PuttResult, path: str | Path) -> None:
    """Write ``result`` as a JSON document at ``path``."""
    Path(path).write_text(json.dumps(putt_result_to_dict(result), indent=2))


def load_putt_result(path: str | Path) -> PuttResult:
    """Read a document written by :func:`save_putt_result` (fail-closed)."""
    return putt_result_from_dict(json.loads(Path(path).read_text()))


def _collision(fields: dict[str, Any]) -> CollisionReport:
    """JSON turns the wrench tuples into lists; restore them."""
    wrench = ("attachment_impulse_n_s", "attachment_moment_n_m_s")
    restored: dict[str, Any] = {
        key: tuple(value) if key in wrench else value for key, value in fields.items()
    }
    return CollisionReport(**restored)
