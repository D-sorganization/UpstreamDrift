"""BunkerShot3D designer workbench, versioned API (issue #9545).

A thin v1 route over the same headless surface the PyQt workbench drives,
:class:`src.tools.bunker_shot_gui.model.WorkbenchModel`. It closes the
``api: null`` gap of ``tools.bunkershot3d_workbench`` in
``src/config/feature_parity.json`` and adds no physics of its own.

Contract
--------

* **Inputs are validated at the boundary.** Every field names its unit, the
  ranges are the desktop panel's (both read ``INPUT_RANGES``), unknown
  fields are refused so a value in the wrong unit cannot be silently
  ignored, and the handedness must be stated. Constructibility is still judged by the domain objects;
  their :class:`~src.tools.bunker_shot_gui.design.WorkbenchInputError` is a
  422, not a 500.
* **The solve is bounded.** One nominal F0 shot at the default study
  settings -- the playability sweep and the A/B ranking, which cost tens of
  shots, are not run here. Long jobs and cancellation belong to #8880/#9472.
* **The verdict travels with every number.** The response carries the tier,
  the verdict, the in-frame stamp the PyQt views draw
  (:func:`~src.tools.bunker_shot_gui.report.validity_stamp`), the source of
  every sand property and a SHA-256 over the canonical JSON of the record.
* **A playability objective is never predictive (#9239).** No
  sand-to-ball transfer qualification ships, so a ``predictive`` objective
  is refused and an exploratory one is reported with ``ranking_permitted``
  false and its degeneracy stated.

The workbench model imports in seconds, so it is imported inside the
handler, never at route discovery (the #8943 lazy-import pattern).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from src.api.middleware.error_handler import handle_api_errors
from src.tools.bunker_shot_gui.input_ranges import INPUT_RANGES

if TYPE_CHECKING:
    from src.tools.bunker_shot_gui.design import SwingSetup
    from src.tools.bunker_shot_gui.model import DesignEvaluation, ShotOutcome

__all__ = [
    "SCHEMA_VERSION",
    "BunkerWorkbenchRequestV1",
    "evaluation_record",
    "record_digest",
    "router",
]

SCHEMA_VERSION = "bunker-workbench-evaluation/1"
"""Version of the response record; bumped on any breaking change."""

MODEL_SOURCE = "src.tools.bunker_shot_gui.model.WorkbenchModel"
"""The headless surface every number in the response comes from."""

_DISPOSITION_UNCALIBRATED = "unavailable-uncalibrated"
"""``bunkershot3d.ball.ObjectiveDisposition.UNAVAILABLE_UNCALIBRATED``.

No ``TransferQualification`` ships, so this is the only disposition the
carry objective can have (#9239). Spelled here so that refusing a
predictive objective does not cost the multi-second model import."""

_LEFT_HANDED_REASON = (
    "the F0 workbench scene is built for a right-handed player (+x toward the "
    "target, +z up, toe on +y; bunkershot3d.metrics.loads). A left-handed "
    "delivery mirrors the scene and flips every signed lateral output, such "
    "as the aim offset and the shaft-axis moment. That mirror is not "
    "implemented, so the request is refused rather than answered in the "
    "wrong frame"
)

CONVENTIONS: dict[str, Any] = {
    "handedness": "right",
    "frame": "+x toward the target, +z up, toe on +y for a right-handed player",
    "bounce": "marketed",
    "attack_angle": "negative for a descending blow",
    "units": {
        "loft_deg": "deg",
        "marketed_bounce_deg": "deg",
        "sole_width_mm": "mm",
        "entry_height_mm": "mm",
        "leading_edge_radius_mm": "mm",
        "camber_area_mm2": "mm^2",
        "firmness_kg_per_cm2": "kg/cm^2",
        "clubhead_speed_mps": "m/s",
        "attack_angle_deg": "deg",
        "face_open_deg": "deg",
        "shaft_lean_deg": "deg",
        "entry_distance_behind_ball_m": "m",
        "ball_depth_m": "m",
        "target_carry_m": "m",
    },
}
"""The conventions every v1 record is expressed in, echoed in each response."""

router = APIRouter(prefix="/tools/bunker-workbench", tags=["bunker-workbench"])


# ------------------------------------------------------------------ request


def _bounded(key: str, *, required: bool = False) -> Any:
    """A pydantic field bounded by the shared ``INPUT_RANGES[key]``.

    Optional fields default to ``None`` (keep the preset or domain default);
    ``required`` fields have no default.
    """
    low, high = INPUT_RANGES[key]
    if required:
        return Field(ge=low, le=high)
    return Field(default=None, ge=low, le=high)


class _Strict(BaseModel):
    """Refuses unknown fields and non-finite numbers."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class BunkerDesignV1(_Strict):
    """A candidate sole; ``None`` keeps the grind preset's value."""

    name: str = Field(min_length=1, max_length=64)
    grind_preset: str | None = Field(default=None, min_length=1, max_length=64)
    loft_deg: float | None = _bounded("loft_deg")
    marketed_bounce_deg: float | None = _bounded("marketed_bounce_deg")
    sole_width_mm: float | None = _bounded("sole_width_mm")
    entry_height_mm: float | None = _bounded("entry_height_mm")
    leading_edge_radius_mm: float | None = _bounded("leading_edge_radius_mm")
    camber_area_mm2: float | None = _bounded("camber_area_mm2")
    heel_relief_fraction: float | None = _bounded("heel_relief_fraction")
    toe_relief_fraction: float | None = _bounded("toe_relief_fraction")


class BunkerSandV1(_Strict):
    """The playing condition; ``None`` keeps the preset's firmness."""

    preset: str | None = Field(default=None, min_length=1, max_length=32)
    firmness_kg_per_cm2: float | None = _bounded("firmness_kg_per_cm2")


class BunkerSwingV1(_Strict):
    """The delivery; ``None`` keeps the workbench's splash-shot default."""

    clubhead_speed_mps: float | None = _bounded("clubhead_speed_mps")
    attack_angle_deg: float | None = _bounded("attack_angle_deg")
    face_open_deg: float | None = _bounded("face_open_deg")
    shaft_lean_deg: float | None = _bounded("shaft_lean_deg")
    entry_distance_behind_ball_m: float | None = _bounded(
        "entry_distance_behind_ball_m"
    )
    ball_depth_m: float | None = _bounded("ball_depth_m")
    dynamic_terms_active: bool | None = None


class BunkerObjectiveV1(_Strict):
    """A carry-target playability objective and the use it is asked for."""

    use: Literal["exploratory", "predictive"]
    target_carry_m: float = _bounded("target_carry_m", required=True)
    tolerance_fraction: float = _bounded("carry_tolerance_fraction", required=True)


class BunkerWorkbenchRequestV1(_Strict):
    """One bounded nominal evaluation."""

    handedness: Literal["right", "left"]
    design: BunkerDesignV1
    sand: BunkerSandV1 = Field(default_factory=BunkerSandV1)
    swing: BunkerSwingV1 = Field(default_factory=BunkerSwingV1)
    objective: BunkerObjectiveV1 | None = None


# ----------------------------------------------------------------- response


class BunkerVerdictV1(BaseModel):
    status: str
    headline: str
    caveats: list[str]
    reasons: list[str]
    reason_count: int


class BunkerShotMetricsV1(BaseModel):
    peak_force_n: float | None
    impulse_n_s: float | None
    entry_speed_mps: float | None
    exit_speed_mps: float | None
    max_depth_m: float | None
    contact_duration_s: float | None
    peak_inertial_fraction: float | None


class BunkerDeliveredV1(BaseModel):
    effective_loft_deg: float
    effective_marketed_bounce_deg: float
    presentation_bounce_deg: float
    aim_offset_deg: float


class BunkerCarryV1(BaseModel):
    carry_m: float
    verdict_status: str
    band_m: list[float] | None
    band_reasons: list[str]


class BunkerSourceStampV1(BaseModel):
    basis: str
    source: str


class BunkerSourcesV1(BaseModel):
    model: str
    sand: dict[str, BunkerSourceStampV1]
    carry_band: str | None


class BunkerObjectiveReportV1(BaseModel):
    use: str
    target_carry_m: float
    tolerance_fraction: float
    disposition: str
    nominal_carry_m: float | None
    degenerate: bool
    ranking_permitted: bool
    reason: str


class BunkerWorkbenchEvaluationV1(BaseModel):
    """The record returned for one evaluation; ``digest`` covers the rest."""

    schema_version: str
    conventions: dict[str, Any]
    inputs: dict[str, Any]
    tier: str
    tier_model: str
    verdict: BunkerVerdictV1
    stamp: str
    refused: bool
    shot: BunkerShotMetricsV1
    delivered: BunkerDeliveredV1
    carry: BunkerCarryV1 | None
    unavailable: list[str]
    sources: BunkerSourcesV1
    objective: BunkerObjectiveReportV1 | None
    predictive: bool
    digest: str


# ------------------------------------------------------------------- record


def record_digest(record: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a record without its digest.

    Args:
        record: The record; a ``digest`` key, if present, is excluded.

    Returns:
        64 lowercase hex characters.
    """
    from bunkershot3d.provenance.hashing import canonical_json

    body = {key: value for key, value in record.items() if key != "digest"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def _verdict(shot: ShotOutcome) -> dict[str, Any]:
    """The verdict block: status, headline, caveats, truncated reasons."""
    from src.tools.bunker_shot_gui.report import MAX_REPORTED_REASONS, status_headline

    verdict = shot.verdict
    reasons = list(verdict.reasons)
    return {
        "status": shot.status.value,
        "headline": status_headline(shot.status),
        "caveats": [caveat.value for caveat in verdict.caveats],
        "reasons": reasons[:MAX_REPORTED_REASONS],
        "reason_count": len(reasons),
    }


def _shot_metrics(shot: ShotOutcome) -> dict[str, float | None]:
    """The scalar shot metrics; all ``None`` for a refused shot."""
    return {
        "peak_force_n": shot.peak_force_n,
        "impulse_n_s": shot.impulse_n_s,
        "entry_speed_mps": shot.entry_speed_mps,
        "exit_speed_mps": shot.exit_speed_mps,
        "max_depth_m": shot.max_depth_m,
        "contact_duration_s": shot.contact_duration_s,
        "peak_inertial_fraction": shot.peak_inertial_fraction,
    }


def _delivered(shot: ShotOutcome) -> dict[str, float]:
    """Effective loft, bounce and aim at impact."""
    delivered = shot.delivered
    return {
        "effective_loft_deg": float(delivered.effective_loft_deg),
        "effective_marketed_bounce_deg": float(delivered.effective_bounce.angle_deg),
        "presentation_bounce_deg": float(delivered.presentation_bounce_deg),
        "aim_offset_deg": float(delivered.aim_offset_deg),
    }


def _carry(shot: ShotOutcome) -> dict[str, Any] | None:
    """The carry with its verdict and band, or ``None`` when there is none."""
    verdict = shot.carry_verdict
    if shot.carry_m is None or verdict is None:
        return None
    band = shot.carry_band
    return {
        "carry_m": float(shot.carry_m),
        "verdict_status": verdict.status.value,
        "band_m": (None if band is None else [band.lower, band.central, band.upper]),
        "band_reasons": list(shot.carry_band_reasons),
    }


def _sources(evaluation: DesignEvaluation) -> dict[str, Any]:
    """Where the model, every sand property and the carry band come from."""
    from src.tools.bunker_shot_gui.uncertainty import CARRY_BAND_SOURCE

    provenance = evaluation.sand.provenance
    entries = provenance.entries
    band = evaluation.shot.carry_band
    return {
        "model": MODEL_SOURCE,
        "sand": {
            name: {"basis": entry.basis.value, "source": entry.source}
            for name, entry in entries.items()
        },
        "carry_band": None if band is None else CARRY_BAND_SOURCE,
    }


def _inputs(evaluation: DesignEvaluation, swing: SwingSetup) -> dict[str, Any]:
    """The resolved inputs the record is bound to."""
    sand = evaluation.sand
    return {
        "design": dataclasses.asdict(evaluation.design),
        "sand": {
            "name": sand.name,
            "firmness_kg_per_cm2": float(sand.firmness_kg_per_cm2),
        },
        "swing": dataclasses.asdict(swing),
    }


def _objective_report(
    objective: BunkerObjectiveV1, carry_m: float | None
) -> dict[str, Any]:
    """An exploratory objective, reported but never allowed to rank.

    Preconditions:
        ``objective.use`` is ``"exploratory"``; a predictive one is refused
        before any solve.
    """
    assert objective.use == "exploratory", "predictive objectives are refused"
    target = objective.target_carry_m
    degenerate = (
        carry_m is None or abs(carry_m - target) > objective.tolerance_fraction * target
    )
    return {
        "use": objective.use,
        "target_carry_m": target,
        "tolerance_fraction": objective.tolerance_fraction,
        "disposition": _DISPOSITION_UNCALIBRATED,
        "nominal_carry_m": carry_m,
        "degenerate": degenerate,
        "ranking_permitted": False,
        "reason": (
            "no sand-to-ball transfer qualification exists (#9239, #9543), so "
            "carry is a placeholder product and cannot rank designs"
        ),
    }


def evaluation_record(
    evaluation: DesignEvaluation,
    swing: SwingSetup,
    *,
    handedness: str,
    objective: BunkerObjectiveV1 | None,
) -> dict[str, Any]:
    """Build the v1 record for one evaluation, digest included.

    A pure function of the model's output, so the same inputs through the
    PyQt workbench and through this route produce the same record.

    Args:
        evaluation: The model's evaluation of one design.
        swing: The delivery it was evaluated at; not kept by the model.
        handedness: The stated handedness; only ``"right"`` is supported.
        objective: An exploratory objective to report, or ``None``.

    Returns:
        The record, with ``digest`` over every other key.

    Raises:
        ValueError: If ``handedness`` is not ``"right"``.
    """
    from src.tools.bunker_shot_gui.report import TIER_MODEL_NAMES, validity_stamp

    if handedness != "right":
        raise ValueError(_LEFT_HANDED_REASON)
    shot = evaluation.shot
    tier = shot.fidelity_tier
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "conventions": CONVENTIONS,
        "inputs": _inputs(evaluation, swing),
        "tier": tier.value,
        "tier_model": TIER_MODEL_NAMES[tier],
        "verdict": _verdict(shot),
        "stamp": validity_stamp(shot.status, tier),
        "refused": shot.refused,
        "shot": _shot_metrics(shot),
        "delivered": _delivered(shot),
        "carry": _carry(shot),
        "unavailable": list(shot.unavailable),
        "sources": _sources(evaluation),
        "objective": (
            None if objective is None else _objective_report(objective, shot.carry_m)
        ),
        "predictive": False,
    }
    record["digest"] = record_digest(record)
    return record


# -------------------------------------------------------------------- route


def _unprocessable(code: str, message: str, **extra: str) -> HTTPException:
    """A 422 with a machine-readable code and a human-readable message."""
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={"code": code, "message": message, **extra},
    )


def _require_answerable(request: BunkerWorkbenchRequestV1) -> None:
    """Refuse, before any solve, a request this route cannot answer honestly.

    Raises:
        HTTPException: 422 for a left-handed request or a predictive objective.
    """
    if request.handedness != "right":
        raise _unprocessable("handedness_unsupported", _LEFT_HANDED_REASON)
    objective = request.objective
    if objective is not None and objective.use == "predictive":
        raise _unprocessable(
            "objective_not_predictive",
            "a carry objective cannot rank designs as predictive: no "
            "sand-to-ball transfer qualification exists (#9239, #9543). "
            "Request it with use='exploratory'.",
            disposition=_DISPOSITION_UNCALIBRATED,
        )


@router.post("/v1/evaluate", response_model=BunkerWorkbenchEvaluationV1)
@handle_api_errors
def evaluate_v1(request: BunkerWorkbenchRequestV1) -> dict[str, Any]:
    """Run one bounded nominal F0 solve and return the stamped record.

    Synchronous on purpose: FastAPI runs it in its threadpool, so the
    second-scale solve does not block the event loop.

    Raises:
        HTTPException: 422 for input the workbench cannot construct, a
            left-handed request or a predictive objective.
    """
    _require_answerable(request)
    from src.tools.bunker_shot_gui.design import (
        SandCondition,
        SwingSetup,
        WedgeDesign,
        WorkbenchInputError,
    )
    from src.tools.bunker_shot_gui.model import WorkbenchModel

    try:
        design = WedgeDesign(**request.design.model_dump(exclude_none=True))
        sand = SandCondition(**request.sand.model_dump(exclude_none=True))
        swing = SwingSetup(**request.swing.model_dump(exclude_none=True))
        evaluation = WorkbenchModel().evaluate(
            design, sand, swing, include_playability=False
        )
    except WorkbenchInputError as error:
        raise _unprocessable("invalid_workbench_input", str(error)) from error
    return evaluation_record(
        evaluation, swing, handedness=request.handedness, objective=request.objective
    )
