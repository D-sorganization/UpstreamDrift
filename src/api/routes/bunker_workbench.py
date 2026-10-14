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
* **The solve is bounded.** ``/v1/evaluate`` runs one nominal F0 shot at
  the default study settings, without the playability sweep. ``/v1/compare``
  runs the A/B ranking, which costs one nominal shot plus the fixed
  ``playability_points ** 2`` sweep per design (52 shots at the defaults);
  no request field can enlarge it. Long jobs and cancellation belong to
  #8880/#9472.
* **The verdict travels with every number.** The response carries the tier,
  the verdict, the in-frame stamp the PyQt views draw
  (:func:`~src.tools.bunker_shot_gui.report.validity_stamp`), the source of
  every sand property and a SHA-256 over the canonical JSON of the record.
* **A playability objective is never predictive (#9239).** No
  sand-to-ball transfer qualification ships, so a ``predictive`` objective
  is refused and an exploratory one is reported with ``ranking_permitted``
  false and its degeneracy stated. The comparison carries the model's own
  :class:`~src.tools.bunker_shot_gui.objective.CarryObjectiveStanding` --
  disposition and reasons in the canonical
  :class:`~bunkershot3d.ball.ObjectiveDisposition` vocabulary -- and names a
  ``winner`` only when that standing is ``supported``; any other lead is an
  ``exploratory_leader``.
* **A saved record can be checked, never re-trusted.** ``/v1/verify`` takes a
  record a client saved and reloaded and accepts it only when its schema is
  known and its digest still covers it, so an edited record -- an exploratory
  leader promoted to ``winner``, say -- is refused as corrupt.

The workbench model imports in seconds, so it is imported inside the
handler, never at route discovery (the #8943 lazy-import pattern).
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, Body, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from src.api.middleware.error_handler import handle_api_errors
from src.tools.bunker_shot_gui.input_ranges import INPUT_RANGES
from src.tools.bunker_shot_gui.objective import (
    SHIPPED_CARRY_DISPOSITION,
    UNCALIBRATED_REASON,
)

if TYPE_CHECKING:
    from src.tools.bunker_shot_gui.design import (
        SandCondition,
        SolverSetup,
        SwingSetup,
        WedgeDesign,
    )
    from src.tools.bunker_shot_gui.model import (
        DesignEvaluation,
        ShotOutcome,
        WorkbenchComparison,
    )
    from src.tools.bunker_shot_gui.objective import CarryObjectiveStanding

__all__ = [
    "COMPARISON_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "BunkerWorkbenchCompareRequestV1",
    "BunkerWorkbenchRequestV1",
    "comparison_record",
    "evaluation_record",
    "record_digest",
    "router",
]

SCHEMA_VERSION = "bunker-workbench-evaluation/1"
"""Version of the response record; bumped on any breaking change."""

COMPARISON_SCHEMA_VERSION = "bunker-workbench-comparison/1"
"""Version of the A/B comparison record; bumped on any breaking change."""

_RECORD_SCHEMAS = frozenset({SCHEMA_VERSION, COMPARISON_SCHEMA_VERSION})
"""The record schemas ``/v1/verify`` knows how to check."""

_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
"""A SHA-256 hex digest as :func:`record_digest` spells it."""

MODEL_SOURCE = "src.tools.bunker_shot_gui.model.WorkbenchModel"
"""The headless surface every number in the response comes from."""

_DISPOSITION_UNCALIBRATED = SHIPPED_CARRY_DISPOSITION.value
"""The shipped carry objective's disposition, from its one owner (#9239).

No ``TransferQualification`` ships, so this is the only disposition the
carry objective can have. :mod:`~src.tools.bunker_shot_gui.objective` is
cheap to import, so refusing a predictive objective still does not cost the
multi-second model import."""

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


class BunkerWorkbenchCompareRequestV1(_Strict):
    """One A/B comparison of two differently named soles.

    ``objective``, when given, sets the target carry and window the ranking
    is judged against; omitted, the workbench defaults apply.
    """

    handedness: Literal["right", "left"]
    left: BunkerDesignV1
    right: BunkerDesignV1
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


class BunkerObjectiveStandingV1(BaseModel):
    """The model's :class:`CarryObjectiveStanding`, field for field."""

    disposition: str
    reasons: list[str]
    predictive: bool


class BunkerRankingV1(BaseModel):
    available: bool
    unavailable_reason: str | None
    shared_points: int
    banded_verdict: str | None
    separation_m: float | None
    separated: bool


class BunkerObjectiveRequestEchoV1(BaseModel):
    use: str
    target_carry_m: float
    tolerance_fraction: float


class BunkerWorkbenchComparisonV1(BaseModel):
    """The record returned for one A/B comparison; ``digest`` covers the rest.

    ``winner`` is ``None`` unless ``objective_standing.predictive``; a lead
    the bands show on an unqualified objective is ``exploratory_leader``.
    """

    schema_version: str
    conventions: dict[str, Any]
    left: BunkerWorkbenchEvaluationV1
    right: BunkerWorkbenchEvaluationV1
    objective: BunkerObjectiveRequestEchoV1 | None
    objective_standing: BunkerObjectiveStandingV1
    ranking: BunkerRankingV1
    predictive: bool
    winner: str | None
    exploratory_leader: str | None
    verdict_statement: str
    digest: str


class BunkerRecordVerificationV1(BaseModel):
    valid: bool
    schema_version: str
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
        "reason": UNCALIBRATED_REASON,
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


def _standing(standing: CarryObjectiveStanding) -> dict[str, Any]:
    """The objective standing in the canonical disposition vocabulary."""
    return {
        "disposition": standing.disposition.value,
        "reasons": list(standing.reasons),
        "predictive": standing.predictive,
    }


def _ranking(comparison: WorkbenchComparison) -> dict[str, Any]:
    """Whether the pair was ranked, on how many conditions, and how far apart."""
    banded = comparison.banded
    return {
        "available": banded is not None,
        "unavailable_reason": (
            comparison.verdict_statement if banded is None else None
        ),
        "shared_points": comparison.shared_points,
        "banded_verdict": None if banded is None else banded.verdict.value,
        "separation_m": None if banded is None else float(banded.separation),
        "separated": comparison.separated,
    }


def comparison_record(
    comparison: WorkbenchComparison,
    swing: SwingSetup,
    *,
    handedness: str,
    objective: BunkerObjectiveV1 | None,
) -> dict[str, Any]:
    """Build the v1 record for one A/B comparison, digest included.

    A pure function of the model's comparison: the standing, the winner and
    the exploratory leader are read off it, never re-derived, so the route
    can claim no more than :class:`WorkbenchComparison` allows (#9239).

    Args:
        comparison: The model's comparison of two designs.
        swing: The nominal delivery both were evaluated at.
        handedness: The stated handedness; only ``"right"`` is supported.
        objective: The exploratory objective the comparison was judged
            against, or ``None`` for the workbench defaults.

    Returns:
        The record, with ``digest`` over every other key. Postcondition:
        ``winner`` is ``None`` unless ``objective_standing.predictive``.

    Raises:
        ValueError: If ``handedness`` is not ``"right"``.
    """
    sides = {
        side: evaluation_record(
            evaluation, swing, handedness=handedness, objective=objective
        )
        for side, evaluation in (
            ("left", comparison.left),
            ("right", comparison.right),
        )
    }
    record: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "conventions": CONVENTIONS,
        **sides,
        "objective": None if objective is None else objective.model_dump(),
        "objective_standing": _standing(comparison.objective),
        "ranking": _ranking(comparison),
        "predictive": comparison.predictive,
        "winner": comparison.winner,
        "exploratory_leader": comparison.exploratory_leader,
        "verdict_statement": comparison.verdict_statement,
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


def _require_answerable(handedness: str, objective: BunkerObjectiveV1 | None) -> None:
    """Refuse, before any solve, a request this route cannot answer honestly.

    Args:
        handedness: The stated handedness.
        objective: The requested objective, if any.

    Raises:
        HTTPException: 422 for a left-handed request or a predictive objective.
    """
    if handedness != "right":
        raise _unprocessable("handedness_unsupported", _LEFT_HANDED_REASON)
    if objective is not None and objective.use == "predictive":
        raise _unprocessable(
            "objective_not_predictive",
            "a carry objective cannot rank designs as predictive: no "
            "sand-to-ball transfer qualification exists (#9239, #9543). "
            "Request it with use='exploratory'.",
            disposition=_DISPOSITION_UNCALIBRATED,
        )


@contextmanager
def _workbench_input() -> Iterator[None]:
    """Relabel the domain's construction failure as a precise 422."""
    from src.tools.bunker_shot_gui.design import WorkbenchInputError

    try:
        yield
    except WorkbenchInputError as error:
        raise _unprocessable("invalid_workbench_input", str(error)) from error


def _wedge(design: BunkerDesignV1) -> WedgeDesign:
    """The domain design; ``None`` fields keep the preset's values."""
    from src.tools.bunker_shot_gui.design import WedgeDesign

    return WedgeDesign(**design.model_dump(exclude_none=True))


def _conditions(
    sand: BunkerSandV1, swing: BunkerSwingV1
) -> tuple[SandCondition, SwingSetup]:
    """The domain sand condition and delivery."""
    from src.tools.bunker_shot_gui.design import SandCondition, SwingSetup

    return (
        SandCondition(**sand.model_dump(exclude_none=True)),
        SwingSetup(**swing.model_dump(exclude_none=True)),
    )


def _study_settings(objective: BunkerObjectiveV1 | None) -> SolverSetup:
    """The study settings, with the objective's target and window if given."""
    from src.tools.bunker_shot_gui.design import SolverSetup

    if objective is None:
        return SolverSetup()
    return SolverSetup(
        target_carry_m=objective.target_carry_m,
        carry_tolerance_fraction=objective.tolerance_fraction,
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
    _require_answerable(request.handedness, request.objective)
    from src.tools.bunker_shot_gui.model import WorkbenchModel

    with _workbench_input():
        design = _wedge(request.design)
        sand, swing = _conditions(request.sand, request.swing)
        evaluation = WorkbenchModel().evaluate(
            design, sand, swing, include_playability=False
        )
    return evaluation_record(
        evaluation, swing, handedness=request.handedness, objective=request.objective
    )


@router.post("/v1/compare", response_model=BunkerWorkbenchComparisonV1)
@handle_api_errors
def compare_v1(request: BunkerWorkbenchCompareRequestV1) -> dict[str, Any]:
    """Compare two soles over the shared delivery sweep, with the standing.

    The ranking is the model's (:meth:`WorkbenchModel.compare`); this route
    adds no judgement of its own. With no transfer qualification shipped the
    standing is ``unavailable-uncalibrated``, so ``winner`` is ``None`` and a
    lead the bands show is reported as ``exploratory_leader`` (#9239).

    Synchronous on purpose, like :func:`evaluate_v1`.

    Raises:
        HTTPException: 422 for two designs with one name, input the workbench
            cannot construct, a left-handed request or a predictive objective.
    """
    _require_answerable(request.handedness, request.objective)
    if request.left.name == request.right.name:
        raise _unprocessable(
            "design_names_not_distinct",
            "the two designs must have different names; the comparison "
            f"reports them by name and both are {request.left.name!r}",
        )
    from src.tools.bunker_shot_gui.model import WorkbenchModel

    with _workbench_input():
        left, right = _wedge(request.left), _wedge(request.right)
        sand, swing = _conditions(request.sand, request.swing)
        model = WorkbenchModel(_study_settings(request.objective))
        comparison = model.compare(left, right, sand, swing)
    return comparison_record(
        comparison, swing, handedness=request.handedness, objective=request.objective
    )


@router.post("/v1/verify", response_model=BunkerRecordVerificationV1)
@handle_api_errors
def verify_v1(record: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Check a saved v1 record before it is trusted again.

    Fail-closed: the record must name a known schema and carry a well-formed
    digest that still covers every other key. Nothing is re-solved.

    Raises:
        HTTPException: 422 ``record_schema_unknown``,
            ``record_digest_invalid`` or ``record_digest_mismatch``.
    """
    schema = record.get("schema_version")
    if schema not in _RECORD_SCHEMAS:
        raise _unprocessable(
            "record_schema_unknown",
            f"schema_version {schema!r} is not one of {sorted(_RECORD_SCHEMAS)}",
        )
    digest = record.get("digest")
    if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
        raise _unprocessable(
            "record_digest_invalid",
            "the record carries no 64-character lowercase SHA-256 digest",
        )
    if not hmac.compare_digest(record_digest(record), digest):
        raise _unprocessable(
            "record_digest_mismatch",
            "the record was changed after it was produced; its digest no "
            "longer covers its contents",
        )
    return {"valid": True, "schema_version": schema, "digest": digest}
