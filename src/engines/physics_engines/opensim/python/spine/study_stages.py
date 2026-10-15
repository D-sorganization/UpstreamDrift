"""Stage runner of the spine study (SPN-13a, #12766).

Internal to :mod:`spine.study`: runs golf input, drive, inverse dynamics,
trunk static optimisation and joint reactions in order, records per-stage
provenance and metrics, writes the ``.mot`` and loads ``.npz`` artifacts and
collects the headline residual, reserve and context-band numbers.  A stage
exception (``ValueError``/``OSError``/``RuntimeError``) stops the run and is
recorded; it is never swallowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from src.engines.physics_engines.opensim.python.spine.drive import (
    FIXTURE_PELVIS,
    GOLF_PELVIS,
    DriveOptions,
    DriveResult,
    drive_segmental_model,
    export_mot,
)
from src.engines.physics_engines.opensim.python.spine.golf_input import (
    read_golf_humanoid_motion,
)
from src.engines.physics_engines.opensim.python.spine.inverse_dynamics import (
    ResidualBudget,
    SpineIDOptions,
    run_spinal_inverse_dynamics,
    spinal_generalized_forces,
)
from src.engines.physics_engines.opensim.python.spine.joint_reactions import (
    JointReactionOptions,
    compute_joint_reactions,
)
from src.engines.physics_engines.opensim.python.spine.static_optimization import (
    ReserveSpec,
    SpineSOOptions,
    run_spine_static_optimization,
)
from src.engines.physics_engines.opensim.python.spine.study_receipt import (
    FileRecord,
    StageRecord,
    round_sig,
)
from src.shared.python.core.constants import GRAVITY_FLOAT
from src.shared.python.pose_estimation.model_files import sha256_of
from src.shared.python.spine.context_bands import annotate
from src.shared.python.spine.coupling_data import coupling_law

logger = logging.getLogger(__name__)

_PELVIS = {"golf": GOLF_PELVIS, "fixture": FIXTURE_PELVIS}
_STAGE_ERRORS = (ValueError, OSError, RuntimeError)
_MAX_ERROR_CHARS = 400
#: Stage order; the failing stage is the first one without a record.
STAGE_NAMES = (
    "golf_input",
    "drive",
    "inverse_dynamics",
    "static_optimization",
    "joint_reactions",
)


@dataclass
class StageOutcome:
    """Mutable accumulator of one run; see :func:`run_stages`."""

    stages: list[StageRecord] = field(default_factory=list)
    artifacts: list[FileRecord] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    context: list[dict[str, Any]] = field(default_factory=list)
    residual: dict[str, Any] = field(default_factory=dict)
    reserve: dict[str, Any] = field(default_factory=dict)
    law: str = "n/a"
    distribution_source: str = "n/a"
    failed: bool = False
    ran: bool = False

    @classmethod
    def skipped(cls, reason: str) -> StageOutcome:
        """Outcome of a study whose stages were not run."""
        return cls(reasons=[f"stages not run: {reason}"])


def _num(value: Any) -> Any:
    if isinstance(value, bool | int | str):
        return value
    return round_sig(float(value))


def _record(
    stage: str, provenance: Mapping[str, str], metrics: Mapping[str, Any], scrub: Any
) -> StageRecord:
    return StageRecord(
        stage=stage,
        status="ok",
        provenance={str(k): scrub(str(v)) for k, v in sorted(provenance.items())},
        metrics={str(k): _num(v) for k, v in sorted(metrics.items())},
    )


def _drive(request: Any, model: Path, scrub: Any, out: StageOutcome) -> DriveResult:
    golf = read_golf_humanoid_motion(request.motion_path, request.golf_model_path)
    out.stages.append(
        _record(
            "golf_input",
            golf.source,
            {"n_frames": golf.time_s.size, "duration_s": golf.time_s[-1]},
            scrub,
        )
    )
    law = (
        None
        if request.coupling_law_id is None
        else coupling_law(request.coupling_law_id)
    )
    options = DriveOptions(
        tolerance_rad=request.drive_tolerance_rad, pelvis=_PELVIS[request.pelvis]
    )
    result = drive_segmental_model(
        golf, model_path=model, model_id=request.model_id, law=law, options=options
    )
    mot = export_mot(result, request.output_dir / f"{request.study_id}_coordinates.mot")
    out.artifacts.append(FileRecord("coordinates_mot", mot.name, sha256_of(mot)))
    n_invalid = int(np.count_nonzero(~result.valid))
    out.law = f"{result.law_id}@{result.law_version}"
    out.distribution_source = result.kinematics.distribution_source
    out.residual.update(
        drive_max_rad=_num(result.max_residual_rad),
        drive_rms_rad=_num(np.sqrt(np.mean(result.residual_rad**2))),
        drive_n_invalid=n_invalid,
    )
    if n_invalid:
        out.reasons.append(f"{n_invalid} drive frames invalid")
    prov = {
        "model_id": result.model_id,
        "model_sha256": result.model_sha256,
        "law": out.law,
        "distribution_source": out.distribution_source,
        **{f"input_{k}": v for k, v in result.input_source.items()},
    }
    out.stages.append(
        _record(
            "drive",
            prov,
            {"max_residual_rad": result.max_residual_rad, "n_invalid": n_invalid},
            scrub,
        )
    )
    return result


def _id_options(request: Any) -> SpineIDOptions:
    budget = ResidualBudget(
        force_fraction_bw=request.residual_force_fraction_bw,
        moment_fraction_bw_height=request.residual_moment_fraction,
    )
    return SpineIDOptions(pelvis=_PELVIS[request.pelvis], budget=budget)


def _inverse_dynamics(
    request: Any, model: Path, drive: DriveResult, scrub: Any, out: StageOutcome
) -> Any:
    common = {
        "external_loads": request.external_loads,
        "filter_hz": request.filter_hz,
        "options": _id_options(request),
    }
    series, budget = run_spinal_inverse_dynamics(
        model, drive, request.model_id, **common
    )
    efforts = spinal_generalized_forces(model, drive, request.model_id, **common)
    out.residual.update(
        id_force_ratio=_num(budget.force_ratio),
        id_moment_ratio=_num(budget.moment_ratio),
        id_peak_force_n=_num(budget.peak_force_n),
        id_peak_moment_nm=_num(budget.peak_moment_nm),
        id_budget_passed=bool(budget.passed),
    )
    if not budget.passed:
        out.reasons.append("inverse-dynamics pelvis residual exceeds the budget")
    metrics = {
        "force_ratio": budget.force_ratio,
        "moment_ratio": budget.moment_ratio,
        "n_frames": series.time_s.size,
    }
    out.stages.append(_record("inverse_dynamics", series.provenance, metrics, scrub))
    return series, efforts, budget


def _static_optimization(
    request: Any, model: Path, efforts: Any, scrub: Any, out: StageOutcome
) -> Any:
    options = SpineSOOptions(
        exponent=request.cost_exponent,
        reserves=ReserveSpec(
            limit_nm=request.reserve_limit_nm, max_share=request.reserve_max_share
        ),
        events=dict(request.events) or None,
        stride=request.stride,
        max_frames=request.max_frames,
    )
    so = run_spine_static_optimization(
        model, efforts, request.model_id, options=options
    )
    n_flagged = int(np.count_nonzero(so.flagged))
    n_failed = int(np.count_nonzero(~so.success))
    reasons = sorted({r for row in so.flag_reasons for r in row})
    out.reserve.update(
        n_frames=int(so.frame_index.size),
        n_flagged=n_flagged,
        n_unsuccessful=n_failed,
        flag_reasons=reasons,
        flagged_labels=[
            lab for lab, f in zip(so.frame_labels, so.flagged, strict=True) if f
        ],
        max_reserve_share=_num(np.max(so.reserve_share)),
        max_abs_reserve_nm=_num(np.max(np.abs(so.reserve_nm))),
    )
    if n_flagged:
        out.reasons.append(f"{n_flagged} frames flagged for reserve use")
    if n_failed:
        out.reasons.append(f"{n_failed} static-optimisation solves unsuccessful")
    metrics = {"n_frames": so.frame_index.size, "n_flagged": n_flagged}
    out.stages.append(_record("static_optimization", so.provenance, metrics, scrub))
    return so


def _joint_reactions(
    request: Any,
    model: Path,
    drive: DriveResult,
    so: Any,
    scrub: Any,
    out: StageOutcome,
) -> Any:
    options = JointReactionOptions(
        pelvis=_PELVIS[request.pelvis],
        external_loads=request.external_loads,
        filter_hz=request.filter_hz,
    )
    series = compute_joint_reactions(
        model, drive, so, request.model_id, options=options
    )
    out.stages.append(
        _record(
            "joint_reactions",
            series.provenance,
            {"n_frames": series.time_s.size, "n_levels": len(series.levels)},
            scrub,
        )
    )
    return series


def _context(series: Any, body_weight_n: float, out: StageOutcome) -> None:
    for item in annotate(series, body_weight_n / GRAVITY_FLOAT):
        band = item.band
        out.context.append(
            {
                "band_id": band.band_id,
                "study": band.study,
                "level": band.level.name,
                "quantity": band.quantity,
                "units": band.units,
                "band_value": _num(band.value),
                "qualifier": band.qualifier,
                "model_peak": _num(item.model_peak),
                "ratio": _num(item.ratio),
                "is_acceptance": item.is_acceptance,
            }
        )


def _write_loads(
    request: Any, id_series: Any, so: Any, jr: Any, out: StageOutcome
) -> None:
    path = request.output_dir / f"{request.study_id}_loads.npz"
    np.savez(
        path,
        time_s=so.time_s,
        frame_index=so.frame_index,
        levels=np.array([lv.name for lv in jr.levels]),
        id_time_s=id_series.time_s,
        id_moment_nm=id_series.moment_nm,
        so_activation=so.activation,
        so_reserve_nm=so.reserve_nm,
        so_required_nm=so.required_nm,
        joint_reaction_force_n=jr.force_n,
        joint_reaction_moment_nm=jr.moment_nm,
    )
    out.artifacts.append(FileRecord("loads_npz", path.name, sha256_of(path)))


def _run_chain(
    request: Any, model: Path, scrub: Callable[[str], str], out: Any
) -> None:
    drive = _drive(request, model, scrub, out)
    id_series, efforts, budget = _inverse_dynamics(request, model, drive, scrub, out)
    so = _static_optimization(request, model, efforts, scrub, out)
    jr = _joint_reactions(request, model, drive, so, scrub, out)
    _write_loads(request, id_series, so, jr, out)
    _context(jr, budget.body_weight_n, out)


def run_stages(request: Any, model: Path, scrub: Callable[[str], str]) -> StageOutcome:
    """Run the five stages; a failing stage is recorded last and sets ``failed``.

    ``scrub`` removes private directory names from every recorded text.
    Postcondition: ``stages`` lists the stages that ran, in order, the last
    one ``failed`` iff ``outcome.failed``.
    """
    out = StageOutcome(ran=True)
    request.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run_chain(request, model, scrub, out)
    except _STAGE_ERRORS as exc:
        logger.exception("spine study stage failed")
        stage = STAGE_NAMES[len(out.stages)]
        message = scrub(f"{type(exc).__name__}: {exc}")[:_MAX_ERROR_CHARS]
        out.stages.append(StageRecord(stage=stage, status="failed", error=message))
        out.failed = True
        out.reasons.append(f"stage {stage} failed")
    return out
