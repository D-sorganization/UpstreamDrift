"""SDK-free, hash-bound local historical research handoff; never publication authority."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from src.shared.python.shadow_tracker.source_records import FrameIdentity
from .artifact_handoff import compute_file_sha256

if TYPE_CHECKING:
    from .necromatcher import NecromatcherLibrary

SCHEMA_ID = "urn:upstreamdrift:historical-player-research:v1"
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[4]
    / "docs/api/contracts/historical-player-research-v1.schema.json"
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_AUDIT_LIMIT = 20_000_000
_SUMMARIES = {
    "necromatcher/normalized-coverage-v15-independent-assessment/1": (
        "both_runs_verified",
        "source_runtime_parent_artifact_evidence_brackets_verified",
    ),
    "necromatcher/geometry-weight-v14-independent-assessment/1": (
        "both_runs_verified",
        "source_runtime_parent_artifact_evidence_brackets_verified",
    ),
    "necromatcher/controlled-v12-independent-assessment/1": (
        "all_four_verified",
        "source_runtime_driver_artifact_reference_brackets_verified",
    ),
}
_METRIC_KEYS = (
    "observed_original_rms_pixels",
    "dense_original_rms_pixels",
    "maximum_sampled_grip_gap_m",
    "maximum_sampled_grip_rotation_rad",
    "maximum_sampled_ground_penetration_m",
    "source_frame_count",
    "dense_observed_point_count",
)


@dataclass(frozen=True)
class ResearchAuditPin:
    """Explicit read-only evidence pins; source paths never enter exported JSON."""

    fit_id: str
    assessment_path: Path
    assessment_sha256: str
    full_audit_path: Path
    full_audit_sha256: str
    source_video_path: Path
    rights_status: str = "unresolved"

    def __post_init__(self) -> None:
        if not isinstance(self.fit_id, str) or not self.fit_id.strip():
            raise ValueError("Research fit identity must be nonempty")
        for value in (self.assessment_sha256, self.full_audit_sha256):
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise ValueError("External audit pins require plain lowercase SHA-256")
        for path in (
            self.assessment_path,
            self.full_audit_path,
            self.source_video_path,
        ):
            if not isinstance(path, Path):
                raise TypeError("Evidence paths must be pathlib.Path objects")
        if self.rights_status not in {
            "unresolved",
            "restricted",
            "documented_local_use",
        }:
            raise ValueError("Rights decision cannot authorize distribution")


def historical_research_schema_bytes() -> bytes:
    """Return the exact versioned provider contract, shared with the consumer."""
    return _SCHEMA_PATH.read_bytes()


def _plain_hash(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("SHA-256 identity must be a string")
    value = value.removeprefix("sha256:")
    if _DIGEST.fullmatch(value) is None:
        raise ValueError("Expected exact SHA-256 identity")
    return value


def _unique_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate audit JSON keys are forbidden")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"Nonfinite audit JSON constant: {value}")


def _read_pin(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _AUDIT_LIMIT:
        raise ValueError("Pinned audit must be a bounded regular file")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("External audit bytes changed from their explicit SHA-256 pin")
    try:
        value = json.loads(
            payload, object_pairs_hook=_unique_json, parse_constant=_nonfinite
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid pinned audit JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Pinned audit must be a JSON object")
    return value


def _research_flags(value: dict[str, Any], fields: Sequence[str]) -> None:
    for name in fields:
        if value.get(name) is not False:
            raise ValueError(f"Research artifact cannot promote {name}")


def _external_audit(
    pin: ResearchAuditPin,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    summary = _read_pin(pin.assessment_path, pin.assessment_sha256)
    audit = _read_pin(pin.full_audit_path, pin.full_audit_sha256)
    keys = _SUMMARIES.get(str(summary.get("schema")))
    if keys is None or any(summary.get(key) is not True for key in keys):
        raise ValueError("Unsupported or unverified independent assessment protocol")
    _research_flags(
        summary,
        (
            "scientific_acceptance",
            "continuous_certified",
            "physical_time_qualified",
            "optimization_performed",
        ),
    )
    if audit.get("schema") != "necromatcher/independent-bounded-trial-audit/1":
        raise ValueError("Unsupported full independent audit schema")
    if audit.get("audit_source_runtime_unchanged") is not True:
        raise ValueError("Independent audit source/runtime bracket was not verified")
    _research_flags(
        audit,
        (
            "scientific_acceptance",
            "continuous_nonlinear_certified",
            "physical_time_qualified",
            "optimization_performed",
            "saved_optimizer_converged",
        ),
    )
    matches = [run for run in summary["runs"] if run.get("fit_id") == pin.fit_id]
    if len(matches) != 1:
        raise ValueError("Assessment must identify exactly one selected fit")
    run = matches[0]
    reference = run["fresh_full_audit"]
    if (
        _plain_hash(reference["sha256"]) != pin.full_audit_sha256
        or Path(reference["path"]).resolve() != pin.full_audit_path.resolve()
    ):
        raise ValueError("Assessment/full audit reference differs from explicit pins")
    for key in (
        *_METRIC_KEYS,
        "fit_id",
        "fit_hash",
        "model_id",
        "model_hash",
        "capture_id",
        "capture_hash",
    ):
        if run["metrics"].get(key) != audit.get(key):
            raise ValueError(
                f"Independent assessment differs from its full audit: {key}"
            )
    if _plain_hash(run["fit_sha256"]) != _plain_hash(audit["fit_hash"]):
        raise ValueError("Assessment fit identity differs from full audit")
    return run, audit, summary["producer"]


def _match_saved(
    library: NecromatcherLibrary,
    pin: ResearchAuditPin,
    run: dict[str, Any],
    audit: dict[str, Any],
    producer: str,
) -> dict[str, Any]:
    fit = library.load_fit(pin.fit_id)
    asset = library.load_asset(pin.fit_id)
    if asset.kind != "kinematic_fit" or audit["fit_id"] != pin.fit_id:
        raise ValueError("Research handoff requires the exact selected kinematic fit")
    if _plain_hash(asset.metadata["hash"]) != _plain_hash(audit["fit_hash"]):
        raise ValueError("Independent audit fit hash differs from saved bytes")
    for name, kind in (("model", "native_model"), ("capture", "image_capture")):
        parent = library.load_asset(fit[f"{name}_id"])
        if (
            parent.kind != kind
            or parent.session_id != asset.session_id
            or audit[f"{name}_id"] != parent.dataset_id
            or audit[f"{name}_hash"] != parent.metadata["hash"]
        ):
            raise ValueError(
                "Independent audit parent binding differs from canonical library"
            )
    if fit.get("qualification") != "monocular_research_hypothesis":
        raise ValueError("Only monocular research hypotheses are supported")
    _research_flags(fit, ("physical_time_qualified", "dynamics_replayed"))
    stamp = fit["provenance"]["execution_stamp"]
    if (
        stamp != audit["producer_execution_stamp"]
        or stamp["source_commit"] != producer
        or not isinstance(producer, str)
        or _COMMIT.fullmatch(producer) is None
    ):
        raise ValueError(
            "Independent audit numerical producer differs from saved provenance"
        )
    original = fit["evidence"]["original_fit"]
    restart = run.get("fresh_strict_restart", run.get("initial_restart_and_images"))
    if (
        not isinstance(restart, dict)
        or restart.get("initial_spline_six_fields_exact") is not True
    ):
        raise ValueError("Fresh restart/image evidence is unavailable")
    for saved, fresh in (
        ("rms_pixels", "training_rms_pixels"),
        ("held_out_rms_pixels", "held_out_rms_pixels"),
        ("dense_rms_pixels", "dense_original_rms_pixels"),
    ):
        expected = (
            restart[fresh] if fresh != "dense_original_rms_pixels" else audit[fresh]
        )
        if not math.isclose(original[saved], expected, rel_tol=0, abs_tol=1e-8):
            raise ValueError(
                "Audited image RMS differs from saved original-fit evidence"
            )
    if not math.isclose(
        original["rms_pixels"],
        audit["observed_original_rms_pixels"],
        rel_tol=0,
        abs_tol=1e-8,
    ):
        raise ValueError("Full audit training RMS differs from saved image evidence")
    if (
        original.get("converged") is not False
        or run.get("optimizer_converged") is not False
    ):
        raise ValueError("v1 requires the recorded nonconverged research cohort")
    _image_counts(original, restart, audit)
    options = fit["provenance"]["request_options"]
    if (
        options.get("unknown_visibility_weight") != 0.5
        or restart["unknown_visibility_weight"] != 0.5
    ):
        raise ValueError("Image metric requires explicit unknown-visibility weight 0.5")
    return fit


def _image_counts(
    original: dict[str, Any], restart: dict[str, Any], audit: dict[str, Any]
) -> None:
    if (
        restart["training_frame_count"] != len(original["frame_indices"])
        or restart["training_frame_count"] > audit["source_frame_count"]
        or restart["training_observed_point_count"]
        > audit["dense_observed_point_count"]
        or restart["dense_frame_count"] != audit["source_frame_count"]
        or restart["dense_observed_point_count"] != audit["dense_observed_point_count"]
        or audit.get("added_image_objective_points") != 0
        or audit.get("saved_optimizer_ran") is not True
        or original.get("optimizer_ran") is not True
    ):
        raise ValueError(
            "Independent image evidence/counts differ from saved computation"
        )


def _sanitized(value: Any) -> None:
    if isinstance(value, str) and re.search(
        r"(?<![A-Za-z])[A-Za-z]:[/\\]|(?:^|\s)/(?:Users|home|tmp)/|\\\\", value
    ):
        raise ValueError("Local machine paths cannot enter sanitized research JSON")
    if isinstance(value, dict):
        for item in value.values():
            _sanitized(item)
    elif isinstance(value, list):
        for item in value:
            _sanitized(item)


def _clock(frames: list[dict[str, Any]]) -> dict[str, Any]:
    identities = [FrameIdentity(**frame) for frame in frames]
    if len(identities) < 2 or any(
        not f.is_timing_exact
        or f.physical_time_s is not None
        or f.timing_mode != "container_pts"
        for f in identities
    ):
        raise ValueError(
            "Research clock requires original exact PTS and unknown physical time"
        )
    times = [frame.presentation_time for frame in identities]
    if times[0] < 0 or any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("Research source PTS must increase")

    def rational(value: Fraction) -> dict[str, int]:
        return {"numerator": value.numerator, "denominator": value.denominator}

    return {
        "start": rational(times[0]),
        "end": rational(times[-1]),
        "origin": "video_presentation_time",
        "frame_count": len(frames),
        "physical_time": "unknown",
    }


def _record(library: NecromatcherLibrary, pin: ResearchAuditPin) -> dict[str, Any]:
    run, audit, producer = _external_audit(pin)
    fit = _match_saved(library, pin, run, audit, producer)
    asset = library.load_asset(pin.fit_id)
    capture = library.load_asset(fit["capture_id"])
    source_hash = _plain_hash(compute_file_sha256(pin.source_video_path))
    if (
        source_hash != capture.metadata.get("source_sha256")
        or source_hash != audit["source_video_identity"]["content_sha256"]
    ):
        raise ValueError(
            "Original downloaded source bytes differ from capture/audit identity"
        )
    swing = next(s for s in library.swings() if s.session_id == asset.session_id)
    player = next(p for p in library.players() if p.subject_id == swing.subject_id)
    clock = _clock(fit["frames"])
    if clock["frame_count"] != audit["source_frame_count"]:
        raise ValueError("Audited source frame count differs from canonical fit")
    restart = run.get("fresh_strict_restart", run.get("initial_restart_and_images"))
    targets = run["targets"].get("fixed_historical_V10_targets", run["targets"])
    _research_flags(targets, ("scientific_acceptance",))
    input_binding = {
        "schema": "historical-research/input-binding/1",
        "fit_sha256": _plain_hash(audit["fit_hash"]),
        "request_options": fit["provenance"]["request_options"],
        "assessment_sha256": pin.assessment_sha256,
        "full_audit_sha256": pin.full_audit_sha256,
    }
    return {
        "player_id": player.subject_id,
        "player_name": player.display_name,
        "swing_id": swing.session_id,
        "fit_id": pin.fit_id,
        "producer_commit": producer,
        "hashes": {
            "source": source_hash,
            "capture": _plain_hash(audit["capture_hash"]),
            "model": _plain_hash(audit["model_hash"]),
            "fit": _plain_hash(audit["fit_hash"]),
            "input": hashlib.sha256(_canonical(input_binding)).hexdigest(),
        },
        "source_clock": clock,
        "execution_status": "succeeded",
        "qualification": "monocular_research_hypothesis",
        "scientific_acceptance": "rejected",
        "optimizer_converged": False,
        "continuous_certified": False,
        "targets": [
            {"id": key.replace("_", "-"), "passed": value}
            for key, value in sorted(targets["checks"].items())
        ],
        "metrics": {
            "training_rms_px": audit["observed_original_rms_pixels"],
            "held_out_rms_px": restart["held_out_rms_pixels"],
            "dense_rms_px": audit["dense_original_rms_pixels"],
            "grip_gap_mm": 1000 * audit["maximum_sampled_grip_gap_m"],
            "grip_angle_deg": math.degrees(audit["maximum_sampled_grip_rotation_rad"]),
            "penetration_mm": 1000 * audit["maximum_sampled_ground_penetration_m"],
        },
        "metric_definition": "Confidence-weighted Euclidean landmark RMS: sqrt(sum(w*||pixel residual||^2)/sum(w)); all positive image observations, confidence times visibility, unknown visibility 0.5. Geometric maxima are finite sampled native hypotheses, not physical measurements or continuous certificates.",
        "unknown_visibility_weight": 0.5,
        "image_counts": {
            "training_frame_count": restart["training_frame_count"],
            "source_frame_count": audit["source_frame_count"],
            "training_observation_count": restart["training_observed_point_count"],
            "dense_observation_count": audit["dense_observed_point_count"],
        },
        "rights": {"status": pin.rights_status, "distribution": "not_authorized"},
        "limitations": [
            "Local research only; source footage distribution is not authorized.",
            "Physical playback scale, camera calibration, and historical anatomy are unqualified.",
            "Generic native body origins and attachment seeds do not reconstruct skin or the visible club shaft.",
            "Authored contact phases and coordinate ranges are operator hypotheses; nonlinear constraints are sampled, not continuously certified.",
            "Optimizer did not converge; finite target passes do not grant scientific acceptance.",
        ],
    }


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _validate_contract(value: dict[str, Any]) -> None:
    """Require optional schema validation only when export is explicitly requested."""
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import ValidationError
    except ImportError as exc:
        raise RuntimeError(
            "Historical research export requires the jsonschema schema-validation dependency"
        ) from exc
    schema = json.loads(historical_research_schema_bytes())
    try:
        Draft202012Validator(schema).validate(value)
    except ValidationError as exc:
        raise ValueError(f"Historical research contract: {exc.message}") from exc


def build_historical_research_package(
    library: NecromatcherLibrary, pins: Sequence[ResearchAuditPin], source_commit: str
) -> bytes:
    """Return deterministic sanitized JSON after exact canonical/audit/source checks.

    Postcondition: no original media/local paths; all records remain rejected research.
    This performs no native computation, library mutation, or public publication.
    """
    if not isinstance(source_commit, str) or _COMMIT.fullmatch(source_commit) is None:
        raise ValueError("Export implementation requires an exact source commit")
    if not pins or any(not isinstance(pin, ResearchAuditPin) for pin in pins):
        raise ValueError("Export requires explicit typed research evidence pins")
    if len({pin.fit_id for pin in pins}) != len(pins):
        raise ValueError("Research fit identities must be unique")
    try:
        records = [
            _record(library, pin) for pin in sorted(pins, key=lambda p: p.fit_id)
        ]
        value = {
            "schema": "upstreamdrift/historical-player-research/v1",
            "provider": {
                "repository": "D-sorganization/UpstreamDrift",
                "source_commit": source_commit,
                "source_url": f"https://github.com/D-sorganization/UpstreamDrift/tree/{source_commit}",
            },
            "distribution": {
                "scope": "local_research_only",
                "decision": "not_authorized_for_publication",
                "contains_original_media": False,
            },
            "records": records,
        }
        _sanitized(value)
        _validate_contract(value)
        payload = _canonical(value)
        # Bracket all canonical and external reads before returning candidate bytes.
        if records != [
            _record(library, pin) for pin in sorted(pins, key=lambda p: p.fit_id)
        ]:
            raise ValueError("Research inputs changed during export")
        return payload
    except (KeyError, TypeError, StopIteration) as exc:
        raise ValueError(f"Invalid research handoff evidence: {exc}") from exc


def export_historical_research_package(
    library: NecromatcherLibrary,
    pins: Sequence[ResearchAuditPin],
    source_commit: str,
    destination: Path,
) -> dict[str, Any]:
    """Write one new sanitized package, refusing original/previous file replacement."""
    if not isinstance(destination, Path):
        raise TypeError("Research destination must be a pathlib.Path")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Research output already exists")
    payload = build_historical_research_package(library, pins, source_commit)
    with destination.open("xb") as handle:
        handle.write(payload)
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "source_commit": source_commit,
        "schema_id": SCHEMA_ID,
        "scientific_acceptance": "rejected",
        "scope": "local_research_only",
    }
