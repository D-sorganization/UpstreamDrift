"""Validate hosted workflow artifacts without interpreting scientific outcomes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np

from scripts.research.proximal_distal_energy.articulated_structural_factorial_evidence import (
    EVIDENCE_SIDECAR_SCHEMA,
    REQUIRED_EVIDENCE_ARRAYS,
    validate_structural_evidence_arrays,
)

_BASE_SCHEMA = "articulated-structural-factorial-artifact-receipt/1.1.0"
_ENRICHED_SCHEMA = "articulated-structural-factorial-artifact-receipt/1.2.0"
_SESSION_SCHEMA = "articulated-structural-factorial-session/1.0.0"
_JOB_NAME = "Structural Runtime Audit or Campaign Slice"
_CAMPAIGN_STEP = "Run Registered Structural Campaign Slice"
_UPLOAD_STEP = "Upload Structural Campaign Checkpoints"
_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_TERMINAL_CONCLUSIONS = {
    "action_required",
    "cancelled",
    "failure",
    "neutral",
    "skipped",
    "stale",
    "success",
    "timed_out",
}


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _selected_fields(
    record: Mapping[str, object], names: Sequence[str], *, prefix: str
) -> dict[str, object]:
    selected: dict[str, object] = {}
    for name in names:
        value = record.get(name)
        if value is None:
            raise ValueError(f"{prefix}.{name} is required")
        selected[name] = value
    return selected


def _terminal_record(record: Mapping[str, object], *, name: str) -> str:
    if record.get("status") != "completed":
        raise ValueError(f"{name} must be terminal")
    conclusion = record.get("conclusion")
    if conclusion not in _TERMINAL_CONCLUSIONS:
        raise ValueError(f"{name} conclusion is not terminal")
    return str(conclusion)


def _exact_named_record(
    values: object, *, expected_name: str, collection_name: str
) -> Mapping[str, object]:
    if not isinstance(values, list):
        raise ValueError(f"{collection_name} must be a list")
    matches = [
        _mapping(value, name=f"{collection_name} record")
        for value in values
        if isinstance(value, Mapping) and value.get("name") == expected_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{collection_name} must contain exactly one {expected_name!r} record"
        )
    return matches[0]


def _read_json_mapping(path: Path, *, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is unreadable") from exc
    return _mapping(value, name=name)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"cannot hash {path}") from exc


def _validate_artifact_tree(
    *,
    extracted_dir: Path,
    expected_execution_revision: str,
    expected_session_sha256: str,
    required_evidence_schema: str | None,
) -> tuple[list[dict[str, str]], int, int]:
    if not extracted_dir.is_dir():
        raise ValueError("extracted artifact directory does not exist")
    entries = tuple(extracted_dir.iterdir())
    if any(not entry.is_file() for entry in entries):
        raise ValueError("extracted artifact contains an unexpected directory")
    allowed = re.compile(r"case-.+\.(?:json|npz)")
    unexpected = {
        entry.name
        for entry in entries
        if entry.name != "execution-session.json"
        and allowed.fullmatch(entry.name) is None
    }
    if unexpected:
        raise ValueError("extracted artifact contains unexpected files")

    session_path = extracted_dir / "execution-session.json"
    if not session_path.is_file():
        raise ValueError("execution session is missing")
    if _sha256(session_path) != expected_session_sha256:
        raise ValueError("execution session digest does not match the expected session")
    session = _read_json_mapping(session_path, name="execution session")
    if (
        session.get("schema_version") != _SESSION_SCHEMA
        or session.get("execution_revision") != expected_execution_revision
    ):
        raise ValueError("execution session identity is invalid")

    json_paths = {path.stem: path for path in extracted_dir.glob("case-*.json")}
    npz_paths = {path.stem: path for path in extracted_dir.glob("case-*.npz")}
    if json_paths.keys() != npz_paths.keys():
        raise ValueError("checkpoint JSON and NPZ sidecars must be exactly paired")
    evidence_sidecars_validated = 0
    for stem in sorted(json_paths):
        checkpoint = _read_json_mapping(json_paths[stem], name=f"checkpoint {stem}")
        identity = _mapping(
            checkpoint.get("identity"), name=f"checkpoint {stem}.identity"
        )
        if identity.get("execution_revision") != expected_execution_revision:
            raise ValueError("checkpoint execution revision does not match the session")
        if required_evidence_schema is not None:
            outcome = checkpoint.get("outcome")
            result = outcome.get("result") if isinstance(outcome, Mapping) else None
            if (
                not isinstance(result, Mapping)
                or result.get("evidence_sidecar_schema") != required_evidence_schema
            ):
                raise ValueError(
                    "checkpoint does not declare the required evidence sidecar schema"
                )
        try:
            with np.load(npz_paths[stem], allow_pickle=False) as archive:
                if not archive.files:
                    raise ValueError("checkpoint NPZ sidecar contains no named arrays")
                if required_evidence_schema is not None:
                    arrays = {name: np.asarray(archive[name]) for name in archive.files}
                    validate_structural_evidence_arrays(arrays)
                    evidence_sidecars_validated += 1
        except (OSError, ValueError) as exc:
            if required_evidence_schema is not None:
                raise ValueError("checkpoint evidence sidecar is invalid") from exc
            raise ValueError("checkpoint NPZ sidecar is unreadable") from exc

    files = [
        {"name": path.name, "sha256": _sha256(path)}
        for path in sorted(entries, key=lambda item: item.name)
    ]
    return files, len(json_paths), evidence_sidecars_validated


def build_structural_artifact_receipt(
    *,
    run: Mapping[str, object],
    jobs: Mapping[str, object],
    artifact: Mapping[str, object],
    archive_path: Path,
    extracted_dir: Path,
    expected_run_id: int,
    expected_dispatch_head: str,
    expected_execution_revision: str,
    expected_session_sha256: str,
    requested_case_start: int,
    requested_case_stop: int,
    required_evidence_schema: str | None = None,
) -> dict[str, object]:
    """Return an outcome-blind receipt for one terminal hosted slice artifact."""

    expected_run_id = _positive_integer(expected_run_id, name="expected_run_id")
    if _SHA40.fullmatch(expected_dispatch_head) is None:
        raise ValueError("expected_dispatch_head must be a lowercase 40-character SHA")
    if _SHA40.fullmatch(expected_execution_revision) is None:
        raise ValueError(
            "expected_execution_revision must be a lowercase 40-character SHA"
        )
    if _SHA256.fullmatch(expected_session_sha256) is None:
        raise ValueError("expected_session_sha256 must be a lowercase SHA-256")
    if required_evidence_schema not in {None, EVIDENCE_SIDECAR_SCHEMA}:
        raise ValueError("required_evidence_schema is not a registered schema")
    if (
        isinstance(requested_case_start, bool)
        or not isinstance(requested_case_start, int)
        or isinstance(requested_case_stop, bool)
        or not isinstance(requested_case_stop, int)
        or requested_case_start < 0
        or requested_case_stop <= requested_case_start
    ):
        raise ValueError("requested case range must satisfy 0 <= start < stop")
    if not isinstance(archive_path, Path) or not archive_path.is_file():
        raise ValueError("artifact archive does not exist")
    if not isinstance(extracted_dir, Path):
        raise TypeError("extracted_dir must be a pathlib.Path")

    run_record = _mapping(run, name="run")
    if run_record.get("id") != expected_run_id:
        raise ValueError("run ID does not match the expected run")
    if run_record.get("head_sha") != expected_dispatch_head:
        raise ValueError("run head does not match the expected dispatch head")
    run_conclusion = _terminal_record(run_record, name="run")

    jobs_record = _mapping(jobs, name="jobs response")
    job = _exact_named_record(
        jobs_record.get("jobs"), expected_name=_JOB_NAME, collection_name="jobs"
    )
    job_conclusion = _terminal_record(job, name="structural job")
    campaign_step = _exact_named_record(
        job.get("steps"), expected_name=_CAMPAIGN_STEP, collection_name="job steps"
    )
    upload_step = _exact_named_record(
        job.get("steps"), expected_name=_UPLOAD_STEP, collection_name="job steps"
    )
    campaign_conclusion = _terminal_record(campaign_step, name="campaign step")
    upload_conclusion = _terminal_record(upload_step, name="artifact upload step")
    if run_conclusion != "success" and job_conclusion == "success":
        raise ValueError("run failure is inconsistent with a successful slice job")
    if run_conclusion == "success" and (
        job_conclusion != "success"
        or campaign_conclusion != "success"
        or upload_conclusion != "success"
    ):
        raise ValueError("successful slice run requires successful job and steps")

    artifact_record = _mapping(artifact, name="artifact")
    artifact_id = _positive_integer(artifact_record.get("id"), name="artifact.id")
    expected_artifact_name = f"structural-checkpoints-{expected_run_id}"
    if artifact_record.get("name") != expected_artifact_name:
        raise ValueError("artifact name does not match the expected run")
    _positive_integer(
        artifact_record.get("size_in_bytes"), name="artifact.size_in_bytes"
    )
    if artifact_record.get("expired") is not False:
        raise ValueError("artifact must be retained and unexpired")
    api_digest = artifact_record.get("digest")
    if (
        not isinstance(api_digest, str)
        or _ARTIFACT_DIGEST.fullmatch(api_digest) is None
    ):
        raise ValueError("GitHub artifact digest must be a lowercase SHA-256")
    archive_sha256 = _sha256(archive_path)
    if api_digest != f"sha256:{archive_sha256}":
        raise ValueError("GitHub artifact digest does not match the retained archive")

    files, checkpoint_pair_count, evidence_sidecars_validated = _validate_artifact_tree(
        extracted_dir=extracted_dir,
        expected_execution_revision=expected_execution_revision,
        expected_session_sha256=expected_session_sha256,
        required_evidence_schema=required_evidence_schema,
    )
    requested_count = requested_case_stop - requested_case_start
    if run_conclusion == "success" and checkpoint_pair_count != requested_count:
        raise ValueError("successful slice artifact has an incomplete checkpoint set")
    if run_conclusion != "success" and checkpoint_pair_count > requested_count:
        raise ValueError("unsuccessful slice artifact exceeds its requested range")

    step_fields = (
        "number",
        "name",
        "status",
        "conclusion",
        "started_at",
        "completed_at",
    )
    receipt: dict[str, object] = {
        "schema_version": (
            _ENRICHED_SCHEMA if required_evidence_schema is not None else _BASE_SCHEMA
        ),
        "classification": "workflow_artifact_provenance_not_scientific_summary",
        "requested_case_range": [requested_case_start, requested_case_stop],
        "execution_revision": expected_execution_revision,
        "execution_session_sha256": expected_session_sha256,
        "run": _selected_fields(
            run_record,
            (
                "id",
                "status",
                "conclusion",
                "head_sha",
                "created_at",
                "run_started_at",
                "updated_at",
                "html_url",
            ),
            prefix="run",
        ),
        "job": {
            **_selected_fields(
                job,
                (
                    "id",
                    "name",
                    "status",
                    "conclusion",
                    "started_at",
                    "completed_at",
                    "runner_name",
                ),
                prefix="job",
            ),
            "steps": [
                _selected_fields(campaign_step, step_fields, prefix="campaign step"),
                _selected_fields(upload_step, step_fields, prefix="upload step"),
            ],
        },
        "artifact": {
            **_selected_fields(
                artifact_record,
                (
                    "id",
                    "name",
                    "size_in_bytes",
                    "expired",
                    "created_at",
                    "updated_at",
                    "archive_download_url",
                    "digest",
                ),
                prefix="artifact",
            ),
            "id": artifact_id,
        },
        "artifact_archive_sha256": archive_sha256,
        "checkpoint_pair_count": checkpoint_pair_count,
        "files": files,
    }
    if required_evidence_schema is not None:
        receipt.update(
            {
                "evidence_sidecar_schema": required_evidence_schema,
                "required_evidence_array_count": len(REQUIRED_EVIDENCE_ARRAYS),
                "evidence_sidecars_validated": evidence_sidecars_validated,
            }
        )
    return receipt


def _write_required_absent_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError("output receipt must not already exist")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError("output receipt must not already exist")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Build one receipt from exact retained GitHub API responses."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--jobs-json", type=Path, required=True)
    parser.add_argument("--artifacts-json", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extracted-dir", type=Path, required=True)
    parser.add_argument("--expected-run-id", type=int, required=True)
    parser.add_argument("--expected-dispatch-head", required=True)
    parser.add_argument("--expected-execution-revision", required=True)
    parser.add_argument("--expected-session-sha256", required=True)
    parser.add_argument("--case-start", type=int, required=True)
    parser.add_argument("--case-stop", type=int, required=True)
    parser.add_argument(
        "--required-evidence-schema",
        choices=[EVIDENCE_SIDECAR_SCHEMA],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    artifact_response = _read_json_mapping(
        args.artifacts_json, name="artifacts response"
    )
    artifact = _exact_named_record(
        artifact_response.get("artifacts"),
        expected_name=f"structural-checkpoints-{args.expected_run_id}",
        collection_name="artifacts",
    )
    receipt = build_structural_artifact_receipt(
        run=_read_json_mapping(args.run_json, name="run response"),
        jobs=_read_json_mapping(args.jobs_json, name="jobs response"),
        artifact=artifact,
        archive_path=args.archive,
        extracted_dir=args.extracted_dir,
        expected_run_id=args.expected_run_id,
        expected_dispatch_head=args.expected_dispatch_head,
        expected_execution_revision=args.expected_execution_revision,
        expected_session_sha256=args.expected_session_sha256,
        requested_case_start=args.case_start,
        requested_case_stop=args.case_stop,
        required_evidence_schema=args.required_evidence_schema,
    )
    _write_required_absent_json(args.output, receipt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_structural_artifact_receipt", "main"]
