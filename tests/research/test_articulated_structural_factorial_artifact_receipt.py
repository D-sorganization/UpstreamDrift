"""Hosted structural artifacts are retained with exact operational provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_artifact_receipt import (
    build_structural_artifact_receipt,
    main,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_evidence import (
    EVIDENCE_SIDECAR_SCHEMA,
    REQUIRED_EVIDENCE_ARRAYS,
)

pytestmark = pytest.mark.scientific
RUN_ID = 33273691711
HEAD_SHA = "a" * 40
EXECUTION_REVISION = "b" * 40
ARCHIVE_SHA256 = hashlib.sha256(b"synthetic archive bytes").hexdigest()


def _run(
    *, status: str = "completed", conclusion: str = "success"
) -> dict[str, object]:
    return {
        "id": RUN_ID,
        "status": status,
        "conclusion": conclusion,
        "head_sha": HEAD_SHA,
        "created_at": "2026-08-29T20:31:06Z",
        "run_started_at": "2026-08-29T20:31:06Z",
        "updated_at": "2026-08-29T21:12:00Z",
        "html_url": f"https://github.example/actions/runs/{RUN_ID}",
    }


def _jobs() -> dict[str, object]:
    return {
        "jobs": [
            {
                "id": 99156607309,
                "name": "Structural Runtime Audit or Campaign Slice",
                "status": "completed",
                "conclusion": "success",
                "started_at": "2026-08-29T20:31:11Z",
                "completed_at": "2026-08-29T21:11:58Z",
                "runner_name": "GitHub Actions 1000410531",
                "steps": [
                    {
                        "number": 11,
                        "name": "Run Registered Structural Campaign Slice",
                        "status": "completed",
                        "conclusion": "success",
                        "started_at": "2026-08-29T20:32:25Z",
                        "completed_at": "2026-08-29T21:11:40Z",
                    },
                    {
                        "number": 12,
                        "name": "Upload Structural Campaign Checkpoints",
                        "status": "completed",
                        "conclusion": "success",
                        "started_at": "2026-08-29T21:11:40Z",
                        "completed_at": "2026-08-29T21:11:55Z",
                    },
                ],
            }
        ]
    }


def _artifact() -> dict[str, object]:
    return {
        "id": 987654321,
        "name": f"structural-checkpoints-{RUN_ID}",
        "size_in_bytes": 12345,
        "expired": False,
        "created_at": "2026-08-29T21:11:55Z",
        "updated_at": "2026-08-29T21:11:55Z",
        "archive_download_url": "https://api.github.example/artifacts/987654321/zip",
        "digest": f"sha256:{ARCHIVE_SHA256}",
    }


def _complete_evidence_arrays() -> dict[str, np.ndarray]:
    arrays = {name: np.zeros(2) for name in REQUIRED_EVIDENCE_ARRAYS}
    arrays["time_s"] = np.array([0.0, 0.1])
    arrays["station_force_on_club_n"] = np.zeros((2, 2, 2, 3))
    arrays["active_station"] = np.zeros((2, 2, 2), dtype=bool)
    arrays["active_set_transition"] = np.zeros(2, dtype=bool)
    arrays["net_club_force_n"] = np.zeros((2, 3))
    arrays["cumulative_contact_impulse_n_s"] = np.zeros((2, 3))
    arrays["active_station_count"] = np.zeros(2, dtype=int)
    return arrays


def _artifact_tree(
    tmp_path: Path, *, count: int = 2, enriched: bool = False
) -> tuple[Path, Path, str]:
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    session = (
        json.dumps(
            {
                "schema_version": "articulated-structural-factorial-session/1.0.0",
                "execution_revision": EXECUTION_REVISION,
                "runtime_identity_sha256": "c" * 64,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    (extracted / "execution-session.json").write_bytes(session)
    for index in range(count):
        stem = f"case-{index:02d}"
        checkpoint: dict[str, object] = {
            "identity": {"execution_revision": EXECUTION_REVISION},
            "case": {"case_key": stem},
        }
        if enriched:
            checkpoint["outcome"] = {
                "result": {"evidence_sidecar_schema": EVIDENCE_SIDECAR_SCHEMA}
            }
        (extracted / f"{stem}.json").write_text(
            json.dumps(checkpoint),
            encoding="utf-8",
        )
        arrays = (
            _complete_evidence_arrays() if enriched else {"time_s": np.array([0.0])}
        )
        np.savez_compressed(extracted / f"{stem}.npz", **arrays)
    archive = tmp_path / "artifact.zip"
    archive.write_bytes(b"synthetic archive bytes")
    return extracted, archive, hashlib.sha256(session).hexdigest()


def test_receipt_binds_terminal_run_job_artifact_and_exact_slice(
    tmp_path: Path,
) -> None:
    extracted, archive, session_digest = _artifact_tree(tmp_path)

    receipt = build_structural_artifact_receipt(
        run=_run(),
        jobs=_jobs(),
        artifact=_artifact(),
        archive_path=archive,
        extracted_dir=extracted,
        expected_run_id=RUN_ID,
        expected_dispatch_head=HEAD_SHA,
        expected_execution_revision=EXECUTION_REVISION,
        expected_session_sha256=session_digest,
        requested_case_start=694,
        requested_case_stop=696,
    )

    assert receipt["classification"] == (
        "workflow_artifact_provenance_not_scientific_summary"
    )
    assert receipt["run"]["id"] == RUN_ID
    assert receipt["job"]["id"] == 99156607309
    assert [step["number"] for step in receipt["job"]["steps"]] == [11, 12]
    assert receipt["artifact"]["id"] == 987654321
    assert receipt["artifact"]["digest"] == f"sha256:{ARCHIVE_SHA256}"
    assert len(receipt["artifact_archive_sha256"]) == 64
    assert receipt["checkpoint_pair_count"] == 2
    assert len(receipt["files"]) == 5


def test_receipt_cli_consumes_retained_github_responses_atomically(
    tmp_path: Path,
) -> None:
    extracted, archive, session_digest = _artifact_tree(tmp_path, enriched=True)
    run_path = tmp_path / "github-run.json"
    jobs_path = tmp_path / "github-jobs.json"
    artifacts_path = tmp_path / "github-artifacts.json"
    output_path = tmp_path / "artifact-receipt.json"
    run_path.write_text(json.dumps(_run()), encoding="utf-8")
    jobs_path.write_text(json.dumps(_jobs()), encoding="utf-8")
    artifacts_path.write_text(
        json.dumps({"total_count": 1, "artifacts": [_artifact()]}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-json",
            str(run_path),
            "--jobs-json",
            str(jobs_path),
            "--artifacts-json",
            str(artifacts_path),
            "--archive",
            str(archive),
            "--extracted-dir",
            str(extracted),
            "--expected-run-id",
            str(RUN_ID),
            "--expected-dispatch-head",
            HEAD_SHA,
            "--expected-execution-revision",
            EXECUTION_REVISION,
            "--expected-session-sha256",
            session_digest,
            "--case-start",
            "694",
            "--case-stop",
            "696",
            "--required-evidence-schema",
            EVIDENCE_SIDECAR_SCHEMA,
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    receipt = json.loads(output_path.read_text(encoding="utf-8"))
    assert receipt["artifact"]["id"] == _artifact()["id"]
    assert receipt["evidence_sidecars_validated"] == 2
    assert not tuple(tmp_path.glob("artifact-receipt.json.tmp-*"))

    with pytest.raises(FileExistsError, match="output receipt"):
        main(
            [
                "--run-json",
                str(run_path),
                "--jobs-json",
                str(jobs_path),
                "--artifacts-json",
                str(artifacts_path),
                "--archive",
                str(archive),
                "--extracted-dir",
                str(extracted),
                "--expected-run-id",
                str(RUN_ID),
                "--expected-dispatch-head",
                HEAD_SHA,
                "--expected-execution-revision",
                EXECUTION_REVISION,
                "--expected-session-sha256",
                session_digest,
                "--case-start",
                "694",
                "--case-stop",
                "696",
                "--output",
                str(output_path),
            ]
        )


def test_enriched_receipt_validates_every_registered_evidence_history(
    tmp_path: Path,
) -> None:
    extracted, archive, session_digest = _artifact_tree(tmp_path, enriched=True)

    receipt = build_structural_artifact_receipt(
        run=_run(),
        jobs=_jobs(),
        artifact=_artifact(),
        archive_path=archive,
        extracted_dir=extracted,
        expected_run_id=RUN_ID,
        expected_dispatch_head=HEAD_SHA,
        expected_execution_revision=EXECUTION_REVISION,
        expected_session_sha256=session_digest,
        requested_case_start=694,
        requested_case_stop=696,
        required_evidence_schema=EVIDENCE_SIDECAR_SCHEMA,
    )

    assert receipt["schema_version"] == (
        "articulated-structural-factorial-artifact-receipt/1.2.0"
    )
    assert receipt["evidence_sidecar_schema"] == EVIDENCE_SIDECAR_SCHEMA
    assert receipt["required_evidence_array_count"] == 37
    assert receipt["evidence_sidecars_validated"] == 2


def test_enriched_receipt_rejects_a_readable_legacy_sidecar(tmp_path: Path) -> None:
    extracted, archive, session_digest = _artifact_tree(tmp_path)

    with pytest.raises(ValueError, match="evidence sidecar schema"):
        build_structural_artifact_receipt(
            run=_run(),
            jobs=_jobs(),
            artifact=_artifact(),
            archive_path=archive,
            extracted_dir=extracted,
            expected_run_id=RUN_ID,
            expected_dispatch_head=HEAD_SHA,
            expected_execution_revision=EXECUTION_REVISION,
            expected_session_sha256=session_digest,
            requested_case_start=694,
            requested_case_stop=696,
            required_evidence_schema=EVIDENCE_SIDECAR_SCHEMA,
        )


def test_receipt_rejects_an_archive_that_disagrees_with_the_api_digest(
    tmp_path: Path,
) -> None:
    extracted, archive, session_digest = _artifact_tree(tmp_path)
    artifact = _artifact()
    artifact["digest"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="GitHub artifact digest"):
        build_structural_artifact_receipt(
            run=_run(),
            jobs=_jobs(),
            artifact=artifact,
            archive_path=archive,
            extracted_dir=extracted,
            expected_run_id=RUN_ID,
            expected_dispatch_head=HEAD_SHA,
            expected_execution_revision=EXECUTION_REVISION,
            expected_session_sha256=session_digest,
            requested_case_start=694,
            requested_case_stop=696,
        )


@pytest.mark.parametrize("defect", ["missing_sidecar", "unexpected", "session_drift"])
def test_receipt_rejects_incomplete_or_contaminated_artifact(
    tmp_path: Path, defect: str
) -> None:
    extracted, archive, session_digest = _artifact_tree(tmp_path)
    if defect == "missing_sidecar":
        next(extracted.glob("case-*.npz")).unlink()
    elif defect == "unexpected":
        (extracted / "partial.tmp").write_text("inflight", encoding="utf-8")
    else:
        session_digest = "0" * 64

    with pytest.raises(ValueError):
        build_structural_artifact_receipt(
            run=_run(),
            jobs=_jobs(),
            artifact=_artifact(),
            archive_path=archive,
            extracted_dir=extracted,
            expected_run_id=RUN_ID,
            expected_dispatch_head=HEAD_SHA,
            expected_execution_revision=EXECUTION_REVISION,
            expected_session_sha256=session_digest,
            requested_case_start=694,
            requested_case_stop=696,
        )


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (_run(status="in_progress", conclusion="success"), "terminal"),
        (_run(status="completed", conclusion="failure"), "successful slice"),
    ],
)
def test_receipt_rejects_nonterminal_or_inconsistent_success(
    tmp_path: Path, run: dict[str, object], message: str
) -> None:
    extracted, archive, session_digest = _artifact_tree(tmp_path)

    with pytest.raises(ValueError, match=message):
        build_structural_artifact_receipt(
            run=run,
            jobs=_jobs(),
            artifact=_artifact(),
            archive_path=archive,
            extracted_dir=extracted,
            expected_run_id=RUN_ID,
            expected_dispatch_head=HEAD_SHA,
            expected_execution_revision=EXECUTION_REVISION,
            expected_session_sha256=session_digest,
            requested_case_start=694,
            requested_case_stop=696,
        )
