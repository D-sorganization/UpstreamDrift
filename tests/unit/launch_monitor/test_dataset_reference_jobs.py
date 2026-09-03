"""Immutable, aggregate-only dataset-reference job contract tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from shared.python.launch_monitor.dataset_reference import (
    DatasetJobRequestV1,
    DatasetOperationV1,
    DatasetReferenceV1,
    dataset_content_sha256,
    dataset_job_contract_json_schema,
)
from src.api.services.launch_monitor_dataset_jobs import (
    DatasetJobService,
    DatasetJobServiceClosedError,
    DatasetRootRegistry,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def job_service(tmp_path: Path) -> Iterator[DatasetJobService]:
    """Provide a service whose workers are always joined after each test."""
    service = DatasetJobService(DatasetRootRegistry({"test-authority": tmp_path}))
    try:
        yield service
    finally:
        service.close()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _write_authority(root: Path, *, rows: int = 261_666) -> DatasetReferenceV1:
    dataset = root / "data/authority/database/shot_corpus_parquet"
    partition = dataset / "source_id=synthetic_trackman"
    partition.mkdir(parents=True)
    table = pa.table(
        {
            "monitor": pa.array(["TrackMan"] * rows),
            "file": pa.array(["synthetic.csv"] * rows),
            "row_index": pa.array(range(rows), type=pa.int64()),
            "club": pa.array(["7-iron"] * rows),
            "club_speed_mph": pa.array([80.0 + (index % 20) for index in range(rows)]),
            "ball_speed_mph": pa.array(
                [122.0 + 1.4 * (index % 20) for index in range(rows)]
            ),
        },
    )
    pq.write_table(table, partition / "part-0.parquet", compression="zstd")
    manifest = {
        "schema_version": 1,
        "sources": {
            "synthetic_trackman": {
                "rows": rows,
                "bytes": (partition / "part-0.parquet").stat().st_size,
                "columns": table.column_names,
            }
        },
        "total_rows": rows,
        "total_bytes": (partition / "part-0.parquet").stat().st_size,
    }
    manifest_path = dataset / "_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    database = root / "data/authority/database"
    acquisition = {
        "schema_version": 1,
        "source_count": 1,
        "parsed_row_count": rows,
        "sources": [
            {
                "source_id": "synthetic_trackman",
                "repository": "https://github.com/example/synthetic.git",
                "resolved_commit": "a" * 40,
                "parsed_rows": rows,
                "files": [
                    {
                        "path": "synthetic.csv",
                        "sha256": "b" * 64,
                        "bytes": 123,
                    }
                ],
            }
        ],
    }
    acquisition_path = database / "acquisition_manifest.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    results = root / "results/v2"
    results.mkdir(parents=True)
    source_summary = results / "source_summary.csv"
    source_summary.write_text(
        "source_id,monitor,vendor_key,rows,redistribution_status,license_spdx\n"
        f"synthetic_trackman,TrackMan,trackman,{rows},reference_only,MIT\n",
        encoding="utf-8",
    )
    qualification = {
        "schema": "launch-monitor-data-qualification-manifest/v1",
        "source_rows": rows,
        "source_count": 1,
        "parquet_manifest_sha256": _sha256(manifest_path),
        "acquisition_manifest_sha256": _sha256(acquisition_path),
        "output_sha256": {"source_summary.csv": _sha256(source_summary)},
    }
    (results / "qualification_manifest.json").write_text(
        json.dumps(qualification), encoding="utf-8"
    )

    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Dataset Contract Test")
    _git(root, "config", "user.email", "dataset-contract@example.invalid")
    _git(
        root,
        "remote",
        "add",
        "origin",
        "https://github.com/D-sorganization/Launch-Monitor-Flight-Model-Campaign.git",
    )
    _git(root, "add", "data", "results")
    _git(root, "commit", "-qm", "synthetic authority")
    commit = _git(root, "rev-parse", "HEAD")
    return DatasetReferenceV1(
        root_id="test-authority",
        repository="D-sorganization/Launch-Monitor-Flight-Model-Campaign",
        commit=commit,
        manifest_sha256=_sha256(manifest_path),
        content_sha256=dataset_content_sha256(dataset),
        expected_row_count=rows,
    )


def _wait(service: DatasetJobService, job_id: str) -> object:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        job = service.status(job_id)
        if job.status not in {"queued", "running"}:
            return job
        time.sleep(0.01)
    pytest.fail("dataset job did not complete within 30 seconds")


def test_large_corpus_runs_by_reference_without_inline_rows(
    tmp_path: Path, job_service: DatasetJobService
) -> None:
    reference = _write_authority(tmp_path)
    service = job_service
    request = DatasetJobRequestV1(
        dataset=reference,
        operation=DatasetOperationV1(
            kind="correlation", metrics=["club_speed", "ball_speed"]
        ),
    )

    started = time.monotonic()
    submitted = service.submit(request)
    job = _wait(service, submitted.job_id)
    elapsed = time.monotonic() - started

    assert job.status == "completed"
    assert job.input_row_count == 261_666
    assert job.result_item_count == 1
    page = service.results(job.job_id, offset=0, limit=20)
    assert page.items[0]["left_metric"] == "club_speed"
    assert page.items[0]["right_metric"] == "ball_speed"
    assert page.items[0]["n"] == 261_666
    assert page.items[0]["correlation"] == pytest.approx(1.0)
    assert "records" not in page.model_dump_json()
    assert elapsed < 30.0


def test_source_summary_joins_content_addressed_backing_without_rows(
    tmp_path: Path, job_service: DatasetJobService
) -> None:
    reference = _write_authority(tmp_path, rows=100)
    service = job_service
    submitted = service.submit(
        DatasetJobRequestV1(
            dataset=reference,
            operation=DatasetOperationV1(kind="source_summary"),
        )
    )
    job = _wait(service, submitted.job_id)
    page = service.results(job.job_id, offset=0, limit=20)

    assert job.status == "completed"
    assert page.items == (
        {
            "source_id": "synthetic_trackman",
            "row_count": 100,
            "vendor_key": "trackman",
            "redistribution_status": "reference_only",
            "license_spdx": "MIT",
            "backing_repository": "example/synthetic",
            "backing_commit": "a" * 40,
            "backing_object_digests": [{"sha256": "b" * 64, "bytes": 123}],
        },
    )
    serialized = page.model_dump_json()
    assert "synthetic.csv" not in serialized
    assert "github.com" not in serialized
    assert str(tmp_path) not in serialized


def test_source_summary_suppresses_subminimum_sources(
    tmp_path: Path, job_service: DatasetJobService
) -> None:
    reference = _write_authority(tmp_path, rows=9)
    job = _wait(
        job_service,
        job_service.submit(
            DatasetJobRequestV1(
                dataset=reference,
                operation=DatasetOperationV1(kind="source_summary"),
            )
        ).job_id,
    )

    assert job.status == "completed"
    assert job.result_item_count == 0


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("commit", "f" * 40, "commit_mismatch"),
        ("manifest_sha256", "f" * 64, "manifest_mismatch"),
        ("content_sha256", "f" * 64, "content_mismatch"),
        ("expected_row_count", 99, "row_count_mismatch"),
    ],
)
def test_reference_mismatch_fails_closed(
    tmp_path: Path,
    job_service: DatasetJobService,
    field: str,
    value: object,
    code: str,
) -> None:
    reference = _write_authority(tmp_path, rows=100)
    bad_reference = reference.model_copy(update={field: value})
    service = job_service

    job = _wait(
        service,
        service.submit(
            DatasetJobRequestV1(
                dataset=bad_reference,
                operation=DatasetOperationV1(kind="source_summary"),
            )
        ).job_id,
    )

    assert job.status == "unavailable"
    assert job.unavailable is not None
    assert job.unavailable.code == code
    assert job.result_item_count == 0


def test_root_alias_cannot_be_used_as_a_path_escape(tmp_path: Path) -> None:
    reference = _write_authority(tmp_path, rows=100).model_copy(
        update={"root_id": "../../private"}
    )
    with pytest.raises(ValueError, match="root_id"):
        DatasetJobRequestV1(
            dataset=reference,
            operation=DatasetOperationV1(kind="source_summary"),
        )


def test_unconfigured_root_is_structured_unavailable(
    tmp_path: Path, job_service: DatasetJobService
) -> None:
    reference = _write_authority(tmp_path, rows=100).model_copy(
        update={"root_id": "not-authorized"}
    )
    service = job_service
    job = _wait(
        service,
        service.submit(
            DatasetJobRequestV1(
                dataset=reference,
                operation=DatasetOperationV1(kind="source_summary"),
            )
        ).job_id,
    )

    assert job.status == "unavailable"
    assert job.unavailable is not None
    assert job.unavailable.code == "root_not_authorized"
    assert str(tmp_path) not in job.unavailable.message


def test_client_cannot_bless_content_modified_after_the_exact_commit(
    tmp_path: Path, job_service: DatasetJobService
) -> None:
    reference = _write_authority(tmp_path, rows=100)
    parquet_path = next(
        (tmp_path / "data/authority/database/shot_corpus_parquet").rglob("*.parquet")
    )
    parquet_path.write_bytes(parquet_path.read_bytes() + b"dirty")
    dirty_hash = dataset_content_sha256(parquet_path.parents[1])
    service = job_service
    submitted = service.submit(
        DatasetJobRequestV1(
            dataset=reference.model_copy(update={"content_sha256": dirty_hash}),
            operation=DatasetOperationV1(kind="source_summary"),
        )
    )

    job = _wait(service, submitted.job_id)

    assert job.status == "unavailable"
    assert job.unavailable is not None
    assert job.unavailable.code == "content_mismatch"


def test_result_paging_is_bounded(
    tmp_path: Path, job_service: DatasetJobService
) -> None:
    reference = _write_authority(tmp_path, rows=100)
    service = job_service
    job = _wait(
        service,
        service.submit(
            DatasetJobRequestV1(
                dataset=reference,
                operation=DatasetOperationV1(
                    kind="metric_summary", metrics=["club_speed", "ball_speed"]
                ),
            )
        ).job_id,
    )

    first = service.results(job.job_id, offset=0, limit=1)
    assert len(first.items) == 1
    assert first.total_items == 2
    assert first.next_offset == 1
    with pytest.raises(ValueError, match="limit"):
        service.results(job.job_id, offset=0, limit=201)


def test_context_manager_joins_workers_and_rejects_new_submissions(
    tmp_path: Path,
) -> None:
    reference = _write_authority(tmp_path, rows=100)
    request = DatasetJobRequestV1(
        dataset=reference,
        operation=DatasetOperationV1(kind="source_summary"),
    )
    with DatasetJobService(
        DatasetRootRegistry({"test-authority": tmp_path})
    ) as service:
        assert _wait(service, service.submit(request).job_id).status == "completed"

    assert service.closed is True
    service.close()
    with pytest.raises(DatasetJobServiceClosedError, match="closed"):
        service.submit(request)


def test_published_dataset_job_schema_matches_python_authority() -> None:
    path = (
        Path(__file__).parents[3]
        / "docs/api/contracts/launch-monitor-dataset-job-v1.schema.json"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == (
        dataset_job_contract_json_schema()
    )
