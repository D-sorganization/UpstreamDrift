"""Bounded process-local jobs for immutable launch-monitor references."""

from __future__ import annotations

import json
import os
import secrets
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.tools.launch_monitor_model.dataset_reference import (
    DATASET_JOB_CONTRACT_VERSION,
    MAX_PAGE_SIZE,
    DatasetJobRequestV1,
    DatasetUnavailableError,
    DatasetUnavailableStateV1,
    execute_dataset_operation,
    verify_dataset_reference,
)

_ROOTS_ENV = "UPSTREAMDRIFT_LAUNCH_MONITOR_DATASET_ROOTS"
_LEGACY_ROOT_ENV = "LAUNCH_MONITOR_DATA_ROOT"
_MAX_JOBS = 64
JobStatus = Literal["queued", "running", "completed", "unavailable", "failed"]


class DatasetJobCapacityError(RuntimeError):
    """Raised when the bounded in-memory queue cannot accept another job."""


class DatasetJobServiceClosedError(RuntimeError):
    """Raised when callers submit work after deterministic shutdown."""


class DatasetJobStatusV1(BaseModel):
    """Data-free state and aggregate counts for one server-side job."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["launch-monitor-dataset-job/1.0.0"] = (
        DATASET_JOB_CONTRACT_VERSION
    )
    job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    status: JobStatus
    submitted_at_utc: datetime
    completed_at_utc: datetime | None = None
    input_row_count: int = Field(ge=0)
    result_item_count: int = Field(ge=0)
    unavailable: DatasetUnavailableStateV1 | None = None


class DatasetJobResultPageV1(BaseModel):
    """Bounded page of aggregate or source-backing records, never shot rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["launch-monitor-dataset-job/1.0.0"] = (
        DATASET_JOB_CONTRACT_VERSION
    )
    job_id: str
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=MAX_PAGE_SIZE)
    total_items: int = Field(ge=0)
    next_offset: int | None = Field(default=None, ge=0)
    items: tuple[dict[str, Any], ...]


class DatasetRootRegistry:
    """Resolve opaque aliases to server-authorized absolute roots."""

    def __init__(self, roots: dict[str, Path]) -> None:
        canonical: dict[str, Path] = {}
        for root_id, path in roots.items():
            if not root_id or Path(root_id).name != root_id or ".." in root_id:
                raise ValueError("root_id must be an opaque alias, not a path")
            if not path.is_absolute():
                raise ValueError("authorized dataset roots must be absolute paths")
            canonical[root_id] = path.resolve(strict=False)
        self._roots = canonical

    @classmethod
    def from_environment(cls) -> DatasetRootRegistry:
        """Load administrator-authorized roots without accepting client paths."""
        roots: dict[str, Path] = {}
        raw = os.environ.get(_ROOTS_ENV)
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{_ROOTS_ENV} must be a JSON object") from exc
            if not isinstance(parsed, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in parsed.items()
            ):
                raise ValueError(f"{_ROOTS_ENV} must map aliases to paths")
            roots.update({key: Path(value) for key, value in parsed.items()})
        legacy = os.environ.get(_LEGACY_ROOT_ENV)
        if legacy and "default" not in roots:
            roots["default"] = Path(legacy)
        return cls(roots)

    def resolve(self, root_id: str) -> Path:
        """Return an authorized root or a data-free unavailable state."""
        if not root_id or Path(root_id).name != root_id or ".." in root_id:
            raise ValueError("root_id must be an opaque alias, not a path")
        try:
            return self._roots[root_id]
        except KeyError as exc:
            raise DatasetUnavailableError(
                DatasetUnavailableStateV1(
                    code="root_not_authorized",
                    message="The requested dataset authority is not authorized on this server.",
                    retryable=False,
                )
            ) from exc


class _MutableJob:
    def __init__(self, request: DatasetJobRequestV1) -> None:
        self.request = request
        self.status: JobStatus = "queued"
        self.submitted_at_utc = datetime.now(UTC)
        self.completed_at_utc: datetime | None = None
        self.results: list[dict[str, Any]] = []
        self.unavailable: DatasetUnavailableStateV1 | None = None


class DatasetJobService:
    """Run verified aggregate jobs with bounded concurrency and retention."""

    def __init__(
        self,
        roots: DatasetRootRegistry,
        *,
        max_jobs: int = _MAX_JOBS,
        max_workers: int = 2,
    ) -> None:
        if max_jobs < 1:
            raise ValueError("max_jobs must be positive")
        if max_workers < 1 or max_workers > 4:
            raise ValueError("max_workers must be between 1 and 4")
        self._roots = roots
        self._max_jobs = max_jobs
        self._jobs: OrderedDict[str, _MutableJob] = OrderedDict()
        self._lock = threading.Lock()
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="launch-monitor-dataset"
        )

    def submit(self, request: DatasetJobRequestV1) -> DatasetJobStatusV1:
        """Queue one job and return immediately without reading private rows."""
        validated = DatasetJobRequestV1.model_validate(request.model_dump())
        job_id = secrets.token_hex(16)
        job = _MutableJob(validated)
        with self._lock:
            if self._closed:
                raise DatasetJobServiceClosedError("dataset job service is closed")
            while len(self._jobs) >= self._max_jobs:
                oldest_id, oldest = next(iter(self._jobs.items()))
                if oldest.status in {"queued", "running"}:
                    raise DatasetJobCapacityError(
                        "dataset job capacity is temporarily exhausted"
                    )
                self._jobs.pop(oldest_id)
            self._jobs[job_id] = job
        self._executor.submit(self._execute, job_id)
        return self.status(job_id)

    @property
    def closed(self) -> bool:
        """Return whether deterministic shutdown has begun."""
        with self._lock:
            return self._closed

    def close(self) -> None:
        """Stop accepting jobs and join workers; safe to call repeatedly."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> DatasetJobService:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _execute(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
        request = job.request
        dataset_ref = request.dataset
        try:
            root = self._roots.resolve(dataset_ref.root_id)
            dataset = verify_dataset_reference(root, dataset_ref)
            results = execute_dataset_operation(dataset, request.operation)
        except DatasetUnavailableError as exc:
            self._finish(job_id, status="unavailable", unavailable=exc.state)
        except (OSError, ValueError, KeyError, TypeError):
            self._finish(
                job_id,
                status="failed",
                unavailable=DatasetUnavailableStateV1(
                    code="internal_execution_error",
                    message="The dataset job failed without returning private data.",
                    retryable=False,
                ),
            )
        else:
            self._finish(job_id, status="completed", results=results)

    def _finish(
        self,
        job_id: str,
        *,
        status: Literal["completed", "unavailable", "failed"],
        results: list[dict[str, Any]] | None = None,
        unavailable: DatasetUnavailableStateV1 | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.results = results or []
            job.unavailable = unavailable
            job.completed_at_utc = datetime.now(UTC)

    def status(self, job_id: str) -> DatasetJobStatusV1:
        """Return a serialization-safe status with no server filesystem path."""
        with self._lock:
            try:
                job = self._jobs[job_id]
            except KeyError as exc:
                raise KeyError("dataset job not found") from exc
            request = job.request
            dataset_ref = request.dataset
            row_count = (
                dataset_ref.expected_row_count if job.status == "completed" else 0
            )
            return DatasetJobStatusV1(
                job_id=job_id,
                status=job.status,
                submitted_at_utc=job.submitted_at_utc,
                completed_at_utc=job.completed_at_utc,
                input_row_count=row_count,
                result_item_count=len(job.results),
                unavailable=job.unavailable,
            )

    def results(
        self, job_id: str, *, offset: int, limit: int
    ) -> DatasetJobResultPageV1:
        """Return one aggregate page after completion; page size is hard-bounded."""
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit < 1 or limit > MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
        with self._lock:
            try:
                job = self._jobs[job_id]
            except KeyError as exc:
                raise KeyError("dataset job not found") from exc
            if job.status != "completed":
                raise ValueError("dataset job results are not available")
            total = len(job.results)
            items = tuple(job.results[offset : offset + limit])
        next_offset = offset + len(items)
        return DatasetJobResultPageV1(
            job_id=job_id,
            offset=offset,
            limit=limit,
            total_items=total,
            next_offset=next_offset if next_offset < total else None,
            items=items,
        )


__all__ = [
    "DatasetJobCapacityError",
    "DatasetJobResultPageV1",
    "DatasetJobService",
    "DatasetJobServiceClosedError",
    "DatasetJobStatusV1",
    "DatasetRootRegistry",
]
