"""Versioned contracts for immutable private-dataset analysis jobs."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.shared.python.launch_monitor.corpus import CORPUS_COLUMN_MAP

DATASET_JOB_CONTRACT_VERSION: Final = "launch-monitor-dataset-job/1.0.0"
MAX_RESULT_ITEMS = 5_000
MAX_PAGE_SIZE = 200
MIN_AGGREGATE_ROWS = 10
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_COMMIT_PATTERN = r"^[0-9a-f]{40}$"
_ROOT_ID_PATTERN = r"^[a-z][a-z0-9-]{0,62}$"
_REPOSITORY_PATTERN = r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"
_ALLOWED_METRICS = frozenset(name for name, _ in CORPUS_COLUMN_MAP.values())
UnavailableCode = Literal[
    "root_not_authorized",
    "authority_unavailable",
    "repository_mismatch",
    "commit_mismatch",
    "manifest_mismatch",
    "content_mismatch",
    "row_count_mismatch",
    "backing_manifest_mismatch",
    "dependency_unavailable",
    "operation_unavailable",
    "internal_execution_error",
]


class DatasetReferenceV1(BaseModel):
    """Content-addressed reference to one authorized authority checkout."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    root_id: str = Field(pattern=_ROOT_ID_PATTERN)
    repository: str = Field(pattern=_REPOSITORY_PATTERN)
    commit: str = Field(pattern=_COMMIT_PATTERN)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_row_count: int = Field(ge=1, le=10_000_000)


class DatasetOperationV1(BaseModel):
    """Allow-listed aggregate operation; arbitrary query text is forbidden."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["source_summary", "metric_summary", "correlation"]
    metrics: tuple[str, ...] = Field(default=(), max_length=12)
    group_by: Literal["source_id", "monitor", "club"] | None = None
    minimum_group_rows: int = Field(MIN_AGGREGATE_ROWS, ge=MIN_AGGREGATE_ROWS)

    @model_validator(mode="after")
    def validate_operation(self) -> DatasetOperationV1:
        """Reject unsupported metrics and ambiguous operation shapes."""
        unknown = set(self.metrics) - _ALLOWED_METRICS
        if unknown:
            raise ValueError(f"unsupported metrics: {sorted(unknown)}")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must be unique")
        if self.kind == "source_summary":
            if self.metrics or self.group_by is not None:
                raise ValueError("source_summary does not accept metrics or group_by")
        elif self.kind == "metric_summary" and not self.metrics:
            raise ValueError("metric_summary requires at least one metric")
        elif self.kind == "correlation" and len(self.metrics) < 2:
            raise ValueError("correlation requires at least two metrics")
        return self


class DatasetJobRequestV1(BaseModel):
    """A private-data job request containing no inline observations."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    contract_version: Literal["launch-monitor-dataset-job/1.0.0"] = (
        DATASET_JOB_CONTRACT_VERSION
    )
    dataset: DatasetReferenceV1
    operation: DatasetOperationV1

    @model_validator(mode="after")
    def validate_nested_reference(self) -> DatasetJobRequestV1:
        """Revalidate root aliases even after unsafe model-copy operations."""
        DatasetReferenceV1.model_validate(self.dataset.model_dump())
        return self


class DatasetUnavailableStateV1(BaseModel):
    """Data-free reason that an immutable reference cannot be used."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    code: UnavailableCode
    message: str = Field(min_length=1, max_length=240)
    retryable: bool = False


class DatasetUnavailableError(RuntimeError):
    """Raised when a reference cannot be verified without weakening policy."""

    def __init__(self, state: DatasetUnavailableStateV1) -> None:
        super().__init__(state.message)
        self.state = state


def unavailable(
    code: UnavailableCode, message: str, *, retryable: bool = False
) -> DatasetUnavailableError:
    """Build a structured fail-closed error without exposing server paths."""
    return DatasetUnavailableError(
        DatasetUnavailableStateV1(code=code, message=message, retryable=retryable)
    )


def dataset_job_contract_json_schema() -> dict[str, object]:
    """Return the stable request schema for non-Python consumers."""
    return DatasetJobRequestV1.model_json_schema()


__all__ = [
    "DATASET_JOB_CONTRACT_VERSION",
    "MAX_PAGE_SIZE",
    "MAX_RESULT_ITEMS",
    "DatasetJobRequestV1",
    "DatasetOperationV1",
    "DatasetReferenceV1",
    "DatasetUnavailableError",
    "DatasetUnavailableStateV1",
    "UnavailableCode",
    "dataset_job_contract_json_schema",
    "unavailable",
]
