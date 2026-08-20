"""Bounded aggregate operations over a verified private dataset."""

from __future__ import annotations

import csv
import itertools
import re
from collections.abc import Sequence
from typing import Any, cast

import pandas as pd

from src.shared.python.launch_monitor.corpus import load_private_corpus
from src.shared.python.launch_monitor.dataset_reference_contract import (
    MAX_RESULT_ITEMS,
    MIN_AGGREGATE_ROWS,
    DatasetOperationV1,
    unavailable,
)
from src.shared.python.launch_monitor.dataset_reference_verification import (
    ACQUISITION_MANIFEST_RELATIVE_PATH,
    SOURCE_SUMMARY_RELATIVE_PATH,
    VerifiedDataset,
    normalize_repository,
    parse_json_bytes,
    safe_fixed_child,
    sha256_file,
)

_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.+ -]{1,128}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _safe_label(value: object, *, label: str) -> str:
    text = str(value)
    if not _SAFE_LABEL.fullmatch(text):
        raise unavailable(
            "backing_manifest_mismatch", f"A source backing {label} is invalid."
        )
    return text


def _backing_object_digests(files: object) -> list[dict[str, int | str]]:
    if not isinstance(files, list):
        raise unavailable(
            "backing_manifest_mismatch", "A source backing object list is invalid."
        )
    digests: list[dict[str, int | str]] = []
    for item in files:
        if not isinstance(item, dict) or not _SHA256.fullmatch(
            str(item.get("sha256", ""))
        ):
            raise unavailable(
                "backing_manifest_mismatch", "A source backing digest is invalid."
            )
        size = item.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise unavailable(
                "backing_manifest_mismatch", "A source backing size is invalid."
            )
        digests.append({"sha256": str(item["sha256"]), "bytes": size})
    return digests


def _verified_backing_rows(dataset: VerifiedDataset) -> dict[str, dict[str, Any]]:
    qualification = dataset.qualification
    acquisition_path = safe_fixed_child(
        dataset.root, ACQUISITION_MANIFEST_RELATIVE_PATH
    )
    summary_path = safe_fixed_child(dataset.root, SOURCE_SUMMARY_RELATIVE_PATH)
    if sha256_file(acquisition_path) != qualification.get(
        "acquisition_manifest_sha256"
    ):
        raise unavailable(
            "backing_manifest_mismatch",
            "Acquisition metadata hash verification failed.",
        )
    output_hashes = qualification.get("output_sha256")
    if not isinstance(output_hashes, dict) or sha256_file(
        summary_path
    ) != output_hashes.get("source_summary.csv"):
        raise unavailable(
            "backing_manifest_mismatch", "Source summary hash verification failed."
        )
    acquisition = parse_json_bytes(
        acquisition_path.read_bytes(), label="acquisition manifest"
    )
    sources = acquisition.get("sources")
    if not isinstance(sources, list):
        raise unavailable(
            "backing_manifest_mismatch", "Acquisition metadata has no source list."
        )
    acquired = {
        str(item["source_id"]): item for item in sources if isinstance(item, dict)
    }
    with summary_path.open(encoding="utf-8", newline="") as stream:
        summaries = {row["source_id"]: row for row in csv.DictReader(stream)}
    if set(acquired) != set(summaries):
        raise unavailable(
            "backing_manifest_mismatch", "Source backing membership does not match."
        )
    return {
        source_id: {"acquisition": acquired[source_id], "summary": summaries[source_id]}
        for source_id in acquired
    }


def _read_dataset_columns(
    dataset: VerifiedDataset, columns: Sequence[str]
) -> pd.DataFrame:
    try:
        import pyarrow.dataset as pyarrow_dataset
    except ImportError as exc:  # pragma: no cover
        raise unavailable(
            "dependency_unavailable",
            "Parquet support is unavailable; install the data extra.",
            retryable=True,
        ) from exc
    table = pyarrow_dataset.dataset(
        dataset.dataset_path, format="parquet", partitioning="hive"
    ).to_table(columns=list(columns))
    return cast(pd.DataFrame, table.to_pandas())


def _source_summary(dataset: VerifiedDataset) -> list[dict[str, Any]]:
    frame = _read_dataset_columns(dataset, ["source_id"])
    observed = (
        frame.groupby("source_id", dropna=False, sort=True)
        .agg(row_count=("source_id", "size"))
        .reset_index()
    )
    backing = _verified_backing_rows(dataset)
    items: list[dict[str, Any]] = []
    for row in observed.to_dict(orient="records"):
        source_id = _safe_label(row["source_id"], label="source identifier")
        if source_id not in backing:
            raise unavailable(
                "backing_manifest_mismatch", "An observed source has no backing record."
            )
        acquisition = backing[source_id]["acquisition"]
        summary = backing[source_id]["summary"]
        if int(summary["rows"]) != int(row["row_count"]):
            raise unavailable(
                "row_count_mismatch", "A source backing row count does not match."
            )
        if int(row["row_count"]) < MIN_AGGREGATE_ROWS:
            continue
        repository = normalize_repository(str(acquisition.get("repository", "")))
        if repository is None:
            raise unavailable(
                "backing_manifest_mismatch",
                "A source backing repository identity is invalid.",
            )
        commit = str(acquisition.get("resolved_commit", ""))
        if not _COMMIT.fullmatch(commit):
            raise unavailable(
                "backing_manifest_mismatch", "A source backing commit is invalid."
            )
        vendor_key = _safe_label(
            summary.get("vendor_key", "unconfirmed"), label="vendor key"
        )
        items.append(
            {
                "source_id": source_id,
                "row_count": int(row["row_count"]),
                "vendor_key": vendor_key,
                "redistribution_status": _safe_label(
                    summary.get("redistribution_status", "unknown"),
                    label="redistribution status",
                ),
                "license_spdx": _safe_label(
                    summary.get("license_spdx", "NOASSERTION"), label="license"
                ),
                "backing_repository": repository,
                "backing_commit": commit,
                "backing_object_digests": _backing_object_digests(
                    acquisition.get("files")
                ),
            }
        )
    return items


def _groups(
    frame: pd.DataFrame, group_by: str | None
) -> Sequence[tuple[str | None, pd.DataFrame]]:
    if group_by is None:
        return [(None, frame)]
    corpus_column = "session_id" if group_by == "source_id" else group_by
    if corpus_column not in frame.columns:
        raise unavailable(
            "operation_unavailable", f"Grouping field {group_by!r} is unavailable."
        )
    return [
        (str(group), subset)
        for group, subset in frame.groupby(corpus_column, dropna=False, sort=True)
    ]


def _metric_summary(
    dataset: VerifiedDataset, operation: DatasetOperationV1
) -> list[dict[str, Any]]:
    frame = load_private_corpus(dataset.root, metrics=list(operation.metrics))
    items: list[dict[str, Any]] = []
    for group, subset in _groups(frame, operation.group_by):
        for metric in operation.metrics:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            if len(values) >= operation.minimum_group_rows:
                items.append(
                    {
                        "group_by": operation.group_by,
                        "group": group,
                        "metric": metric,
                        "n": int(len(values)),
                        "mean": float(values.mean()),
                        "standard_deviation": float(values.std(ddof=1)),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                    }
                )
    return items


def _correlations(
    dataset: VerifiedDataset, operation: DatasetOperationV1
) -> list[dict[str, Any]]:
    frame = load_private_corpus(dataset.root, metrics=list(operation.metrics))
    items: list[dict[str, Any]] = []
    for group, subset in _groups(frame, operation.group_by):
        for left, right in itertools.combinations(operation.metrics, 2):
            pairs = subset[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pairs) >= operation.minimum_group_rows:
                items.append(
                    {
                        "group_by": operation.group_by,
                        "group": group,
                        "left_metric": left,
                        "right_metric": right,
                        "n": int(len(pairs)),
                        "correlation": float(pairs[left].corr(pairs[right])),
                    }
                )
    return items


def execute_dataset_operation(
    dataset: VerifiedDataset, operation: DatasetOperationV1
) -> list[dict[str, Any]]:
    """Execute one aggregate allow-listed operation with bounded output."""
    if operation.kind == "source_summary":
        items = _source_summary(dataset)
    elif operation.kind == "metric_summary":
        items = _metric_summary(dataset, operation)
    else:
        items = _correlations(dataset, operation)
    if len(items) > MAX_RESULT_ITEMS:
        raise unavailable(
            "operation_unavailable",
            "The requested grouping exceeds the bounded result limit.",
        )
    return items


__all__ = ["execute_dataset_operation"]
