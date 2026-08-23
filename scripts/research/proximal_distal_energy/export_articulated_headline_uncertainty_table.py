"""Export completed articulated headline uncertainty evidence as deterministic CSV."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
DEFAULT_RECORD = DATA / "articulated_headline_uncertainty.json"
DEFAULT_OUTPUT = DATA / "articulated_headline_uncertainty.csv"
TABLE_SCHEMA_VERSION = "articulated-headline-uncertainty-table/v1"
EVIDENCE_SCOPE = "summary_counts_not_trajectory_data"
PATHWAYS = ("shaft", "ground")
FIELDNAMES = (
    "table_schema_version",
    "study_id",
    "campaign_record_sha256",
    "source_set_sha256",
    "method",
    "evidence_scope",
    "corner_index",
    "corner_id",
    "axis_name",
    "level",
    "value",
    "pathway",
    "pathway_registered",
    "status",
    "failure_class",
    "failure_message",
    "matched_cell_count",
    "matched_cell_count_change_from_nominal",
    "total_cell_count",
    "all_registered_gates_passed",
)


def audit_headline_record(path: Path) -> dict[str, Any]:
    """Load the scientific auditor only when an export is requested."""

    from scripts.research.proximal_distal_energy.articulated_headline_record_audit import (
        audit_headline_record as audit,
    )

    return audit(path)


def _csv_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _rows(
    record: Mapping[str, Any],
    *,
    record_sha256: str,
    source_set_sha256: str,
) -> list[dict[str, str]]:
    common = {
        "table_schema_version": TABLE_SCHEMA_VERSION,
        "study_id": _csv_scalar(record["study_id"]),
        "campaign_record_sha256": record_sha256,
        "source_set_sha256": source_set_sha256,
        "method": _csv_scalar(record["design"]["method"]),
        "evidence_scope": EVIDENCE_SCOPE,
    }
    result: list[dict[str, str]] = []
    for corner_index, corner in enumerate(record["corners"], start=1):
        registered = set(corner["pathways"])
        for pathway in PATHWAYS:
            pathway_result = corner[pathway]
            result.append(
                {
                    **common,
                    "corner_index": str(corner_index),
                    "corner_id": _csv_scalar(corner["corner_id"]),
                    "axis_name": _csv_scalar(corner["axis_name"]),
                    "level": _csv_scalar(corner["level"]),
                    "value": _csv_scalar(corner["value"]),
                    "pathway": pathway,
                    "pathway_registered": _csv_scalar(pathway in registered),
                    "status": _csv_scalar(pathway_result["status"]),
                    "failure_class": _csv_scalar(pathway_result.get("failure_class")),
                    "failure_message": _csv_scalar(
                        pathway_result.get("failure_message")
                    ),
                    "matched_cell_count": _csv_scalar(
                        pathway_result["matched_cell_count"]
                    ),
                    "matched_cell_count_change_from_nominal": _csv_scalar(
                        pathway_result["matched_cell_count_change_from_nominal"]
                    ),
                    "total_cell_count": _csv_scalar(pathway_result["total_cell_count"]),
                    "all_registered_gates_passed": _csv_scalar(
                        pathway_result["all_registered_gates_passed"]
                    ),
                }
            )
    return result


def export_headline_uncertainty_table(
    record_path: Path = DEFAULT_RECORD,
    output_path: Path = DEFAULT_OUTPUT,
) -> int:
    """Write the completed campaign's governed summary projection.

    The CSV contains one row for every corner-pathway pair, including retained
    failures and pathways registered as not affected. Empty nullable cells mean
    JSON ``null``. The output is summary evidence, not trajectory-level data.

    Postcondition: the returned count equals the number of data rows written.
    """

    if not isinstance(record_path, Path) or not isinstance(output_path, Path):
        raise TypeError("record_path and output_path must be pathlib.Path values")
    if output_path.suffix.lower() != ".csv":
        raise ValueError("output_path must use the .csv suffix")

    audit = audit_headline_record(record_path)
    if audit["status"] != "complete":
        raise ValueError("table export requires a complete campaign record")
    raw = record_path.read_bytes()
    observed_digest = hashlib.sha256(raw).hexdigest()
    if observed_digest != audit["record_sha256"]:
        raise RuntimeError("campaign record changed during export")
    record = json.loads(raw.decode("utf-8"))
    rows = _rows(
        record,
        record_sha256=observed_digest,
        source_set_sha256=audit["source_set_sha256"],
    )

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp")
    temporary.write_text(stream.getvalue(), encoding="utf-8", newline="")
    temporary.replace(output_path)
    return len(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", nargs="?", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    row_count = export_headline_uncertainty_table(args.record, args.output)
    print(f"Saved {row_count} rows: {args.output}")


if __name__ == "__main__":
    main()


__all__ = ["export_headline_uncertainty_table", "main"]
