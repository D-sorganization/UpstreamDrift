"""Extract club-focused surrogate datasets from the 3D golf parquet export.

The source parquet has club-head position and velocity, but not explicit
club-head acceleration. This extractor uses ``time`` only inside each parquet
row group to differentiate club-head velocity, then excludes time from the
training metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DEFAULT_SOURCE = Path(r"C:\Users\diete\Repositories\data\TenThousandFiles.parquet")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BODY_MANIFEST = SCRIPT_DIR / "column_manifest_inverse_ready.json"
DEFAULT_DIRECT_OUTPUT = (
    SCRIPT_DIR / "data" / "processed" / "club_direct_dynamics.parquet"
)
DEFAULT_BODY_OUTPUT = (
    SCRIPT_DIR / "data" / "processed" / "body_to_club_kinematics.parquet"
)
LOGGER = logging.getLogger(__name__)

CLUB_POSITION_COLUMNS = [
    "ClubLogs_CHGlobalPosition_1",
    "ClubLogs_CHGlobalPosition_2",
    "ClubLogs_CHGlobalPosition_3",
]
CLUB_VELOCITY_COLUMNS = [
    "ClubLogs_CHGlobalVelocity_1",
    "ClubLogs_CHGlobalVelocity_2",
    "ClubLogs_CHGlobalVelocity_3",
]
CLUB_ACCELERATION_COLUMNS = [
    "ClubLogs_CHGlobalAcceleration_1",
    "ClubLogs_CHGlobalAcceleration_2",
    "ClubLogs_CHGlobalAcceleration_3",
]


def _flatten(groups: dict[str, list[str]]) -> list[str]:
    columns: list[str] = []
    for group in groups.values():
        columns.extend(group)
    return columns


def _dedupe(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _cast_string_column_to_float32(
    column: pa.ChunkedArray | pa.Array,
) -> pa.ChunkedArray:
    chunks = column.chunks if isinstance(column, pa.ChunkedArray) else [column]
    converted = []
    for chunk in chunks:
        blank = pc.equal(chunk, "")
        cleaned = pc.if_else(blank, pa.scalar(None, pa.string()), chunk)
        converted.append(pc.cast(cleaned, pa.float32(), safe=False))
    return pa.chunked_array(converted, type=pa.float32())


def _numeric_table(table: pa.Table, columns: list[str]) -> pa.Table:
    arrays = [_cast_string_column_to_float32(table[column]) for column in columns]
    return pa.Table.from_arrays(arrays, names=columns)


def _club_acceleration(table: pa.Table) -> pa.Table:
    time = table["time"].to_numpy(zero_copy_only=False).astype(np.float64)
    arrays = []
    for velocity_column in CLUB_VELOCITY_COLUMNS:
        velocity = (
            table[velocity_column].to_numpy(zero_copy_only=False).astype(np.float64)
        )
        if len(velocity) > 1 and np.all(np.isfinite(time)) and len(np.unique(time)) > 1:
            acceleration = np.gradient(velocity, time, edge_order=1)
        else:
            acceleration = np.full_like(velocity, np.nan, dtype=np.float64)
        arrays.append(pa.array(acceleration.astype(np.float32), type=pa.float32()))
    return pa.Table.from_arrays(arrays, names=CLUB_ACCELERATION_COLUMNS)


def _write_dataset(
    source: Path,
    output: Path,
    body_manifest: dict[str, Any],
    input_columns: list[str],
    target_columns: list[str],
    compression: str,
) -> None:
    source_columns = _dedupe(
        ["time", *input_columns, *CLUB_POSITION_COLUMNS, *CLUB_VELOCITY_COLUMNS]
    )
    parquet_file = pq.ParquetFile(source)
    available = set(parquet_file.schema_arrow.names)
    missing = [column for column in source_columns if column not in available]
    if missing:
        raise ValueError(f"Source parquet is missing selected columns: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    metadata = {
        b"source_parquet": str(source).encode("utf-8"),
        b"body_manifest": json.dumps(body_manifest).encode("utf-8"),
        b"input_columns": json.dumps(input_columns).encode("utf-8"),
        b"target_columns": json.dumps(target_columns).encode("utf-8"),
        b"derived_columns": json.dumps(CLUB_ACCELERATION_COLUMNS).encode("utf-8"),
        b"time_usage": (
            b"time is used only to derive club acceleration within row groups"
        ),
    }

    selected_columns = _dedupe([*input_columns, *target_columns])
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    for row_group in range(parquet_file.num_row_groups):
        raw = parquet_file.read_row_group(row_group, columns=source_columns)
        numeric = _numeric_table(raw, source_columns)
        acceleration = _club_acceleration(numeric)
        full = numeric.append_column(
            CLUB_ACCELERATION_COLUMNS[0], acceleration[CLUB_ACCELERATION_COLUMNS[0]]
        )
        full = full.append_column(
            CLUB_ACCELERATION_COLUMNS[1], acceleration[CLUB_ACCELERATION_COLUMNS[1]]
        )
        full = full.append_column(
            CLUB_ACCELERATION_COLUMNS[2], acceleration[CLUB_ACCELERATION_COLUMNS[2]]
        )
        output_table = full.select(selected_columns)

        if writer is None:
            schema = output_table.schema.with_metadata(metadata)
            writer = pq.ParquetWriter(output, schema=schema, compression=compression)
            output_table = output_table.cast(schema)
        writer.write_table(output_table)
        rows_written += output_table.num_rows

    if writer is not None:
        writer.close()

    summary = {
        "source": str(source),
        "output": str(output),
        "rows": rows_written,
        "columns": len(selected_columns),
        "input_columns": input_columns,
        "target_columns": target_columns,
        "derived_columns": CLUB_ACCELERATION_COLUMNS,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOGGER.info("%s", json.dumps(summary, indent=2))


def extract_club_datasets(
    source: Path,
    body_manifest_path: Path,
    direct_output: Path,
    body_output: Path,
    mode: str,
    compression: str,
) -> None:
    body_manifest = json.loads(body_manifest_path.read_text(encoding="utf-8"))
    positions = list(body_manifest["input_columns"]["positions"])
    velocities = list(body_manifest["input_columns"]["velocities"])
    controls = list(body_manifest["input_columns"]["applied_controls"])
    body_accelerations = list(body_manifest["target_columns"]["accelerations"])

    club_targets = [
        *CLUB_POSITION_COLUMNS,
        *CLUB_VELOCITY_COLUMNS,
        *CLUB_ACCELERATION_COLUMNS,
    ]

    if mode in {"direct", "both"}:
        _write_dataset(
            source=source,
            output=direct_output,
            body_manifest=body_manifest,
            input_columns=[*positions, *velocities, *controls],
            target_columns=club_targets,
            compression=compression,
        )
    if mode in {"body-to-club", "both"}:
        _write_dataset(
            source=source,
            output=body_output,
            body_manifest=body_manifest,
            input_columns=[*positions, *velocities, *body_accelerations],
            target_columns=club_targets,
            compression=compression,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--body-manifest", type=Path, default=DEFAULT_BODY_MANIFEST)
    parser.add_argument("--direct-output", type=Path, default=DEFAULT_DIRECT_OUTPUT)
    parser.add_argument("--body-output", type=Path, default=DEFAULT_BODY_OUTPUT)
    parser.add_argument(
        "--mode", choices=["direct", "body-to-club", "both"], default="both"
    )
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    extract_club_datasets(
        source=args.source,
        body_manifest_path=args.body_manifest,
        direct_output=args.direct_output,
        body_output=args.body_output,
        mode=args.mode,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
