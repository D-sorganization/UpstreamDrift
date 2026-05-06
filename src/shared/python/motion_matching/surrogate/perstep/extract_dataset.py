"""Extract a slim numeric dynamics dataset from the 3D golf parquet export.

The source parquet stores all columns as strings and includes many simulator
outputs that are not needed for the first surrogate model. This script projects
only the manifest-selected columns, casts them to float32, and writes a much
smaller training parquet.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DEFAULT_SOURCE = Path(r"C:\Users\diete\Repositories\data\TenThousandFiles.parquet")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "column_manifest_inverse_ready.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "processed" / "golf_dynamics_slim.parquet"
LOGGER = logging.getLogger(__name__)


def _flatten_manifest_columns(
    manifest: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    input_columns: list[str] = []
    for group in manifest["input_columns"].values():
        input_columns.extend(group)

    target_columns: list[str] = []
    for group in manifest["target_columns"].values():
        target_columns.extend(group)

    # Preserve order while guarding accidental duplicates.
    seen: set[str] = set()
    ordered = []
    for column in [*input_columns, *target_columns]:
        if column not in seen:
            ordered.append(column)
            seen.add(column)

    return ordered, input_columns, target_columns


def _cast_string_column_to_float32(
    column: pa.ChunkedArray | pa.Array,
) -> pa.ChunkedArray:
    """Convert string numeric columns to float32, treating blanks as nulls."""
    chunks = column.chunks if isinstance(column, pa.ChunkedArray) else [column]
    converted = []
    for chunk in chunks:
        blank = pc.equal(chunk, "")
        cleaned = pc.if_else(blank, pa.scalar(None, pa.string()), chunk)
        converted.append(pc.cast(cleaned, pa.float32(), safe=False))
    return pa.chunked_array(converted, type=pa.float32())


def extract_dataset(
    source: Path,
    output: Path,
    manifest_path: Path,
    compression: str,
    row_group_size: int,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_columns, input_columns, target_columns = _flatten_manifest_columns(
        manifest
    )

    parquet_file = pq.ParquetFile(source)
    available = set(parquet_file.schema_arrow.names)
    missing = [column for column in selected_columns if column not in available]
    if missing:
        raise ValueError(f"Source parquet is missing selected columns: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    metadata = {
        b"source_parquet": str(source).encode("utf-8"),
        b"manifest": str(manifest_path).encode("utf-8"),
        b"input_columns": json.dumps(input_columns).encode("utf-8"),
        b"target_columns": json.dumps(target_columns).encode("utf-8"),
    }

    writer: pq.ParquetWriter | None = None
    rows_written = 0
    for batch in parquet_file.iter_batches(
        columns=selected_columns,
        batch_size=row_group_size,
        use_threads=True,
    ):
        columns = [
            _cast_string_column_to_float32(batch.column(i))
            for i in range(batch.num_columns)
        ]
        table = pa.Table.from_arrays(columns, names=selected_columns)
        if writer is None:
            schema = table.schema.with_metadata(metadata)
            writer = pq.ParquetWriter(output, schema=schema, compression=compression)
            table = table.cast(schema)
        writer.write_table(table, row_group_size=row_group_size)
        rows_written += table.num_rows

    if writer is not None:
        writer.close()

    summary = {
        "source": str(source),
        "output": str(output),
        "rows": rows_written,
        "columns": len(selected_columns),
        "input_columns": input_columns,
        "target_columns": target_columns,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("%s", json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--row-group-size", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    extract_dataset(
        source=args.source,
        output=args.output,
        manifest_path=args.manifest,
        compression=args.compression,
        row_group_size=args.row_group_size,
    )


if __name__ == "__main__":
    main()
