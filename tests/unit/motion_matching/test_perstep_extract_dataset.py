"""Coverage tests for ``surrogate.perstep.extract_dataset``."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")
from src.shared.python.motion_matching.surrogate.perstep.extract_dataset import (
    extract_dataset,
)


def _write_synthetic_manifest(path: Path, columns: list[str]) -> None:
    payload = {
        "input_columns": {"a": columns[: len(columns) // 2]},
        "target_columns": {"b": columns[len(columns) // 2 :]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_synthetic_parquet(path: Path, columns: list[str]) -> None:
    rows = 10
    data = {col: [str(i + j) for i in range(rows)] for j, col in enumerate(columns)}
    table = pa.table(data)
    pq.write_table(table, path)


def test_extract_dataset_round_trip(tmp_path: Path) -> None:
    """Pin: extract converts strings to float32 and writes a summary."""
    columns = ["c1", "c2", "c3", "c4"]
    src = tmp_path / "src.parquet"
    out = tmp_path / "out" / "slim.parquet"
    manifest = tmp_path / "manifest.json"
    _write_synthetic_parquet(src, columns)
    _write_synthetic_manifest(manifest, columns)

    extract_dataset(
        source=src,
        output=out,
        manifest_path=manifest,
        compression="snappy",
        row_group_size=4,
    )

    assert out.exists()
    summary = out.with_suffix(".summary.json")
    assert summary.exists()
    parsed = json.loads(summary.read_text(encoding="utf-8"))
    assert parsed["rows"] == 10
    assert parsed["columns"] == 4
    # All columns must now be float32.
    table = pq.read_table(out)
    for col in columns:
        assert table.column(col).type == pa.float32()


def test_extract_dataset_missing_columns(tmp_path: Path) -> None:
    """Pin: manifest naming columns absent from the source raises."""
    src = tmp_path / "src.parquet"
    out = tmp_path / "out.parquet"
    manifest = tmp_path / "m.json"
    _write_synthetic_parquet(src, ["a", "b"])
    manifest.write_text(
        json.dumps(
            {
                "input_columns": {"x": ["does_not_exist"]},
                "target_columns": {"y": []},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing selected columns"):
        extract_dataset(
            source=src,
            output=out,
            manifest_path=manifest,
            compression="snappy",
            row_group_size=4,
        )


def test_extract_dataset_overwrites_existing(tmp_path: Path) -> None:
    """Pin: a pre-existing output file is overwritten."""
    columns = ["a", "b"]
    src = tmp_path / "src.parquet"
    out = tmp_path / "slim.parquet"
    manifest = tmp_path / "m.json"
    _write_synthetic_parquet(src, columns)
    _write_synthetic_manifest(manifest, columns)
    out.write_bytes(b"stale")
    extract_dataset(
        source=src,
        output=out,
        manifest_path=manifest,
        compression="snappy",
        row_group_size=4,
    )
    # Now must be a real parquet.
    pq.read_table(out)


def test_extract_dataset_handles_blank_strings(tmp_path: Path) -> None:
    """Pin: blank strings are converted to nulls before float32 cast."""
    src = tmp_path / "src.parquet"
    out = tmp_path / "out.parquet"
    manifest = tmp_path / "m.json"
    table = pa.table({"a": ["1", "", "3"], "b": ["", "2", "3"]})
    pq.write_table(table, src)
    _write_synthetic_manifest(manifest, ["a", "b"])
    extract_dataset(
        source=src,
        output=out,
        manifest_path=manifest,
        compression="snappy",
        row_group_size=4,
    )
    parsed = pq.read_table(out)
    a_vals = parsed.column("a").to_pylist()
    assert a_vals[1] is None
