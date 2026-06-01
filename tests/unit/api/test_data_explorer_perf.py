"""Regression tests for data-explorer perf/resource fixes.

Covers:
- #6990: GET /datasets lists columns without a full ``json.load``.
- #6991: paginated read endpoint + streamed stats/filter.
- #6989: DatasetStorage enforces TTL retention (bounded DB growth).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.routes.data_explorer as de
from src.api.routes.data_explorer import DatasetStorage, router


@pytest.fixture
def temp_dataset_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        os.environ["DATASET_DIR"] = tmpdir
        yield tmpdir


@pytest.fixture
def client(temp_dataset_dir) -> TestClient:
    test_app = FastAPI()
    test_app.include_router(router)
    return TestClient(test_app)


# ── #6990: list columns without full json.load ──


def test_list_datasets_does_not_full_load_json(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    """Listing must read only the header/first object, not json.load() all of it."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    # Array-of-objects JSON; columns come from the first element.
    payload = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    (out_dir / "data.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(de, "_get_output_dir", lambda: out_dir)

    # Fail loudly if anything full-parses the file during listing.
    def _boom(*_args, **_kwargs):
        raise AssertionError("json.load must not be called when listing datasets")

    monkeypatch.setattr(de.json, "load", _boom)

    resp = client.get("/tools/data-explorer/datasets")
    assert resp.status_code == 200
    entry = next(d for d in resp.json()["datasets"] if d["name"] == "data.json")
    assert entry["columns"] == ["a", "b"]


def test_stream_json_columns_is_bounded(tmp_path, monkeypatch) -> None:
    """_stream_json_columns reads at most the sniff window, never the whole file."""
    big = tmp_path / "big.json"
    rows = [{"x": i, "y": i * 2} for i in range(10000)]
    big.write_text(json.dumps(rows), encoding="utf-8")

    monkeypatch.setattr(de, "_JSON_COLUMN_SNIFF_BYTES", 4096)
    cols = de._stream_json_columns(big)
    assert cols == ["x", "y"]


# ── #6991: paginated read endpoint + streamed stats/filter ──


def test_paginated_rows_endpoint_returns_correct_slices(client: TestClient) -> None:
    """GET /datasets/{id}/rows returns exactly the requested offset/limit slice."""
    rows = [{"i": str(n)} for n in range(10)]
    storage = de.get_dataset_storage()
    dataset_id = storage.store_dataset(
        filename="paged.csv",
        fmt="csv",
        columns=["i"],
        rows=rows,
        size_bytes=100,
    )

    resp = client.get(
        f"/tools/data-explorer/datasets/{dataset_id}/rows",
        params={"offset": 3, "limit": 4},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_rows"] == 10
    assert body["rows"] == [{"i": "3"}, {"i": "4"}, {"i": "5"}, {"i": "6"}]

    # A second page continues correctly.
    resp2 = client.get(
        f"/tools/data-explorer/datasets/{dataset_id}/rows",
        params={"offset": 8, "limit": 4},
    )
    assert resp2.json()["rows"] == [{"i": "8"}, {"i": "9"}]


def test_paginated_rows_unknown_dataset_404(client: TestClient) -> None:
    resp = client.get("/tools/data-explorer/datasets/does-not-exist/rows")
    assert resp.status_code == 404


def test_stats_streams_csv_without_materializing(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    """Stats must stream rows from disk, never read the whole CSV into memory."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    csv_path = out_dir / "nums.csv"
    csv_path.write_text("v\n1\n2\n3\n4\n", encoding="utf-8")
    monkeypatch.setattr(de, "_get_output_dir", lambda: out_dir)

    # Reading the entire file at once is the anti-pattern we are fixing.
    real_read_text = Path.read_text

    def _guard_read_text(self, *args, **kwargs):
        if self == csv_path:
            raise AssertionError("CSV stats must stream, not read_text the whole file")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _guard_read_text)

    resp = client.get("/tools/data-explorer/datasets/nums.csv/stats")
    assert resp.status_code == 200
    stats = resp.json()["stats"]["v"]
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0
    assert stats["mean"] == 2.5
    assert resp.json()["row_count"] == 4


def test_filter_stops_at_limit(client: TestClient, tmp_path, monkeypatch) -> None:
    """Filter streams and stops once the requested page is full."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    lines = "g\n" + "\n".join(["1"] * 100) + "\n"
    (out_dir / "f.csv").write_text(lines, encoding="utf-8")
    monkeypatch.setattr(de, "_get_output_dir", lambda: out_dir)

    resp = client.post(
        "/tools/data-explorer/datasets/f.csv/filter",
        json={"column": "g", "operator": "eq", "value": "1", "limit": 5},
    )
    assert resp.status_code == 200
    assert len(resp.json()["rows"]) == 5


# ── #6989: bounded DB growth via TTL retention ──


def test_store_dataset_purges_expired(tmp_path) -> None:
    """Expired datasets are removed on the next store -> DB stays bounded."""
    db = tmp_path / "datasets.db"
    storage = DatasetStorage(db_path=db)

    # Store one dataset that is already expired.
    old_id = storage.store_dataset(
        filename="old.csv",
        fmt="csv",
        columns=["a"],
        rows=[{"a": "1"}, {"a": "2"}],
        size_bytes=10,
    )
    # Force its TTL into the past.
    with storage._lock, storage._transaction() as conn:
        conn.execute(
            "UPDATE datasets SET ttl_at = ? WHERE dataset_id = ?",
            (time.time() - 1, old_id),
        )

    # A subsequent store triggers retention cleanup of the expired entry.
    new_id = storage.store_dataset(
        filename="new.csv",
        fmt="csv",
        columns=["a"],
        rows=[{"a": "9"}],
        size_bytes=10,
    )

    assert storage.get_dataset_metadata(old_id) is None
    assert storage.get_dataset_metadata(new_id) is not None
    # The expired dataset's rows are gone too (no orphan growth).
    assert storage.get_dataset_rows(old_id) == []
    assert storage.get_dataset_rows(new_id) == [{"a": "9"}]


def test_cleanup_expired_counts_removed(tmp_path) -> None:
    storage = DatasetStorage(db_path=tmp_path / "d.db")
    ds_id = storage.store_dataset(
        filename="x.csv", fmt="csv", columns=["a"], rows=[{"a": "1"}], size_bytes=5
    )
    with storage._lock, storage._transaction() as conn:
        conn.execute(
            "UPDATE datasets SET ttl_at = ? WHERE dataset_id = ?",
            (time.time() - 1, ds_id),
        )
    assert storage.cleanup_expired() == 1
    assert storage.cleanup_expired() == 0
