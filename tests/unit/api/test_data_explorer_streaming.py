"""Tests for incremental dataset loading and OOM prevention in Data Explorer router (#6923)."""

from __future__ import annotations

import csv
import json
import pytest
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.data_explorer import (
    router,
    _preview_dataset_from_path,
)


@pytest.fixture
def temp_csv(tmp_path: Path) -> Path:
    """Create a temporary CSV file with 100 rows."""
    filepath = tmp_path / "test_large.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "value"])
        for i in range(100):
            writer.writerow([i, f"item_{i}", i * 1.5])
    return filepath


@pytest.fixture
def temp_json(tmp_path: Path) -> Path:
    """Create a temporary JSON file with 100 list elements."""
    filepath = tmp_path / "test_large.json"
    data = [{"id": i, "name": f"item_{i}", "value": i * 1.5} for i in range(100)]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return filepath


def test_load_csv_incremental(temp_csv: Path) -> None:
    """Streaming preview must read only the limit rows for CSV but return correct total."""
    columns, rows, total = _preview_dataset_from_path(temp_csv, 15)

    assert columns == ["id", "name", "value"]
    assert len(rows) == 15
    assert total == 100
    assert rows[0]["id"] == "0"
    assert rows[14]["id"] == "14"


def test_load_json_limit(temp_json: Path) -> None:
    """Streaming preview must limit loaded rows for JSON and return correct total."""
    columns, rows, total = _preview_dataset_from_path(temp_json, 15)

    assert columns == ["id", "name", "value"]
    assert len(rows) == 15
    assert total == 100
    assert rows[0]["id"] == 0
    assert rows[14]["id"] == 14


def test_preview_endpoint_limits(temp_csv: Path) -> None:
    """Preview endpoint must return limited rows and correct total_rows."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # Mock _find_dataset_path to return our temp CSV path
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.api.routes.data_explorer._find_dataset_path",
            lambda name: temp_csv,
        )

        response = client.get(
            "/tools/data-explorer/datasets/test_large.csv/preview?limit=10"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_large.csv"
        assert len(data["rows"]) == 10
        assert data["total_rows"] == 100
        assert data["columns"] == ["id", "name", "value"]
