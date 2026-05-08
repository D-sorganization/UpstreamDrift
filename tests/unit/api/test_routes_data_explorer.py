"""Unit tests for the data explorer API route."""

import pytest
import tempfile
import os
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.data_explorer import router, get_dataset_storage


@pytest.fixture
def temp_dataset_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        os.environ["DATASET_DIR"] = tmpdir
        yield tmpdir


@pytest.fixture
def app(temp_dataset_dir) -> FastAPI:
    """Create a FastAPI app with the data explorer router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_list_datasets(client: TestClient, temp_dataset_dir) -> None:
    """Test listing datasets."""
    response = client.get("/tools/data-explorer/datasets")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    assert "total" in data


def test_get_export_formats(client: TestClient) -> None:
    """Test getting export formats."""
    response = client.get("/tools/data-explorer/export-formats")
    assert response.status_code == 200
    data = response.json()
    assert any(f["format"] == "csv" for f in data)


def test_import_dataset_csv(client: TestClient, temp_dataset_dir) -> None:
    """Test importing a CSV dataset."""
    csv_content = b"a,b,c\n1,2,3\n4,5,6"
    files = {"file": ("test.csv", csv_content, "text/csv")}
    response = client.post("/tools/data-explorer/import", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test.csv"
    assert data["row_count"] == 2
    assert "a" in data["columns"]


def test_import_dataset_json(client: TestClient, temp_dataset_dir) -> None:
    """Test importing a JSON dataset."""
    json_content = b'[{"a": 1, "b": 2}, {"a": 3, "b": 4}]'
    files = {"file": ("test.json", json_content, "application/json")}
    response = client.post("/tools/data-explorer/import", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test.json"
    assert data["row_count"] == 2
    assert "a" in data["columns"]
