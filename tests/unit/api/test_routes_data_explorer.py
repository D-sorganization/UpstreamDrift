"""Unit tests for the data explorer API route."""

import pytest
import tempfile
import os
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.data_explorer import router


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


def test_list_datasets_does_not_leak_absolute_paths(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    """Issue #6636 F6: dataset listing must not expose absolute server paths."""
    from pathlib import Path

    import src.api.routes.data_explorer as de

    # Build an output dir with a nested dataset file.
    out_dir = tmp_path / "output"
    (out_dir / "runs").mkdir(parents=True)
    (out_dir / "runs" / "swing.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    monkeypatch.setattr(de, "_get_output_dir", lambda: out_dir)

    response = client.get("/tools/data-explorer/datasets")
    assert response.status_code == 200
    data = response.json()

    abs_marker = str(out_dir)
    # No absolute server path may appear in any dataset path or search_dir.
    for ds in data["datasets"]:
        assert abs_marker not in ds["path"]
        assert not Path(ds["path"]).is_absolute()
    assert abs_marker not in data["search_dir"]

    # The relative path is still usable (points at the nested file).
    csv_entries = [d for d in data["datasets"] if d["name"] == "swing.csv"]
    assert csv_entries
    assert csv_entries[0]["path"] == str(Path("runs") / "swing.csv")


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


def test_list_datasets_no_absolute_paths(client: TestClient, temp_dataset_dir) -> None:
    """Test that listing datasets does not return absolute paths."""
    from pathlib import Path
    from unittest.mock import patch

    output_dir = Path(temp_dataset_dir)
    with patch("src.api.routes.data_explorer._get_output_dir", return_value=output_dir):
        filepath = output_dir / "dummy_dataset.csv"
        filepath.write_text("a,b,c\n1,2,3", encoding="utf-8")

        response = client.get("/tools/data-explorer/datasets")
        assert response.status_code == 200
        data = response.json()

        # Check search_dir
        assert data["search_dir"] == output_dir.name

        # Check path
        assert len(data["datasets"]) > 0
        ds = next(d for d in data["datasets"] if d["name"] == "dummy_dataset.csv")
        assert not Path(ds["path"]).is_absolute()
        assert ds["path"] == "dummy_dataset.csv"


def test_preview_streams_only_limit_rows_for_large_csv(
    client: TestClient, temp_dataset_dir
) -> None:
    """Issue #6923: preview must not read an entire large CSV into memory.

    It streams only the header plus ``limit`` rows from a file handle while
    still reporting the true total row count.
    """
    from pathlib import Path
    from unittest.mock import patch

    output_dir = Path(temp_dataset_dir)
    big_csv = output_dir / "big.csv"
    total = 5000
    lines = ["a,b"]
    lines.extend(f"{i},{i * 2}" for i in range(total))
    big_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (
        patch("src.api.routes.data_explorer._get_output_dir", return_value=output_dir),
        patch.object(
            Path, "read_text", side_effect=AssertionError("must not read whole file")
        ),
    ):
        response = client.get("/tools/data-explorer/datasets/big.csv/preview?limit=10")

    assert response.status_code == 200
    data = response.json()
    assert data["columns"] == ["a", "b"]
    assert len(data["rows"]) == 10
    assert data["total_rows"] == total
    assert data["rows"][0] == {"a": "0", "b": "0"}


def test_preview_csv_skips_blank_lines(client: TestClient, temp_dataset_dir) -> None:
    """Issue #6956: blank lines in a CSV must not inflate total_rows or add empty rows.

    csv.reader yields an empty list for blank lines; the streaming path must
    skip them so the row count and preview match what csv.DictReader (used for
    stats/filtering) reports for the same file.
    """
    from pathlib import Path
    from unittest.mock import patch

    output_dir = Path(temp_dataset_dir)
    csv_file = output_dir / "blanks.csv"
    # 3 real data rows, 2 blank lines interspersed
    csv_file.write_text("a,b\n1,2\n\n3,4\n\n5,6\n", encoding="utf-8")

    with patch("src.api.routes.data_explorer._get_output_dir", return_value=output_dir):
        response = client.get(
            "/tools/data-explorer/datasets/blanks.csv/preview?limit=10"
        )

    assert response.status_code == 200
    data = response.json()
    assert data["columns"] == ["a", "b"]
    assert data["total_rows"] == 3
    assert len(data["rows"]) == 3
    assert {"a": "1", "b": "2"} in data["rows"]
    assert {"a": "3", "b": "4"} in data["rows"]
    assert {"a": "5", "b": "6"} in data["rows"]
    assert {} not in data["rows"]


def test_preview_streams_only_limit_rows_for_large_json(
    client: TestClient, temp_dataset_dir
) -> None:
    """Issue #6923: JSON preview returns at most ``limit`` rows."""
    import json
    from pathlib import Path
    from unittest.mock import patch

    output_dir = Path(temp_dataset_dir)
    big_json = output_dir / "big.json"
    total = 2000
    records = [{"x": i, "y": i * 2} for i in range(total)]
    big_json.write_text(json.dumps(records), encoding="utf-8")

    with patch("src.api.routes.data_explorer._get_output_dir", return_value=output_dir):
        response = client.get("/tools/data-explorer/datasets/big.json/preview?limit=5")

    assert response.status_code == 200
    data = response.json()
    assert data["columns"] == ["x", "y"]
    assert len(data["rows"]) == 5
    assert data["total_rows"] == total
