"""Unit tests for the data explorer API route."""

import pytest
import tempfile
import os
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.data_explorer import router

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_loaded_dataset_cache():
    """Reset the module-global in-memory dataset cache between tests.

    ``_loaded_datasets`` is process-global, so without this an imported file
    name leaks across tests (e.g. duplicate-import 409s, ghost list entries).
    """
    import src.api.routes.data_explorer as de

    de._loaded_datasets.clear()
    yield
    de._loaded_datasets.clear()


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
    assert data["dataset_id"]
    assert data["row_count"] == 2
    assert "a" in data["columns"]


def test_imported_dataset_list_exposes_dataset_id(
    client: TestClient, temp_dataset_dir
) -> None:
    """Imported dataset list entries expose the durable storage identifier."""
    csv_content = b"a,b\n1,2\n"
    files = {"file": ("listed.csv", csv_content, "text/csv")}
    import_response = client.post("/tools/data-explorer/import", files=files)
    assert import_response.status_code == 200
    dataset_id = import_response.json()["dataset_id"]

    response = client.get("/tools/data-explorer/datasets")

    assert response.status_code == 200
    listed = next(
        ds for ds in response.json()["datasets"] if ds["name"] == "listed.csv"
    )
    assert listed["dataset_id"] == dataset_id


def test_list_datasets_parses_quoted_csv_header(
    client: TestClient, temp_dataset_dir
) -> None:
    """CSV dataset listing must parse quoted header fields with csv semantics."""
    from pathlib import Path
    from unittest.mock import patch

    output_dir = Path(temp_dataset_dir)
    csv_file = output_dir / "quoted_header.csv"
    csv_file.write_text('"club,head",speed\n7i,92\n', encoding="utf-8")

    with patch("src.api.routes.data_explorer._get_output_dir", return_value=output_dir):
        response = client.get("/tools/data-explorer/datasets")

    assert response.status_code == 200
    listed = next(
        ds for ds in response.json()["datasets"] if ds["name"] == csv_file.name
    )
    assert listed["columns"] == ["club,head", "speed"]


def test_import_dataset_json(client: TestClient, temp_dataset_dir) -> None:
    """Test importing a JSON dataset."""
    json_content = b'[{"a": 1, "b": 2}, {"a": 3, "b": 4}]'
    files = {"file": ("test.json", json_content, "application/json")}
    response = client.post("/tools/data-explorer/import", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test.json"
    assert data["dataset_id"]
    assert data["row_count"] == 2
    assert "a" in data["columns"]


def test_filter_rejects_invalid_operator(client: TestClient, temp_dataset_dir) -> None:
    """Invalid filter operators fail request validation instead of matching no rows."""
    csv_content = b"a,b\n1,2\n"
    files = {"file": ("filter.csv", csv_content, "text/csv")}
    assert client.post("/tools/data-explorer/import", files=files).status_code == 200

    response = client.post(
        "/tools/data-explorer/datasets/filter.csv/filter",
        json={"column": "a", "operator": "starts_with", "value": "1"},
    )

    assert response.status_code == 422


def test_stats_ignores_non_finite_cells(client: TestClient, temp_dataset_dir) -> None:
    """Issue #7732: 'inf'/'nan' textual cells must not poison stats."""
    csv_content = b"a\n1\n2\ninf\nnan\n3\n"
    files = {"file": ("finite.csv", csv_content, "text/csv")}
    assert client.post("/tools/data-explorer/import", files=files).status_code == 200

    response = client.get("/tools/data-explorer/datasets/finite.csv/stats")
    assert response.status_code == 200
    # Response must be strictly valid JSON (no bare NaN/Infinity tokens).
    stats = response.json()["stats"]
    col = stats["a"]
    import math as _math

    assert _math.isfinite(col["min"])
    assert _math.isfinite(col["max"])
    assert _math.isfinite(col["mean"])
    # Only the finite cells (1, 2, 3) are aggregated.
    assert col["min"] == 1.0
    assert col["max"] == 3.0
    assert col["mean"] == 2.0


def test_row_matches_filter_rejects_non_finite_operands() -> None:
    """Issue #7732: 'inf'/'nan' filter operands must not match rows."""
    from src.api.routes.data_explorer import (
        DatasetFilterRequest,
        _row_matches_filter,
    )

    # A filter value of 'inf' must not match an ordinary numeric row.
    req = DatasetFilterRequest(column="a", operator="lt", value="inf")
    assert _row_matches_filter({"a": "5"}, req) is False

    # A non-finite cell value must not match a finite numeric filter.
    req2 = DatasetFilterRequest(column="a", operator="gt", value="0")
    assert _row_matches_filter({"a": "inf"}, req2) is False

    # Sanity: a finite comparison still works.
    req3 = DatasetFilterRequest(column="a", operator="gt", value="0")
    assert _row_matches_filter({"a": "5"}, req3) is True


def test_filter_wraps_midstream_csv_decode_errors(
    app: FastAPI, tmp_path, monkeypatch
) -> None:
    """Corrupt streamed CSV rows return a structured API error."""
    import src.api.routes.data_explorer as de

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    csv_path = output_dir / "corrupt.csv"
    csv_path.write_bytes(b"g\n1\n\xff\n")
    monkeypatch.setattr(de, "_get_output_dir", lambda: output_dir)
    no_raise_client = TestClient(app, raise_server_exceptions=False)

    response = no_raise_client.post(
        "/tools/data-explorer/datasets/corrupt.csv/filter",
        json={"column": "g", "operator": "eq", "value": "1"},
    )

    assert response.status_code == 400
    assert "detail" in response.json()


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


# ---------------------------------------------------------------------------
# Filter operator correctness + edge cases (finding #7740 C)
# ---------------------------------------------------------------------------


def _filter(client: TestClient, name: str, **body: object) -> dict:
    resp = client.post(f"/tools/data-explorer/datasets/{name}/filter", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _import_csv(client: TestClient, name: str, content: bytes) -> None:
    files = {"file": (name, content, "text/csv")}
    assert client.post("/tools/data-explorer/import", files=files).status_code == 200


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        ("eq", "2", [{"a": "2", "b": "y"}]),
        ("ne", "2", [{"a": "1", "b": "x"}, {"a": "3", "b": "z"}]),
        ("gt", "2", [{"a": "3", "b": "z"}]),
        ("lt", "2", [{"a": "1", "b": "x"}]),
        ("gte", "2", [{"a": "2", "b": "y"}, {"a": "3", "b": "z"}]),
        ("lte", "2", [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]),
    ],
)
def test_filter_numeric_operators(
    client: TestClient,
    temp_dataset_dir,
    operator: str,
    value: str,
    expected: list,
) -> None:
    """ne/gt/lt/gte/lte each select the correct rows on column 'a'."""
    _import_csv(client, "ops.csv", b"a,b\n1,x\n2,y\n3,z\n")
    data = _filter(client, "ops.csv", column="a", operator=operator, value=value)
    assert data["rows"] == expected


def test_filter_contains_is_case_insensitive(
    client: TestClient, temp_dataset_dir
) -> None:
    """The 'contains' operator matches case-insensitively on substrings."""
    _import_csv(client, "names.csv", b"name\nSwingAlpha\nbeta\nGAMMA\n")
    data = _filter(client, "names.csv", column="name", operator="contains", value="a")
    # All three contain an 'a'/'A' somewhere.
    assert {r["name"] for r in data["rows"]} == {"SwingAlpha", "beta", "GAMMA"}

    data2 = _filter(
        client, "names.csv", column="name", operator="contains", value="swing"
    )
    assert [r["name"] for r in data2["rows"]] == ["SwingAlpha"]


def test_filter_non_numeric_cells_excluded_from_numeric_ops(
    client: TestClient, temp_dataset_dir
) -> None:
    """gt/lt on a column with non-numeric cells silently skips those rows."""
    _import_csv(client, "mixed.csv", b"a\n1\nfoo\n5\n\n")
    data = _filter(client, "mixed.csv", column="a", operator="gt", value="0")
    # Only the numeric rows 1 and 5 qualify; 'foo' and the empty cell are out.
    assert [r["a"] for r in data["rows"]] == ["1", "5"]


def test_filter_empty_string_eq_matches_blank_cells(
    client: TestClient, temp_dataset_dir
) -> None:
    """An eq filter with an empty value matches genuinely-empty cells."""
    _import_csv(client, "blank.csv", b"a,b\n1,\n2,present\n")
    data = _filter(client, "blank.csv", column="b", operator="eq", value="")
    assert [r["a"] for r in data["rows"]] == ["1"]


def test_filter_unicode_values(client: TestClient, temp_dataset_dir) -> None:
    """Filtering matches unicode cell content exactly and via contains."""
    _import_csv(
        client,
        "uni.csv",
        "city\nMünchen\nZürich\nLondon\n".encode(),
    )
    eq_data = _filter(client, "uni.csv", column="city", operator="eq", value="München")
    assert [r["city"] for r in eq_data["rows"]] == ["München"]

    contains_data = _filter(
        client, "uni.csv", column="city", operator="contains", value="ü"
    )
    assert {r["city"] for r in contains_data["rows"]} == {"München", "Zürich"}


def test_filter_unknown_column_returns_400(
    client: TestClient, temp_dataset_dir
) -> None:
    """Filtering on a missing column is a structured 400, not a silent empty."""
    _import_csv(client, "cols.csv", b"a\n1\n")
    resp = client.post(
        "/tools/data-explorer/datasets/cols.csv/filter",
        json={"column": "nope", "operator": "eq", "value": "1"},
    )
    assert resp.status_code == 400


def test_row_matches_filter_operator_matrix() -> None:
    """Direct _row_matches_filter coverage for every operator branch."""
    from src.api.routes.data_explorer import (
        DatasetFilterRequest,
        _row_matches_filter,
    )

    row = {"a": "5", "s": "Hello"}

    def req(op: str, value: str, col: str = "a") -> DatasetFilterRequest:
        return DatasetFilterRequest(column=col, operator=op, value=value)

    assert _row_matches_filter(row, req("eq", "5")) is True
    assert _row_matches_filter(row, req("eq", "6")) is False
    assert _row_matches_filter(row, req("ne", "6")) is True
    assert _row_matches_filter(row, req("ne", "5")) is False
    assert _row_matches_filter(row, req("gt", "4")) is True
    assert _row_matches_filter(row, req("gt", "5")) is False
    assert _row_matches_filter(row, req("lt", "6")) is True
    assert _row_matches_filter(row, req("gte", "5")) is True
    assert _row_matches_filter(row, req("lte", "5")) is True
    assert _row_matches_filter(row, req("contains", "ell", col="s")) is True
    assert _row_matches_filter(row, req("contains", "xyz", col="s")) is False
    # Missing column -> empty string, so eq "" matches.
    assert _row_matches_filter(row, req("eq", "", col="missing")) is True


# ---------------------------------------------------------------------------
# _find_dataset_path: glob rejection + ambiguous-name 409 (finding #7740 F)
# ---------------------------------------------------------------------------


def test_filter_rejects_glob_metacharacters(
    client: TestClient, temp_dataset_dir
) -> None:
    """A name with glob metacharacters must be rejected, not expanded (#7740 F)."""
    from pathlib import Path
    from unittest.mock import patch

    output_dir = Path(temp_dataset_dir)
    (output_dir / "swing.csv").write_text("a\n1\n", encoding="utf-8")

    with patch("src.api.routes.data_explorer._get_output_dir", return_value=output_dir):
        resp = client.post(
            "/tools/data-explorer/datasets/*.csv/filter",
            json={"column": "a", "operator": "eq", "value": "1"},
        )
    # The wildcard must not silently match swing.csv; it is rejected outright.
    assert resp.status_code == 400


def test_find_dataset_path_rejects_glob_pattern() -> None:
    """_find_dataset_path raises 400 for names containing * ? [ ]."""
    from fastapi import HTTPException

    from src.api.routes.data_explorer import _find_dataset_path

    for bad in ("*.csv", "swing*", "data?.csv", "set[1].csv"):
        with pytest.raises(HTTPException) as exc:
            _find_dataset_path(bad)
        assert exc.value.status_code == 400


def test_find_dataset_path_matches_exact_filename_only(
    temp_dataset_dir,
) -> None:
    """A literal name must match by exact filename, not as a glob."""
    from pathlib import Path
    from unittest.mock import patch

    output_dir = Path(temp_dataset_dir)
    (output_dir / "swing.csv").write_text("a\n1\n", encoding="utf-8")

    from src.api.routes.data_explorer import _find_dataset_path

    with patch("src.api.routes.data_explorer._get_output_dir", return_value=output_dir):
        resolved = _find_dataset_path("swing.csv")
    assert resolved.name == "swing.csv"


def test_find_dataset_path_ambiguous_name_409(temp_dataset_dir) -> None:
    """Two files with the same name in different subdirs yield a 409."""
    from pathlib import Path
    from unittest.mock import patch

    from fastapi import HTTPException

    output_dir = Path(temp_dataset_dir)
    (output_dir / "a").mkdir()
    (output_dir / "b").mkdir()
    (output_dir / "a" / "dup.csv").write_text("x\n1\n", encoding="utf-8")
    (output_dir / "b" / "dup.csv").write_text("x\n2\n", encoding="utf-8")

    from src.api.routes.data_explorer import _find_dataset_path

    with (
        patch("src.api.routes.data_explorer._get_output_dir", return_value=output_dir),
        pytest.raises(HTTPException) as exc,
    ):
        _find_dataset_path("dup.csv")
    assert exc.value.status_code == 409


def test_stats_ambiguous_name_returns_409(client: TestClient, temp_dataset_dir) -> None:
    """The ambiguous-name 409 surfaces through an endpoint, not just the helper."""
    from pathlib import Path
    from unittest.mock import patch

    output_dir = Path(temp_dataset_dir)
    (output_dir / "x").mkdir()
    (output_dir / "y").mkdir()
    (output_dir / "x" / "dup.csv").write_text("v\n1\n", encoding="utf-8")
    (output_dir / "y" / "dup.csv").write_text("v\n2\n", encoding="utf-8")

    no_raise = TestClient(client.app, raise_server_exceptions=False)
    with patch("src.api.routes.data_explorer._get_output_dir", return_value=output_dir):
        resp = no_raise.get("/tools/data-explorer/datasets/dup.csv/stats")
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# list_datasets pagination + bounded scan (finding #7740 H)
# ---------------------------------------------------------------------------


def test_list_datasets_pagination_offset_limit(
    client: TestClient, temp_dataset_dir
) -> None:
    """offset/limit return a stable, non-overlapping window of on-disk files."""
    from pathlib import Path
    from unittest.mock import patch

    output_dir = Path(temp_dataset_dir)
    for i in range(10):
        (output_dir / f"set_{i:02d}.csv").write_text("a\n1\n", encoding="utf-8")

    with patch("src.api.routes.data_explorer._get_output_dir", return_value=output_dir):
        page1 = client.get("/tools/data-explorer/datasets?offset=0&limit=4").json()
        page2 = client.get("/tools/data-explorer/datasets?offset=4&limit=4").json()

    names1 = [d["name"] for d in page1["datasets"]]
    names2 = [d["name"] for d in page2["datasets"]]
    assert len(names1) == 4
    assert len(names2) == 4
    # Stable sorted order, no overlap between pages.
    assert set(names1).isdisjoint(names2)
    assert names1 == sorted(names1)


def test_list_datasets_truncates_at_scan_cap(
    client: TestClient, temp_dataset_dir, monkeypatch
) -> None:
    """When the tree exceeds the hard scan cap, truncated=True is reported."""
    from pathlib import Path
    from unittest.mock import patch

    import src.api.routes.data_explorer as de

    output_dir = Path(temp_dataset_dir)
    for i in range(6):
        (output_dir / f"f_{i}.csv").write_text("a\n1\n", encoding="utf-8")

    # Force a tiny cap so the scan is provably bounded.
    monkeypatch.setattr(de, "MAX_DATASET_LIST_SCAN", 3)

    with patch("src.api.routes.data_explorer._get_output_dir", return_value=output_dir):
        resp = client.get("/tools/data-explorer/datasets?limit=3")
    body = resp.json()
    assert body["truncated"] is True
    assert len(body["datasets"]) <= 3
