"""Data Explorer tool API routes.

Provides REST endpoints for browsing, filtering, and visualizing
simulation datasets in the React Data Explorer tool page.

See issue #1206
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/tools/data-explorer", tags=["data-explorer"])


# ── Request / Response Models ──


class DatasetInfo(BaseModel):
    """Information about a discovered dataset file."""

    name: str
    path: str
    format: str
    size_bytes: int
    columns: list[str] = Field(default_factory=list)


class DatasetListResponse(BaseModel):
    """Response listing available datasets."""

    datasets: list[DatasetInfo]
    total: int
    search_dir: str


class DatasetPreviewResponse(BaseModel):
    """Response with a preview of dataset contents."""

    name: str
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int
    format: str


class DatasetStatsResponse(BaseModel):
    """Response with summary statistics for a dataset."""

    name: str
    columns: list[str]
    row_count: int
    stats: dict[str, dict[str, float | None]]


class DatasetFilterRequest(BaseModel):
    """Request to filter dataset rows."""

    column: str = Field(..., description="Column name to filter on")
    operator: str = Field(
        "eq",
        description="Filter operator: eq, ne, gt, lt, gte, lte, contains",
    )
    value: str = Field(..., description="Filter value (string-encoded)")
    limit: int = Field(100, ge=1, le=10000)


class ImportResponse(BaseModel):
    """Response after importing a dataset."""

    name: str
    format: str
    columns: list[str]
    row_count: int


# ── In-memory dataset cache ──

_loaded_datasets: dict[str, dict[str, Any]] = {}


def _get_output_dir() -> Path:
    """Get the project output directory."""
    return Path(__file__).parent.parent.parent.parent / "output"


def _parse_csv_content(content: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Parse CSV content into columns and rows."""
    reader = csv.DictReader(io.StringIO(content))
    columns = reader.fieldnames or []
    rows = list(reader)
    return list(columns), rows


def _parse_json_content(content: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Parse JSON content into columns and rows."""
    data = json.loads(content)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        columns = list(data[0].keys())
        return columns, data
    if isinstance(data, dict):
        columns = list(data.keys())
        return columns, [data]
    return [], []


# ── Endpoints ──


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets() -> DatasetListResponse:
    """List available datasets in the output directory.

    See issue #1206
    """
    output_dir = _get_output_dir()
    datasets: list[DatasetInfo] = []

    if output_dir.exists():
        supported = {".csv", ".json", ".hdf5", ".h5", ".c3d"}
        for filepath in sorted(output_dir.rglob("*")):
            if filepath.suffix.lower() in supported and filepath.is_file():
                columns: list[str] = []
                try:
                    if filepath.suffix.lower() == ".csv":
                        with open(filepath, encoding="utf-8") as f:
                            header = f.readline().strip()
                        columns = [c.strip().strip('"') for c in header.split(",")]
                    elif filepath.suffix.lower() == ".json":
                        with open(filepath, encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, dict):
                            columns = list(data.keys())
                except (FileNotFoundError, PermissionError, OSError):
                    pass

                datasets.append(
                    DatasetInfo(
                        name=filepath.name,
                        path=str(filepath),
                        format=filepath.suffix.lstrip("."),
                        size_bytes=filepath.stat().st_size,
                        columns=columns,
                    )
                )

    # Also include any loaded (imported) datasets
    for name, ds in _loaded_datasets.items():
        if not any(d.name == name for d in datasets):
            datasets.append(
                DatasetInfo(
                    name=name,
                    path="(imported)",
                    format=ds.get("format", "unknown"),
                    size_bytes=0,
                    columns=ds.get("columns", []),
                )
            )

    return DatasetListResponse(
        datasets=datasets,
        total=len(datasets),
        search_dir=str(output_dir),
    )


@router.get("/datasets/{name}/preview", response_model=DatasetPreviewResponse)
async def preview_dataset(name: str, limit: int = 50) -> DatasetPreviewResponse:
    """Get a preview of dataset contents.

    See issue #1206
    """
    # Check in-memory cache first
    if name in _loaded_datasets:
        ds = _loaded_datasets[name]
        rows = ds["rows"][:limit]
        return DatasetPreviewResponse(
            name=name,
            columns=ds["columns"],
            rows=rows,
            total_rows=len(ds["rows"]),
            format=ds["format"],
        )

    # Try to load from disk
    output_dir = _get_output_dir()
    matches = list(output_dir.rglob(name))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    filepath = matches[0]
    try:
        content = filepath.read_text(encoding="utf-8")
        if filepath.suffix.lower() == ".csv":
            columns, rows = _parse_csv_content(content)
        elif filepath.suffix.lower() == ".json":
            columns, rows = _parse_json_content(content)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Preview not supported for {filepath.suffix} format",
            )

        return DatasetPreviewResponse(
            name=name,
            columns=columns,
            rows=rows[:limit],
            total_rows=len(rows),
            format=filepath.suffix.lstrip("."),
        )

    except HTTPException:
        raise
    except (RuntimeError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/datasets/{name}/stats", response_model=DatasetStatsResponse)
async def dataset_stats(name: str) -> DatasetStatsResponse:
    """Get summary statistics for a dataset.

    See issue #1206
    """
    # Get dataset rows
    if name in _loaded_datasets:
        ds = _loaded_datasets[name]
        columns = ds["columns"]
        rows = ds["rows"]
    else:
        output_dir = _get_output_dir()
        matches = list(output_dir.rglob(name))
        if not matches:
            raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
        filepath = matches[0]
        try:
            content = filepath.read_text(encoding="utf-8")
            if filepath.suffix.lower() == ".csv":
                columns, rows = _parse_csv_content(content)
            elif filepath.suffix.lower() == ".json":
                columns, rows = _parse_json_content(content)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Stats not supported for {filepath.suffix}",
                )
        except HTTPException:
            raise
        except (RuntimeError, TypeError, AttributeError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Compute statistics per column
    stats: dict[str, dict[str, float | None]] = {}
    for col in columns:
        values: list[float] = []
        for row in rows:
            val = row.get(col)
            if val is not None:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    pass

        if values:
            stats[col] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "count": float(len(values)),
            }
        else:
            stats[col] = {"min": None, "max": None, "mean": None, "count": 0.0}

    return DatasetStatsResponse(
        name=name,
        columns=columns,
        row_count=len(rows),
        stats=stats,
    )


@router.post("/import", response_model=ImportResponse)
async def import_dataset(file: UploadFile) -> ImportResponse:
    """Import a CSV or JSON dataset.

    See issue #1206
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".csv", ".json"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {suffix}. Use .csv or .json",
        )

    try:
        content = (await file.read()).decode("utf-8")

        if suffix == ".csv":
            columns, rows = _parse_csv_content(content)
        else:
            columns, rows = _parse_json_content(content)

        _loaded_datasets[file.filename] = {
            "columns": columns,
            "rows": rows,
            "format": suffix.lstrip("."),
        }

        return ImportResponse(
            name=file.filename,
            format=suffix.lstrip("."),
            columns=columns,
            row_count=len(rows),
        )

    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/datasets/{name}/filter")
async def filter_dataset(
    name: str, request: DatasetFilterRequest
) -> DatasetPreviewResponse:
    """Filter a dataset by column value.

    See issue #1206
    """
    # First get the dataset
    if name in _loaded_datasets:
        ds = _loaded_datasets[name]
        columns = ds["columns"]
        rows = ds["rows"]
        fmt = ds["format"]
    else:
        output_dir = _get_output_dir()
        matches = list(output_dir.rglob(name))
        if not matches:
            raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
        filepath = matches[0]
        content = filepath.read_text(encoding="utf-8")
        if filepath.suffix.lower() == ".csv":
            columns, rows = _parse_csv_content(content)
        elif filepath.suffix.lower() == ".json":
            columns, rows = _parse_json_content(content)
        else:
            raise HTTPException(
                status_code=400, detail="Filter not supported for this format"
            )
        fmt = filepath.suffix.lstrip(".")

    if request.column not in columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{request.column}' not found. Available: {columns}",
        )

    # Apply filter
    filtered: list[dict[str, Any]] = []
    for row in rows:
        val = str(row.get(request.column, ""))
        match = False
        if request.operator == "eq":
            match = val == request.value
        elif request.operator == "ne":
            match = val != request.value
        elif request.operator == "contains":
            match = request.value.lower() in val.lower()
        elif request.operator in ("gt", "lt", "gte", "lte"):
            try:
                num_val = float(val)
                num_filter = float(request.value)
                if request.operator == "gt":
                    match = num_val > num_filter
                elif request.operator == "lt":
                    match = num_val < num_filter
                elif request.operator == "gte":
                    match = num_val >= num_filter
                elif request.operator == "lte":
                    match = num_val <= num_filter
            except (ValueError, TypeError):
                pass

        if match:
            filtered.append(row)
            if len(filtered) >= request.limit:
                break

    return DatasetPreviewResponse(
        name=name,
        columns=columns,
        rows=filtered,
        total_rows=len(filtered),
        format=fmt,
    )


@router.get("/export-formats")
async def get_export_formats() -> list[dict[str, str]]:
    """List supported export formats.

    See issue #1206
    """
    return [
        {"format": "csv", "description": "Comma-separated values"},
        {"format": "json", "description": "JSON array of objects"},
    ]
