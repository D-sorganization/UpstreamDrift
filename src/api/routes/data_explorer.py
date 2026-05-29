"""Data Explorer tool API routes.

Provides REST endpoints for browsing, filtering, and visualizing
simulation datasets in the React Data Explorer tool page.

See issue #1206
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.api.middleware.error_handler import handle_api_errors
from src.api.middleware.upload_limits import read_upload_file_bytes
from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/tools/data-explorer", tags=["data-explorer"])


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


# ── Dataset storage configuration ──
# Production hardening (issue #3943):
# - Use durable SQLite storage instead of process-global memory
# - Enforce size, row count, and column count limits
# - Use stable dataset IDs instead of raw filenames
# - Support pagination for preview/stats operations

# Configuration constants for upload limits
MAX_DATASET_SIZE_BYTES = 50 * 1024 * 1024  # 50MB max file size
MAX_DATASET_ROWS = 100000  # 100K rows max
MAX_DATASET_COLUMNS = 500  # 500 columns max
MAX_CACHE_AGE_SECONDS = 3600  # 1 hour cache TTL


@dataclass
class DatasetRecord:
    """Dataset metadata and content for durable storage."""

    dataset_id: str
    original_filename: str
    format: str
    columns: list[str]
    row_count: int
    size_bytes: int
    content_hash: str
    created_at: float
    ttl_at: float


class DatasetStorage:
    """SQLite-backed dataset storage with limits and pagination.

    Replaces the unbounded process-global cache with durable,
    size-limited storage that survives process restarts.
    """

    def __init__(self, db_path: Path | None = None):
        """Initialize dataset storage.

        Args:
            db_path: Path to SQLite database. Defaults to ARTIFACT_DIR/datasets.db
        """
        if db_path is None:
            artifact_dir = os.environ.get(
                "DATASET_DIR",
                os.path.join(tempfile.gettempdir(), "upstream_drift_datasets"),
            )
            os.makedirs(artifact_dir, exist_ok=True)
            db_path = Path(artifact_dir) / "datasets.db"

        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                isolation_level=None,
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn  # type: ignore[no-any-return]

    @contextlib.contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database transactions."""
        conn = self._get_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock, self._transaction() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    format TEXT NOT NULL,
                    columns TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    ttl_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_rows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    row_data TEXT NOT NULL,
                    row_index INTEGER NOT NULL,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rows_dataset"
                " ON dataset_rows(dataset_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_datasets_ttl ON datasets(ttl_at)"
            )

    def store_dataset(
        self,
        filename: str,
        fmt: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        size_bytes: int,
    ) -> str:
        """Store a dataset with limits enforcement.

        Args:
            filename: Original filename
            fmt: Dataset format (csv, json)
            columns: Column names
            rows: All rows (validated against limits)
            size_bytes: Size in bytes

        Returns:
            Stable dataset ID

        Raises:
            ValueError: If limits are exceeded
        """
        if len(rows) > MAX_DATASET_ROWS:
            raise ValueError(
                f"Dataset exceeds maximum row limit ({len(rows)} > {MAX_DATASET_ROWS})"
            )
        if len(columns) > MAX_DATASET_COLUMNS:
            raise ValueError(
                f"Dataset exceeds maximum column limit"
                f" ({len(columns)} > {MAX_DATASET_COLUMNS})"
            )
        if size_bytes > MAX_DATASET_SIZE_BYTES:
            raise ValueError(
                f"Dataset exceeds maximum size"
                f" ({size_bytes} > {MAX_DATASET_SIZE_BYTES})"
            )

        dataset_id = str(uuid.uuid4())
        content_hash = hashlib.sha256(
            json.dumps({"columns": columns, "rows": rows}, sort_keys=True).encode()
        ).hexdigest()[:16]
        now = time.time()

        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO datasets (
                    dataset_id, original_filename, format, columns,
                    row_count, size_bytes, content_hash, created_at, ttl_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    filename,
                    fmt,
                    json.dumps(columns),
                    len(rows),
                    size_bytes,
                    content_hash,
                    now,
                    now + MAX_CACHE_AGE_SECONDS,
                ),
            )
            for i, row in enumerate(rows):
                conn.execute(
                    "INSERT INTO dataset_rows"
                    " (dataset_id, row_data, row_index) VALUES (?, ?, ?)",
                    (dataset_id, json.dumps(row), i),
                )

        logger.info(
            "Stored dataset %s (%s): %d rows, %d columns",
            dataset_id,
            filename,
            len(rows),
            len(columns),
        )
        return dataset_id

    def get_dataset_metadata(self, dataset_id: str) -> DatasetRecord | None:
        """Get dataset metadata."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
            )
            row = cursor.fetchone()
            if row:
                return DatasetRecord(
                    dataset_id=row[0],
                    original_filename=row[1],
                    format=row[2],
                    columns=json.loads(row[3]),
                    row_count=row[4],
                    size_bytes=row[5],
                    content_hash=row[6],
                    created_at=row[7],
                    ttl_at=row[8],
                )
            return None

    def get_dataset_rows(
        self,
        dataset_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get paginated dataset rows."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "SELECT row_data FROM dataset_rows"
                " WHERE dataset_id = ? ORDER BY row_index LIMIT ? OFFSET ?",
                (dataset_id, limit, offset),
            )
            return [json.loads(row[0]) for row in cursor.fetchall()]

    def cleanup_expired(self) -> int:
        """Remove expired datasets."""
        now = time.time()
        with self._lock, self._transaction() as conn:
            cursor = conn.execute(
                "SELECT dataset_id FROM datasets WHERE ttl_at < ?", (now,)
            )
            expired_ids = [row[0] for row in cursor.fetchall()]
            for ds_id in expired_ids:
                conn.execute("DELETE FROM dataset_rows WHERE dataset_id = ?", (ds_id,))
                conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (ds_id,))
            return len(expired_ids)


# Global storage instance (lazy initialized)
_dataset_storage: DatasetStorage | None = None

# In-memory LRU cache for backward compatibility and fast repeated access
# (issue #3943). New imports also write to durable SQLite storage.
MAX_LOADED_DATASETS = 32
_loaded_datasets: OrderedDict[str, dict[str, Any]] = OrderedDict()
_cache_lock = asyncio.Lock()


def get_dataset_storage() -> DatasetStorage:
    """Get or create dataset storage instance."""
    global _dataset_storage
    if _dataset_storage is None:
        _dataset_storage = DatasetStorage()
    return _dataset_storage


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


def _enforce_loaded_dataset_limit_locked() -> None:
    """Evict least-recently-used imported datasets beyond the cache ceiling."""
    while len(_loaded_datasets) > MAX_LOADED_DATASETS:
        evicted_name, _ = _loaded_datasets.popitem(last=False)
        logger.debug("Evicted imported dataset from cache: %s", evicted_name)


async def _get_cached_dataset(
    name: str,
) -> tuple[list[str], list[dict[str, Any]], str] | None:
    """Return a copy of an imported dataset and refresh its LRU position."""
    async with _cache_lock:
        dataset = _loaded_datasets.get(name)
        if dataset is None:
            return None
        _loaded_datasets.move_to_end(name)
        columns = list(dataset["columns"])
        rows = list(dataset["rows"])
        dataset_format = str(dataset["format"])
    return columns, rows, dataset_format


async def _store_cached_dataset(
    name: str, columns: list[str], rows: list[dict[str, Any]], dataset_format: str
) -> None:
    """Store an imported dataset with duplicate rejection and LRU eviction."""
    async with _cache_lock:
        if name in _loaded_datasets:
            raise HTTPException(
                status_code=409,
                detail=f"Dataset '{name}' is already imported",
            )
        _loaded_datasets[name] = {
            "columns": columns,
            "rows": rows,
            "format": dataset_format,
        }
        _enforce_loaded_dataset_limit_locked()


def _find_dataset_path(name: str) -> Path:
    """Resolve a dataset filename from output, rejecting ambiguous matches."""
    output_dir = _get_output_dir()
    matches = sorted(path for path in output_dir.rglob(name) if path.is_file())
    if not matches:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    if len(matches) > 1:
        raise HTTPException(
            status_code=409,
            detail=f"Dataset name '{name}' is ambiguous; {len(matches)} matches found",
        )
    return matches[0]


def _load_dataset_from_path(filepath: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Load previewable dataset rows from a resolved path."""
    content = filepath.read_text(encoding="utf-8")
    if filepath.suffix.lower() == ".csv":
        return _parse_csv_content(content)
    if filepath.suffix.lower() == ".json":
        return _parse_json_content(content)
    raise HTTPException(
        status_code=400,
        detail=f"Preview not supported for {filepath.suffix} format",
    )


async def _load_dataset_for_operation(
    name: str, unsupported_detail: str
) -> tuple[list[str], list[dict[str, Any]], str]:
    """Load an imported or disk dataset for endpoint-specific operations."""
    cached_dataset = await _get_cached_dataset(name)
    if cached_dataset is not None:
        return cached_dataset

    filepath = _find_dataset_path(name)
    try:
        columns, rows = _load_dataset_from_path(filepath)
    except HTTPException as exc:
        if exc.status_code == 400:
            raise HTTPException(status_code=400, detail=unsupported_detail) from exc
        raise
    return columns, rows, filepath.suffix.lstrip(".")


def _row_matches_filter(row: dict[str, Any], request: DatasetFilterRequest) -> bool:
    """Return whether a row satisfies the requested filter expression."""
    val = str(row.get(request.column, ""))
    if request.operator == "eq":
        return val == request.value
    if request.operator == "ne":
        return val != request.value
    if request.operator == "contains":
        return request.value.lower() in val.lower()
    if request.operator not in ("gt", "lt", "gte", "lte"):
        return False

    try:
        num_val = float(val)
        num_filter = float(request.value)
    except (ValueError, TypeError) as exc:
        logger.debug("Non-numeric value in filter comparison: %s", exc)
        return False

    if request.operator == "gt":
        return num_val > num_filter
    if request.operator == "lt":
        return num_val < num_filter
    if request.operator == "gte":
        return num_val >= num_filter
    return num_val <= num_filter


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
                except (FileNotFoundError, PermissionError, OSError) as exc:
                    logger.debug(
                        "Could not read columns from %s: %s", filepath.name, exc
                    )

                datasets.append(
                    DatasetInfo(
                        name=filepath.name,
                        path=str(filepath.relative_to(output_dir)),
                        format=filepath.suffix.lstrip("."),
                        size_bytes=filepath.stat().st_size,
                        columns=columns,
                    )
                )

    # Also include any loaded (imported) datasets
    async with _cache_lock:
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
        search_dir=output_dir.name,
    )


@router.get("/datasets/{name}/preview", response_model=DatasetPreviewResponse)
@precondition(
    lambda name, limit=50: name is not None and len(name.strip()) > 0 and limit > 0,
    "Dataset name must be non-empty and limit must be positive",
)
@handle_api_errors
async def preview_dataset(name: str, limit: int = 50) -> DatasetPreviewResponse:
    """Get a preview of dataset contents.

    See issue #1206
    """
    # Check in-memory cache first
    cached_dataset = await _get_cached_dataset(name)
    if cached_dataset is not None:
        columns, rows, dataset_format = cached_dataset
        return DatasetPreviewResponse(
            name=name,
            columns=columns,
            rows=rows[:limit],
            total_rows=len(rows),
            format=dataset_format,
        )

    filepath = _find_dataset_path(name)
    columns, rows = _load_dataset_from_path(filepath)

    return DatasetPreviewResponse(
        name=name,
        columns=columns,
        rows=rows[:limit],
        total_rows=len(rows),
        format=filepath.suffix.lstrip("."),
    )


@router.get("/datasets/{name}/stats", response_model=DatasetStatsResponse)
@precondition(
    lambda name: name is not None and len(name.strip()) > 0,
    "Dataset name must be a non-empty string",
)
@handle_api_errors
async def dataset_stats(name: str) -> DatasetStatsResponse:
    """Get summary statistics for a dataset.

    See issue #1206
    """
    columns, rows, _ = await _load_dataset_for_operation(
        name, "Stats not supported for this format"
    )

    # Compute statistics per column
    stats: dict[str, dict[str, float | None]] = {}
    for col in columns:
        values: list[float] = []
        for row in rows:
            val = row.get(col)
            if val is not None:
                with contextlib.suppress(ValueError, TypeError):
                    values.append(float(val))

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
@handle_api_errors
async def import_dataset(file: UploadFile) -> ImportResponse:
    """Import a CSV or JSON dataset.

    Production hardening (issue #3943):
    - Validates upload size, row count, column count limits
    - Stores in durable SQLite storage with stable dataset ID
    - Returns dataset_id for subsequent operations

    See issue #1206, #3943
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".csv", ".json"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {suffix}. Use .csv or .json",
        )
    if await _get_cached_dataset(file.filename) is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Dataset '{file.filename}' is already imported",
        )

    content_bytes = await read_upload_file_bytes(file)

    # Size validation
    if len(content_bytes) > MAX_DATASET_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({len(content_bytes)} bytes,"
                f" max {MAX_DATASET_SIZE_BYTES})"
            ),
        )

    content = content_bytes.decode("utf-8")

    if suffix == ".csv":
        columns, rows = _parse_csv_content(content)
    else:
        columns, rows = _parse_json_content(content)

    # Store in durable storage with limits enforcement
    storage = get_dataset_storage()
    try:
        dataset_id = storage.store_dataset(
            filename=file.filename,
            fmt=suffix.lstrip("."),
            columns=columns,
            rows=rows,
            size_bytes=len(content_bytes),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Also keep in LRU cache for fast repeated access
    async with _cache_lock:
        _loaded_datasets[file.filename] = {
            "columns": columns,
            "rows": rows,
            "format": suffix.lstrip("."),
            "dataset_id": dataset_id,
        }
        _enforce_loaded_dataset_limit_locked()

    return ImportResponse(
        name=file.filename,
        format=suffix.lstrip("."),
        columns=columns,
        row_count=len(rows),
    )


@router.post("/datasets/{name}/filter")
@precondition(
    lambda name, request: name is not None and len(name.strip()) > 0,
    "Dataset name must be a non-empty string",
)
async def filter_dataset(
    name: str, request: DatasetFilterRequest
) -> DatasetPreviewResponse:
    """Filter a dataset by column value.

    See issue #1206
    """
    columns, rows, fmt = await _load_dataset_for_operation(
        name, "Filter not supported for this format"
    )

    if request.column not in columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{request.column}' not found. Available: {columns}",
        )

    filtered: list[dict[str, Any]] = []
    for row in rows:
        if _row_matches_filter(row, request):
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
