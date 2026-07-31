"""Python wrapper for the Rust/Polars bulk-I/O engine (issue #2989, Phase 2).

Tries to import the native ``data_processor_core`` PyO3 extension built by
maturin.  If the wheel is not present (e.g. during CI on plain CPython without
a compiled wheel), every function falls back to a pure-pandas implementation
so contract tests can run without a Rust toolchain.

Public functions match the contract defined in the epic:

- ``inspect(path) -> SchemaInfo``
- ``preview(path, nrows=100, columns=None) -> pd.DataFrame``
- ``convert(src, dst, format) -> ConversionReport``
- ``scan_batch(path, batch_size, columns=None) -> Iterator[pd.DataFrame]``
- ``filter_export(path, dst, predicate, columns=None) -> int``
- ``cancel()`` — signal cancellation (no-op for pandas fallback)

All paths must be ``str`` or ``pathlib.Path``-like.  ``ValueError`` is raised
for contract violations; ``NotImplementedError`` for operations not yet
available in the selected backend.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd

logger = logging.getLogger(__name__)


# ── Try to import the native extension ───────────────────────────────────────

_RUST_AVAILABLE = False
_rust_mod = None

_DISABLE_NATIVE = os.environ.get("DATA_PROCESSOR_IO_DISABLE_NATIVE", "").lower() in {
    "1",
    "true",
    "yes",
}

if not _DISABLE_NATIVE:
    try:
        import data_processor_core as _rust_mod_lib

        _rust_mod = _rust_mod_lib

        _RUST_AVAILABLE = True
        logger.debug("data_processor_core native extension loaded")
    except ImportError:
        logger.debug(
            "data_processor_core native extension not found; using pandas fallback"
        )
else:
    logger.debug("data_processor_core native extension disabled by environment")

# ── Public data types ─────────────────────────────────────────────────────────


@dataclass
class SchemaInfo:
    """Column schema and file metadata returned by :func:`inspect`."""

    columns: list[str] = field(default_factory=list)
    column_types: dict[str, str] = field(default_factory=dict)
    row_count_estimate: int = 0
    file_size_bytes: int = 0
    format: str = ""


@dataclass
class ConversionReport:
    """Statistics returned by :func:`convert`."""

    source: str = ""
    destination: str = ""
    output_format: str = ""
    rows_written: int = 0
    columns: list[str] = field(default_factory=list)
    bytes_written: int = 0


# ── Cancellation token ────────────────────────────────────────────────────────


class OperationCancelled(RuntimeError):
    """Raised when an operation is stopped via its :class:`CancellationToken`."""


class CancellationToken:
    """Per-operation cancellation handle.

    Each long-running operation is scoped to its own token, so cancelling one
    operation never affects another running concurrently. Call :meth:`cancel`
    (from any thread) to stop the bound operation at its next batch boundary.

    Replaces the former process-global ``_cancelled`` flag (issue #3679), which
    let concurrent conversions cancel each other.
    """

    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the operation bound to this token."""
        self._cancelled = True
        logger.info("Engine cancellation requested (token %#x)", id(self))

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested for this token."""
        return self._cancelled

    def reset(self) -> None:
        """Clear the cancellation request so the token can be reused."""
        self._cancelled = False


# Backwards-compatible process-global token used only by the legacy ``cancel()``
# below; new code should pass an explicit per-operation ``CancellationToken``.
_global_token = CancellationToken()


def cancel() -> None:
    """Signal cancellation via the legacy process-global token.

    Deprecated: prefer passing a per-operation :class:`CancellationToken` to
    ``scan_batch``/``filter_export`` so concurrent operations do not cancel each
    other.  This affects only operations that were *not* given an explicit token.
    """
    _global_token.cancel()


def _resolve_token(token: CancellationToken | None) -> CancellationToken:
    """Return the token an operation should observe.

    An explicit per-operation token is used as-is; otherwise the global token is
    reset so the operation starts un-cancelled regardless of a prior ``cancel()``.
    """
    if token is not None:
        return token
    _global_token.reset()
    return _global_token


# ── Helpers ───────────────────────────────────────────────────────────────────


def _str_path(path: str | os.PathLike[str]) -> str:
    s = os.fspath(path) if isinstance(path, os.PathLike) else path
    return str(Path(s)) if s else ""


def _require_path(path: str) -> None:
    if not path:
        raise ValueError("path must not be empty")
    if not Path(path).is_file():
        raise FileNotFoundError(f"file does not exist: {path}")


def _detect_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    raise ValueError(
        f"unsupported format '{suffix}': only csv and parquet are supported"
    )


def _validate_output_format(fmt: str) -> None:
    if fmt not in ("csv", "parquet"):
        raise ValueError(
            f"unsupported output format '{fmt}': only csv and parquet are supported"
        )


def _select_columns(df: pd.DataFrame, columns: Sequence[str] | None) -> pd.DataFrame:
    if columns is None:
        return df
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"column not found: {missing[0]}")
    return df[list(columns)]


# ── inspect ───────────────────────────────────────────────────────────────────


def inspect(path: str | os.PathLike[str]) -> SchemaInfo:
    """Return column names, inferred types, row count estimate, and file size.

    Preconditions:
        - ``path`` must be a non-empty string or Path-like.
        - The file must exist and be readable.
        - Format must be ``csv`` or ``parquet``.

    Raises:
        ValueError: if path is empty or format is unsupported.
        FileNotFoundError: if the file does not exist.
    """
    _global_token.reset()
    p = _str_path(path)
    _require_path(p)
    fmt = _detect_format(p)

    if _RUST_AVAILABLE:
        raw: dict = cast(Any, _rust_mod).py_inspect(p)
        return SchemaInfo(
            columns=raw["columns"],
            column_types=raw["column_types"],
            row_count_estimate=raw["row_count_estimate"],
            file_size_bytes=raw["file_size_bytes"],
            format=raw["format"],
        )

    # ── pandas fallback ──────────────────────────────────────────────────────
    file_size_bytes = Path(p).stat().st_size

    if fmt == "csv":
        df_header = pd.read_csv(p, nrows=0)
        columns_list = list(df_header.columns)
        # Full scan for row count (cheap for CI test fixtures)
        df_full = pd.read_csv(p)
        row_count = len(df_full)
        column_types = {col: str(dtype) for col, dtype in df_full.dtypes.items()}
    else:  # parquet
        df_full = pd.read_parquet(p)
        columns_list = list(df_full.columns)
        row_count = len(df_full)
        column_types = {col: str(dtype) for col, dtype in df_full.dtypes.items()}

    return SchemaInfo(
        columns=columns_list,
        column_types=column_types,
        row_count_estimate=row_count,
        file_size_bytes=file_size_bytes,
        format=fmt,
    )


# ── preview ───────────────────────────────────────────────────────────────────


def preview(
    path: str | os.PathLike[str],
    nrows: int = 100,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the first ``nrows`` rows as a DataFrame.

    Preconditions:
        - ``path`` must exist and be a supported format.
        - ``nrows`` must be greater than zero.
        - ``columns``, when provided, must all be present in the file.

    Raises:
        ValueError: for contract violations.
        FileNotFoundError: if the file does not exist.
    """
    _global_token.reset()
    p = _str_path(path)
    _require_path(p)
    fmt = _detect_format(p)
    if nrows <= 0:
        raise ValueError("nrows must be greater than zero")

    if _RUST_AVAILABLE:
        rows: list[dict] = cast(Any, _rust_mod).py_preview(
            p, nrows, list(columns) if columns is not None else None
        )
        return pd.DataFrame(rows, columns=list(columns) if columns else None)

    # ── pandas fallback ──────────────────────────────────────────────────────
    df = pd.read_csv(p, nrows=nrows) if fmt == "csv" else pd.read_parquet(p).head(nrows)

    return _select_columns(df, columns)


# ── convert ───────────────────────────────────────────────────────────────────


def convert(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    format: str,  # noqa: A002
    token: CancellationToken | None = None,
) -> ConversionReport:
    """Convert ``src`` to ``dst`` in the given ``format``.

    Preconditions:
        - ``src`` must exist and be a supported format.
        - ``format`` must be ``"csv"`` or ``"parquet"``.

    Args:
        token: Optional per-operation cancellation handle (issue #3679).
            Cancelling it before the write stops this conversion only.

    Raises:
        ValueError: for contract violations or unsupported format combinations.
        FileNotFoundError: if ``src`` does not exist.
        NotImplementedError: for format paths not yet implemented.
        OperationCancelled: if ``token`` is cancelled before the output is written.
    """
    active_token = _resolve_token(token)
    p_src = _str_path(src)
    p_dst = _str_path(dst)
    _require_path(p_src)
    _detect_format(p_src)
    _validate_output_format(format)

    if _RUST_AVAILABLE:
        raw: dict = cast(Any, _rust_mod).py_convert(p_src, p_dst, format)
        return ConversionReport(
            source=raw["source"],
            destination=raw["destination"],
            output_format=raw["output_format"],
            rows_written=raw["rows_written"],
            columns=raw["columns"],
            bytes_written=raw["bytes_written"],
        )

    # ── pandas fallback ──────────────────────────────────────────────────────
    src_fmt = _detect_format(p_src)

    df = pd.read_csv(p_src) if src_fmt == "csv" else pd.read_parquet(p_src)

    if active_token.is_cancelled():
        raise OperationCancelled("convert cancelled before write")

    Path(p_dst).parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        df.to_csv(p_dst, index=False)
    elif format == "parquet":
        df.to_parquet(p_dst, index=False)

    bytes_written = Path(p_dst).stat().st_size
    return ConversionReport(
        source=p_src,
        destination=p_dst,
        output_format=format,
        rows_written=len(df),
        columns=list(df.columns),
        bytes_written=bytes_written,
    )


# ── scan_batch ────────────────────────────────────────────────────────────────


def scan_batch(
    path: str | os.PathLike[str],
    batch_size: int,
    columns: Sequence[str] | None = None,
    token: CancellationToken | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield DataFrames of ``batch_size`` rows until the file is exhausted.

    Preconditions:
        - ``path`` must exist and be a supported format.
        - ``batch_size`` must be greater than zero.

    Args:
        token: Optional per-operation cancellation handle (issue #3679), checked
            at each batch boundary. Cancelling it stops this scan only.

    Raises:
        ValueError: for contract violations.
        FileNotFoundError: if ``path`` does not exist.

    Yields:
        pd.DataFrame: successive batches of at most ``batch_size`` rows.
    """
    active_token = _resolve_token(token)
    p = _str_path(path)
    _require_path(p)
    fmt = _detect_format(p)
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if _RUST_AVAILABLE:
        # Phase 2: Rust scan_batch raises NotImplementedError — fall through to
        # the pandas fallback so CI can run contract tests without Phase 3.
        try:
            cast(Any, _rust_mod).py_scan_batch(
                p, batch_size, list(columns) if columns is not None else None
            )
            return
        except NotImplementedError:
            pass

    # ── pandas fallback ──────────────────────────────────────────────────────
    if fmt == "csv":
        reader: Generator[pd.DataFrame, None, None] = pd.read_csv(
            p, chunksize=batch_size
        )
        for chunk in reader:
            if active_token.is_cancelled():
                logger.info("scan_batch cancelled after %d rows", len(chunk))
                return
            yield _select_columns(chunk, columns).reset_index(drop=True)
    else:
        df = pd.read_parquet(p)
        df = _select_columns(df, columns)
        for start in range(0, len(df), batch_size):
            if active_token.is_cancelled():
                return
            yield df.iloc[start : start + batch_size].reset_index(drop=True)


# ── filter_export ─────────────────────────────────────────────────────────────


def filter_export(
    path: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    predicate: str,
    columns: Sequence[str] | None = None,
    token: CancellationToken | None = None,
) -> int:
    """Filter rows matching ``predicate`` (pandas query string) and export.

    Preconditions:
        - ``path`` must exist and be a supported format.
        - ``dst`` must be a non-empty path with a supported format extension.
        - ``predicate`` must be a non-empty string.

    Args:
        token: Optional per-operation cancellation handle (issue #3679).
            Cancelling it before the export writes stops this operation only.

    Returns:
        Number of rows written to ``dst``.

    Raises:
        ValueError: for contract violations.
        FileNotFoundError: if ``path`` does not exist.
        OperationCancelled: if ``token`` is cancelled before the output is written.
    """
    active_token = _resolve_token(token)
    p = _str_path(path)
    p_dst = _str_path(dst)
    _require_path(p)
    if not p_dst:
        raise ValueError("dst must not be empty")
    if not predicate or not predicate.strip():
        raise ValueError("predicate must not be empty")
    fmt = _detect_format(p)
    dst_fmt = _detect_format(p_dst)

    if _RUST_AVAILABLE:
        try:
            return int(
                cast(Any, _rust_mod).py_filter_export(
                    p,
                    p_dst,
                    predicate,
                    list(columns) if columns is not None else None,
                )
            )
        except NotImplementedError:
            pass

    # ── pandas fallback ──────────────────────────────────────────────────────
    df = pd.read_csv(p) if fmt == "csv" else pd.read_parquet(p)

    df = _select_columns(df, columns)
    filtered = df.query(predicate)

    if active_token.is_cancelled():
        raise OperationCancelled("filter_export cancelled before write")

    Path(p_dst).parent.mkdir(parents=True, exist_ok=True)

    if dst_fmt == "csv":
        filtered.to_csv(p_dst, index=False)
    else:
        filtered.to_parquet(p_dst, index=False)

    return len(filtered)


from .bulk_facade import DataProcessorRustError, RustBulkDataEngine  # noqa: E402,F401
