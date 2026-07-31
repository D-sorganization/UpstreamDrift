"""Compatibility facade for the shared data-processor fallback package."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class DataProcessorRustError(RuntimeError):
    """Raised when the Rust data engine rejects or cannot complete a request."""


@dataclass(frozen=True)
class DatasetMetadata:
    """Compatibility metadata returned by :class:`RustBulkDataEngine`."""

    format: str
    row_count: int
    columns: list[str]
    byte_size: int


@dataclass(frozen=True)
class PreviewTable:
    """Compatibility preview returned by :class:`RustBulkDataEngine`."""

    columns: list[str]
    rows: list[dict[str, str]]
    rows_returned: int


@dataclass(frozen=True)
class BulkConversionReport:
    """Compatibility conversion report returned by :class:`RustBulkDataEngine`."""

    input: str
    output: str
    output_format: str
    rows_read: int
    rows_written: int
    columns: list[str]
    bytes_written: int


class RustBulkDataEngine:
    """Compatibility facade over the pandas fallback bulk-data functions."""

    def __init__(self, *, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root

    @classmethod
    def from_repo_root(cls, repo_root: Path | None = None) -> RustBulkDataEngine:
        """Construct the compatibility engine using the current repository layout."""
        return cls(repo_root=repo_root)

    def is_available(self) -> bool:
        """Return whether this process can service bulk-data requests."""
        return True

    def inspect(self, path: str | os.PathLike[str]) -> DatasetMetadata:
        """Inspect a supported dataset through the fallback implementation."""
        from . import rust_engine

        info = rust_engine.inspect(path)
        return DatasetMetadata(
            format=info.format,
            row_count=info.row_count_estimate,
            columns=info.columns,
            byte_size=info.file_size_bytes,
        )

    def preview(
        self,
        path: str | os.PathLike[str],
        *,
        rows: int = 100,
        columns: list[str] | None = None,
    ) -> PreviewTable:
        """Preview rows through the fallback implementation."""
        from . import rust_engine

        frame = rust_engine.preview(path, nrows=rows, columns=columns)
        return PreviewTable(
            columns=[str(column) for column in frame.columns],
            rows=[
                {str(key): str(value) for key, value in row.items()}
                for row in frame.to_dict(orient="records")
            ],
            rows_returned=len(frame),
        )

    def convert(
        self,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        *,
        output_format: str = "csv",
        columns: list[str] | None = None,
    ) -> BulkConversionReport:
        """Convert a dataset through the fallback implementation."""
        if columns is None:
            return self._convert_all_columns(input_path, output_path, output_format)
        return self._convert_selected_columns(
            input_path,
            output_path,
            output_format,
            columns,
        )

    def _convert_all_columns(
        self,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        output_format: str,
    ) -> BulkConversionReport:
        from . import rust_engine

        report = rust_engine.convert(input_path, output_path, output_format)
        return BulkConversionReport(
            input=str(input_path),
            output=str(output_path),
            output_format=output_format,
            rows_read=report.rows_written,
            rows_written=report.rows_written,
            columns=report.columns,
            bytes_written=report.bytes_written,
        )

    def _convert_selected_columns(
        self,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        output_format: str,
        columns: list[str],
    ) -> BulkConversionReport:
        from . import rust_engine

        frame = rust_engine.preview(input_path, nrows=10**12, columns=columns)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if output_format == "csv":
            frame.to_csv(output_path, index=False)
        elif output_format == "parquet":
            frame.to_parquet(output_path, index=False)
        else:
            raise ValueError(
                f"unsupported output format '{output_format}': "
                "only csv and parquet are supported"
            )

        return BulkConversionReport(
            input=str(input_path),
            output=str(output_path),
            output_format=output_format,
            rows_read=len(frame),
            rows_written=len(frame),
            columns=[str(column) for column in frame.columns],
            bytes_written=Path(output_path).stat().st_size,
        )
