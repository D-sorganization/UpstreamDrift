"""Supported Sidekick data I/O format registry."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_FORMATS = frozenset(
    {
        "csv",
        "tsv",
        "excel",
        "parquet",
        "json",
        "numpy",
        "matlab",
        "sqlite",
    }
)


class FileFormatDetector:
    """Utility for detecting implemented file formats."""

    _FORMAT_MAP = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".txt": "tsv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".json": "json",
        ".npy": "numpy",
        ".mat": "matlab",
        ".db": "sqlite",
        ".sqlite": "sqlite",
    }

    @classmethod
    def detect_format(cls, file_path: str | Path) -> str | None:
        """Detect format from extension."""
        if file_path is None:
            raise ValueError("file_path must be provided")
        path = Path(file_path)
        return cls._FORMAT_MAP.get(path.suffix.lower())

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get list of supported extensions."""
        return list(cls._FORMAT_MAP.keys())
