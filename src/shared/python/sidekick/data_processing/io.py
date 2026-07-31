# mypy: ignore-errors
"""Data I/O utilities for data processing.

Supported formats are limited to handlers implemented below: CSV, TSV, Excel,
Parquet, JSON, MATLAB, SQLite, and NumPy. Pickle input/output is intentionally
disabled because loading pickle files executes untrusted payloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.shared.python.sidekick.data_processing.formats import FileFormatDetector

# Optional imports
try:
    import scipy.io

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pyarrow  # noqa: F401

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


class DataReader:
    """Class for reading data files in various formats."""

    @staticmethod
    def read_file(
        file_path: str | Path, format_type: str | None = None, **kwargs: Any
    ) -> pd.DataFrame:
        """Read a data file based on its format.

        Args:
            file_path: Path to the file.
            format_type: Optional format override. If None, detected from extension.
            **kwargs: Passed to pd.read_* functions.
        """
        path = Path(file_path)
        fmt = (format_type or FileFormatDetector.detect_format(path) or "").lower()

        if fmt == "csv":
            return pd.read_csv(path, **kwargs)
        if fmt == "tsv":
            return pd.read_csv(path, sep="\t", **kwargs)
        if fmt == "excel":
            return pd.read_excel(path, **kwargs)
        if fmt == "parquet":
            if not PYARROW_AVAILABLE:
                raise ImportError("PyArrow is required for Parquet files")
            return pd.read_parquet(path, **kwargs)
        if fmt == "json":
            return pd.read_json(path, **kwargs)
        if fmt == "pickle":
            raise ValueError(
                "Pickle format is disabled for security reasons (CWE-502)."
            )
        if fmt == "numpy":
            data = np.load(path, allow_pickle=False)
            if isinstance(data, np.ndarray):
                return pd.DataFrame(data)
            return pd.DataFrame(data.item())
        if fmt == "matlab":
            if not SCIPY_AVAILABLE:
                raise ImportError("SciPy is required for MATLAB files")
            data = scipy.io.loadmat(path)
            data_keys = [k for k in data if not k.startswith("__")]
            if len(data_keys) == 1:
                return pd.DataFrame(data[data_keys[0]])
            return pd.DataFrame(
                {k: v for k, v in data.items() if not k.startswith("__")}
            )
        if fmt == "sqlite":
            import sqlite3
            from contextlib import closing

            # closing() guarantees the connection is closed even if the query
            # raises (malformed SQL, missing table, locked/corrupt DB), so the
            # connection / file descriptor is not leaked. The lifecycle is
            # bounded to this call.
            with closing(sqlite3.connect(str(path))) as conn:
                return pd.read_sql_query(
                    kwargs.get("query", "SELECT * FROM data"), conn
                )

        raise ValueError(f"Unsupported or undetected format for: {path}")


class DataWriter:
    """Class for writing data files in various formats."""

    @staticmethod
    def write_file(
        df: pd.DataFrame,
        file_path: str | Path,
        format_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Write a DataFrame to a file."""
        if df is None:
            raise ValueError("df must be provided")
        path = Path(file_path)
        fmt = (format_type or FileFormatDetector.detect_format(path) or "").lower()
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "csv":
            df.to_csv(path, index=False, **kwargs)
        elif fmt == "tsv":
            df.to_csv(path, sep="\t", index=False, **kwargs)
        elif fmt == "excel":
            df.to_excel(path, index=False, **kwargs)
        elif fmt == "parquet":
            if not PYARROW_AVAILABLE:
                raise ImportError("PyArrow is required for Parquet files")
            df.to_parquet(path, index=False, **kwargs)
        elif fmt == "json":
            df.to_json(path, orient="records", indent=2, **kwargs)
        elif fmt == "pickle":
            raise ValueError(
                "Pickle format is disabled for security reasons (CWE-502)."
            )
        elif fmt == "numpy":
            np.save(str(path), df.values)
        elif fmt == "sqlite":
            import sqlite3
            from contextlib import closing

            # closing() guarantees the connection is closed even if to_sql
            # raises (type-coercion error, locked DB), so no connection / file
            # descriptor is leaked. The lifecycle is bounded to this call.
            with closing(sqlite3.connect(str(path))) as conn:
                df.to_sql(
                    kwargs.get("table_name", "data"),
                    conn,
                    if_exists="replace",
                    index=False,
                )
        else:
            raise ValueError(f"Unsupported or undetected format for: {path}")


__all__ = [
    "DataReader",
    "DataWriter",
    "FileFormatDetector",
]
