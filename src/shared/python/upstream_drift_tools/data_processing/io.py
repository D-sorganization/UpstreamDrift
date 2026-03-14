"""Data I/O utilities for data processing.

Supports reading/writing various formats: CSV, TSV, Parquet, Excel, JSON,
Matlab, Arrow, SQLite, NumPy, and Pickles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Optional imports
try:
    import scipy.io

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
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
            return pd.read_pickle(path)
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

            conn = sqlite3.connect(str(path))
            df = pd.read_sql_query(kwargs.get("query", "SELECT * FROM data"), conn)
            conn.close()
            return df

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
        assert df is not None, "df must be provided"
        assert df is not None, "df must be provided"
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
            df.to_pickle(path)
        elif fmt == "numpy":
            np.save(str(path), df.values)
        elif fmt == "sqlite":
            import sqlite3

            conn = sqlite3.connect(str(path))
            df.to_sql(
                kwargs.get("table_name", "data"), conn, if_exists="replace", index=False
            )
            conn.close()
        else:
            raise ValueError(f"Unsupported or undetected format for: {path}")


class FileFormatDetector:
    """Utility for detecting file formats."""

    _FORMAT_MAP = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".txt": "tsv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".json": "json",
        ".pkl": "pickle",
        ".pickle": "pickle",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".feather": "feather",
        ".npy": "numpy",
        ".mat": "matlab",
        ".db": "sqlite",
        ".sqlite": "sqlite",
    }

    @classmethod
    def detect_format(cls, file_path: str | Path) -> str | None:
        """Detect format from extension."""
        assert file_path is not None, "file_path must be provided"
        assert file_path is not None, "file_path must be provided"
        path = Path(file_path)
        return cls._FORMAT_MAP.get(path.suffix.lower())

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get list of supported extensions."""
        return list(cls._FORMAT_MAP.keys())


__all__ = ["DataReader", "DataWriter", "FileFormatDetector"]
