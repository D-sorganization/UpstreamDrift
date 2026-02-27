"""Backward-compatible data reader supporting CSV and Parquet.

Provides ``read_data()`` which transparently reads Parquet if available,
falling back to CSV. This allows incremental migration from CSV to Parquet
without breaking existing code.

See issue #565.

Usage:
    from upstream_drift_tools.data_io import read_data

    # Reads .parquet if it exists, otherwise .csv
    df = read_data("path/to/data.csv")

    # Explicitly read a specific format
    df = read_data("path/to/data.parquet")
"""

from __future__ import annotations

import logging
from pathlib import Path

from contracts import ensure, require

logger = logging.getLogger(__name__)

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def read_data(
    file_path: str | Path,
    *,
    prefer_parquet: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Read a data file, preferring Parquet over CSV when available.

    Args:
        file_path: Path to a CSV or Parquet file. If a CSV path is given
            and ``prefer_parquet`` is True, the function first checks for
            a ``.parquet`` sibling file and reads that instead.
        prefer_parquet: If True (default), try the Parquet equivalent first.
        **kwargs: Passed to ``pd.read_csv`` or ``pd.read_parquet``.

    Returns:
        pandas DataFrame.

    Raises:
        ImportError: If pandas is not installed.
        FileNotFoundError: If neither the file nor its Parquet sibling exist.
    """
    require(file_path is not None, "file_path must not be None")
    require(
        isinstance(file_path, (str, Path)),
        "file_path must be a string or Path",
        value=type(file_path).__name__,
    )

    if not _HAS_PANDAS:
        msg = "pandas is required for read_data(). Install with: pip install pandas pyarrow"
        raise ImportError(msg)

    path = Path(file_path)
    require(
        path.suffix.lower() in {".csv", ".tsv", ".txt", ".parquet"},
        f"Unsupported file extension: {path.suffix}",
        value=path.suffix,
    )

    # If asked for CSV, check if Parquet sibling exists
    if prefer_parquet and path.suffix.lower() == ".csv":
        parquet_path = path.with_suffix(".parquet")
        if parquet_path.exists():
            logger.debug("Reading Parquet sibling: %s", parquet_path)
            return pd.read_parquet(parquet_path, **kwargs)

    # Read the file directly based on extension
    if path.suffix.lower() == ".parquet":
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        return pd.read_parquet(path, **kwargs)

    if path.suffix.lower() in (".csv", ".tsv", ".txt"):
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        delimiter = kwargs.pop(
            "delimiter", "," if path.suffix.lower() != ".tsv" else "\t"
        )
        return pd.read_csv(path, delimiter=delimiter, **kwargs)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def write_data(
    df: pd.DataFrame,
    file_path: str | Path,
    *,
    also_csv: bool = False,
    **kwargs,
) -> Path:
    """Write a DataFrame to Parquet (default) or CSV.

    Args:
        df: DataFrame to write.
        file_path: Output path. Format is determined by extension.
        also_csv: If True and writing Parquet, also write a CSV sibling.
        **kwargs: Passed to ``df.to_parquet`` or ``df.to_csv``.

    Returns:
        Path to the written file.
    """
    require(df is not None, "df must not be None")
    require(file_path is not None, "file_path must not be None")
    require(
        isinstance(file_path, (str, Path)),
        "file_path must be a string or Path",
        value=type(file_path).__name__,
    )

    if not _HAS_PANDAS:
        msg = "pandas is required for write_data()"
        raise ImportError(msg)

    path = Path(file_path)
    ensure(
        path.suffix.lower() in {".csv", ".tsv", ".txt", ".parquet"},
        f"Output file extension must be CSV or Parquet, got: {path.suffix}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, engine="pyarrow", index=False, **kwargs)
        if also_csv:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
    else:
        df.to_csv(path, index=False, **kwargs)

    return path


__all__ = ["read_data", "write_data"]
