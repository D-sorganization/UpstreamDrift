"""DataProcessor facade -- clean API over the extracted core modules.

This is the main entry point for programmatic data processing without any GUI.
It delegates to the existing ``data_processor.core`` modules but presents a
simplified, chainable interface suitable for scripting, API backends, and tests.

See issue #407.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)
SUPPORTED_FILTER_TYPES = {"butterworth", "moving_average", "median", "savgol"}


@dataclass
class DatasetInfo:
    """Metadata about a loaded dataset."""

    name: str = ""
    source_path: str = ""
    num_rows: int = 0
    num_columns: int = 0
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    memory_mb: float = 0.0


class DataProcessor:
    """Facade for data loading, transformation, analysis, and export.

    Example::

        dp = DataProcessor()
        dp.load("data.csv")
        dp.trim_time(0, 100)
        dp.apply_filter("butterworth", cutoff=10, order=4, columns=["v1"])
        dp.resample(target_rate=100)
        stats = dp.describe()
        dp.export("out.parquet")
    """

    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None
        self._source_path: str = ""
        self._history: list[str] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the current working DataFrame (raises if nothing loaded)."""
        if self._df is None:
            raise RuntimeError("No data loaded. Call load() first.")
        return self._df

    @dataframe.setter
    def dataframe(self, value: pd.DataFrame) -> None:
        self._df = value

    @property
    def info(self) -> DatasetInfo:
        """Return metadata about the current dataset."""
        if self._df is None:
            return DatasetInfo()
        return DatasetInfo(
            name=Path(self._source_path).stem if self._source_path else "untitled",
            source_path=self._source_path,
            num_rows=len(self._df),
            num_columns=len(self._df.columns),
            columns=list(self._df.columns),
            dtypes={str(col): str(dt) for col, dt in self._df.dtypes.items()},
            memory_mb=self._df.memory_usage(deep=True).sum() / 1e6,
        )

    @property
    def history(self) -> list[str]:
        """Return the processing history (list of operation descriptions)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        path: str | Path,
        *,
        sheet_name: str | int = 0,
        encoding: str = "utf-8",
    ) -> DataProcessor:
        """Load data from a file (CSV, Excel, Parquet, DAT).

        Returns *self* for method chaining.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in (".csv", ".tsv", ".txt"):
            sep = "\t" if suffix == ".tsv" else ","
            self._df = pd.read_csv(path, sep=sep, encoding=encoding)
        elif suffix in (".xlsx", ".xls"):
            self._df = pd.read_excel(path, sheet_name=sheet_name)
        elif suffix == ".parquet":
            self._df = pd.read_parquet(path)
        elif suffix == ".dat":
            # Delegate to the core DAT importer if available
            try:
                from data_processor.core.dat_importer import read_dat_file

                self._df = read_dat_file(str(path))
            except ImportError:
                # Fallback: try whitespace-delimited
                self._df = pd.read_csv(path, sep=r"\s+", encoding=encoding)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        self._source_path = str(path)
        self._history = [
            f"Loaded {path.name} ({len(self._df)} rows, {len(self._df.columns)} cols)"
        ]
        logger.info(
            "Loaded %s: %d rows x %d cols",
            path.name,
            len(self._df),
            len(self._df.columns),
        )
        return self

    def load_dataframe(self, df: pd.DataFrame, name: str = "inline") -> DataProcessor:
        """Load from an existing DataFrame."""
        self._df = df.copy()
        self._source_path = ""
        self._history = [
            f"Loaded DataFrame '{name}' ({len(df)} rows, {len(df.columns)} cols)"
        ]
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def trim_time(
        self,
        start: float,
        end: float,
        time_column: str | None = None,
    ) -> DataProcessor:
        """Trim data to a time range.  Auto-detects the time column if not given."""
        df = self.dataframe
        if time_column is None:
            time_column = self._detect_time_column(df)
        mask = (df[time_column] >= start) & (df[time_column] <= end)
        self._df = df.loc[mask].reset_index(drop=True)
        self._history.append(f"Trimmed time [{start}, {end}] on '{time_column}'")
        return self

    def resample(
        self,
        target_rate: float,
        time_column: str | None = None,
        method: str = "linear",
    ) -> DataProcessor:
        """Resample data to a target sample rate (Hz).

        Uses the core ``resample_data`` when available, else falls back to
        pandas interpolation.
        """
        df = self.dataframe
        if time_column is None:
            time_column = self._detect_time_column(df)

        try:
            from data_processor.core.signal_processing import resample_data

            self._df = resample_data(
                df, target_rate, time_col=time_column, method=method
            )
        except ImportError:
            # Fallback using pandas
            import numpy as np

            t = df[time_column].values
            t_new = np.arange(t[0], t[-1], 1.0 / target_rate)
            new_df = pd.DataFrame({time_column: t_new})
            for col in df.columns:
                if col == time_column:
                    continue
                new_df[col] = np.interp(
                    t_new, np.asarray(t), np.asarray(df[col].values)
                )
            self._df = new_df

        self._history.append(f"Resampled to {target_rate} Hz ({method})")
        return self

    def apply_filter(
        self,
        filter_type: str,
        *,
        columns: list[str] | None = None,
        cutoff: float = 10.0,
        order: int = 4,
        window_size: int = 11,
    ) -> DataProcessor:
        """Apply a signal filter (butterworth, moving_average, median, savgol).

        Parameters
        ----------
        filter_type : str
            One of 'butterworth', 'moving_average', 'median', 'savgol'.
        columns : list[str] | None
            Columns to filter.  Defaults to all numeric columns.
        cutoff : float
            Cutoff frequency for butterworth [Hz].
        order : int
            Filter order.
        window_size : int
            Window size for moving_average / median / savgol.
        """
        self._validate_filter_contract(filter_type, window_size)
        df = self.dataframe
        selected_columns = self._resolve_filter_columns(df, columns)
        self._apply_filter_impl(
            df=df,
            filter_type=filter_type,
            columns=selected_columns,
            cutoff=cutoff,
            order=order,
            window_size=window_size,
        )
        self._df = df
        self._history.append(
            f"Applied {filter_type} filter to {len(selected_columns)} columns"
        )
        return self

    def _validate_filter_contract(self, filter_type: str, window_size: int) -> None:
        """Validate filter preconditions at API boundary."""
        if filter_type not in SUPPORTED_FILTER_TYPES:
            raise ValueError(f"Unknown filter type: {filter_type}")
        if window_size <= 0:
            raise ValueError("window_size must be positive")

    def _resolve_filter_columns(
        self, df: pd.DataFrame, columns: list[str] | None
    ) -> list[str]:
        """Resolve and validate target columns for filtering."""
        selected_columns = (
            list(df.select_dtypes(include="number").columns)
            if columns is None
            else [column for column in columns if column in df.columns]
        )
        if not selected_columns:
            raise ValueError("No valid columns to filter")
        return selected_columns

    def _apply_filter_impl(
        self,
        df: pd.DataFrame,
        filter_type: str,
        columns: list[str],
        cutoff: float,
        order: int,
        window_size: int,
    ) -> None:
        """Apply filter using SciPy implementation when available."""
        try:
            self._apply_filter_with_scipy(
                df=df,
                filter_type=filter_type,
                columns=columns,
                cutoff=cutoff,
                order=order,
                window_size=window_size,
            )
        except ImportError:
            self._apply_filter_fallback(df=df, columns=columns, window_size=window_size)

    def _apply_filter_with_scipy(
        self,
        df: pd.DataFrame,
        filter_type: str,
        columns: list[str],
        cutoff: float,
        order: int,
        window_size: int,
    ) -> None:
        """Apply filter implementation backed by scipy.signal."""
        from scipy.signal import butter, filtfilt, medfilt, savgol_filter

        for column in columns:
            values = df[column].values.astype(float)
            if filter_type == "butterworth":
                b, a = butter(order, cutoff, btype="low", fs=1000)
                df[column] = filtfilt(b, a, values)
            elif filter_type == "moving_average":
                df[column] = (
                    pd.Series(values).rolling(window_size, center=True).mean().values
                )
            elif filter_type == "median":
                kernel = window_size if window_size % 2 == 1 else window_size + 1
                df[column] = medfilt(values, kernel_size=kernel)
            else:
                df[column] = savgol_filter(
                    values, window_size, min(order, window_size - 1)
                )

    def _apply_filter_fallback(
        self, df: pd.DataFrame, columns: list[str], window_size: int
    ) -> None:
        """Fallback filter implementation (moving average) without SciPy."""
        for column in columns:
            df[column] = (
                pd.Series(df[column]).rolling(window_size, center=True).mean().values
            )

    def apply_formula(
        self,
        new_column: str,
        expression: str,
    ) -> DataProcessor:
        """Create a new column from a pandas-eval expression.

        Example: ``dp.apply_formula("speed", "distance / time")``
        """
        df = self.dataframe
        # pandas DataFrame.eval() is safe -- it only resolves column names
        # within the dataframe and does not execute arbitrary Python code.
        df[new_column] = df.eval(expression)
        self._history.append(f"Created column '{new_column}' = {expression}")
        return self

    def drop_columns(self, columns: list[str]) -> DataProcessor:
        """Drop specified columns."""
        self._df = self.dataframe.drop(columns=columns, errors="ignore")
        self._history.append(f"Dropped columns: {columns}")
        return self

    def rename_columns(self, mapping: dict[str, str]) -> DataProcessor:
        """Rename columns."""
        self._df = self.dataframe.rename(columns=mapping)
        self._history.append(f"Renamed {len(mapping)} columns")
        return self

    def sort(self, by: str, ascending: bool = True) -> DataProcessor:
        """Sort by a column."""
        self._df = self.dataframe.sort_values(by=by, ascending=ascending).reset_index(
            drop=True
        )
        self._history.append(f"Sorted by '{by}' ({'asc' if ascending else 'desc'})")
        return self

    def dropna(self, columns: list[str] | None = None) -> DataProcessor:
        """Drop rows with NaN values."""
        if columns:
            self._df = self.dataframe.dropna(subset=columns).reset_index(drop=True)
        else:
            self._df = self.dataframe.dropna().reset_index(drop=True)
        self._history.append("Dropped NaN rows")
        return self

    # ------------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Return descriptive statistics for all numeric columns."""
        df = self.dataframe
        stats = df.describe().to_dict()
        return {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "statistics": stats,
        }

    def correlate(self, method: str = "pearson") -> pd.DataFrame:
        """Return correlation matrix."""
        result: pd.DataFrame = self.dataframe.select_dtypes(include="number").corr(
            method=method
        )
        return result

    def detect_outliers(
        self,
        columns: list[str] | None = None,
        method: str = "zscore",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """Detect outliers and return a boolean mask DataFrame.

        Delegates to ``data_processor.core.outlier_detection`` when available.
        """
        df = self.dataframe
        if columns is None:
            columns = list(df.select_dtypes(include="number").columns)

        try:
            from data_processor.core.outlier_detection import (
                OutlierConfig,
                OutlierDetector,
            )

            detector = OutlierDetector(OutlierConfig(threshold=threshold))
            result = detector.detect(df[columns])
            # outlier_mask is 1D boolean (per row), broadcast to columns
            mask_1d = result.outlier_mask.astype(bool)
            outlier_mask = pd.DataFrame(
                dict.fromkeys(columns, mask_1d),
                index=df.index,
            )
            return outlier_mask
        except ImportError:
            # Fallback: simple z-score
            from scipy import stats as sp_stats

            mask = pd.DataFrame(False, index=df.index, columns=columns)
            for col in columns:
                z = sp_stats.zscore(df[col].dropna())
                mask.loc[df[col].dropna().index, col] = abs(z) > threshold
            return mask

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(
        self,
        path: str | Path,
        *,
        index: bool = False,
        encoding: str = "utf-8",
    ) -> Path:
        """Export the current DataFrame to a file.

        Supported formats: .csv, .xlsx, .parquet, .json
        """
        path = Path(path)
        suffix = path.suffix.lower()
        df = self.dataframe

        if suffix == ".csv":
            df.to_csv(path, index=index, encoding=encoding)
        elif suffix in (".xlsx", ".xls"):
            df.to_excel(path, index=index)
        elif suffix == ".parquet":
            df.to_parquet(path, index=index)
        elif suffix == ".json":
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported export format: {suffix}")

        self._history.append(f"Exported to {path.name}")
        logger.info("Exported %d rows to %s", len(df), path)
        return path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_time_column(df: pd.DataFrame) -> str:
        """Heuristic to detect the time column."""
        candidates = ["time", "Time", "TIME", "t", "T", "timestamp", "Timestamp"]
        for c in candidates:
            if c in df.columns:
                return c
        # Fall back to first column
        return str(df.columns[0])
