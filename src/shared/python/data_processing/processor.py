# mypy: ignore-errors
# ruff: noqa: E501
"""DataProcessor facade -- clean API over the extracted core modules.

This is the main entry point for programmatic data processing without any GUI.
It delegates to the existing ``data_processor.core`` modules but presents a
simplified, chainable interface suitable for scripting, API backends, and tests.

See issue #407.
"""

from __future__ import annotations  # noqa: E402, F404

import logging  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import pandas as pd

from src.shared.python.contracts import require
from src.shared.python.safe_pandas_eval import (
    log_formula_rejected,
    validate_pandas_formula,
)

logger = logging.getLogger(__name__)
SUPPORTED_FILTER_TYPES = {"butterworth", "moving_average", "median", "savgol"}


def _eval_with_optional_numexpr(df: pd.DataFrame, expression: str) -> pd.Series:
    """Validate and evaluate a formula expression against a DataFrame.

    Validation via ``validate_pandas_formula`` runs before any engine is
    invoked.  Direct use of ``engine="python"`` with untrusted input is unsafe
    (no resource limits), but after the allow-list validation above it is safe
    to use as a fallback when numexpr is not installed.
    """

    try:
        validate_pandas_formula(expression, allowed_columns=df.columns)
    except ValueError as error:
        log_formula_rejected(expression, error)
        raise

    try:
        return df.eval(expression, engine="numexpr")
    except ImportError:
        # numexpr is not installed.  The expression has already been validated
        # by validate_pandas_formula, so it is safe to evaluate with the
        # Python engine as a fallback.
        try:
            return df.eval(expression, engine="python")
        except ImportError as exc:
            raise RuntimeError(
                "Formula evaluation requires numexpr. Install it with: pip install numexpr"  # noqa: E501
            ) from exc


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
        """Replace the working DataFrame directly (used by tests and pipeline code)."""
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
            # Delegate to the core DAT importer if available; fall back to
            # whitespace-delimited CSV parsing when the package is absent.
            try:
                from data_processor.core.dat_importer import read_dat_file

                self._df = read_dat_file(str(path))
            except ImportError:
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
        require(isinstance(df, pd.DataFrame), "df must be a pandas DataFrame")
        require(isinstance(name, str) and bool(name), "name must be a non-empty string")
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
        require(isinstance(start, int | float), "start must be numeric")
        require(isinstance(end, int | float), "end must be numeric")
        require(end >= start, "end must be >= start")
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

        Args:
            target_rate: Target sample rate in Hz. Must be positive.
            time_column: Column containing time values. Auto-detected if None.
            method: Interpolation method ('linear', 'cubic', etc.).
        """
        require(
            isinstance(target_rate, int | float) and target_rate > 0,
            "target_rate must be a positive number",
        )
        df = self.dataframe
        if time_column is None:
            time_column = self._detect_time_column(df)

        try:
            from data_processor.core.signal_processing import resample_data

            self._df = resample_data(
                df, target_rate, time_col=time_column, method=method
            )
        except ImportError:
            import numpy as np

            t = np.asarray(df[time_column].values)
            # Validate monotonicity
            if not np.all(np.diff(t) > 0):
                raise ValueError(
                    f"Time column '{time_column}' is not strictly monotonically"
                    " increasing; resampling requires sorted timestamps."
                ) from None
            n_samples = int(round((t[-1] - t[0]) * target_rate)) + 1
            t_new = np.linspace(t[0], t[-1], n_samples)
            new_df = pd.DataFrame({time_column: t_new})
            for col in df.columns:
                if col == time_column:
                    continue
                new_df[col] = np.interp(t_new, t, np.asarray(df[col].values))
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
        sample_rate: float = 1000.0,
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
        sample_rate : float
            Sample rate of the data [Hz].  Used by the Butterworth filter
            to correctly interpret the cutoff frequency.  Defaults to 1000.
        """
        self._validate_filter_contract(filter_type, window_size)
        df = self.dataframe
        selected_columns = self._resolve_filter_columns(df, columns)
        effective_sample_rate = self._resolve_sample_rate(df, filter_type, sample_rate)
        self._apply_filter_impl(
            df=df,
            filter_type=filter_type,
            columns=selected_columns,
            cutoff=cutoff,
            order=order,
            window_size=window_size,
            sample_rate=effective_sample_rate,
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

    def _resolve_sample_rate(
        self, df: pd.DataFrame, filter_type: str, sample_rate: float
    ) -> float:
        """Use detected timestamp spacing for Butterworth filters when available."""
        if filter_type != "butterworth":
            return sample_rate
        try:
            time_column = self._detect_time_column(df)
        except ValueError:
            return sample_rate
        intervals = pd.Series(df[time_column]).diff().dropna()
        median_interval = float(intervals.median()) if not intervals.empty else 0.0
        return 1.0 / median_interval if median_interval > 0.0 else sample_rate

    def _apply_filter_impl(
        self,
        df: pd.DataFrame,
        filter_type: str,
        columns: list[str],
        cutoff: float,
        order: int,
        window_size: int,
        sample_rate: float = 1000.0,
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
                sample_rate=sample_rate,
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
        sample_rate: float = 1000.0,
    ) -> None:
        """Apply filter implementation backed by scipy.signal."""
        from scipy.signal import butter, filtfilt, medfilt, savgol_filter

        for column in columns:
            values = df[column].values.astype(float)
            if filter_type == "butterworth":
                nyquist = sample_rate / 2.0
                if cutoff >= nyquist:
                    raise ValueError(
                        f"Filter cutoff {cutoff} Hz must be below the Nyquist"
                        f" frequency ({nyquist} Hz = sample_rate/2)."
                        " Reduce cutoff or increase sample_rate."
                    )
                b, a = butter(order, cutoff, btype="low", fs=sample_rate)
                min_len = 3 * (max(len(b), len(a)) - 1)
                if len(values) < min_len:
                    raise ValueError(
                        f"Column has {len(values)} samples but filtfilt requires"
                        f" at least {min_len} samples for a filter of this order."
                        " Use a lower filter order."
                    )
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
        require(
            isinstance(new_column, str) and bool(new_column),
            "new_column must be a non-empty string",
        )
        require(
            isinstance(expression, str) and bool(expression),
            "expression must be a non-empty string",
        )
        df = self.dataframe
        df[new_column] = _eval_with_optional_numexpr(df, expression)
        self._history.append(f"Created column '{new_column}' = {expression}")
        return self

    def drop_columns(self, columns: list[str]) -> DataProcessor:
        """Drop specified columns."""
        require(
            isinstance(columns, list) and bool(columns),
            "columns must be a non-empty list",
        )
        self._df = self.dataframe.drop(columns=columns, errors="ignore")
        self._history.append(f"Dropped columns: {columns}")
        return self

    def rename_columns(self, mapping: dict[str, str]) -> DataProcessor:
        """Rename columns."""
        require(
            isinstance(mapping, dict) and bool(mapping),
            "mapping must be a non-empty dict",
        )
        self._df = self.dataframe.rename(columns=mapping)
        self._history.append(f"Renamed {len(mapping)} columns")
        return self

    def sort(self, by: str, ascending: bool = True) -> DataProcessor:
        """Sort by a column."""
        require(isinstance(by, str) and bool(by), "by must be a non-empty string")
        require(isinstance(ascending, bool), "ascending must be a boolean")
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
        require(method is not None, "method must be provided")
        require(
            isinstance(method, str) and bool(method),
            "method must be a non-empty string",
        )
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
        require(method is not None, "method must be provided")
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
            import importlib.util

            if (
                importlib.util.find_spec("pyarrow") is None
                and importlib.util.find_spec("fastparquet") is None
            ):
                raise ImportError(
                    "Parquet export requires pyarrow or fastparquet:"
                    " pip install pyarrow"
                )
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
