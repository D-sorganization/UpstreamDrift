"""Signal import and export utilities.

This module provides functionality for importing signals from various
file formats (CSV, JSON, numpy) and exporting signals to those formats.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.core.contracts import require
from src.shared.python.signal_toolkit.core import Signal

logger = logging.getLogger(__name__)


class SignalImporter:
    """Import signals from various file formats."""

    @staticmethod
    def _read_csv_rows(
        file_path: Path, delimiter: str, encoding: str
    ) -> list[list[str]]:
        """Read all rows from a CSV file.

        Raises:
            ValueError: If the file is empty.
        """
        with open(file_path, encoding=encoding) as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)
        if not rows:
            msg = f"Empty CSV file: {file_path}"
            raise ValueError(msg)
        return rows

    @staticmethod
    def _resolve_column(col: str | int, header: list[str]) -> int:
        """Convert a column name or index to a numeric column index.

        Raises:
            ValueError: If a named column is not found in the header.
        """
        if isinstance(col, int):
            return col
        try:
            return header.index(col)
        except ValueError:
            msg = f"Column '{col}' not found in header: {header}"
            raise ValueError(msg) from None

    @staticmethod
    def _resolve_value_columns(
        value_columns: str | int | list[str | int] | None,
        header: list[str],
        time_idx: int,
    ) -> tuple[list[int], list[str]]:
        """Resolve value column specs into indices and names."""
        if value_columns is None:
            value_indices = [i for i in range(len(header)) if i != time_idx]
            value_names = [header[i] for i in value_indices]
        elif isinstance(value_columns, (str, int)):
            idx = SignalImporter._resolve_column(value_columns, header)
            value_indices = [idx]
            value_names = [header[idx]]
        else:
            value_indices = [
                SignalImporter._resolve_column(c, header) for c in value_columns
            ]
            value_names = [header[i] for i in value_indices]
        return value_indices, value_names

    @staticmethod
    def _parse_data_rows(
        data_rows: list[list[str]],
        time_idx: int,
        value_indices: list[int],
        time_scale: float,
    ) -> tuple[np.ndarray, dict[int, list[float]]]:
        """Parse numeric data from CSV rows into time and value arrays."""
        time_data: list[float] = []
        value_data: dict[int, list[float]] = {i: [] for i in value_indices}

        for row in data_rows:
            if len(row) <= time_idx:
                continue
            try:
                time_data.append(float(row[time_idx]) * time_scale)
                for idx in value_indices:
                    if idx < len(row):
                        value_data[idx].append(float(row[idx]))
                    else:
                        value_data[idx].append(np.nan)
            except ValueError:
                continue  # Skip rows with non-numeric data

        return np.array(time_data), value_data

    @staticmethod
    def from_csv(
        file_path: str | Path,
        time_column: str | int = 0,
        value_columns: str | int | list[str | int] | None = None,
        delimiter: str = ",",
        skip_header: bool = True,
        time_scale: float = 1.0,
        encoding: str = "utf-8",
    ) -> Signal | list[Signal]:
        """Import signal(s) from a CSV file.

        Args:
            file_path: Path to the CSV file.
            time_column: Column name or index for time data.
            value_columns: Column name(s) or index(es) for value data.
                If None, imports all columns except time.
            delimiter: CSV delimiter.
            skip_header: Whether the CSV has a header row.
            time_scale: Scale factor for time values (e.g., 0.001 for ms to s).
            encoding: File encoding.

        Returns:
            Single Signal if one value column, list of Signals otherwise.

        Raises:
            ValueError: If file is empty, time_scale is non-positive, or column not found.
        """
        if not isinstance(time_scale, (int, float)) or time_scale <= 0:
            raise ValueError(f"time_scale must be positive, got {time_scale}")

        file_path = Path(file_path)
        rows = SignalImporter._read_csv_rows(file_path, delimiter, encoding)

        # Parse header
        if skip_header:
            header = rows[0]
            data_rows = rows[1:]
        else:
            header = [str(i) for i in range(len(rows[0]))]
            data_rows = rows

        time_idx = SignalImporter._resolve_column(time_column, header)
        value_indices, value_names = SignalImporter._resolve_value_columns(
            value_columns, header, time_idx
        )

        time_array, value_data = SignalImporter._parse_data_rows(
            data_rows, time_idx, value_indices, time_scale
        )

        # Create signals
        signals = []
        for idx, name in zip(value_indices, value_names, strict=False):
            sig = Signal(
                time=time_array.copy(),
                values=np.array(value_data[idx]),
                name=name,
                metadata={"source_file": str(file_path), "column": name},
            )
            signals.append(sig)

        if len(signals) == 1:
            return signals[0]
        return signals

    @staticmethod
    def from_numpy(
        time: np.ndarray,
        values: np.ndarray,
        name: str = "imported_signal",
        units: str = "",
    ) -> Signal:
        """Create a Signal from numpy arrays.

        Args:
            time: Time array.
            values: Values array.
            name: Signal name.
            units: Signal units.

        Returns:
            Signal object.
        """
        require(len(time) > 0, "time array must be non-empty")
        require(
            len(time) == len(values),
            "time and values must have same length",
            {"time_len": len(time), "values_len": len(values)},
        )
        return Signal(time=time, values=values, name=name, units=units)

    @staticmethod
    def from_npz(
        file_path: str | Path,
        time_key: str = "time",
        value_key: str = "values",
        name: str | None = None,
    ) -> Signal:
        """Import a signal from a numpy .npz file.

        Args:
            file_path: Path to the .npz file.
            time_key: Key for time array in the archive.
            value_key: Key for values array in the archive.
            name: Signal name (defaults to value_key).

        Returns:
            Signal object.
        """
        file_path = Path(file_path)
        data = np.load(file_path)

        time = data[time_key]
        values = data[value_key]

        return Signal(
            time=time,
            values=values,
            name=name or value_key,
            metadata={"source_file": str(file_path)},
        )

    @staticmethod
    def from_json(
        file_path: str | Path,
        time_key: str = "time",
        value_key: str = "values",
    ) -> Signal:
        """Import a signal from a JSON file.

        Args:
            file_path: Path to the JSON file.
            time_key: Key for time data.
            value_key: Key for values data.

        Returns:
            Signal object.
        """
        file_path = Path(file_path)

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        time = np.array(data[time_key])
        values = np.array(data[value_key])

        name = data.get("name", file_path.stem)
        units = data.get("units", "")
        metadata = data.get("metadata", {})
        metadata["source_file"] = str(file_path)

        return Signal(
            time=time, values=values, name=name, units=units, metadata=metadata
        )

    @staticmethod
    def from_dict(
        data: dict[str, Any],
        time_key: str = "time",
        value_key: str = "values",
    ) -> Signal:
        """Create a Signal from a dictionary.

        Args:
            data: Dictionary with time and values.
            time_key: Key for time data.
            value_key: Key for values data.

        Returns:
            Signal object.
        """
        time = np.array(data[time_key])
        values = np.array(data[value_key])

        return Signal(
            time=time,
            values=values,
            name=data.get("name", "signal"),
            units=data.get("units", ""),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def from_mat(
        file_path: str | Path,
        time_var: str = "t",
        value_var: str = "y",
        name: str | None = None,
    ) -> Signal:
        """Import a signal from a MATLAB .mat file.

        Args:
            file_path: Path to the .mat file.
            time_var: Variable name for time.
            value_var: Variable name for values.
            name: Signal name (defaults to value_var).

        Returns:
            Signal object.
        """
        from scipy.io import loadmat

        file_path = Path(file_path)
        data = loadmat(file_path)

        time = np.asarray(data[time_var]).flatten()
        values = np.asarray(data[value_var]).flatten()

        return Signal(
            time=time,
            values=values,
            name=name or value_var,
            metadata={"source_file": str(file_path)},
        )


class SignalExporter:
    """Export signals to various file formats."""

    @staticmethod
    def to_csv(
        signal: Signal | list[Signal],
        file_path: str | Path,
        time_column_name: str = "time",
        delimiter: str = ",",
        include_header: bool = True,
        precision: int = 6,
    ) -> None:
        """Export signal(s) to a CSV file.

        Args:
            signal: Signal or list of Signals to export.
            file_path: Output file path.
            time_column_name: Name for the time column.
            delimiter: CSV delimiter.
            include_header: Whether to include header row.
            precision: Number of decimal places.
        """
        file_path = Path(file_path)

        signals = [signal] if isinstance(signal, Signal) else signal

        # Ensure all signals have the same time array
        time = signals[0].time

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=delimiter)

            if include_header:
                header = [time_column_name] + [s.name for s in signals]
                writer.writerow(header)

            for i in range(len(time)):
                row = [round(time[i], precision)]
                for sig in signals:
                    row.append(round(sig.values[i], precision))
                writer.writerow(row)

    @staticmethod
    def to_numpy(
        signal: Signal,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Export signal to numpy arrays.

        Args:
            signal: Signal to export.

        Returns:
            Tuple of (time, values) arrays.
        """
        return signal.time.copy(), signal.values.copy()

    @staticmethod
    def to_npz(
        signal: Signal | list[Signal],
        file_path: str | Path,
        compressed: bool = True,
    ) -> None:
        """Export signal(s) to a numpy .npz file.

        Args:
            signal: Signal or list of Signals to export.
            file_path: Output file path.
            compressed: Whether to use compressed format.
        """
        file_path = Path(file_path)

        signals = [signal] if isinstance(signal, Signal) else signal

        data = {"time": signals[0].time}
        for sig in signals:
            data[sig.name] = sig.values

        if compressed:
            np.savez_compressed(file_path, **data)  # type: ignore[arg-type]
        else:
            np.savez(file_path, **data)  # type: ignore[arg-type]

    @staticmethod
    def to_json(
        signal: Signal,
        file_path: str | Path,
        precision: int = 6,
        indent: int = 2,
    ) -> None:
        """Export signal to a JSON file.

        Args:
            signal: Signal to export.
            file_path: Output file path.
            precision: Number of decimal places.
            indent: JSON indentation.
        """
        file_path = Path(file_path)

        data = {
            "name": signal.name,
            "units": signal.units,
            "time": [round(t, precision) for t in signal.time.tolist()],
            "values": [round(v, precision) for v in signal.values.tolist()],
            "metadata": signal.metadata,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)

    @staticmethod
    def to_dict(
        signal: Signal,
    ) -> dict[str, Any]:
        """Export signal to a dictionary.

        Args:
            signal: Signal to export.

        Returns:
            Dictionary representation of the signal.
        """
        return {
            "name": signal.name,
            "units": signal.units,
            "time": signal.time.tolist(),
            "values": signal.values.tolist(),
            "metadata": signal.metadata,
        }

    @staticmethod
    def to_mat(
        signal: Signal | list[Signal],
        file_path: str | Path,
        time_var: str = "t",
    ) -> None:
        """Export signal(s) to a MATLAB .mat file.

        Args:
            signal: Signal or list of Signals to export.
            file_path: Output file path.
            time_var: Variable name for time.
        """
        from scipy.io import savemat

        file_path = Path(file_path)

        signals = [signal] if isinstance(signal, Signal) else signal

        data = {time_var: signals[0].time}
        for sig in signals:
            # MATLAB variable names can't have some characters
            safe_name = sig.name.replace(" ", "_").replace("-", "_")
            data[safe_name] = sig.values

        savemat(file_path, data)


# Convenience functions


def import_from_csv(
    file_path: str | Path,
    time_column: str | int = 0,
    value_columns: str | int | list[str | int] | None = None,
    **kwargs,
) -> Signal | list[Signal]:
    """Import signal(s) from a CSV file (convenience function).

    Args:
        file_path: Path to the CSV file.
        time_column: Column name or index for time data.
        value_columns: Column name(s) or index(es) for value data.
        **kwargs: Additional arguments for SignalImporter.from_csv.

    Returns:
        Single Signal if one value column, list of Signals otherwise.
    """
    return SignalImporter.from_csv(file_path, time_column, value_columns, **kwargs)


def export_to_csv(
    signal: Signal | list[Signal],
    file_path: str | Path,
    **kwargs,
) -> None:
    """Export signal(s) to a CSV file (convenience function).

    Args:
        signal: Signal or list of Signals to export.
        file_path: Output file path.
        **kwargs: Additional arguments for SignalExporter.to_csv.
    """
    SignalExporter.to_csv(signal, file_path, **kwargs)


class SignalLoader:
    """High-level signal loading with automatic format detection."""

    SUPPORTED_EXTENSIONS = {
        ".csv": "csv",
        ".txt": "csv",
        ".tsv": "csv",
        ".json": "json",
        ".npz": "npz",
        ".npy": "npy",
        ".mat": "mat",
    }

    @classmethod
    def load(
        cls,
        file_path: str | Path,
        **kwargs,
    ) -> Signal | list[Signal]:
        """Load signal(s) from a file with automatic format detection.

        Args:
            file_path: Path to the file.
            **kwargs: Format-specific arguments.

        Returns:
            Signal or list of Signals.
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext not in cls.SUPPORTED_EXTENSIONS:
            msg = f"Unsupported file format: {ext}"
            raise ValueError(msg)

        fmt = cls.SUPPORTED_EXTENSIONS[ext]

        if fmt == "csv":
            delimiter = kwargs.pop("delimiter", "," if ext != ".tsv" else "\t")
            return SignalImporter.from_csv(file_path, delimiter=delimiter, **kwargs)

        if fmt == "json":
            return SignalImporter.from_json(file_path, **kwargs)

        if fmt == "npz":
            return SignalImporter.from_npz(file_path, **kwargs)

        if fmt == "npy":
            # .npy files contain a single array
            data = np.load(file_path)
            if data.ndim == 1:
                # Assume uniform time sampling
                time = np.arange(len(data))
                return Signal(time=time, values=data, name=file_path.stem)
            if data.ndim == 2:
                # Assume first column is time
                time = data[:, 0]
                values = data[:, 1]
                return Signal(time=time, values=values, name=file_path.stem)
            msg = f"Unsupported array shape: {data.shape}"
            raise ValueError(msg)

        if fmt == "mat":
            return SignalImporter.from_mat(file_path, **kwargs)

        msg = f"Format handler not implemented: {fmt}"
        raise NotImplementedError(msg)


class BatchProcessor:
    """Process multiple signal files in batch."""

    def __init__(self, input_dir: str | Path) -> None:
        """Initialize the batch processor.

        Args:
            input_dir: Directory containing signal files.
        """
        self.input_dir = Path(input_dir)

    def find_files(
        self,
        pattern: str = "*.csv",
    ) -> list[Path]:
        """Find all files matching a pattern.

        Args:
            pattern: Glob pattern for files.

        Returns:
            List of matching file paths.
        """
        return sorted(self.input_dir.glob(pattern))

    def load_all(
        self,
        pattern: str = "*.csv",
        **kwargs,
    ) -> dict[str, Signal | list[Signal]]:
        """Load all signals from matching files.

        Args:
            pattern: Glob pattern for files.
            **kwargs: Arguments for SignalLoader.load.

        Returns:
            Dictionary mapping file names to signals.
        """
        files = self.find_files(pattern)
        signals = {}

        for file_path in files:
            try:
                signals[file_path.stem] = SignalLoader.load(file_path, **kwargs)
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning("Failed to load %s: %s", file_path, e)

        return signals

    def process_all(
        self,
        processor: Callable[[Signal], Signal],
        pattern: str = "*.csv",
        output_dir: str | Path | None = None,
        output_format: str = "csv",
        **kwargs,
    ) -> dict[str, Signal | list[Signal]]:
        """Load, process, and optionally save all signals.

        Args:
            processor: Function to apply to each signal.
            pattern: Glob pattern for input files.
            output_dir: Directory for output files (None = don't save).
            output_format: Output format ('csv', 'json', 'npz').
            **kwargs: Arguments for SignalLoader.load.

        Returns:
            Dictionary mapping file names to processed signals.
        """
        files = self.find_files(pattern)
        results: dict[str, Signal | list[Signal]] = {}

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            try:
                signal = SignalLoader.load(file_path, **kwargs)

                # Handle multiple signals
                processed: Signal | list[Signal]
                if isinstance(signal, list):
                    processed = [processor(s) for s in signal]
                else:
                    processed = processor(signal)

                results[file_path.stem] = processed

                # Save if output_dir specified
                if output_dir:
                    output_path = output_dir / f"{file_path.stem}.{output_format}"
                    if output_format == "csv":
                        SignalExporter.to_csv(processed, output_path)
                    elif output_format == "json":
                        json_signal = (
                            processed[0] if isinstance(processed, list) else processed
                        )
                        SignalExporter.to_json(json_signal, output_path)
                    elif output_format == "npz":
                        SignalExporter.to_npz(processed, output_path)

            except (RuntimeError, ValueError, OSError) as e:
                logger.error("Warning: Failed to process %s: %s", file_path, e)

        return results
