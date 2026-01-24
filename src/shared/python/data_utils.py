"""Data loading and processing utilities.

This module provides reusable data loading patterns to eliminate duplication.

Usage:
    from src.shared.python.data_utils import (
        load_csv_data,
        load_json_data,
        save_csv_data,
        DataLoader,
    )

    # Load data with error handling
    data = load_csv_data("data.csv")

    # Use data loader with caching
    loader = DataLoader("data.csv")
    data = loader.load()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.shared.python.error_decorators import log_errors
from src.shared.python.logging_config import get_logger
from src.shared.python.validation_utils import validate_file_exists

logger = get_logger(__name__)


@log_errors("Failed to load CSV data", reraise=False, default_return=None)
def load_csv_data(
    path: str | Path,
    **kwargs: Any,
) -> pd.DataFrame | None:
    """Load CSV data with error handling.

    Args:
        path: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv()

    Returns:
        DataFrame or None if failed

    Example:
        data = load_csv_data("trajectory.csv")
        data = load_csv_data("data.csv", sep=";", header=0)
    """
    path_obj = validate_file_exists(path, "CSV file")
    logger.debug(f"Loading CSV data from {path}")

    data = pd.read_csv(path_obj, **kwargs)
    logger.info(f"Loaded {len(data)} rows from {path}")

    return data


@log_errors("Failed to save CSV data", reraise=False)
def save_csv_data(
    data: pd.DataFrame,
    path: str | Path,
    **kwargs: Any,
) -> bool:
    """Save CSV data with error handling.

    Args:
        data: DataFrame to save
        path: Path to CSV file
        **kwargs: Additional arguments for DataFrame.to_csv()

    Returns:
        True if successful, False otherwise

    Example:
        save_csv_data(results, "output.csv", index=False)
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Saving CSV data to {path}")
    data.to_csv(path_obj, **kwargs)
    logger.info(f"Saved {len(data)} rows to {path}")

    return True


@log_errors("Failed to load JSON data", reraise=False, default_return=None)
def load_json_data(
    path: str | Path,
) -> dict[str, Any] | list | None:
    """Load JSON data with error handling.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data or None if failed

    Example:
        config = load_json_data("config.json")
    """
    path_obj = validate_file_exists(path, "JSON file")
    logger.debug(f"Loading JSON data from {path}")

    with path_obj.open("r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded JSON data from {path}")
    return data


@log_errors("Failed to save JSON data", reraise=False)
def save_json_data(
    data: dict[str, Any] | list,
    path: str | Path,
    indent: int = 2,
) -> bool:
    """Save JSON data with error handling.

    Args:
        data: Data to save
        path: Path to JSON file
        indent: JSON indentation level

    Returns:
        True if successful, False otherwise

    Example:
        save_json_data(results, "output.json")
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Saving JSON data to {path}")

    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)

    logger.info(f"Saved JSON data to {path}")
    return True


@log_errors("Failed to load numpy data", reraise=False, default_return=None)
def load_numpy_data(
    path: str | Path,
) -> np.ndarray | None:
    """Load numpy array with error handling.

    Args:
        path: Path to .npy or .npz file

    Returns:
        Numpy array or None if failed

    Example:
        data = load_numpy_data("trajectory.npy")
    """
    path_obj = validate_file_exists(path, "numpy file")
    logger.debug(f"Loading numpy data from {path}")

    if path_obj.suffix == ".npz":
        data = np.load(path_obj)
        logger.info(f"Loaded npz archive from {path}")
    else:
        data = np.load(path_obj)
        logger.info(f"Loaded numpy array from {path} with shape {data.shape}")

    return data


@log_errors("Failed to save numpy data", reraise=False)
def save_numpy_data(
    data: np.ndarray,
    path: str | Path,
    compressed: bool = False,
) -> bool:
    """Save numpy array with error handling.

    Args:
        data: Array to save
        path: Path to .npy file
        compressed: Whether to use compression

    Returns:
        True if successful, False otherwise

    Example:
        save_numpy_data(trajectory, "output.npy")
        save_numpy_data(large_data, "output.npz", compressed=True)
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Saving numpy data to {path}")

    if compressed:
        np.savez_compressed(path_obj, data=data)
    else:
        np.save(path_obj, data)

    logger.info(f"Saved numpy array to {path} with shape {data.shape}")
    return True


class DataLoader:
    """Data loader with caching and format detection.

    Example:
        loader = DataLoader("data.csv")
        data = loader.load()

        # Reload without cache
        data = loader.load(use_cache=False)
    """

    def __init__(self, path: str | Path):
        """Initialize data loader.

        Args:
            path: Path to data file
        """
        self.path = Path(path)
        self._cache: Any = None
        self._format = self._detect_format()

    def _detect_format(self) -> str:
        """Detect data format from file extension."""
        suffix = self.path.suffix.lower()

        format_map = {
            ".csv": "csv",
            ".json": "json",
            ".npy": "numpy",
            ".npz": "numpy",
            ".txt": "text",
            ".xlsx": "excel",
            ".xls": "excel",
        }

        return format_map.get(suffix, "unknown")

    def load(self, use_cache: bool = True, **kwargs: Any) -> Any:
        """Load data with caching.

        Args:
            use_cache: Whether to use cached data
            **kwargs: Format-specific loading arguments

        Returns:
            Loaded data

        Raises:
            ValueError: If format is unknown
        """
        if use_cache and self._cache is not None:
            logger.debug(f"Using cached data for {self.path}")
            return self._cache

        logger.info(f"Loading data from {self.path} (format: {self._format})")

        if self._format == "csv":
            data = load_csv_data(self.path, **kwargs)
        elif self._format == "json":
            data = load_json_data(self.path)
        elif self._format == "numpy":
            data = load_numpy_data(self.path)
        elif self._format == "text":
            data = self.path.read_text(encoding="utf-8")
        elif self._format == "excel":
            data = pd.read_excel(self.path, **kwargs)
        else:
            raise ValueError(f"Unknown format: {self._format}")

        self._cache = data
        return data

    def clear_cache(self) -> None:
        """Clear cached data."""
        self._cache = None

    @property
    def format(self) -> str:
        """Get detected format."""
        return self._format


def load_c3d_data(path: str | Path) -> dict[str, Any]:
    """Load C3D motion capture data.

    Args:
        path: Path to C3D file

    Returns:
        Dictionary with marker data and metadata

    Example:
        data = load_c3d_data("motion.c3d")
        markers = data["markers"]
    """
    try:
        import c3d
    except ImportError:
        logger.error("c3d library not installed. Install with: pip install c3d")
        raise

    path_obj = validate_file_exists(path, "C3D file")
    logger.info(f"Loading C3D data from {path}")

    with path_obj.open("rb") as f:
        reader = c3d.Reader(f)

        # Extract marker data
        markers = {}
        for i, points, _analog in reader.read_frames():
            # points shape: (n_markers, 4) where columns are [x, y, z, confidence]
            if i == 0:
                # Initialize marker arrays
                n_frames = reader.header.last_frame - reader.header.first_frame + 1
                n_markers = points.shape[0]
                for j in range(n_markers):
                    markers[f"marker_{j}"] = np.zeros((n_frames, 3))

            # Store marker positions
            for j in range(points.shape[0]):
                markers[f"marker_{j}"][i] = points[j, :3]

        metadata = {
            "frame_rate": reader.header.frame_rate,
            "n_frames": reader.header.last_frame - reader.header.first_frame + 1,
            "n_markers": reader.header.point_count,
        }

        return {
            "markers": markers,
            "metadata": metadata,
        }


def convert_to_dataframe(
    data: dict[str, np.ndarray],
    time: np.ndarray | None = None,
) -> pd.DataFrame:
    """Convert dictionary of arrays to DataFrame.

    Args:
        data: Dictionary mapping column names to arrays
        time: Optional time array for index

    Returns:
        DataFrame

    Example:
        data = {"x": x_array, "y": y_array, "z": z_array}
        df = convert_to_dataframe(data, time=time_array)
    """
    df = pd.DataFrame(data)

    if time is not None:
        df.index = time
        df.index.name = "time"

    return df


def resample_data(
    data: pd.DataFrame,
    target_rate: float,
    method: str = "linear",
) -> pd.DataFrame:
    """Resample time series data to target rate.

    Args:
        data: DataFrame with time index
        target_rate: Target sampling rate [Hz]
        method: Interpolation method

    Returns:
        Resampled DataFrame

    Example:
        resampled = resample_data(data, target_rate=100.0)
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        # Assume index is time in seconds
        time = data.index.values
        dt = 1.0 / target_rate
        new_time = np.arange(time[0], time[-1], dt)

        # Interpolate
        resampled_data = {}
        for col in data.columns:
            resampled_data[col] = np.interp(new_time, time, data[col].values)

        return pd.DataFrame(resampled_data, index=new_time)
    else:
        # Use pandas resample for datetime index
        return data.resample(f"{1000 / target_rate:.0f}ms").interpolate(method=method)
