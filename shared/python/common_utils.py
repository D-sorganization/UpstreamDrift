"""Common utilities shared across all golf modeling engines."""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up consistent logging across all engines.

    Args:
        name: Logger name (typically __name__)
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def ensure_output_dir(engine_name: str, subdir: Optional[str] = None) -> Path:
    """Ensure output directory exists for an engine.

    Args:
        engine_name: Name of the physics engine
        subdir: Optional subdirectory name

    Returns:
        Path to the output directory
    """
    from . import OUTPUT_ROOT

    output_path = OUTPUT_ROOT / engine_name
    if subdir:
        output_path = output_path / subdir

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def load_golf_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load golf swing data from various formats.

    Args:
        data_path: Path to data file

    Returns:
        DataFrame with golf swing data

    Raises:
        ValueError: If file format not supported
    """
    data_path = Path(data_path)

    if data_path.suffix.lower() == ".csv":
        return pd.read_csv(data_path)
    elif data_path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(data_path)
    elif data_path.suffix.lower() == ".json":
        return pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")


def save_golf_data(
    data: pd.DataFrame, output_path: Union[str, Path], format: str = "csv"
) -> None:
    """Save golf swing data in specified format.

    Args:
        data: DataFrame to save
        output_path: Output file path
        format: Output format ('csv', 'excel', 'json')
    """
    output_path = Path(output_path)

    if format.lower() == "csv":
        data.to_csv(output_path, index=False)
    elif format.lower() == "excel":
        data.to_excel(output_path, index=False)
    elif format.lower() == "json":
        data.to_json(output_path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def standardize_joint_angles(
    angles: np.ndarray, angle_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Standardize joint angle data across engines.

    Args:
        angles: Joint angle array (time x joints)
        angle_names: Optional joint names

    Returns:
        Standardized DataFrame with joint angles
    """
    if angle_names is None:
        angle_names = [f"joint_{i}" for i in range(angles.shape[1])]

    df = pd.DataFrame(angles, columns=angle_names)
    df["time"] = np.linspace(0, len(df) * 0.01, len(df))  # Assume 100Hz

    return df


def plot_joint_trajectories(
    data: pd.DataFrame,
    title: str = "Joint Trajectories",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create standardized joint trajectory plots.

    Args:
        data: DataFrame with time and joint columns
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    joint_cols = [col for col in data.columns if col != "time"]

    for i, joint in enumerate(joint_cols[:4]):  # Plot first 4 joints
        if i < len(axes):
            axes[i].plot(data["time"], data[joint])
            axes[i].set_title(f"{joint.replace('_', ' ').title()}")
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel("Angle (rad)")
            axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(joint_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between common golf modeling units.

    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted value
    """
    # Angle conversions
    if from_unit == "deg" and to_unit == "rad":
        return np.deg2rad(value)
    elif from_unit == "rad" and to_unit == "deg":
        return np.rad2deg(value)

    # Length conversions
    elif from_unit == "m" and to_unit == "mm":
        return value * 1000
    elif from_unit == "mm" and to_unit == "m":
        return value / 1000

    # Velocity conversions
    elif from_unit == "m/s" and to_unit == "mph":
        return value * 2.237
    elif from_unit == "mph" and to_unit == "m/s":
        return value / 2.237

    else:
        raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported")


class GolfModelingError(Exception):
    """Base exception for golf modeling suite."""

    pass


class EngineNotFoundError(GolfModelingError):
    """Raised when a physics engine is not found or not properly installed."""

    pass


class DataFormatError(GolfModelingError):
    """Raised when data format is invalid or unsupported."""

    pass
