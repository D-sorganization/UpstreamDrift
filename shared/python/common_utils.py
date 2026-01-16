"""Common utilities shared across all golf modeling engines."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

# Explicitly import OUTPUT_ROOT from the package to avoid circular dependency issues
# while keeping the import at module level for clarity.
# Note: shared/python/__init__.py defines OUTPUT_ROOT but does NOT import this module.
from shared.python import OUTPUT_ROOT

# Import core utilities (exceptions, logging) from the lightweight module
from shared.python.constants import (
    DEG_TO_RAD,
    KG_TO_LB,
    M_TO_FT,
    M_TO_YARD,
    MPS_TO_KPH,
    MPS_TO_MPH,
    RAD_TO_DEG,
)
from shared.python.core import (
    DataFormatError,
    EngineNotFoundError,
    GolfModelingError,
    get_logger,
    setup_logging,
    setup_structured_logging,
)

# Re-export them for backward compatibility
__all__ = [
    "DataFormatError",
    "EngineNotFoundError",
    "GolfModelingError",
    "setup_logging",
    "setup_structured_logging",
    "get_logger",
    "ensure_output_dir",
    "load_golf_data",
    "save_golf_data",
    "standardize_joint_angles",
    "plot_joint_trajectories",
    "convert_units",
    "get_shared_urdf_path",
    "normalize_z_score",
]

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


# Centralized conversion factors for maintainability (DRY, Orthogonality)
# Format: (from_unit, to_unit): factor
# Usage: value * factor
CONVERSION_FACTORS: dict[tuple[str, str], float] = {
    # Angle
    ("deg", "rad"): float(DEG_TO_RAD),
    ("rad", "deg"): float(RAD_TO_DEG),
    # Length
    ("m", "mm"): 1000.0,
    ("mm", "m"): 0.001,
    ("m", "ft"): float(M_TO_FT),
    ("ft", "m"): 1.0 / float(M_TO_FT),
    ("m", "yd"): float(M_TO_YARD),
    ("yd", "m"): 1.0 / float(M_TO_YARD),
    # Velocity
    ("m/s", "mph"): float(MPS_TO_MPH),
    ("mph", "m/s"): 1.0 / float(MPS_TO_MPH),
    ("m/s", "km/h"): float(MPS_TO_KPH),
    ("km/h", "m/s"): 1.0 / float(MPS_TO_KPH),
    # Mass
    ("kg", "lb"): float(KG_TO_LB),
    ("lb", "kg"): 1.0 / float(KG_TO_LB),
}


def ensure_output_dir(engine_name: str, subdir: str | None = None) -> Path:
    """Ensure output directory exists for an engine.

    Args:
        engine_name: Name of the physics engine
        subdir: Optional subdirectory name

    Returns:
        Path to the output directory
    """
    output_path = OUTPUT_ROOT / engine_name
    if subdir:
        output_path = output_path / subdir

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def load_golf_data(data_path: str | Path) -> pd.DataFrame:
    """Load golf swing data from various formats.

    Args:
        data_path: Path to data file

    Returns:
        DataFrame with golf swing data

    Raises:
        ValueError: If file format not supported
    """
    data_path = Path(data_path)
    import pandas as pd

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(data_path)
    elif suffix == ".json":
        return pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def save_golf_data(
    data: pd.DataFrame, output_path: str | Path, format: str = "csv"
) -> None:
    """Save golf swing data in specified format.

    Args:
        data: DataFrame to save
        output_path: Output file path
        format: Output format ('csv', 'excel', 'json')
    """
    output_path = Path(output_path)
    format = format.lower()

    if format == "csv":
        data.to_csv(output_path, index=False)
    elif format == "excel":
        data.to_excel(output_path, index=False)
    elif format == "json":
        data.to_json(output_path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def normalize_z_score(data: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """Normalize data using Z-score standardization.

    Args:
        data: Input array
        epsilon: Small constant to avoid division by zero

    Returns:
        Normalized array
    """
    result = (data - np.mean(data)) / (np.std(data) + epsilon)
    return np.asarray(result)


def standardize_joint_angles(
    angles: np.ndarray,
    angle_names: list[str] | None = None,
    time_step: float = 0.01,
) -> pd.DataFrame:
    """Standardize joint angle data across engines.

    Args:
        angles: Joint angle array (time x joints)
        angle_names: Optional joint names
        time_step: Time step in seconds (default: 0.01s = 100Hz)

    Returns:
        Standardized DataFrame with joint angles
    """
    if angle_names is None:
        angle_names = [f"joint_{i}" for i in range(angles.shape[1])]

    df = pd.DataFrame(angles, columns=angle_names)
    # Use endpoint=False to generate N time steps from 0 to T-dt
    # For N=10, dt=0.01, we want 0.00, 0.01, ..., 0.09
    df["time"] = np.linspace(0, len(df) * time_step, len(df), endpoint=False)

    return df


def plot_joint_trajectories(
    data: pd.DataFrame,
    title: str = "Joint Trajectories",
    save_path: Path | None = None,
) -> plt.Figure:
    """Create standardized joint trajectory plots.

    Args:
        data: DataFrame with time and joint columns
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

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

    Raises:
        ValueError: If conversion is not supported
    """
    if from_unit == to_unit:
        return value

    try:
        factor = CONVERSION_FACTORS[(from_unit, to_unit)]
        return value * factor
    except KeyError:
        raise ValueError(
            f"Conversion from {from_unit} to {to_unit} not supported"
        ) from None


def get_shared_urdf_path() -> Path | None:
    """Get the path to the shared URDF directory.

    Returns:
        Path to shared/urdf directory if found, None otherwise.
    """
    # Attempt to locate shared/urdf relative to this file
    # shared/python/common_utils.py -> shared/python -> shared -> root -> shared/urdf
    # This logic assumes standard repo structure.
    try:
        current_file = Path(__file__).resolve()
        # Go up to 'shared' dir: common_utils.py (parent) -> python (parent) -> shared (parent)
        # Note: current_file.parents[0] is 'shared/python'
        #       current_file.parents[1] is 'shared'
        shared_dir = current_file.parents[1]

        # Check if we are actually in the shared directory structure
        if shared_dir.name != "shared":
            # Fallback: traverse up until we find 'shared' or root
            for parent in current_file.parents:
                if (parent / "shared" / "urdf").exists():
                    return parent / "shared" / "urdf"
            return None

        urdf_dir = shared_dir / "urdf"
        if urdf_dir.exists():
            return urdf_dir

    except Exception:
        pass

    return None
