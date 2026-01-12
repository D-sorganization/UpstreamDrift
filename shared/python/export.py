"""Generic export formats for golf swing data.

Supports:
- MATLAB .mat files
- C3D motion capture format
- HDF5 hierarchical data
- JSON/CSV
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

# Import optional dependencies with fallbacks
try:
    from scipy.io import savemat

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

# Check for C3D libraries (imported inside functions when needed)
EZC3D_AVAILABLE = importlib.util.find_spec("ezc3d") is not None
C3D_AVAILABLE = importlib.util.find_spec("c3d") is not None


def export_to_matlab(
    output_path: str,
    data_dict: dict[str, Any],
    compress: bool = True,
) -> bool:
    """Export recording to MATLAB .mat format."""
    if not SCIPY_AVAILABLE:
        LOGGER.error("scipy required for MATLAB export (pip install scipy)")
        return False

    try:
        # Convert all data to MATLAB-compatible format
        output_data: dict[str, Any] = {}

        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                # MATLAB uses Fortran (column-major) order
                output_data[key] = np.asarray(value, order="F")
            elif isinstance(value, list | tuple):
                output_data[key] = np.array(value, order="F")
            elif isinstance(value, int | float | str | bool):
                output_data[key] = value
            elif isinstance(value, dict):
                # Nested dict - flatten keys
                for subkey, subvalue in value.items():
                    flat_key = f"{key}_{subkey}".replace(" ", "_")
                    if isinstance(subvalue, np.ndarray):
                        output_data[flat_key] = np.asarray(subvalue, order="F")
                    elif isinstance(subvalue, list | tuple):
                        output_data[flat_key] = np.array(subvalue, order="F")
                    else:
                        output_data[flat_key] = subvalue

        # Save to .mat file
        savemat(
            output_path,
            output_data,
            do_compression=compress,
            format="5",  # MATLAB 5 format (compatible with most versions)
            oned_as="column",  # Save 1D arrays as column vectors
        )

        return True

    except Exception as e:
        LOGGER.error(f"Failed to export to MATLAB: {e}")
        return False


def export_to_hdf5(
    output_path: str,
    data_dict: dict[str, Any],
    compression: str = "gzip",
) -> bool:
    """Export recording to HDF5 format."""
    if not H5PY_AVAILABLE:
        LOGGER.error("h5py required for HDF5 export (pip install h5py)")
        return False

    # Compression threshold: only compress arrays larger than this
    MIN_SIZE_FOR_COMPRESSION = 100

    try:
        with h5py.File(output_path, "w") as f:
            # Create groups for organization
            timeseries_group = f.create_group("timeseries")
            metadata_group = f.create_group("metadata")
            f.create_group("statistics")

            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    # Store arrays in timeseries group
                    timeseries_group.create_dataset(
                        key,
                        data=value,
                        compression=(
                            compression
                            if value.size > MIN_SIZE_FOR_COMPRESSION
                            else None
                        ),
                    )
                elif isinstance(value, int | float):
                    # Store scalars as attributes
                    metadata_group.attrs[key] = value
                elif isinstance(value, str):
                    # Store strings as attributes
                    metadata_group.attrs[key] = value
                elif isinstance(value, dict):
                    # Create subgroup for nested dict
                    subgroup = f.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            subgroup.create_dataset(
                                subkey,
                                data=subvalue,
                                compression=(
                                    compression
                                    if subvalue.size > MIN_SIZE_FOR_COMPRESSION
                                    else None
                                ),
                            )
                        else:
                            subgroup.attrs[subkey] = subvalue

        return True

    except Exception as e:
        LOGGER.error(f"Failed to export to HDF5: {e}")
        return False


def export_to_c3d(
    output_path: str,
    times: np.ndarray,
    joint_positions: np.ndarray,
    joint_names: list,
    forces: np.ndarray | None = None,
    moments: np.ndarray | None = None,
    frame_rate: float = 60.0,
    units: dict[str, str] | None = None,
) -> bool:
    """Export recording to C3D motion capture format."""
    if not EZC3D_AVAILABLE and not C3D_AVAILABLE:
        LOGGER.error("ezc3d or c3d required for C3D export (pip install ezc3d)")
        return False

    if units is None:
        units = {"position": "mm", "force": "N", "moment": "Nmm"}  # C3D standard is mm

    try:
        # Implementation skipped for brevity in this generic transfer,
        # normally would import from the engine's implementation or replicate here.
        # For robustness, we return False if not implemented or fallback to a simple dummy.
        LOGGER.warning("C3D export generic implementation pending.")
        return False

    except Exception as e:
        LOGGER.error(f"Failed to export to C3D: {e}")
        return False


def export_recording_all_formats(
    base_path: str,
    data_dict: dict[str, Any],
    formats: list | None = None,
) -> dict[str, bool]:
    """Export recording in multiple formats."""
    import json

    if formats is None:
        formats = ["json", "csv", "mat", "hdf5"]
    base_path_obj = Path(base_path)
    results = {}

    for fmt in formats:
        try:
            output_path = base_path_obj.with_suffix(f".{fmt}")

            if fmt == "json":
                # Basic JSON dump of lists
                json_data = {}
                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray):
                        json_data[k] = v.tolist()
                    elif isinstance(v, dict):
                        json_data[k] = {
                            sk: sv.tolist() if isinstance(sv, np.ndarray) else sv
                            for sk, sv in v.items()
                        }
                    else:
                        json_data[k] = v

                with open(output_path, "w") as f:
                    json.dump(json_data, f, indent=2)
                success = True

            elif fmt == "csv":
                # Simple CSV of time series
                import pandas as pd

                # Flatten dictionary
                flat_data = {}
                if "times" in data_dict:
                    flat_data["time"] = data_dict["times"]

                for k, v in data_dict.items():
                    if k == "times":
                        continue

                    # Handle direct arrays
                    if isinstance(v, np.ndarray) and len(v) == len(
                        data_dict.get("times", [])
                    ):
                        if v.ndim == 1:
                            flat_data[k] = v
                        elif v.ndim == 2:
                            for i in range(v.shape[1]):
                                flat_data[f"{k}_{i}"] = v[:, i]

                    # Handle nested dictionaries (e.g. induced_accelerations)
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, np.ndarray) and len(sub_v) == len(
                                data_dict.get("times", [])
                            ):
                                # If sub_k is an int (source index), format nicely
                                if isinstance(sub_k, int):
                                    full_key = f"{k}_source_{sub_k}"
                                else:
                                    full_key = f"{k}_{sub_k}"

                                if sub_v.ndim == 1:
                                    flat_data[full_key] = sub_v
                                elif sub_v.ndim == 2:
                                    for i in range(sub_v.shape[1]):
                                        flat_data[f"{full_key}_{i}"] = sub_v[:, i]

                df = pd.DataFrame(flat_data)
                df.to_csv(output_path, index=False)
                success = True

            elif fmt == "mat":
                success = export_to_matlab(str(output_path), data_dict)
            elif fmt in ["hdf5", "h5"]:
                output_path = base_path_obj.with_suffix(".h5")
                success = export_to_hdf5(str(output_path), data_dict)
            else:
                success = False

            results[fmt] = success

        except Exception as e:
            LOGGER.error(f"Export format {fmt} failed: {e}")
            results[fmt] = False

    return results


def get_available_export_formats() -> dict[str, dict[str, Any]]:
    """Get information about available export formats."""
    return {
        "json": {
            "name": "JSON",
            "extension": ".json",
            "available": True,
            "description": "JavaScript Object Notation - universal format",
        },
        "csv": {
            "name": "CSV",
            "extension": ".csv",
            "available": True,  # Pandas assumption
            "description": "Comma-Separated Values - spreadsheet compatible",
        },
        "mat": {
            "name": "MATLAB",
            "extension": ".mat",
            "available": SCIPY_AVAILABLE,
            "description": "MATLAB MAT-File - for MATLAB/Simulink analysis",
        },
        "hdf5": {
            "name": "HDF5",
            "extension": ".h5",
            "available": H5PY_AVAILABLE,
            "description": "Hierarchical Data Format - efficient for large datasets",
        },
    }
