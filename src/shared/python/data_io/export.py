"""Generic export formats for golf swing data.

Supports:
- MATLAB .mat files
- C3D motion capture format
- HDF5 hierarchical data
- JSON/CSV
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.core.contracts import precondition
from src.shared.python.engine_core.engine_availability import (
    C3D_AVAILABLE,
    EZC3D_AVAILABLE,
    SCIPY_AVAILABLE,
)
from src.shared.python.engine_core.engine_availability import (
    HDF5_AVAILABLE as H5PY_AVAILABLE,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Conditional imports for optional dependencies
if SCIPY_AVAILABLE:
    from scipy.io import savemat

if H5PY_AVAILABLE:
    import h5py


@precondition(
    lambda output_path, data_dict, compress=True: output_path is not None
    and len(output_path) > 0,
    "Output path must be a non-empty string",
)
@precondition(
    lambda output_path, data_dict, compress=True: data_dict is not None,
    "Data dictionary must not be None",
)
def export_to_matlab(
    output_path: str,
    data_dict: dict[str, Any],
    compress: bool = True,
) -> bool:
    """Export recording to MATLAB .mat format."""
    if not SCIPY_AVAILABLE:
        logger.error("scipy required for MATLAB export (pip install scipy)")
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

    except (OSError, ValueError, TypeError) as e:
        logger.error(f"Failed to export to MATLAB: {e}")
        return False


@precondition(
    lambda output_path, data_dict, compression="gzip": output_path is not None
    and len(output_path) > 0,
    "Output path must be a non-empty string",
)
@precondition(
    lambda output_path, data_dict, compression="gzip": data_dict is not None,
    "Data dictionary must not be None",
)
def export_to_hdf5(
    output_path: str,
    data_dict: dict[str, Any],
    compression: str = "gzip",
) -> bool:
    """Export recording to HDF5 format."""
    if not H5PY_AVAILABLE:
        logger.error("h5py required for HDF5 export (pip install h5py)")
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

    except (OSError, ValueError, TypeError) as e:
        logger.error(f"Failed to export to HDF5: {e}")
        return False


@dataclass
class C3DExportData:
    """Grouped motion capture data for C3D export, reducing PLR0913.

    Attributes:
        times: Time array (N,).
        joint_positions: Joint positions (N, nq).
        joint_names: Names of joints.
        forces: Optional force data (N, nforces, 3).
        moments: Optional moment data (N, nforces, 3).
        frame_rate: Sampling rate in Hz.
        units: Dictionary of units (position, force, moment).
    """

    times: np.ndarray
    joint_positions: np.ndarray
    joint_names: list
    forces: np.ndarray | None = None
    moments: np.ndarray | None = None
    frame_rate: float = 60.0
    units: dict[str, str] = field(default_factory=lambda: {
        "position": "mm", "force": "N", "moment": "Nmm",
    })


@precondition(
    lambda output_path,
    times,
    joint_positions,
    joint_names,
    forces=None,
    moments=None,
    frame_rate=60.0,
    units=None: output_path is not None and len(output_path) > 0,
    "Output path must be a non-empty string",
)
@precondition(
    lambda output_path,
    times,
    joint_positions,
    joint_names,
    forces=None,
    moments=None,
    frame_rate=60.0,
    units=None: frame_rate > 0,
    "Frame rate must be positive",
)
def export_to_c3d(
    output_path: str,
    times: np.ndarray,
    joint_positions: np.ndarray,
    joint_names: list,
    forces: np.ndarray | None = None,
    moments: np.ndarray | None = None,
    frame_rate: float = 60.0,
    units: dict[str, str] | None = None,  # noqa: PLR0913
) -> bool:
    """Export recording to C3D motion capture format.

    Args:
        output_path: Output .c3d file path
        times: Time array (N,)
        joint_positions: Joint positions (N, nq)
        joint_names: Names of joints
        forces: Optional force data (N, nforces, 3)
        moments: Optional moment data (N, nforces, 3)
        frame_rate: Sampling rate in Hz
        units: Dictionary of units (position, force, moment)

    Returns:
        True if successful

    .. tip::
        For new code, prefer constructing a ``C3DExportData`` and calling
        ``export_to_c3d_from_data`` instead of passing individual args.
    """
    if not EZC3D_AVAILABLE and not C3D_AVAILABLE:
        logger.error("ezc3d or c3d required for C3D export (pip install ezc3d)")
        return False

    data = C3DExportData(
        times=times,
        joint_positions=joint_positions,
        joint_names=joint_names,
        forces=forces,
        moments=moments,
        frame_rate=frame_rate,
        units=units or {"position": "mm", "force": "N", "moment": "Nmm"},
    )

    try:
        if EZC3D_AVAILABLE:
            return _export_to_c3d_ezc3d(output_path, data)
        return _export_to_c3d_py(output_path, data)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Failed to export to C3D: {e}")
        return False


def _export_to_c3d_ezc3d(
    output_path: str,
    data: C3DExportData,
) -> bool:
    """Export using ezc3d library."""
    import ezc3d

    c = ezc3d.c3d()
    c["parameters"]["POINT"]["RATE"]["value"] = [data.frame_rate]
    c["parameters"]["POINT"]["UNITS"]["value"] = [data.units["position"]]

    num_frames = len(data.times)
    num_markers = data.joint_positions.shape[1]
    c["parameters"]["POINT"]["LABELS"]["value"] = data.joint_names[:num_markers]

    # Point data: [X, Y, Z, residual] for each marker
    points = np.zeros((4, num_markers, num_frames))
    for i in range(num_markers):
        angles = data.joint_positions[:, i]
        radius = (i + 1) * 100  # mm
        points[0, i, :] = radius * np.cos(angles)
        points[1, i, :] = radius * np.sin(angles)
        points[2, i, :] = np.arange(num_frames) * 10
        points[3, i, :] = 0  # Residual
    c["data"]["points"] = points

    # Analog data (forces/moments)
    if data.forces is not None or data.moments is not None:
        analog_data = []
        analog_labels = []
        if data.forces is not None:
            for fp in range(data.forces.shape[1]):
                for axis, label in enumerate(["X", "Y", "Z"]):
                    analog_data.append(data.forces[:, fp, axis])
                    analog_labels.append(f"Force{fp + 1}_{label}")
        if data.moments is not None:
            for mp in range(data.moments.shape[1]):
                for axis, label in enumerate(["X", "Y", "Z"]):
                    analog_data.append(data.moments[:, mp, axis])
                    analog_labels.append(f"Moment{mp + 1}_{label}")
        if analog_data:
            c["data"]["analogs"] = np.array(analog_data)
            c["parameters"]["ANALOG"]["LABELS"]["value"] = analog_labels
            c["parameters"]["ANALOG"]["RATE"]["value"] = [data.frame_rate]
            c["parameters"]["FORCE_PLATFORM"]["UNITS"]["value"] = [data.units["force"]]

    c.write(output_path)
    return True


def _export_to_c3d_py(
    output_path: str,
    data: C3DExportData,
) -> bool:
    """Export using c3d library (fallback)."""
    import c3d

    writer = c3d.Writer(point_rate=data.frame_rate)
    num_frames = len(data.times)
    num_markers = data.joint_positions.shape[1]

    for frame_idx in range(num_frames):
        frame_points = []
        for marker_idx in range(num_markers):
            angle = data.joint_positions[frame_idx, marker_idx]
            radius = (marker_idx + 1) * 100
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = frame_idx * 10
            frame_points.append([x, y, z, 0.0, 0.0])
        writer.add_frames([(np.array(frame_points), np.array([]))])

    with open(output_path, "wb") as f:
        writer.write(f)
    return True


def _export_json(output_path: Path, data_dict: dict[str, Any]) -> bool:
    """Export data dictionary to JSON format.

    Converts numpy arrays to lists for JSON serialization.
    """
    import json

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
    return True


def _flatten_dict_for_csv(data_dict: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested data dictionary into a flat dictionary for CSV export.

    Handles time series arrays and nested dictionaries of arrays.
    """
    flat_data: dict[str, Any] = {}
    if "times" in data_dict:
        flat_data["time"] = data_dict["times"]

    n_times = len(data_dict.get("times", []))

    for k, v in data_dict.items():
        if k == "times":
            continue

        # Handle direct arrays matching time length
        if isinstance(v, np.ndarray) and len(v) == n_times:
            if v.ndim == 1:
                flat_data[k] = v
            elif v.ndim == 2:
                for i in range(v.shape[1]):
                    flat_data[f"{k}_{i}"] = v[:, i]

        # Handle nested dictionaries (e.g. induced_accelerations)
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, np.ndarray) and len(sub_v) == n_times:
                    full_key = f"{k}_source_{sub_k}" if isinstance(sub_k, int) else f"{k}_{sub_k}"
                    if sub_v.ndim == 1:
                        flat_data[full_key] = sub_v
                    elif sub_v.ndim == 2:
                        for i in range(sub_v.shape[1]):
                            flat_data[f"{full_key}_{i}"] = sub_v[:, i]

    return flat_data


def _export_csv(output_path: Path, data_dict: dict[str, Any]) -> bool:
    """Export data dictionary to CSV format."""
    import pandas as pd

    flat_data = _flatten_dict_for_csv(data_dict)
    df = pd.DataFrame(flat_data)
    df.to_csv(output_path, index=False)
    return True


def export_recording_all_formats(
    base_path: str,
    data_dict: dict[str, Any],
    formats: list | None = None,
) -> dict[str, bool]:
    """Export recording in multiple formats.

    Args:
        base_path: Base file path (without extension).
        data_dict: Data dictionary to export.
        formats: List of format strings. Defaults to ["json", "csv", "mat", "hdf5"].

    Returns:
        Dictionary mapping format names to success booleans.

    Raises:
        ValueError: If base_path is empty or data_dict is empty.
    """
    if not base_path:
        raise ValueError("base_path must be a non-empty string")
    if not isinstance(data_dict, dict):
        raise TypeError(f"data_dict must be a dict, got {type(data_dict).__name__}")

    if formats is None:
        formats = ["json", "csv", "mat", "hdf5"]
    base_path_obj = Path(base_path)
    results = {}

    for fmt in formats:
        try:
            output_path = base_path_obj.with_suffix(f".{fmt}")

            if fmt == "json":
                success = _export_json(output_path, data_dict)
            elif fmt == "csv":
                success = _export_csv(output_path, data_dict)
            elif fmt == "mat":
                success = export_to_matlab(str(output_path), data_dict)
            elif fmt in ["hdf5", "h5"]:
                output_path = base_path_obj.with_suffix(".h5")
                success = export_to_hdf5(str(output_path), data_dict)
            else:
                success = False

            results[fmt] = success

        except ImportError as e:
            logger.error(f"Export format {fmt} failed: {e}")
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
        "c3d": {
            "name": "C3D",
            "extension": ".c3d",
            "available": EZC3D_AVAILABLE or C3D_AVAILABLE,
            "description": "Motion Capture Standard - compatible with Vicon, etc.",
        },
    }
