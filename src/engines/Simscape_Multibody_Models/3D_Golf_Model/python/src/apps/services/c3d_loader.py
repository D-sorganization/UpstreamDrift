"""Service for loading C3D files into application data models."""

import os
import sys
from pathlib import Path

import numpy as np

from ..core.models import AnalogData, C3DDataModel, MarkerData

# Ensure we can import the shared reader
try:
    # Try relative import from src/c3d_reader
    # from src.apps.services -> src is ../..
    from ...c3d_reader import C3DDataReader  # type: ignore
    from ...logger_utils import log_execution_time
except (ImportError, ValueError):
    # Fallback for direct execution
    # current: src/apps/services
    # target: src/
    src_path = Path(__file__).resolve().parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    try:
        from c3d_reader import C3DDataReader  # type: ignore[no-redef]
        from logger_utils import log_execution_time
    except ImportError as e:
        raise ImportError("Could not find c3d_reader or logger_utils module.") from e


def load_c3d_file(filepath: str) -> C3DDataModel:
    """Load and parse a C3D file using the C3DDataReader.

    Args:
        filepath: Absolute path to the .c3d file

    Returns:
        Populated C3DDataModel

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If parsing fails
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with log_execution_time(f"load_c3d_{os.path.basename(filepath)}"):
        reader = C3DDataReader(filepath)
        metadata_obj = reader.get_metadata()

        # Load Points Data
        df_points = reader.points_dataframe(include_time=False)

    # Build markers dict (PERF-001: Optimized from O(n²) to O(n) using groupby)
    markers: dict[str, MarkerData] = {}
    marker_names = metadata_obj.marker_labels

    # Group by marker name once - O(n) instead of O(n²)
    if not df_points.empty:
        grouped = df_points.groupby("marker")
        for name, group in grouped:
            pos = group[["x", "y", "z"]].to_numpy()
            res = group["residual"].to_numpy()
            markers[name] = MarkerData(name=name, position=pos, residuals=res)

    # Add empty markers for labels that had no data
    for name in marker_names:
        if name not in markers:
            markers[name] = MarkerData(
                name=name, position=np.empty((0, 3)), residuals=np.empty((0,))
            )

    # Load Analog Data
    df_analog = reader.analog_dataframe(include_time=False)
    analog: dict[str, AnalogData] = {}

    units_map = dict(
        zip(metadata_obj.analog_labels, metadata_obj.analog_units, strict=False)
    )

    if not df_analog.empty and "channel" in df_analog.columns:
        for name in df_analog["channel"].unique():
            mask = df_analog["channel"] == name
            vals = df_analog.loc[mask, "value"].to_numpy()
            unit = units_map.get(name, "")
            analog[name] = AnalogData(name=name, values=vals, unit=unit)

    # Time vectors
    frame_time = (
        np.arange(metadata_obj.frame_count) / metadata_obj.frame_rate
        if metadata_obj.frame_rate > 0
        else None
    )

    analog_time = None
    if metadata_obj.analog_rate and metadata_obj.analog_rate > 0:
        if analog:
            first_analog = next(iter(analog.values()))
            n_samples = len(first_analog.values)
            analog_time = np.arange(n_samples) / metadata_obj.analog_rate

    # Metadata dict for UI
    metadata_ui = {
        "File": os.path.basename(filepath),
        "Path": filepath,
        "Point rate (Hz)": f"{metadata_obj.frame_rate:.3f}",
        "Analog rate (Hz)": (
            f"{metadata_obj.analog_rate:.3f}" if metadata_obj.analog_rate else "N/A"
        ),
        "Frames": str(metadata_obj.frame_count),
        "Points": str(metadata_obj.marker_count),
        "Units (POINT)": metadata_obj.units,
    }

    if metadata_obj.events:
        events_str = ", ".join(
            [f"{e.label} ({e.time:.2f}s)" for e in metadata_obj.events]
        )
        metadata_ui["Events"] = events_str

    return C3DDataModel(
        filepath=filepath,
        markers=markers,
        analog=analog,
        point_rate=metadata_obj.frame_rate,
        analog_rate=metadata_obj.analog_rate or 0.0,
        point_time=frame_time,
        analog_time=analog_time,
        metadata=metadata_ui,
    )
