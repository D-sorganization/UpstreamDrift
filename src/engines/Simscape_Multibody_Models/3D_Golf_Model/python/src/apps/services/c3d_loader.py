"""Service for loading C3D files into application data models."""

import os

import numpy as np

from ...c3d_reader import C3DDataReader  # type: ignore
from ...logger_utils import log_execution_time
from ..core.models import AnalogData, C3DDataModel, MarkerData


def _build_markers(df_points, marker_names: list[str]) -> dict[str, MarkerData]:
    """Build marker data dictionary from points dataframe.

    Args:
        df_points: Dataframe with marker point data.
        marker_names: List of expected marker labels.

    Returns:
        Dictionary mapping marker names to MarkerData.
    """
    markers: dict[str, MarkerData] = {}
    if not df_points.empty:
        grouped = df_points.groupby("marker")
        for name, group in grouped:
            pos = group[["x", "y", "z"]].to_numpy()
            res = group["residual"].to_numpy()
            markers[name] = MarkerData(name=name, position=pos, residuals=res)

    for name in marker_names:
        if name not in markers:
            markers[name] = MarkerData(
                name=name, position=np.empty((0, 3)), residuals=np.empty((0,))
            )
    return markers


def _build_analog(df_analog, metadata_obj) -> dict[str, AnalogData]:
    """Build analog channel data dictionary from analog dataframe.

    Args:
        df_analog: Dataframe with analog channel data.
        metadata_obj: C3D metadata with labels and units.

    Returns:
        Dictionary mapping channel names to AnalogData.
    """
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
    return analog


def _build_metadata_ui(filepath: str, metadata_obj) -> dict[str, str]:
    """Build UI-friendly metadata dictionary from C3D metadata.

    Args:
        filepath: Path to the C3D file.
        metadata_obj: C3D metadata object.

    Returns:
        Dictionary of display-friendly metadata key-value pairs.
    """
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
    return metadata_ui


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
        df_points = reader.points_dataframe(include_time=False)

    markers = _build_markers(df_points, metadata_obj.marker_labels)

    df_analog = reader.analog_dataframe(include_time=False)
    analog = _build_analog(df_analog, metadata_obj)

    frame_time = (
        np.arange(metadata_obj.frame_count) / metadata_obj.frame_rate
        if metadata_obj.frame_rate > 0
        else None
    )

    analog_time = None
    if metadata_obj.analog_rate and metadata_obj.analog_rate > 0 and analog:
        first_analog = next(iter(analog.values()))
        analog_time = np.arange(len(first_analog.values)) / metadata_obj.analog_rate

    return C3DDataModel(
        filepath=filepath,
        markers=markers,
        analog=analog,
        point_rate=metadata_obj.frame_rate,
        analog_rate=metadata_obj.analog_rate or 0.0,
        point_time=frame_time,
        analog_time=analog_time,
        metadata=_build_metadata_ui(filepath, metadata_obj),
    )
