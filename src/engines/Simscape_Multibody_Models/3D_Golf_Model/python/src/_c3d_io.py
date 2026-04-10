"""C3D file loading and parameter extraction helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

try:
    import ezc3d
except ImportError:
    ezc3d = None  # type: ignore[assignment, unused-ignore]

import numpy as np
from _c3d_models import C3DEvent, C3DMapping, C3DMetadata


def load_c3d_file(file_path: Path) -> C3DMapping:
    if ezc3d is None:
        raise ImportError(
            "ezc3d is required for C3D file reading. "
            "Install it with: pip install ezc3d\n"
            "Note: ezc3d requires Python >=3.10. "
            "For Python 3.9, this functionality is not available."
        )
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return ezc3d.c3d(str(file_path))


def get_point_parameters(c3d_data: C3DMapping, file_path: Path) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], c3d_data["parameters"]["POINT"])
    except KeyError as error:
        raise ValueError(
            f"POINT parameters missing from C3D file: {file_path}"
        ) from error


def get_analog_parameters(c3d_data: C3DMapping) -> dict[str, Any] | None:
    analog_params = c3d_data["parameters"].get("ANALOG")
    return cast(dict[str, Any], analog_params) if analog_params is not None else None


def get_analog_details(
    c3d_data: C3DMapping,
) -> tuple[list[str], float | None, list[str]]:
    analog_parameters = get_analog_parameters(c3d_data)
    analog_array = c3d_data["data"]["analogs"]
    channel_count = analog_array.shape[1]

    if analog_parameters is None:
        labels: list[str] = []
        units: list[str] = []
        analog_rate = None
    else:
        labels = [
            label.strip()
            for label in analog_parameters.get("LABELS", {}).get("value", [])
        ]
        units = [
            unit.strip() for unit in analog_parameters.get("UNITS", {}).get("value", [])
        ]
        analog_rate = float(analog_parameters.get("RATE", {}).get("value", [0])[0])

    if not labels and channel_count > 0:
        labels = [f"Analog_{idx + 1}" for idx in range(channel_count)]

    if len(units) < len(labels):
        units.extend([""] * (len(labels) - len(units)))
    elif len(units) > len(labels):
        units = units[: len(labels)]

    return labels, analog_rate, units


def get_events(c3d_data: C3DMapping) -> list[C3DEvent]:
    event_parameters = c3d_data["parameters"].get("EVENT")
    if not event_parameters:
        return []

    labels_raw: Iterable[str] = event_parameters.get("LABELS", {}).get("value", [])
    times = event_parameters.get("TIMES", {}).get("value")
    if times is None:
        return []

    times_array = np.asarray(times)
    if times_array.ndim == 2:
        times_array = times_array[1, :]

    events: list[C3DEvent] = []
    for idx, label in enumerate(labels_raw):
        time_value = float(times_array[idx]) if idx < len(times_array) else np.nan
        if np.isfinite(time_value):
            events.append(C3DEvent(label=str(label).strip(), time=time_value))

    return events


def build_metadata(c3d_data: C3DMapping, file_path: Path) -> C3DMetadata:
    point_parameters = get_point_parameters(c3d_data, file_path)
    marker_labels = [label.strip() for label in point_parameters["LABELS"]["value"]]
    frame_count = int(point_parameters["FRAMES"]["value"][0])
    frame_rate = float(point_parameters["RATE"]["value"][0])
    units = str(point_parameters["UNITS"]["value"][0])
    analog_labels, analog_rate, analog_units = get_analog_details(c3d_data)
    events = get_events(c3d_data)
    return C3DMetadata(
        marker_labels=marker_labels,
        frame_count=frame_count,
        frame_rate=frame_rate,
        units=units,
        analog_labels=analog_labels,
        analog_units=analog_units,
        analog_rate=analog_rate,
        events=events,
    )
